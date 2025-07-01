"""
Semantischer Diff – *Satzreihenfolge = new.json*
================================================
Dieses Skript vergleicht zwei OCR-JSON-Dateien von Azure Document Intelligence (ehem. Form Recognizer)
und erzeugt einen HTML-Report mit

1. **semantischem Satz-Diff** ohne Tabellen-Zeilen im Fließtext
2. **Tabellen-Diff** auf Zellebene mit semantischer Tabellen-Zuordnung

## Neuerungen
* **Keine Doppelzählung** – Textzeilen, deren Mittelpunkt in einer Tabellen-Bounding-Box liegt, werden bei der Satzanalyse übersprungen.
* **Robuste Bounding-Box-Erkennung** – unterstützt `boundingPolygon`, `polygon` *und* das ältere `boundingBox`-Array.
* **Semantisches Table-Matching** mittels SBERT-Cosine-Similarity + Hungarian Algorithmus.
"""
from __future__ import annotations

import datetime
import html
import itertools
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

import numpy as np
import spacy
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util

###############################################################################
# Regex & Normalisierung
###############################################################################
_bullet_rx = re.compile(r"^[\s\u00A0]*[-–—•][\s\u00A0]*(.*)$")
_soft_hyphen = "\u00ad"
_join_hyphen_rx = re.compile(r"([A-Za-zÄÖÜäöüß]{2,})-\s+([A-Za-zÄÖÜäöüß]{2,})")
_hyphen_space_rx = re.compile(r"-\s+")
_space_before_punct_rx = re.compile(r"\s+([%.,;:!?)]])")


def _strip_bullet_prefix(s: str) -> str:
    m = _bullet_rx.match(s)
    return m.group(1) if m else s


def _join_soft_hyphens(s: str) -> str:
    s = s.replace(_soft_hyphen, "")
    s = _join_hyphen_rx.sub(r"\1\2", s)        # Behand- lung → Behandlung
    s = _hyphen_space_rx.sub("-", s)             # Plaque- Psoriasis → Plaque-Psoriasis
    return s


def _collapse_whitespace(s: str) -> str:
    s = s.replace("\u00A0", " ")
    return re.sub(r"\s+", " ", s)


def _fix_space_before_punct(s: str) -> str:
    return _space_before_punct_rx.sub(r"\1", s)


def normalize(s: str) -> str:
    """String-Normalisierung für Satz- und Zell-Vergleich"""
    s = _strip_bullet_prefix(s)
    s = _join_soft_hyphens(s)
    s = _collapse_whitespace(s)
    s = _fix_space_before_punct(s)
    return s.strip()

###############################################################################
# OCR-Parsing Utilities
###############################################################################
_nlp = spacy.load("de_core_news_sm")

# ── Polygon / Bounding-Box Helfer ────────────────────────────────────────────

def _poly_to_points(poly: Any) -> List[Dict[str, float]]:
    """Konvertiert `polygon`, `boundingPolygon` oder `boundingBox` zu Punkten."""
    if not poly:
        return []
    # Variante 1: list[{x,y}]
    if isinstance(poly[0], dict):
        return poly  # type: ignore[return-value]
    # Variante 2: list[float] → [x0,y0,x1,y1,...]
    if isinstance(poly[0], (int, float)):
        return [{"x": poly[i], "y": poly[i + 1]} for i in range(0, len(poly), 2)]  # type: ignore[arg-type]
    return []


def _bbox_from_polygon(poly: Any) -> Tuple[float, float, float, float]:
    pts = _poly_to_points(poly)
    if not pts:
        return 0.0, 0.0, 0.0, 0.0
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _centroid(poly: Any) -> Tuple[float, float]:
    pts = _poly_to_points(poly)
    if not pts:
        return 0.0, 0.0
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)

# ── JSON-Helper ─────────────────────────────────────────────────────────────

def _pull_pages(di: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "pages" in di:
        return di["pages"]
    if "analyzeResult" in di and "pages" in di["analyzeResult"]:
        return di["analyzeResult"]["pages"]
    if "readResults" in di:  # v2.x
        return di["readResults"]
    raise KeyError("pages / readResults / analyzeResult.pages fehlt")


def _pull_tables(di: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "tables" in di:
        return di["tables"]
    if "analyzeResult" in di and "tables" in di["analyzeResult"]:
        return di["analyzeResult"]["tables"]
    return []


def _build_table_matrix(tbl: Dict[str, Any]) -> List[List[str]]:
    rc = tbl.get("rowCount") or (max(c["rowIndex"] for c in tbl["cells"]) + 1)
    cc = tbl.get("columnCount") or (max(c["columnIndex"] for c in tbl["cells"]) + 1)
    grid = [["" for _ in range(cc)] for _ in range(rc)]
    for cell in tbl["cells"]:
        r, c = cell["rowIndex"], cell["columnIndex"]
        grid[r][c] = cell.get("content") or cell.get("text") or ""
    return grid

# ── Tabellen-Bounding-Boxen pro Seite ───────────────────────────────────────

def _table_bboxes_by_page(tables: List[Dict[str, Any]]) -> DefaultDict[int, List[Tuple[float, float, float, float]]]:
    res: DefaultDict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
    for tbl in tables:
        for region in tbl.get("boundingRegions", []):
            page_no: int = region.get("pageNumber", 1)
            poly = region.get("boundingPolygon") or region.get("polygon") or region.get("boundingBox") or []
            res[page_no].append(_bbox_from_polygon(poly))
    return res

# ── Text-Sammlung außerhalb von Tabellen ───────────────────────────────────

def _item_poly(item: Dict[str, Any]) -> Any:
    return item.get("boundingPolygon") or item.get("polygon") or item.get("boundingBox") or []


def _collect_line_text(page: Dict[str, Any], tbl_boxes: List[Tuple[float, float, float, float]]) -> List[str]:
    def _in_table(poly: Any) -> bool:
        if not poly:
            return False
        cx, cy = _centroid(poly)
        return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in tbl_boxes)

    # modern Layout → lines
    if "lines" in page:
        return [ln.get("content") or ln.get("text") for ln in page["lines"] if not _in_table(_item_poly(ln))]

    # paragraphs
    if "paragraphs" in page:
        out: List[str] = []
        for para in page["paragraphs"]:
            poly_any = (para.get("boundingRegions") or [{}])[0]
            if _in_table(_item_poly(poly_any)):
                continue
            out.append(para["content"])
        return out

    # fallback words
    if "words" in page:
        words = [w for w in page["words"] if not _in_table(_item_poly(w))]
        words_sorted = sorted(words, key=lambda w: (w["boundingBox"][1], w["boundingBox"][0])) if words and "boundingBox" in words[0] else words
        return [" ".join(w["content"] for w in words_sorted)] if words_sorted else []

    return []

# ── Dokument laden ─────────────────────────────────────────────────────────

def load_document(path: str | Path) -> Tuple[List[str], List[List[List[str]]]]:
    di = json.loads(Path(path).read_text(encoding="utf-8"))

    tables_raw = _pull_tables(di)
    tbl_matrices = [_build_table_matrix(t) for t in tables_raw]

    # Normalisierte Zell-Strings zwecks späterem Fließtext-Filter
    tbl_cells_norm = {normalize(c) for t in tbl_matrices for row in t for c in row if c}

    # Bounding-Boxen nach Seite
    tbl_bboxes_by_page = _table_bboxes_by_page(tables_raw)

    # Fließtext sammeln
    pages = _pull_pages(di)
    texts: List[str] = []
    for idx, p in enumerate(pages, 1):
        texts.extend(_collect_line_text(p, tbl_bboxes_by_page.get(idx, [])))

    plain_text = " ".join(texts)
    plain_text = unicodedata.normalize("NFC", plain_text)
    plain_text = re.sub(r"-\s*\n", "", plain_text)
    plain_text = re.sub(r"\s*\n\s*", " ", plain_text)

    sentences = [s.text.strip() for s in _nlp(plain_text).sents if s.text.strip()]
    sentences = [s for s in sentences if normalize(s) not in tbl_cells_norm]

    return sentences, tbl_matrices

###############################################################################
# Satz-Diff
###############################################################################

def _assignment_pairs(sim: np.ndarray, thr: float) -> Dict[int, int]:
    cost = 1 - sim
    cost[sim < thr] = 2.0  # hohe Kosten unter Schwelle
    r, c = linear_sum_assignment(cost)
    return {j: i for i, j in zip(r, c) if sim[i, j] >= thr}


def diff_sentences(old: List[str], new: List[str], thr: float = 0.70):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb_old = model.encode(old, convert_to_tensor=True)
    emb_new = model.encode(new, convert_to_tensor=True)
    sim = util.cos_sim(emb_old, emb_new).cpu().numpy()

    pair = _assignment_pairs(sim, thr)
    used_old = set(pair.values())
    last_old = -1
    for j, new_raw in enumerate(new):
        if j not in pair:
            yield ("added", "", new_raw, 0.0)
            continue
        i = pair[j]
        for k in range(last_old + 1, i):
            if k not in used_old:
                yield ("unmatched", old[k], "", 0.0)
        old_raw = old[i]
        old_norm, new_norm = normalize(old_raw), normalize(new_raw)
        score = float(sim[i, j])
        in_order = i > last_old
        equal = old_norm == new_norm
        tag = ("same" if equal else "changed") if in_order else ("moved_same" if equal else "moved_changed")
        if tag.endswith("changed"):
            diff = [{"tag": t, "a_fragment": old_norm[a1:a2], "b_fragment": new_norm[b1:b2]}
                    for t, a1, a2, b1, b2 in SequenceMatcher(None, old_norm, new_norm).get_opcodes() if t != "equal"]
            if not diff:
                tag = tag.replace("changed", "same")
                yield (tag, old_raw, new_raw, score)
            else:
                yield (tag, old_raw, new_raw, score, diff)
        else:
            yield (tag, old_raw, new_raw, score)
        last_old = max(last_old, i)
    for k in range(last_old + 1, len(old)):
        if k not in used_old:
            yield ("unmatched", old[k], "", 0.0)

###############################################################################
# Tabellen-Diff
###############################################################################

def _table_signature(tbl: List[List[str]]) -> str:
    return " ".join(normalize(c) for row in tbl for c in row if c)


def _match_table_pairs(old: List[List[List[str]]], new: List[List[List[str]]], thr: float = 0.30) -> Dict[int, int]:
    if not old or not new:
        return {}
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb_old = model.encode([_table_signature(t) for t in old], convert_to_tensor=True)
    emb_new = model.encode([_table_signature(t) for t in new], convert_to_tensor=True)
    sim = util.cos_sim(emb_old, emb_new).cpu().numpy()
    return _assignment_pairs(sim, thr)  # Mapping: new-idx → old-idx


def _diff_table_cells(a: List[List[str]], b: List[List[str]]):
    max_r = max(len(a), len(b))
    max_c = max(len(a[0]) if a else 0, len(b[0]) if b else 0)
    grid = []
    for r in range(max_r):
        row: List[Tuple[str, str, str]] = []
        for c in range(max_c):
            x = a[r][c] if r < len(a) and c < len(a[r]) else ""
            y = b[r][c] if r < len(b) and c < len(b[r]) else ""
            status = "same" if normalize(x) == normalize(y) else "changed"
            row.append((x, y, status))
        grid.append(row)
    return grid


def diff_tables(old: List[List[List[str]]], new: List[List[List[str]]], thr: float = 0.30):
    pair = _match_table_pairs(old, new, thr)
    matched_old = set(pair.values())

    # Reihenfolge orientiert sich an new.json
    for j, tbl_new in enumerate(new):
        if j in pair:
            i = pair[j]
            yield (j + 1, _diff_table_cells(old[i], tbl_new))
        else:
            yield (j + 1, _diff_table_cells([], tbl_new))  # neue Tabelle

    # alte Tabellen ohne Match anhängen
    for i, tbl_old in enumerate(old):
        if i not in matched_old:
            yield (len(new) + i + 1, _diff_table_cells(tbl_old, []))

###############################################################################
# HTML-Report
###############################################################################

def write_html_report(sent_rep, tbl_rep, outfile: str = "diff_report.html") -> None:
    css = """
    <style>
      body{font-family:Arial, sans-serif;margin:2rem;}
      table{border-collapse:collapse;width:100%;margin-bottom:2rem;}
      th,td{border:1px solid #ccc;padding:6px;vertical-align:top;}
      tr.same,tr.moved_same{background:#f5f5f5;}
      tr.changed,tr.moved_changed{background:#fffbe6;}
      tr.added{background:#e8f5e9;}
      tr.unmatched{background:#fdecea;}
      .score{font-size:0.8em;color:#666;}
      del{background:#ffb3b3;text-decoration:line-through;}
      ins{background:#b3ffb3;text-decoration:none;}
      .cell_same{}
      .cell_changed{background:#ffeacc;}
    </style>
    """
    # Satz-Tabelle
    sent_rows = []
    for rec in sent_rep:
        status, old, new, score, *rest = rec
        changes = rest[0] if rest else []
        diff_html = ("<ul>" + "".join(
            f"<li><b>{html.escape(c['tag'])}</b>: <del>{html.escape(c['a_fragment'])}</del> → <ins>{html.escape(c['b_fragment'])}</ins></li>" for c in changes) + "</ul>") if changes else ""
        sent_rows.append(f"<tr class='{status}'><td>{html.escape(old)}</td><td>{html.escape(new)}</td><td class='score'>{score:.2f}</td><td>{diff_html}</td></tr>")
    sent_html = "<h2>Satz-Vergleich</h2><table><thead><tr><th>Alt</th><th>Neu</th><th>Score</th><th>Änderungen</th></tr></thead><tbody>" + "\n".join(sent_rows) + "</tbody></table>"

    # Tabellen-Sektionen
    tbl_secs = []
    for idx, grid in tbl_rep:
        rows_html = []
        for r in grid:
            cells_html = []
            for o, n, st in r:
                if st == "same":
                    cells_html.append(f"<td class='cell_same'>{html.escape(n or o)}</td>")
                else:
                    cells_html.append(f"<td class='cell_changed'><del>{html.escape(o)}</del><br/>→ <ins>{html.escape(n)}</ins></td>")
            rows_html.append("<tr>" + "".join(cells_html) + "</tr>")
        tbl_secs.append(f"<h3>Tabelle {idx}</h3><table>{''.join(rows_html)}</table>")

    tbl_html = "<h2>Tabellen-Vergleich</h2>" + "".join(tbl_secs)

    Path(outfile).write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Semantischer Diff – {datetime.datetime.now():%Y-%m-%d %H:%M}</title>" + css + "</head><body>" + sent_html + tbl_html + "</body></html>",
        encoding="utf-8")

###############################################################################
# main
###############################################################################

def main(path_old: str = "old.json", path_new: str = "new.json", thr_sent: float = 0.75, thr_tbl: float = 0.30):
    sents_old, tbls_old = load_document(path_old)
    sents_new, tbls_new = load_document(path_new)

    sent_report = list(diff_sentences(sents_old, sents_new, thr=thr_sent))
    tbl_report = list(diff_tables(tbls_old, tbls_new, thr=thr_tbl))

    write_html_report(sent_report, tbl_report, "diff_report.html")
    print("✅ HTML-Report geschrieben: diff_report.html")


if __name__ == "__main__":
    main()
