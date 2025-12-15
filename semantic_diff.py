"""
Semantischer Diff – *Satzreihenfolge = new.json*
================================================
Dieses Skript vergleicht zwei OCR-JSON-Dateien von Azure Document Intelligence (ehem. Form Recognizer)
und erzeugt einen HTML-Report mit

1. **semantischem Satz-Diff** – Chronologie und Spalten sind strikt an der Reihenfolge des *neuen* Dokuments orientiert.
2. **Tabellen-Diff** auf Zellebene – ebenfalls aus Sicht des *neuen* Dokuments.

## Neuerungen (2025-07-30)
* **Keine Doppelzählung** von Tabellenzeilen im Fließtext.
* **Robuste Bounding-Box-Erkennung** – unterstützt `boundingPolygon`, `polygon` *und* das ältere `boundingBox`.
* **Smarter Matching** – Matrix-Maskierung verhindert Low-Sim-Zuordnungen; adaptiver Schwellwert.
* **Model-Caching** – `SentenceTransformer` wird nur einmal geladen.
* **Token-Level Diff** – Wort-Einfügungen/Löschungen per `<ins>/<del>`.
* **Reihenfolge fix** – Alt-|-Neu-Spalten zeigen nun exakt die Abfolge des *neuen* Dokuments; entfallene alte Sätze werden anschliessend gelistet.
"""
from __future__ import annotations

import datetime
import html
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
# Globale Helfer & Caches
###############################################################################
_nlp = spacy.load("de_core_news_sm")
_model_cache: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lädt SBERT nur einmal (Lazy-Loading)."""
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model_cache

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
    s = _join_hyphen_rx.sub(r"\1\2", s)
    s = _hyphen_space_rx.sub("-", s)
    return s


def _collapse_whitespace(s: str) -> str:
    s = s.replace("\u00A0", " ")
    return re.sub(r"\s+", " ", s)


def _fix_space_before_punct(s: str) -> str:
    return _space_before_punct_rx.sub(r"\1", s)


def normalize(s: str) -> str:
    s = _strip_bullet_prefix(s)
    s = _join_soft_hyphens(s)
    s = _collapse_whitespace(s)
    s = _fix_space_before_punct(s)
    return s.strip()

###############################################################################
# OCR-Parsing Utilities
###############################################################################

def _poly_to_points(poly: Any) -> List[Dict[str, float]]:
    if not poly:
        return []
    if isinstance(poly[0], dict):
        return poly  # type: ignore[return-value]
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


# JSON-Helper

def _pull_pages(di: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "pages" in di:
        return di["pages"]
    if "analyzeResult" in di and "pages" in di["analyzeResult"]:
        return di["analyzeResult"]["pages"]
    if "readResults" in di:
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


def _table_bboxes_by_page(tables: List[Dict[str, Any]]) -> DefaultDict[int, List[Tuple[float, float, float, float]]]:
    res: DefaultDict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
    for tbl in tables:
        for region in tbl.get("boundingRegions", []):
            page_no = region.get("pageNumber", 1)
            poly = region.get("boundingPolygon") or region.get("polygon") or region.get("boundingBox") or []
            res[page_no].append(_bbox_from_polygon(poly))
    return res


def _item_poly(item: Dict[str, Any]) -> Any:
    return item.get("boundingPolygon") or item.get("polygon") or item.get("boundingBox") or []


def _collect_line_text(page: Dict[str, Any], tbl_boxes: List[Tuple[float, float, float, float]]) -> List[str]:
    def _in_table(poly: Any) -> bool:
        if not poly:
            return False
        cx, cy = _centroid(poly)
        return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in tbl_boxes)

    if "lines" in page:
        return [ln.get("content") or ln.get("text") for ln in page["lines"] if not _in_table(_item_poly(ln))]

    if "paragraphs" in page:
        out: List[str] = []
        for para in page["paragraphs"]:
            poly_any = (para.get("boundingRegions") or [{}])[0]
            if _in_table(_item_poly(poly_any)):
                continue
            out.append(para["content"])
        return out

    if "words" in page:
        words = [w for w in page["words"] if not _in_table(_item_poly(w))]
        words_sorted = sorted(words, key=lambda w: (w["boundingBox"][1], w["boundingBox"][0])) if words and "boundingBox" in words[0] else words
        return [" ".join(w["content"] for w in words_sorted)] if words_sorted else []

    return []


def load_document(path: str | Path) -> Tuple[List[str], List[List[List[str]]]]:
    di = json.loads(Path(path).read_text(encoding="utf-8"))

    tables_raw = _pull_tables(di)
    tbl_matrices = [_build_table_matrix(t) for t in tables_raw]

    tbl_cells_norm = {normalize(c) for t in tbl_matrices for row in t for c in row if c}

    tbl_bboxes_by_page = _table_bboxes_by_page(tables_raw)

    texts: List[str] = []
    for idx, p in enumerate(_pull_pages(di), 1):
        texts.extend(_collect_line_text(p, tbl_bboxes_by_page.get(idx, [])))

    plain_text = " ".join(texts)
    plain_text = unicodedata.normalize("NFC", plain_text)
    plain_text = re.sub(r"-\s*\n", "", plain_text)
    plain_text = re.sub(r"\s*\n\s*", " ", plain_text)

    sentences = [s.text.strip() for s in _nlp(plain_text).sents if s.text.strip()]
    sentences = [s for s in sentences if normalize(s) not in tbl_cells_norm]

    return sentences, tbl_matrices

###############################################################################
# Satz-Diff (Reihenfolge = neues Dokument)
###############################################################################
def _pre_align_exact(old: List[str], new: List[str]) -> Dict[int, int]:
    """Mapping *new_idx → old_idx* für identische Normalformen (1‑zu‑1, ohne Duplikate)."""
    old_map: Dict[str, int] = {}
    for i, s in enumerate(old):
        key = normalize(s)
        # erstes Vorkommen behalten (duplikate ignorieren)
        if key not in old_map:
            old_map[key] = i
    mapping: Dict[int, int] = {}
    used_old: set[int] = set()
    for j, s in enumerate(new):
        key = normalize(s)
        if key in old_map and old_map[key] not in used_old:
            mapping[j] = old_map[key]
            used_old.add(old_map[key])
    return mapping

def _hungarian_pairs_new_to_old(sim: np.ndarray, thr: float) -> Dict[int, int]:
    """Optimal-Matching (Hungarian) → Mapping *new_idx → old_idx* über Threshold."""
    cost = 1 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: Dict[int, int] = {}
    for i_old, j_new in zip(row_ind, col_ind):
        if sim[i_old, j_new] >= thr:
            mapping[j_new] = i_old
    return mapping


def _token_diff(a: str, b: str) -> str:
    aw, bw = a.split(), b.split()
    sm = SequenceMatcher(None, aw, bw)
    parts: List[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            parts.append(" ".join(aw[i1:i2]))
        elif op == "delete":
            parts.append(f"<del>{html.escape(' '.join(aw[i1:i2]))}</del>")
        elif op == "insert":
            parts.append(f"<ins>{html.escape(' '.join(bw[j1:j2]))}</ins>")
        elif op == "replace":
            parts.append(f"<del>{html.escape(' '.join(aw[i1:i2]))}</del>")
            parts.append(f"<ins>{html.escape(' '.join(bw[j1:j2]))}</ins>")
    return " ".join(parts)


def diff_sentences(old: List[str], new: List[str], thr: float = 0.60):
    # 1) Vorab exakte Matches (Normalform)
    pre_map = _pre_align_exact(old, new)  # new→old
    matched_old = set(pre_map.values())
    matched_new = set(pre_map.keys())

    # 2) Embedding‑Similarities für verbleibende Sätze
    if len(matched_old) < len(old) and len(matched_new) < len(new):
        model = _get_model()
        emb_old = model.encode(old, convert_to_tensor=True)
        emb_new = model.encode(new, convert_to_tensor=True)
        sim = util.cos_sim(emb_old, emb_new).cpu().numpy()

        idx_old_un = [i for i in range(len(old)) if i not in matched_old]
        idx_new_un = [j for j in range(len(new)) if j not in matched_new]

        if idx_old_un and idx_new_un:
            sim_sub = sim[np.ix_(idx_old_un, idx_new_un)]
            sub_map = _hungarian_pairs_new_to_old(sim_sub, thr)  # new_sub→old_sub
            # Index‑Übersetzung sub→global
            for j_sub, i_sub in sub_map.items():
                pre_map[idx_new_un[j_sub]] = idx_old_un[i_sub]
                matched_old.add(idx_old_un[i_sub])

        score_matrix = sim  # zur Score‑Anzeige später
    else:
        score_matrix = None  # type: ignore[assignment]

    # 3) Report‑Generation in Reihenfolge *new*
    last_old_idx = -1
    for j_new, new_raw in enumerate(new):
        if j_new in pre_map:
            i_old = pre_map[j_new]
            old_raw = old[i_old]
            old_norm, new_norm = normalize(old_raw), normalize(new_raw)
            score = float(score_matrix[i_old, j_new]) if score_matrix is not None else 1.0
            equal = old_norm == new_norm
            in_order = i_old > last_old_idx
            tag = (
                "same" if equal and in_order else
                "changed" if not equal and in_order else
                "moved_same" if equal else "moved_changed"
            )
            diff_html = "" if equal else _token_diff(old_norm, new_norm)
            yield (tag, old_raw, new_raw, score, diff_html)
            last_old_idx = i_old
        else:
            # neuer Satz ohne Gegenstück
            yield ("added", "", new_raw, 0.0, "")

    # 4) Entfernte alte Sätze
    for i_old, old_raw in enumerate(old):
        if i_old not in matched_old:
            yield ("removed", old_raw, "", 0.0, "")

###############################################################################
# Tabellen-Diff (Reihenfolge = neues Dokument)
###############################################################################

def _table_signature(tbl: List[List[str]]) -> str:
    return " ".join(normalize(c) for row in tbl for c in row if c)


def _match_table_pairs(new: List[List[List[str]]], old: List[List[List[str]]], thr: float = 0.25) -> Dict[int, int]:
    if not new or not old:
        return {}
    model = _get_model()
    emb_new = model.encode([_table_signature(t) for t in new], convert_to_tensor=True)
    emb_old = model.encode([_table_signature(t) for t in old], convert_to_tensor=True)
    sim = util.cos_sim(emb_old, emb_new).cpu().numpy()  # old x new
    return _hungarian_pairs_new_to_old(sim, thr)  # new→old (Achtung sim transponiert oben)


def _diff_table_cells(a: List[List[str]], b: List[List[str]]):
    max_r = max(len(a), len(b))
    max_c = max(len(a[0]) if a else 0, len(b[0]) if b else 0)
    grid = []
    for r in range(max_r):
        row: List[Tuple[str, str, str]] = []
        for c in range(max_c):
            x = a[r][c] if r < len(a) and c < len(a[r]) else ""
            y = b[r][c] if r < len(b) and c < len(b[r]) else ""
            st = "same" if normalize(x) == normalize(y) else "changed"
            row.append((x, y, st))
        grid.append(row)
    return grid


def diff_tables(old: List[List[List[str]]], new: List[List[List[str]]], thr: float = 0.25):
    mapping = _match_table_pairs(new, old, thr)  # new→old
    matched_old = set(mapping.values())

    for j_new, tbl_new in enumerate(new):
        if j_new in mapping:
            i_old = mapping[j_new]
            yield (j_new + 1, _diff_table_cells(old[i_old], tbl_new))
        else:
            yield (j_new + 1, _diff_table_cells([], tbl_new))

    # entfernte alte Tabellen
    for i_old, tbl_old in enumerate(old):
        if i_old not in matched_old:
            yield (len(new) + i_old + 1, _diff_table_cells(tbl_old, []))

###############################################################################
# HTML-Report
###############################################################################

def write_html_report(sent_rep, tbl_rep, outfile: str = "diff_report.html") -> None:
    css = """
    <style>
      body{font-family:Arial, sans-serif;margin:2rem;}
      table{border-collapse:collapse;width:100%;margin-bottom:2rem;}
      th,td{border:1px solid #ccc;padding:6px;vertical-align:top;}
      tr.same, tr.moved_same{background:#f5f5f5;}
      tr.changed, tr.moved_changed{background:#fffbe6;}
      tr.added{background:#e8f5e9;}
      tr.removed{background:#fdecea;}
      .score{font-size:0.8em;color:#666;}
      del{background:#ffb3b3;text-decoration:line-through;}
      ins{background:#b3ffb3;text-decoration:none;}
      .cell_same{}
      .cell_changed{background:#ffeacc;}
    </style>
    """
    sent_rows = []
    for status, old, new, score, diff_html in sent_rep:
        sent_rows.append(
            f"<tr class='{status}'><td>{html.escape(old)}</td><td>{html.escape(new)}</td>"
            f"<td class='score'>{score:.2f}</td><td>{diff_html}</td></tr>")
    sent_html = (
        "<h2>Satz-Vergleich (Reihenfolge = neu)</h2><table><thead><tr><th>Alt</th><th>Neu</th>"
        "<th>Score</th><th>Änderungen</th></tr></thead><tbody>" +
        "\n".join(sent_rows) + "</tbody></table>")

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

    tbl_html = "<h2>Tabellen-Vergleich (Reihenfolge = neu)</h2>" + "".join(tbl_secs)

    Path(outfile).write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>" +
        f"<title>Semantischer Diff – {datetime.datetime.now():%Y-%m-%d %H:%M}</title>" +
        css + "</head><body>" + sent_html + tbl_html + "</body></html>",
        encoding="utf-8")

###############################################################################
# main
###############################################################################

def _pre_align_exact(old, new):
    """liefert mapping_new→old für identische Normalformen"""
    old_map = {normalize(s): i for i, s in enumerate(old)}
    mapping = {}
    for j, s in enumerate(new):
        key = normalize(s)
        if key in old_map:
            mapping[j] = old_map[key]
    # nicht-doppeltes Entfernen sicherstellen
    used_old = set()
    finally_map = {}
    for j, i in mapping.items():
        if i not in used_old:
            finally_map[j] = i
            used_old.add(i)
    return finally_map

def main(path_old: str = "old.json", path_new: str = "new.json", thr_sent: float = 0.60, thr_tbl: float = 0.25):
    sents_old, tbls_old = load_document(path_old)
    sents_new, tbls_new = load_document(path_new)

    sent_report = list(diff_sentences(sents_old, sents_new, thr=thr_sent))
    tbl_report = list(diff_tables(tbls_old, tbls_new, thr=thr_tbl))

    write_html_report(sent_report, tbl_report, "diff_report.html")
    print("✅ HTML-Report geschrieben: diff_report.html")


if __name__ == "__main__":
    main()