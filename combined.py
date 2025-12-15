#!/usr/bin/env python3
"""
doc_diff_combined.py – unified paragraph- and table-diff for Azure Document
Intelligence OCR exports (JSON or PDF).

────────────────────────────────────────────────────────────────────────────
* First section  … AI semantic diff  (paragraph-level, page-grouped)
* Second section … Table comparison  (cell-level, new-doc order)

Usage
─────
    python doc_diff_combined.py OLD.json NEW.json \
        --out combined_report.html --tau 0.78 --thr 0.25 --debug

Dependencies
────────────
    pip install openai tenacity tqdm markdown \
                sentence-transformers spacy scipy numpy
    python -m spacy download de_core_news_sm
"""

from __future__ import annotations

# ————————————————————————————————————————————————————— imports ———
import argparse
import datetime
import hashlib
import html
import json
import logging
import os
import pathlib
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openai
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
import markdown

# ——————————————————————————————————  Shared helpers / constants ———
EMBED_MODEL        = "text-embedding-3-small"
GPT_MODEL          = "gpt-4o-mini"
MAX_PAR_TOKENS     = 8_000
ALIGN_LOG_EVERY    = 100
DEFAULT_SIM_TAU    = 0.78
DEFAULT_MIN_WORDS  = 4
DEFAULT_TABLE_THR  = 0.25

sha256 = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]


# ╭──────────────────────── 1. Paragraph-diff section ─────────────────────╮
@dataclass
class Chunk:
    id: str
    text: str
    hash: str
    emb: Optional[List[float]] = None
    embedding_failed: bool = False     # flagged if embedding API fails


@dataclass
class Diff:
    kind: str                          # added | deleted | modified
    new: Optional[Chunk] = None
    old: Optional[Chunk] = None
    details: Optional[str] = None      # GPT enrichment


def tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)      # ≈ heuristic


# ▸ OpenAI helpers
@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=20))
def embed(texts: List[str]):
    return [d.embedding for d in openai.embeddings.create(model=EMBED_MODEL, input=texts).data]


@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=20))
def gpt_diff(old: str, new: str):
    prompt = (
        "You are an expert editor. Compare the OLD and NEW paragraphs and list exact "
        "changes (additions, deletions, re-phrasings). Use concise bullet points.\n\n"
        f"OLD:\n{old}\n\nNEW:\n{new}"
    )
    r = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.0,
    )
    return r.choices[0].message.content.strip()


# ▸ File loading & normalisation
def load_pages(path: pathlib.Path, logger) -> List[str]:
    if path.suffix.lower() != ".json":
        return path.read_text("utf-8").split("\f")

    data = json.loads(path.read_text(encoding="utf-8"))
    if "analyzeResult" in data:
        ar = data["analyzeResult"]
        if "pages" in ar:                   # 2024 schema
            return [
                pg.get("content")
                or "\n".join(l["content"] for l in pg.get("lines", []))
                for pg in ar["pages"]
            ]
        elif "readResults" in ar:           # legacy schema
            return ["\n".join(l["text"] for l in pg["lines"]) for pg in ar["readResults"]]
    if "pages" in data:                     # very old preview
        return [
            pg.get("content")
            or "\n".join(l["content"] for l in pg.get("lines", []))
            for pg in data["pages"]
        ]
    return [data.get("content", "")]


def normalise(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_paragraphs(page: str) -> List[str]:
    return [p.strip() for p in re.split(r"(?:\n\s*){2,}", page) if p.strip()]


def chunk_pages(
    pages: List[str],
    label: str,
    min_words: int,
    logger,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    dropped_short = 0
    oversize_split = 0

    for pg, pg_txt in enumerate(pages, 1):
        for pr, para in enumerate(split_paragraphs(normalise(pg_txt)), 1):
            if len(para.split()) < min_words:
                dropped_short += 1
                continue
            if tokens_approx(para) > MAX_PAR_TOKENS:
                # Split oversized paragraph in half at nearest sentence boundary
                oversize_split += 1
                mid = len(para) // 2
                split_idx = para.find(". ", mid) or mid
                parts = [para[:split_idx + 1], para[split_idx + 1 :]]
                for idx, part in enumerate(parts, 1):
                    cid = f"{label}-p{pg}-para{pr}-{idx}"
                    chunks.append(Chunk(cid, part, sha256(part)))
            else:
                cid = f"{label}-p{pg}-para{pr}"
                chunks.append(Chunk(cid, para, sha256(para)))

    logger.info("%s: %d chunks kept, %d dropped (short), %d split (oversize)",
                label, len(chunks), dropped_short, oversize_split)
    return chunks


# ▸ Alignment
def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5) + 1e-9)


def align(
    old_chunks: List[Chunk],
    new_chunks: List[Chunk],
    tau: float,
    logger,
) -> List[Diff]:
    # Embeddings
    for ch in tqdm(old_chunks + new_chunks, desc="Embedding", unit="chunk",
                   disable=logger.level > logging.INFO):
        if ch.emb is not None or ch.embedding_failed:
            continue
        try:
            ch.emb = embed([ch.text])[0]
        except Exception as e:
            logger.warning("Embedding failed for %s (%s tokens) – %s",
                           ch.id, tokens_approx(ch.text), e)
            ch.embedding_failed = True

    unmatched_old = {c.id: c for c in old_chunks}
    diffs: List[Diff] = []

    for idx, nw in enumerate(tqdm(new_chunks, desc="Aligning", unit="chunk",
                                  disable=logger.level > logging.INFO), 1):
        if idx % ALIGN_LOG_EVERY == 0:
            logger.debug("Alignment progress %d/%d, unmatched_old=%d",
                         idx, len(new_chunks), len(unmatched_old))

        best_id, best_sim = None, 0.0
        for oid, od in unmatched_old.items():
            if nw.emb is None or od.emb is None:
                continue
            sim = cosine(nw.emb, od.emb)
            if sim > best_sim:
                best_sim, best_id = sim, oid
        if best_sim >= tau:
            od = unmatched_old.pop(best_id)
            if od.hash != nw.hash:
                diffs.append(Diff("modified", new=nw, old=od))
        else:
            diffs.append(Diff("added", new=nw))

    diffs.extend(Diff("deleted", old=o) for o in unmatched_old.values())
    return diffs


def enrich_with_gpt(diffs: List[Diff], logger) -> None:
    mods = [d for d in diffs if d.kind == "modified"]
    for d in tqdm(mods, desc="GPT-enrich", unit="chunk",
                  disable=logger.level > logging.INFO):
        try:
            d.details = gpt_diff(d.old.text, d.new.text)
        except Exception as e:
            logger.error("GPT diff failed for %s → %s: %s",
                         d.old.id, d.new.id, e)
            d.details = f"<GPT error: {e}>"


def pos_key(chunk_id: str) -> Tuple[int, int, int]:
    m = re.match(r".*-p(\d+)-para(\d+)(?:-(\d+))?", chunk_id)
    if not m:
        return (0, 0, 0)
    return tuple(int(x or 0) for x in m.groups())


def write_paragraph_md(diffs: List[Diff]) -> str:
    diffs_sorted = sorted(diffs, key=lambda d: pos_key((d.new or d.old).id))
    lines = ["# AI Semantic Diff Report\n"]

    current_page = None
    for d in diffs_sorted:
        page, *_ = pos_key((d.new or d.old).id)
        if page != current_page:
            current_page = page
            lines.append(f"## Page {page}\n")

        if d.kind == "added":
            flag = "⚠️ (embedding failed) " if d.new.embedding_failed else ""
            lines.append(f"* ➕ {flag}**{d.new.id}** – {d.new.text[:160]}…")
        elif d.kind == "deleted":
            flag = "⚠️ (embedding failed) " if d.old.embedding_failed else ""
            lines.append(f"* ➖ {flag}~~{d.old.id}~~ – {d.old.text[:160]}…")
        else:  # modified
            lines.append(
                f"* ✏️ **{d.old.id} → {d.new.id}**\n  {d.details}"
            )

    return "\n".join(lines)


def paragraph_diff_html(
    old_path: pathlib.Path,
    new_path: pathlib.Path,
    tau: float,
    min_words: int,
    logger,
) -> str:
    old_pages = load_pages(old_path, logger)
    new_pages = load_pages(new_path, logger)
    logger.info("Pages – old: %d, new: %d", len(old_pages), len(new_pages))

    old_chunks = chunk_pages(old_pages, "v1", min_words, logger)
    new_chunks = chunk_pages(new_pages, "v2", min_words, logger)

    diffs = align(old_chunks, new_chunks, tau, logger)
    enrich_with_gpt(diffs, logger)

    md_text = write_paragraph_md(diffs)
    body = markdown.markdown(md_text, extensions=["extra", "smarty"])
    return (
        "<h1>Paragraph-level Differences</h1>\n"
        "<link rel='stylesheet' "
        "href='https://cdn.jsdelivr.net/npm/github-markdown-css@5.4.0/"
        "github-markdown.min.css'>\n"
        "<div class='markdown-body'>\n" + body + "\n</div>"
    )


# ╭──────────────────────── 2. Table-diff section ────────────────────────╮
_model_cache: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model_cache


# ── normalisation helpers
_soft_hyphen = "\u00ad"
_join_hyphen_rx = re.compile(r"([A-Za-zÄÖÜäöüß]{2,})-\s+([A-Za-zÄÖÜäöüß]{2,})")
_hyphen_space_rx = re.compile(r"-\s+")
_space_before_punct_rx = re.compile(r"\s+([%.,;:!?)]])")


def _join_soft_hyphens(s: str) -> str:
    s = s.replace(_soft_hyphen, "")
    s = _join_hyphen_rx.sub(r"\1\2", s)
    s = _hyphen_space_rx.sub("-", s)
    return s


def _collapse_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\u00A0", " "))


def _fix_space_before_punct(s: str) -> str:
    return _space_before_punct_rx.sub(r"\1", s)


def normalize_cell(s: str) -> str:
    return _fix_space_before_punct(_collapse_whitespace(_join_soft_hyphens(s))).strip()


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


def load_tables(path: pathlib.Path) -> List[List[List[str]]]:
    return [_build_table_matrix(t) for t in _pull_tables(json.loads(path.read_text("utf-8")))]


def _table_signature(tbl: List[List[str]]) -> str:
    return " ".join(normalize_cell(c) for row in tbl for c in row if c)


def _hungarian_pairs_new_to_old(sim: np.ndarray, thr: float) -> Dict[int, int]:
    cost = 1 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: Dict[int, int] = {}
    for i_old, j_new in zip(row_ind, col_ind):
        if sim[i_old, j_new] >= thr:
            mapping[j_new] = i_old
    return mapping


def _match_table_pairs(
    new: List[List[List[str]]],
    old: List[List[List[str]]],
    thr: float,
) -> Dict[int, int]:
    if not new or not old:
        return {}
    model = _get_model()
    emb_new = model.encode([_table_signature(t) for t in new], convert_to_tensor=True)
    emb_old = model.encode([_table_signature(t) for t in old], convert_to_tensor=True)
    sim = util.cos_sim(emb_old, emb_new).cpu().numpy()  # old × new
    return _hungarian_pairs_new_to_old(sim, thr)


def _diff_table_cells(
    a: List[List[str]],
    b: List[List[str]],
) -> List[List[Tuple[str, str, str]]]:
    max_r = max(len(a), len(b))
    max_c = max(len(a[0]) if a else 0, len(b[0]) if b else 0)
    grid = []
    for r in range(max_r):
        row = []
        for c in range(max_c):
            x = a[r][c] if r < len(a) and c < len(a[r]) else ""
            y = b[r][c] if r < len(b) and c < len(b[r]) else ""
            state = "same" if normalize_cell(x) == normalize_cell(y) else "changed"
            row.append((x, y, state))
        grid.append(row)
    return grid


def diff_tables(
    old: List[List[List[str]]],
    new: List[List[List[str]]],
    thr: float,
):
    mapping = _match_table_pairs(new, old, thr)
    matched_old = set(mapping.values())

    # tables that exist in new
    for j_new, tbl_new in enumerate(new):
        if j_new in mapping:
            i_old = mapping[j_new]
            yield (j_new + 1, _diff_table_cells(old[i_old], tbl_new))
        else:
            yield (j_new + 1, _diff_table_cells([], tbl_new))

    # tables removed from old
    for i_old, tbl_old in enumerate(old):
        if i_old not in matched_old:
            yield (len(new) + i_old + 1, _diff_table_cells(tbl_old, []))


_TABLE_CSS = """
<style>
  table.diff{border-collapse:collapse;width:100%;margin:1.5rem 0;}
  table.diff th,table.diff td{border:1px solid #ccc;padding:6px;vertical-align:top;}
  td.same{}
  td.changed{background:#ffeacc;}
  del{background:#ffb3b3;text-decoration:line-through;}
  ins{background:#b3ffb3;text-decoration:none;}
</style>
"""


def _cell_html(old: str, new: str, st: str) -> str:
    if st == "same":
        return f"<td class='same'>{html.escape(new or old)}</td>"
    return (f"<td class='changed'><del>{html.escape(old)}</del><br>→ "
            f"<ins>{html.escape(new)}</ins></td>")


def tables_diff_html(
    old_path: pathlib.Path,
    new_path: pathlib.Path,
    thr: float,
) -> str:
    tbls_old = load_tables(old_path)
    tbls_new = load_tables(new_path)
    tbl_rep = list(diff_tables(tbls_old, tbls_new, thr))

    sections = []
    for idx, grid in tbl_rep:
        rows = ["<tr>" + "".join(_cell_html(o, n, st) for o, n, st in row) + "</tr>"
                for row in grid]
        sections.append(f"<h3>Table {idx}</h3><table class='diff'>\n"
                        + "".join(rows) + "\n</table>")

    return (
        "<h1>Table Comparison</h1>\n" + _TABLE_CSS + "\n"
        + "".join(sections)
    )


# ╭──────────────────────── 3. Combined driver / CLI ─────────────────────╮
def build_combined_report(
    old_path: pathlib.Path,
    new_path: pathlib.Path,
    out_html: pathlib.Path,
    tau: float,
    thr: float,
    min_words: int,
    debug: bool,
    api_key: Optional[str] = None,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(message)s" if debug else "%(message)s",
    )
    logger = logging.getLogger("doc-diff")

    # OpenAI key
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"

    # Build both sections
    logger.info("▶ AI paragraph diff …")
    ai_html = paragraph_diff_html(old_path, new_path, tau, min_words, logger)

    logger.info("▶ Table diff …")
    tbl_html = tables_diff_html(old_path, new_path, thr)

    # Compose single HTML doc
    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Document diff report</title>"
        "</head><body style='margin:2rem;'>\n"
        + ai_html
        + "<hr style='margin:4rem 0;'>\n"
        + tbl_html
        + "\n</body></html>"
    )
    out_html.write_text(html_doc, encoding="utf-8")
    logger.info("✅ Combined report written → %s", out_html.resolve())


def main() -> None:
    p = argparse.ArgumentParser(
        description="Paragraph + Table diff for Azure Document Intelligence exports"
    )
    p.add_argument("old", type=pathlib.Path, help="OLD OCR JSON (or PDF)")
    p.add_argument("new", type=pathlib.Path, help="NEW OCR JSON (or PDF)")
    p.add_argument(
        "-o", "--out",
        type=pathlib.Path,
        default="combined_report.html",
        help="Output HTML filename",
    )
    p.add_argument("--tau", type=float, default=DEFAULT_SIM_TAU,
                   help="Similarity threshold (0–1) for paragraph alignment")
    p.add_argument("--thr", type=float, default=DEFAULT_TABLE_THR,
                   help="Cosine-similarity threshold (0–1) for table matching")
    p.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS,
                   help="Paragraphs shorter than this are ignored")
    p.add_argument("--debug", action="store_true", help="Verbose logging")

    args = p.parse_args()
    build_combined_report(
        old_path=args.old,
        new_path=args.new,
        out_html=args.out,
        tau=args.tau,
        thr=args.thr,
        min_words=args.min_words,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
