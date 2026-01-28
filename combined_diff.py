#!/usr/bin/env python3
"""
Doc Diff Combined – unified sentence- and table-diff for Azure Document Intelligence OCR
exports (JSON).

Supports both classic `analyzeResult` JSON and the newer (2025-11-01) wrapper format
`[{..., result: { contents: [...] }}]` as produced by `prebuilt-layout`.

This refactored version drops all command-line handling so the module can be
imported and called directly from other Python code. The public entry point is
``diff_documents``; all previous parameters of the CLI are now regular function
arguments with sensible defaults.

Example
-------
```python
from doc_diff_combined_refactored import diff_documents

diff_documents(
    old="v1.json",
    new="v2.json",
    out="combined_report.html",
    tau=0.78,
    thr=0.25,
    min_words=4,
    debug=False,
    api_key="sk-...",  # or set OPENAI_API_KEY in the environment
)
```

Dependencies
------------
``pip install openai tenacity tqdm markdown sentence-transformers spacy scipy numpy``

If you plan to diff German documents you will also need a German spaCy model:
``python -m spacy download de_core_news_sm``
"""

from __future__ import annotations

# ————————————————————————————————————————————————————— imports ———
from collections import defaultdict
import bisect
import datetime
from difflib import SequenceMatcher
import hashlib
import html
import json
import logging
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # type: ignore

# ——————————————————————————————————  Shared helpers / constants ———
EMBED_MODEL        = "text-embedding-3-small"
GPT_MODEL          = "gpt-4o-mini"
MAX_PAR_TOKENS     = 8_000
ALIGN_LOG_EVERY    = 100
DEFAULT_SIM_TAU    = 0.78
DEFAULT_MIN_WORDS  = 4
DEFAULT_TABLE_THR  = 0.25
DEFAULT_POST_MATCH_WINDOW = 3
DEFAULT_POST_MATCH_SIM = 0.80
DEFAULT_POST_MATCH_CONTAIN = 0.75

sha256 = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]


# ╭──────────────────────── 1. Paragraph-diff section ─────────────────────╮
@dataclass
class Chunk:
    id: str
    page: int
    order: int
    para_id: str
    para_order: int
    text: str
    norm: str
    hash: str
    emb: Optional[List[float]] = None
    embedding_failed: bool = False     # flagged if embedding API fails


@dataclass
class ParagraphBlock:
    id: str
    page: int
    order: int
    text: str
    source_index: int | None = None


@dataclass
class Diff:
    kind: str                          # same | added | deleted | modified
    new: Optional[Chunk] = None
    old: Optional[Chunk] = None
    sim: Optional[float] = None
    details: Optional[str] = None      # GPT enrichment


def tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)      # ≈ heuristic


# ▸ OpenAI helpers (client-scoped for thread safety)
@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=20))
def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    return [d.embedding for d in client.embeddings.create(model=EMBED_MODEL, input=texts).data]


@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=20))
def gpt_diff(client: OpenAI, old: str, new: str) -> str:
    prompt = (
        "You are an expert editor. Compare the OLD and NEW paragraphs and list exact "
        "changes (additions, deletions, re-phrasings). Use concise bullet points.\n\n"
        f"OLD:\n{old}\n\nNEW:\n{new}"
    )
    r = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.0,
    )
    return r.choices[0].message.content.strip()


# ▸ File loading & normalisation (supports multiple Azure DI schemas)

_SOURCE_PAGE_RX = re.compile(r"^[A-Z]\((\d+),")


def _page_from_source(source: str | None) -> int:
    if not source:
        return 0
    m = _SOURCE_PAGE_RX.match(source.strip())
    return int(m.group(1)) if m else 0


def _load_di_root(path: pathlib.Path) -> Dict[str, Any]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))

    # Newer Azure DI output (2025-11-01): list[{..., result: {contents: [...]}}]
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            res = first.get("result")
            if isinstance(res, dict):
                contents = res.get("contents")
                if isinstance(contents, list) and contents:
                    return contents[0]
                return res

    # Classic output: { analyzeResult: { pages, paragraphs, tables, content, ... } }
    if isinstance(data, dict):
        if "analyzeResult" in data and isinstance(data["analyzeResult"], dict):
            return data["analyzeResult"]
        # Some tools already store the "inner" object.
        if any(k in data for k in ("pages", "paragraphs", "tables", "content")):
            return data

    raise ValueError(f"Unrecognised Azure DI JSON structure: {path}")


def _base_text(di: Dict[str, Any]) -> str:
    if isinstance(di.get("markdown"), str):
        return di["markdown"]
    if isinstance(di.get("content"), str):
        return di["content"]
    return ""


def _span_slices(span_or_spans: Any) -> List[Tuple[int, int]]:
    if isinstance(span_or_spans, dict):
        off = span_or_spans.get("offset")
        ln = span_or_spans.get("length")
        if isinstance(off, int) and isinstance(ln, int) and off >= 0 and ln > 0:
            return [(off, ln)]
        return []
    if isinstance(span_or_spans, list):
        out: list[Tuple[int, int]] = []
        for sp in span_or_spans:
            out.extend(_span_slices(sp))
        return out
    return []


def _text_from_spans(di: Dict[str, Any], span_or_spans: Any) -> str:
    base = _base_text(di)
    if not base:
        return ""
    parts: list[str] = []
    for off, ln in _span_slices(span_or_spans):
        if off >= len(base):
            continue
        parts.append(base[off : off + ln])
    return "".join(parts)


def _item_text(di: Dict[str, Any], item: Dict[str, Any]) -> str:
    for key in ("content", "text"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val
    if "span" in item:
        return _text_from_spans(di, item.get("span"))
    if "spans" in item:
        return _text_from_spans(di, item.get("spans"))
    return ""


def _table_paragraph_indices(tables: List[Dict[str, Any]]) -> set[int]:
    indices: set[int] = set()
    for tbl in tables:
        for cell in tbl.get("cells", []) or []:
            for el in cell.get("elements", []) or []:
                if not isinstance(el, str):
                    continue
                m = re.search(r"/paragraphs/(\d+)", el)
                if m:
                    indices.add(int(m.group(1)))
    return indices


def _paragraph_page(paragraph: Dict[str, Any]) -> int:
    br = paragraph.get("boundingRegions")
    if isinstance(br, list) and br and isinstance(br[0], dict) and "pageNumber" in br[0]:
        return int(br[0].get("pageNumber") or 0)
    return _page_from_source(paragraph.get("source"))


_EXCLUDED_ROLES = {"pageHeader", "pageFooter", "pageNumber"}


def extract_pages_text(di: Dict[str, Any], logger) -> List[Tuple[int, str]]:
    paragraphs = di.get("paragraphs") if isinstance(di.get("paragraphs"), list) else []
    tables = di.get("tables") if isinstance(di.get("tables"), list) else []

    # Prefer paragraphs (more stable than line grouping) and drop table-contained paragraphs.
    by_page: dict[int, list[str]] = defaultdict(list)
    if paragraphs:
        tbl_para_idx = _table_paragraph_indices(tables)
        for idx, pr in enumerate(paragraphs):
            if idx in tbl_para_idx:
                continue
            if not isinstance(pr, dict):
                continue
            if pr.get("role") in _EXCLUDED_ROLES:
                continue
            content = _item_text(di, pr).strip()
            if not content:
                continue
            page = _paragraph_page(pr) or 0
            by_page[page].append(content)

    if by_page:
        pages = sorted(by_page.items(), key=lambda kv: kv[0])
        # drop unknown page 0 if we also have real page numbers
        if len(pages) > 1 and pages[0][0] == 0:
            pages = pages[1:]
        return [(p, "\n".join(parts)) for p, parts in pages]

    # Fallback: extract from page lines/content.
    pages = di.get("pages") if isinstance(di.get("pages"), list) else []
    out: list[Tuple[int, str]] = []
    for idx, pg in enumerate(pages, 1):
        if not isinstance(pg, dict):
            continue
        page_no = int(pg.get("pageNumber") or pg.get("page") or idx)
        if "lines" in pg and isinstance(pg["lines"], list):
            lines = []
            for ln in pg["lines"]:
                if not isinstance(ln, dict):
                    continue
                txt = _item_text(di, ln).strip()
                if txt:
                    lines.append(txt)
            if lines:
                out.append((page_no, "\n".join(lines)))
                continue
            page_txt = _text_from_spans(di, pg.get("spans"))
            out.append((page_no, page_txt))
        elif isinstance(pg.get("content"), str):
            out.append((page_no, pg["content"]))
        else:
            page_txt = _text_from_spans(di, pg.get("spans"))
            if not page_txt and isinstance(pg.get("words"), list):
                words = []
                for w in pg["words"]:
                    if isinstance(w, dict):
                        txt = _item_text(di, w).strip()
                        if txt:
                            words.append(txt)
                page_txt = " ".join(words)
            out.append((page_no, page_txt))
    logger.info("Pages extracted via fallback pages[] – %d pages", len(out))
    if out and all(not t.strip() for _, t in out):
        logger.warning(
            "All extracted pages are empty; ensure the JSON includes text fields "
            "or a root `content`/`markdown` string with spans."
        )
    return out


def _split_page_into_paragraphs(text: str) -> List[str]:
    raw = text.strip()
    if not raw:
        return []
    blocks = [b.strip() for b in re.split(r"\n{2,}", raw) if b.strip()]
    if len(blocks) > 1:
        return blocks
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) > 1:
        return lines
    return [raw]


def extract_paragraph_blocks(di: Dict[str, Any], logger) -> List[ParagraphBlock]:
    paragraphs = di.get("paragraphs") if isinstance(di.get("paragraphs"), list) else []
    tables = di.get("tables") if isinstance(di.get("tables"), list) else []

    blocks: list[ParagraphBlock] = []
    if paragraphs:
        tbl_para_idx = _table_paragraph_indices(tables)
        order_by_page: dict[int, int] = defaultdict(int)
        for idx, pr in enumerate(paragraphs):
            if idx in tbl_para_idx:
                continue
            if not isinstance(pr, dict):
                continue
            if pr.get("role") in _EXCLUDED_ROLES:
                continue
            content = _item_text(di, pr).strip()
            if not content:
                continue
            page = _paragraph_page(pr) or 0
            order_by_page[page] += 1
            pid = f"p{page}-para{order_by_page[page]}"
            blocks.append(ParagraphBlock(pid, page, order_by_page[page], content, idx))

        if blocks:
            pages = {b.page for b in blocks}
            if len(pages) > 1 and 0 in pages:
                blocks = [b for b in blocks if b.page != 0]
            blocks.sort(key=lambda b: (b.page, b.order))
            return blocks

    # Fallback: split per page text into paragraph-like blocks.
    pages = extract_pages_text(di, logger)
    order_by_page: dict[int, int] = defaultdict(int)
    for page_no, page_txt in pages:
        for para in _split_page_into_paragraphs(page_txt):
            order_by_page[page_no] += 1
            pid = f"p{page_no}-para{order_by_page[page_no]}"
            blocks.append(ParagraphBlock(pid, page_no, order_by_page[page_no], para, None))
    logger.info("Paragraphs extracted via fallback pages[] – %d blocks", len(blocks))
    return blocks

_soft_hyphen = "\u00ad"
# Normalize hyphen + whitespace between word parts (often OCR line breaks).
_join_hyphen_break_rx = re.compile(r"(\w)-\s+(\w)")
_space_before_punct_rx = re.compile(r"\s+([%.,;:!?\)\]])")

def _normalize_hyphen_breaks(text: str) -> str:
    def _repl(m: re.Match[str]) -> str:
        left = m.group(1)
        right = m.group(2)
        # If the next part starts lowercase, treat as discretionary hyphenation.
        if right.islower():
            return f"{left}{right}"
        # Otherwise keep the hyphen but drop whitespace.
        return f"{left}-{right}"
    return _join_hyphen_break_rx.sub(_repl, text)


def normalize_text(text: str) -> str:
    text = text.replace(_soft_hyphen, "")
    text = _normalize_hyphen_breaks(text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = _space_before_punct_rx.sub(r"\1", text)
    return text.strip()

def normalize_for_segmentation(text: str) -> str:
    text = text.replace(_soft_hyphen, "")
    text = _normalize_hyphen_breaks(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return text.strip()


_sentencizer = None


def _get_sentencizer():
    global _sentencizer
    if _sentencizer is not None:
        return _sentencizer
    if spacy is None:  # pragma: no cover
        _sentencizer = None
        return None
    nlp = spacy.blank("xx")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    _sentencizer = nlp
    return _sentencizer


def split_sentences(text: str) -> List[str]:
    text = normalize_for_segmentation(text)
    if not text:
        return []
    nlp = _get_sentencizer()
    blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]
    out: list[str] = []
    numbered_heading_rx = re.compile(r"^\d+(?:\.\d+)*\.\s+\S|^\d+(?:\.\d+)+\s+\S")
    for block in blocks:
        for line in (ln.strip() for ln in block.splitlines()):
            if not line:
                continue
            # Keep section headings like "3. DARREICHUNGSFORM" intact; sentence tokenizers
            # tend to split after the numeric prefix ("3."), which would drop headings.
            if numbered_heading_rx.match(line):
                out.append(line)
                continue
            if nlp is None:  # pragma: no cover
                out.extend([s.strip() for s in re.split(r"(?<=[.!?])\\s+", line) if s.strip()])
            else:
                doc = nlp(line)
                out.extend([s.text.strip() for s in doc.sents if s.text.strip()])
    return out


def chunk_pages(
    paragraphs: List[ParagraphBlock],
    label: str,
    min_words: int,
    logger,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    dropped_short = 0
    oversize_split = 0
    order_by_page: dict[int, int] = defaultdict(int)

    for para in paragraphs:
        page_no = para.page
        for sent in split_sentences(para.text):
            norm = normalize_text(sent)
            if not norm:
                dropped_short += 1
                continue
            words = norm.split()
            has_letter = any(ch.isalpha() for ch in norm)
            if (len(words) < min_words) and not (has_letter and len(norm) >= 4):
                dropped_short += 1
                continue
            if tokens_approx(norm) > MAX_PAR_TOKENS:
                # Split oversized sentence in half at nearest punctuation boundary
                oversize_split += 1
                mid = len(sent) // 2
                split_idx = max(
                    sent.rfind(". ", 0, mid),
                    sent.rfind("! ", 0, mid),
                    sent.rfind("? ", 0, mid),
                )
                if split_idx <= 0:
                    split_idx = mid
                parts = [sent[: split_idx + 1], sent[split_idx + 1 :]]
                for idx, part in enumerate(parts, 1):
                    order_by_page[page_no] += 1
                    norm = normalize_text(part)
                    cid = f"{label}-p{page_no}-sent{order_by_page[page_no]}-{idx}"
                    chunks.append(
                        Chunk(
                            cid,
                            page_no,
                            order_by_page[page_no],
                            para.id,
                            para.order,
                            part,
                            norm,
                            sha256(norm),
                        )
                    )
            else:
                order_by_page[page_no] += 1
                cid = f"{label}-p{page_no}-sent{order_by_page[page_no]}"
                chunks.append(
                    Chunk(
                        cid,
                        page_no,
                        order_by_page[page_no],
                        para.id,
                        para.order,
                        sent,
                        norm,
                        sha256(norm),
                    )
                )

    logger.info("%s: %d chunks kept, %d dropped (short), %d split (oversize)",
                label, len(chunks), dropped_short, oversize_split)
    return chunks


# ▸ Alignment
def _assign_embeddings(
    client: OpenAI,
    chunks: List[Chunk],
    logger,
    *,
    batch_size: int = 96,
) -> None:
    pending = [c for c in chunks if c.emb is None and not c.embedding_failed]
    if not pending:
        return

    texts = [c.norm for c in pending]
    total = len(texts)
    logger.info("Embedding %d chunks (batch_size=%d)", total, batch_size)

    for start in tqdm(
        range(0, total, batch_size),
        desc="Embedding",
        unit="batch",
        disable=logger.level > logging.INFO,
    ):
        batch_texts = texts[start : start + batch_size]
        try:
            batch_embs = embed_batch(client, batch_texts)
            if len(batch_embs) != len(batch_texts):
                raise RuntimeError(f"Embedding length mismatch: {len(batch_embs)} != {len(batch_texts)}")
        except Exception as e:
            logger.warning(
                "Embedding batch failed (%d items) – falling back to single calls: %s",
                len(batch_texts),
                e,
            )
            batch_embs = []
            for t in batch_texts:
                try:
                    batch_embs.append(embed_batch(client, [t])[0])
                except Exception as e2:
                    logger.warning("Embedding failed for %d tokens – %s", tokens_approx(t), e2)
                    batch_embs.append(None)

        for off, emb in enumerate(batch_embs):
            ch = pending[start + off]
            if emb is None:
                ch.embedding_failed = True
            else:
                ch.emb = emb


def align(
    client: OpenAI,
    old_chunks: List[Chunk],
    new_chunks: List[Chunk],
    tau: float,
    logger,
) -> List[Diff]:
    _assign_embeddings(client, old_chunks + new_chunks, logger)

    old_ok = [c for c in old_chunks if c.emb is not None]
    new_ok = [c for c in new_chunks if c.emb is not None]

    diffs: List[Diff] = []

    if old_ok and new_ok:
        old_mat = np.asarray([c.emb for c in old_ok], dtype=np.float32)
        new_mat = np.asarray([c.emb for c in new_ok], dtype=np.float32)

        old_norm = old_mat / (np.linalg.norm(old_mat, axis=1, keepdims=True) + 1e-9)
        new_norm = new_mat / (np.linalg.norm(new_mat, axis=1, keepdims=True) + 1e-9)

        unmatched_old = np.ones(len(old_ok), dtype=bool)

        for idx, nw in enumerate(
            tqdm(new_ok, desc="Aligning", unit="chunk", disable=logger.level > logging.INFO),
            1,
        ):
            if idx % ALIGN_LOG_EVERY == 0:
                logger.debug(
                    "Alignment progress %d/%d, unmatched_old=%d",
                    idx,
                    len(new_ok),
                    int(unmatched_old.sum()),
                )

            sims = old_norm @ new_norm[idx - 1]
            sims = np.where(unmatched_old, sims, -np.inf)
            best_i = int(np.argmax(sims)) if sims.size else -1
            best_sim = float(sims[best_i]) if best_i >= 0 else -1.0

            if best_sim >= tau and best_i >= 0:
                unmatched_old[best_i] = False
                od = old_ok[best_i]
                if od.hash != nw.hash:
                    diffs.append(Diff("modified", new=nw, old=od))
            else:
                diffs.append(Diff("added", new=nw))

        for i, od in enumerate(old_ok):
            if unmatched_old[i]:
                diffs.append(Diff("deleted", old=od))

    else:
        diffs.extend(Diff("added", new=nw) for nw in new_ok)
        diffs.extend(Diff("deleted", old=od) for od in old_ok)

    # Items we could not embed are still reported deterministically.
    diffs.extend(Diff("added", new=c) for c in new_chunks if c.emb is None)
    diffs.extend(Diff("deleted", old=c) for c in old_chunks if c.emb is None)

    return diffs


def enrich_with_gpt(client: OpenAI, diffs: List[Diff], logger, *, max_items: int | None = 200) -> None:
    mods = [d for d in diffs if d.kind == "modified"]
    if max_items is not None:
        mods = mods[:max_items]
    for d in tqdm(mods, desc="GPT-enrich", unit="chunk",
                  disable=logger.level > logging.INFO):
        try:
            d.details = gpt_diff(client, d.old.text, d.new.text)
        except Exception as e:
            logger.error("GPT diff failed for %s → %s: %s",
                         d.old.id, d.new.id, e)
            d.details = f"<GPT error: {e}>"


def align_dp(
    client: OpenAI,
    old_chunks: List[Chunk],
    new_chunks: List[Chunk],
    tau: float,
    logger,
    *,
    gap_penalty: float = -0.15,
) -> List[Diff]:
    """Order-preserving sequence alignment for a 2-column diff table.

    Produces a single alignment over (old, new) that:
    - keeps old sequence intact (left column)
    - inserts new-only chunks where they appear in new (right column)
    """

    old_seq = sorted(old_chunks, key=lambda c: (c.page, c.order))
    new_seq = sorted(new_chunks, key=lambda c: (c.page, c.order))

    _assign_embeddings(client, old_seq + new_seq, logger)

    n, m = len(old_seq), len(new_seq)
    if n == 0 and m == 0:
        return []

    sim = np.full((n, m), -np.inf, dtype=np.float32)

    # Exact matches (even if embeddings failed)
    by_hash_new: dict[str, list[int]] = defaultdict(list)
    for j, c in enumerate(new_seq):
        by_hash_new[c.hash].append(j)
    for i, c in enumerate(old_seq):
        js = by_hash_new.get(c.hash)
        if js:
            sim[i, js] = 1.0

    # Embedding similarities (cosine via normalized dot product)
    idx_old = [i for i, c in enumerate(old_seq) if c.emb is not None]
    idx_new = [j for j, c in enumerate(new_seq) if c.emb is not None]
    if idx_old and idx_new:
        old_mat = np.asarray([old_seq[i].emb for i in idx_old], dtype=np.float32)
        new_mat = np.asarray([new_seq[j].emb for j in idx_new], dtype=np.float32)
        old_norm = old_mat / (np.linalg.norm(old_mat, axis=1, keepdims=True) + 1e-9)
        new_norm = new_mat / (np.linalg.norm(new_mat, axis=1, keepdims=True) + 1e-9)
        sim_sub = old_norm @ new_norm.T
        ix = np.ix_(idx_old, idx_new)
        sim[ix] = np.maximum(sim[ix], sim_sub)

    NEG = np.float32(-1e9)
    match = np.where(sim >= np.float32(tau), sim, NEG)

    dp = np.full((n + 1, m + 1), NEG, dtype=np.float32)
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)  # 1=diag,2=up(del old),3=left(ins new)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + np.float32(gap_penalty)
        trace[i, 0] = 2
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + np.float32(gap_penalty)
        trace[0, j] = 3

    for i in range(1, n + 1):
        dpi = dp[i]
        dpim1 = dp[i - 1]
        for j in range(1, m + 1):
            diag = dpim1[j - 1] + match[i - 1, j - 1]
            up = dpim1[j] + np.float32(gap_penalty)
            left = dpi[j - 1] + np.float32(gap_penalty)
            if diag >= up and diag >= left:
                dpi[j] = diag
                trace[i, j] = 1
            elif up >= left:
                dpi[j] = up
                trace[i, j] = 2
            else:
                dpi[j] = left
                trace[i, j] = 3

    # backtrace
    out: list[Diff] = []
    i, j = n, m
    while i > 0 or j > 0:
        t = trace[i, j]
        if t == 1:
            od = old_seq[i - 1]
            nw = new_seq[j - 1]
            out.append(
                Diff(
                    "same" if od.hash == nw.hash else "modified",
                    old=od,
                    new=nw,
                    sim=float(sim[i - 1, j - 1]),
                )
            )
            i -= 1
            j -= 1
        elif t == 2:
            out.append(Diff("deleted", old=old_seq[i - 1]))
            i -= 1
        else:
            out.append(Diff("added", new=new_seq[j - 1]))
            j -= 1
    out.reverse()
    return out


_cmp_strip_rx = re.compile(r"[^\w\s]")
_heading_rx = re.compile(r"^\s*(?:#+\s*)?\d+(?:\.\d+)*\.\s+\S")


def _compare_key(text: str) -> str:
    base = normalize_text(text).lower()
    base = _cmp_strip_rx.sub("", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def _containment_ratio(old_cmp: str, new_cmp: str) -> float:
    if not old_cmp:
        return 0.0
    sm = SequenceMatcher(None, old_cmp, new_cmp)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / max(1, len(old_cmp))


def _pages_close(old: Chunk, new: Chunk, *, max_delta: int = 1) -> bool:
    if old.page <= 0 or new.page <= 0:
        return True
    return abs(old.page - new.page) <= max_delta


def _is_heading_text(text: str) -> bool:
    if not text:
        return False
    s = normalize_text(text)
    if not s:
        return False
    if _heading_rx.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 6:
        upper = sum(1 for c in letters if c.isupper())
        if upper / len(letters) >= 0.7:
            return True
    return False


def postprocess_unmatched_pairs(
    diffs: List[Diff],
    logger,
    *,
    window: int = DEFAULT_POST_MATCH_WINDOW,
    sim: float = DEFAULT_POST_MATCH_SIM,
    containment: float = DEFAULT_POST_MATCH_CONTAIN,
    same_page: bool = True,
) -> List[Diff]:
    if window <= 0 or not diffs:
        return diffs

    used: set[int] = set()
    pairs: dict[int, tuple[int, Diff]] = {}
    skip: set[int] = set()

    for i, d in enumerate(diffs):
        if i in used:
            continue
        if d.kind not in ("added", "deleted"):
            continue

        best_j = None
        best_ratio = 0.0
        best_contain = 0.0

        for j in range(i + 1, min(len(diffs), i + window + 1)):
            if j in used:
                continue
            cand = diffs[j]
            if cand.kind == d.kind or cand.kind not in ("added", "deleted"):
                continue

            if d.kind == "deleted":
                old = d.old
                new = cand.new
            else:
                old = cand.old
                new = d.new

            if old is None or new is None:
                continue
            if same_page and not _pages_close(old, new, max_delta=1):
                continue

            old_cmp = _compare_key(old.text)
            new_cmp = _compare_key(new.text)
            if not old_cmp or not new_cmp:
                continue

            ratio = SequenceMatcher(None, old_cmp, new_cmp).ratio()
            contain = _containment_ratio(old_cmp, new_cmp)
            substring = old_cmp in new_cmp and (len(old_cmp) / max(1, len(new_cmp)) >= 0.5)

            if ratio < sim and contain < containment and not substring:
                continue

            if (ratio > best_ratio) or (ratio == best_ratio and contain > best_contain):
                best_j = j
                best_ratio = ratio
                best_contain = contain

        if best_j is not None:
            used.add(i)
            used.add(best_j)
            other = diffs[best_j]
            if d.kind == "deleted":
                old = d.old
                new = other.new
            else:
                old = other.old
                new = d.new
            merged = Diff("modified", old=old, new=new, sim=best_ratio)
            pairs[i] = (best_j, merged)
            skip.add(best_j)
            logger.debug(
                "Post-match paired %s with %s (ratio=%.3f, contain=%.3f, window=%d)",
                old.id if old else "?",
                new.id if new else "?",
                best_ratio,
                best_contain,
                best_j - i,
            )

    if not pairs:
        return diffs

    out: list[Diff] = []
    for i, d in enumerate(diffs):
        if i in pairs:
            out.append(pairs[i][1])
            continue
        if i in skip:
            continue
        out.append(d)
    return out


_word_start_rx = re.compile(r"\w")


def _is_hyphen_break(left: str, right: str | None) -> bool:
    if not right:
        return False
    if not left.endswith("-") or len(left) < 2:
        return False
    if not _word_start_rx.search(left[:-1]):
        return False
    return bool(_word_start_rx.match(right))


def _group_tokens_for_diff(text: str) -> list[tuple[str, str]]:
    tokens = text.split()
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if _is_hyphen_break(tok, nxt):
            display = f"{tok} {nxt}"
            if nxt and nxt[0].islower():
                compare = f"{tok[:-1]}{nxt}"
            else:
                compare = f"{tok}{nxt}"
            compare = compare.replace(_soft_hyphen, "")
            out.append((display, compare))
            i += 2
            continue
        out.append((tok, tok.replace(_soft_hyphen, "")))
        i += 1
    return out


def _token_diff_two_col_html(old: str, new: str) -> tuple[str, str]:
    old_groups = _group_tokens_for_diff(old)
    new_groups = _group_tokens_for_diff(new)
    old_keys = [k for _, k in old_groups]
    new_keys = [k for _, k in new_groups]
    sm = SequenceMatcher(None, old_keys, new_keys)
    out_old: list[str] = []
    out_new: list[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            old_txt = html.escape(" ".join(t for t, _ in old_groups[i1:i2]))
            new_txt = html.escape(" ".join(t for t, _ in new_groups[j1:j2]))
            if old_txt:
                out_old.append(old_txt)
            if new_txt:
                out_new.append(new_txt)
        elif op == "delete":
            txt = html.escape(" ".join(t for t, _ in old_groups[i1:i2]))
            if txt:
                out_old.append(f"<del>{txt}</del>")
        elif op == "insert":
            txt = html.escape(" ".join(t for t, _ in new_groups[j1:j2]))
            if txt:
                out_new.append(f"<ins>{txt}</ins>")
        elif op == "replace":
            old_txt = html.escape(" ".join(t for t, _ in old_groups[i1:i2]))
            new_txt = html.escape(" ".join(t for t, _ in new_groups[j1:j2]))
            if old_txt:
                out_old.append(f"<del>{old_txt}</del>")
            if new_txt:
                out_new.append(f"<ins>{new_txt}</ins>")
    return " ".join(x for x in out_old if x), " ".join(x for x in out_new if x)


def _paragraph_group_info(d: Diff) -> tuple[str, str]:
    old = d.old
    new = d.new
    if old and new and old.para_id and new.para_id and old.para_id != new.para_id:
        label = f"Paragraph {old.para_id} (page {old.page}) → {new.para_id} (page {new.page})"
        return f"{old.para_id}->{new.para_id}", label
    ch = old or new
    if ch and ch.para_id:
        label = f"Paragraph {ch.para_id}"
        if ch.page:
            label += f" (page {ch.page})"
        return ch.para_id, label
    return "unknown", "Paragraph (unknown)"


def _group_diffs_by_paragraph(diffs: List[Diff]) -> List[tuple[str, str, List[Diff]]]:
    if not diffs:
        return []
    groups: list[tuple[str, str, list[Diff]]] = []
    current_key = None
    current_label = None
    current: list[Diff] = []
    for d in diffs:
        key, label = _paragraph_group_info(d)
        if key != current_key:
            if current:
                groups.append((current_key or "unknown", current_label or "", current))
            current_key = key
            current_label = label
            current = [d]
        else:
            current.append(d)
    if current:
        groups.append((current_key or "unknown", current_label or "", current))
    return groups


def _merge_same_rows(rows: List[dict], *, break_para_ids: set[str] | None = None) -> List[dict]:
    if not rows:
        return rows
    merged: list[dict] = []
    i = 0
    while i < len(rows):
        cur = rows[i]
        if cur["kind"] not in {"same", "added", "deleted"} or cur["is_heading"]:
            merged.append(cur)
            i += 1
            continue
        j = i + 1
        while j < len(rows):
            nxt = rows[j]
            if nxt["kind"] != cur["kind"] or nxt["is_heading"]:
                break
            if break_para_ids:
                cur_ids = set(cur.get("old_para_ids", []) + cur.get("new_para_ids", []))
                nxt_ids = set(nxt.get("old_para_ids", []) + nxt.get("new_para_ids", []))
                if cur_ids & break_para_ids:
                    break
                if nxt_ids & break_para_ids:
                    break
            if cur["old_page"] is not None and nxt["old_page"] is not None:
                if cur["old_page"] != nxt["old_page"]:
                    break
            if cur["new_page"] is not None and nxt["new_page"] is not None:
                if cur["new_page"] != nxt["new_page"]:
                    break
            cur["old_text"] = (cur["old_text"] + "\n\n" + nxt["old_text"]).strip()
            cur["new_text"] = (cur["new_text"] + "\n\n" + nxt["new_text"]).strip()
            cur["old_para_ids"].extend(nxt["old_para_ids"])
            cur["new_para_ids"].extend(nxt["new_para_ids"])
            j += 1
        merged.append(cur)
        i = j
    return merged


def _interleave_rows_and_tables(
    rows: List[dict],
    table_entries: List[TableEntry],
) -> List[tuple[str, object]]:
    if not table_entries:
        return [("text", row) for row in rows]

    old_index: dict[str, int] = {}
    new_index: dict[str, int] = {}
    page_index: dict[int, int] = {}
    for idx, row in enumerate(rows):
        for pid in row.get("old_para_ids", []):
            old_index.setdefault(pid, idx)
        for pid in row.get("new_para_ids", []):
            new_index.setdefault(pid, idx)
        if row.get("old_page"):
            page_index[row["old_page"]] = idx
        if row.get("new_page"):
            page_index[row["new_page"]] = idx

    items: list[tuple[float, str, object]] = []
    for idx, row in enumerate(rows):
        items.append((float(idx), "text", row))

    for entry in table_entries:
        anchor_idx = None
        if entry.anchor_id:
            if entry.anchor_side == "new":
                anchor_idx = new_index.get(entry.anchor_id)
            elif entry.anchor_side == "old":
                anchor_idx = old_index.get(entry.anchor_id)
            else:
                anchor_idx = new_index.get(entry.anchor_id) or old_index.get(entry.anchor_id)
        if anchor_idx is None and entry.anchor_page:
            anchor_idx = page_index.get(entry.anchor_page)
        if anchor_idx is None:
            anchor_idx = len(rows) - 1 if rows else 0
        pos = float(anchor_idx) + 0.5 + (entry.order * 1e-3)
        items.append((pos, "table", entry))

    items.sort(key=lambda t: t[0])
    return [(kind, obj) for _, kind, obj in items]


_SENTENCE_CSS = """
<style>
  table.docdiff{border-collapse:collapse;width:100%;table-layout:fixed;}
  table.docdiff th,table.docdiff td{border:1px solid #ddd;padding:8px;vertical-align:top;}
  table.docdiff th{position:sticky;top:0;background:#f7f7f7;z-index:1;}
  table.docdiff td{white-space:pre-wrap;word-break:break-word;}
  table.docdiff .meta{display:block;color:#666;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:0.82em;margin-bottom:0.35rem;}
  table.docdiff tr.same{background:#fff;}
  table.docdiff tr.modified{background:#fff6db;}
  table.docdiff tr.added{background:#eaffea;}
  table.docdiff tr.deleted{background:#ffecec;}
  table.docdiff tr.table-row td{background:#fafafa;}
  table.docdiff del{background:#ffb3b3;text-decoration:line-through;}
  table.docdiff ins{background:#b3ffb3;text-decoration:none;}
  table.docdiff details{margin-top:0.35rem;}
  table.docdiff pre{white-space:pre-wrap;margin:0.25rem 0 0 0;}
  .table-embed{white-space:normal;}
  .table-embed .table-title{font-weight:600;margin:0 0 0.5rem 0;}
  .table-embed table.diff{margin:0.5rem 0 0 0;width:100%;}
  .table-embed table.diff td{white-space:normal;}
</style>
"""


def sentence_diff_html(
    old_path: pathlib.Path,
    new_path: pathlib.Path,
    client: OpenAI,
    tau: float,
    min_words: int,
    logger,
    *,
    gpt_enrich: bool = False,
    max_gpt_items: int | None = 200,
    post_match: bool = True,
    post_match_window: int = DEFAULT_POST_MATCH_WINDOW,
    post_match_sim: float = DEFAULT_POST_MATCH_SIM,
    post_match_containment: float = DEFAULT_POST_MATCH_CONTAIN,
    post_match_same_page: bool = True,
    embed_tables: bool = True,
    table_thr: float = DEFAULT_TABLE_THR,
) -> str:
    old_di = _load_di_root(old_path)
    new_di = _load_di_root(new_path)
    old_paras = extract_paragraph_blocks(old_di, logger)
    new_paras = extract_paragraph_blocks(new_di, logger)
    logger.info("Paragraphs – old: %d, new: %d", len(old_paras), len(new_paras))

    old_chunks = chunk_pages(old_paras, "v1", min_words, logger)
    new_chunks = chunk_pages(new_paras, "v2", min_words, logger)

    diffs = align_dp(client, old_chunks, new_chunks, tau, logger)
    if post_match:
        diffs = postprocess_unmatched_pairs(
            diffs,
            logger,
            window=post_match_window,
            sim=post_match_sim,
            containment=post_match_containment,
            same_page=post_match_same_page,
        )
    if gpt_enrich:
        enrich_with_gpt(client, diffs, logger, max_items=max_gpt_items)

    parts: list[str] = [
        "<h1>Document Comparison (Old → New)</h1>\n",
        _SENTENCE_CSS,
        _TABLE_CSS,
        "<table class='docdiff'>\n",
        "<thead><tr><th style='width:50%;'>Old</th><th style='width:50%;'>New</th></tr></thead>\n",
        "<tbody>\n",
    ]

    rows: list[dict] = []
    for _, group_label, group in _group_diffs_by_paragraph(diffs):
        old_texts = [d.old.text for d in group if d.old is not None]
        new_texts = [d.new.text for d in group if d.new is not None]
        old_text = "\n\n".join(t.strip() for t in old_texts if t.strip()).strip()
        new_text = "\n\n".join(t.strip() for t in new_texts if t.strip()).strip()

        has_old = bool(old_text)
        has_new = bool(new_text)
        if has_old and has_new:
            kind = "same" if all(d.kind == "same" for d in group) else "modified"
        elif has_new:
            kind = "added"
        else:
            kind = "deleted"

        old_meta_chunk = next((d.old for d in group if d.old is not None), None)
        new_meta_chunk = next((d.new for d in group if d.new is not None), None)

        details = [d for d in group if d.details]

        rows.append(
            {
                "kind": kind,
                "old_text": old_text,
                "new_text": new_text,
                "old_para_ids": [old_meta_chunk.para_id] if old_meta_chunk else [],
                "new_para_ids": [new_meta_chunk.para_id] if new_meta_chunk else [],
                "old_page": old_meta_chunk.page if old_meta_chunk else None,
                "new_page": new_meta_chunk.page if new_meta_chunk else None,
                "details": details,
                "group_label": group_label,
                "is_heading": _is_heading_text(old_text or new_text),
            }
        )

    table_entries: list[TableEntry] = []
    anchor_ids: set[str] = set()
    if embed_tables:
        old_tables = _extract_tables_with_anchor(old_di, old_paras)
        new_tables = _extract_tables_with_anchor(new_di, new_paras)
        table_entries = _build_table_entries(old_tables, new_tables, table_thr)
        anchor_ids = {e.anchor_id for e in table_entries if e.anchor_id}

    rows = _merge_same_rows(rows, break_para_ids=anchor_ids)

    for kind, item in _interleave_rows_and_tables(rows, table_entries):
        if kind == "table":
            entry = item
            table_html = _render_table_entry_html(entry)
            parts.append(
                "<tr class='table-row'>"
                f"<td colspan='2'>{table_html}</td>"
                "</tr>\n"
            )
            continue
        row = item
        left = ""
        right = ""

        if row["kind"] == "modified":
            left, right = _token_diff_two_col_html(row["old_text"], row["new_text"])
        elif row["kind"] == "added":
            right = html.escape(row["new_text"])
        elif row["kind"] == "deleted":
            left = html.escape(row["old_text"])
        else:
            left = html.escape(row["old_text"])
            right = html.escape(row["new_text"])

        meta_left = ""
        meta_right = ""

        if row["details"]:
            summary_lines = []
            for d in row["details"]:
                old_id = d.old.id if d.old else ""
                new_id = d.new.id if d.new else ""
                header = f"{old_id} → {new_id}".strip()
                if header:
                    summary_lines.append(header)
                summary_lines.append(d.details or "")
            right += (
                "<details><summary>LLM summary</summary>"
                f"<pre>{html.escape('\\n\\n'.join(summary_lines))}</pre>"
                "</details>"
            )

        row_class = html.escape(row["kind"])
        meta_left_html = f"<span class='meta'>{html.escape(meta_left)}</span>" if meta_left else ""
        meta_right_html = f"<span class='meta'>{html.escape(meta_right)}</span>" if meta_right else ""
        parts.append(
            f"<tr class='{row_class}'>"
            f"<td>{meta_left_html}{left}</td>"
            f"<td>{meta_right_html}{right}</td>"
            "</tr>\n"
        )

    parts.append("</tbody></table>\n")
    return "".join(parts)


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
_space_before_punct_rx = re.compile(r"\s+([%.,;:!?\)\]])")


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


@dataclass
class TableInfo:
    index: int
    page: int
    para_id: str | None
    matrix: List[List[str]]


@dataclass
class TableEntry:
    status: str  # matched | added | removed
    old: TableInfo | None
    new: TableInfo | None
    anchor_id: str | None
    anchor_page: int | None
    anchor_side: str  # new | old | none
    order: int


def _extract_tables_with_anchor(
    di: Dict[str, Any],
    paragraphs: List[ParagraphBlock],
) -> List[TableInfo]:
    tables = _pull_tables(di)
    if not tables:
        return []
    para_by_idx = {p.source_index: p for p in paragraphs if p.source_index is not None}
    available_idx = sorted(k for k in para_by_idx.keys() if k is not None)
    infos: list[TableInfo] = []
    for idx, tbl in enumerate(tables):
        para_id = None
        page = 0
        para_indices: set[int] = set()
        for cell in tbl.get("cells", []) or []:
            for el in cell.get("elements", []) or []:
                if not isinstance(el, str):
                    continue
                m = re.search(r"/paragraphs/(\d+)", el)
                if m:
                    para_indices.add(int(m.group(1)))
        if para_indices:
            for pi in sorted(para_indices):
                pb = para_by_idx.get(pi)
                if pb is not None:
                    para_id = pb.id
                    page = pb.page
                    break
            if para_id is None and available_idx:
                min_idx = min(para_indices)
                pos = bisect.bisect_right(available_idx, min_idx) - 1
                if pos >= 0:
                    pb = para_by_idx.get(available_idx[pos])
                else:
                    pb = para_by_idx.get(available_idx[0])
                if pb is not None:
                    para_id = pb.id
                    page = pb.page
        if page == 0:
            br = tbl.get("boundingRegions") if isinstance(tbl.get("boundingRegions"), list) else []
            if br and isinstance(br[0], dict) and "pageNumber" in br[0]:
                page = int(br[0].get("pageNumber") or 0)
        infos.append(TableInfo(idx, page, para_id, _build_table_matrix(tbl)))
    return infos


def _build_table_entries(
    old_tables: List[TableInfo],
    new_tables: List[TableInfo],
    thr: float,
) -> List[TableEntry]:
    if not old_tables and not new_tables:
        return []
    mapping = _match_table_pairs([t.matrix for t in new_tables], [t.matrix for t in old_tables], thr)
    matched_old = set(mapping.values())
    entries: list[TableEntry] = []

    for j_new, new_tbl in enumerate(new_tables):
        if j_new in mapping:
            i_old = mapping[j_new]
            old_tbl = old_tables[i_old]
            anchor_id = new_tbl.para_id or old_tbl.para_id
            anchor_page = new_tbl.page or old_tbl.page
            anchor_side = "new" if new_tbl.para_id else "old"
            entries.append(
                TableEntry(
                    "matched",
                    old_tbl,
                    new_tbl,
                    anchor_id,
                    anchor_page or None,
                    anchor_side,
                    new_tbl.index,
                )
            )
        else:
            anchor_id = new_tbl.para_id
            anchor_page = new_tbl.page
            entries.append(
                TableEntry(
                    "added",
                    None,
                    new_tbl,
                    anchor_id,
                    anchor_page or None,
                    "new",
                    new_tbl.index,
                )
            )

    for i_old, old_tbl in enumerate(old_tables):
        if i_old not in matched_old:
            anchor_id = old_tbl.para_id
            anchor_page = old_tbl.page
            entries.append(
                TableEntry(
                    "removed",
                    old_tbl,
                    None,
                    anchor_id,
                    anchor_page or None,
                    "old",
                    len(new_tables) + old_tbl.index,
                )
            )
    return entries


def _build_table_matrix(tbl: Dict[str, Any]) -> List[List[str]]:
    rc = tbl.get("rowCount") or (max(c["rowIndex"] for c in tbl["cells"]) + 1)
    cc = tbl.get("columnCount") or (max(c["columnIndex"] for c in tbl["cells"]) + 1)
    grid = [["" for _ in range(cc)] for _ in range(rc)]
    for cell in tbl["cells"]:
        r, c = cell["rowIndex"], cell["columnIndex"]
        grid[r][c] = cell.get("content") or cell.get("text") or ""
    return grid


def load_tables(path: pathlib.Path) -> List[List[List[str]]]:
    di = _load_di_root(path)
    return [_build_table_matrix(t) for t in _pull_tables(di)]


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
            yield (j_new + 1, _diff_table_cells(old[i_old], tbl_new), "matched")
        else:
            yield (j_new + 1, _diff_table_cells([], tbl_new), "added")

    # tables removed from old
    for i_old, tbl_old in enumerate(old):
        if i_old not in matched_old:
            yield (len(new) + i_old + 1, _diff_table_cells(tbl_old, []), "removed")


_TABLE_CSS = """
<style>
  table.diff{border-collapse:collapse;width:100%;margin:1.5rem 0;}
  table.diff th,table.diff td{border:1px solid #ccc;padding:6px;vertical-align:top;}
  td.same{}
  td.changed{background:#ffeacc;}
  del{background:#ffb3b3;text-decoration:line-through;}
  ins{background:#b3ffb3;text-decoration:none;}
  .table-removed{padding:0.75rem 1rem;border:1px dashed #cc6666;background:#fff3f3;}
</style>
"""


def _cell_html(old: str, new: str, st: str) -> str:
    if st == "same":
        return f"<td class='same'>{html.escape(new or old)}</td>"
    return (f"<td class='changed'><del>{html.escape(old)}</del><br>→ "
            f"<ins>{html.escape(new)}</ins></td>")


def _table_title(entry: TableEntry) -> str:
    if entry.new is not None:
        idx = entry.new.index + 1
    elif entry.old is not None:
        idx = entry.old.index + 1
    else:
        idx = 0
    label = f"Table {idx}" if idx else "Table"
    if entry.status == "added":
        label += " (added)"
    elif entry.status == "removed":
        label += " (removed)"
    return label


def _render_table_entry_html(entry: TableEntry) -> str:
    title = _table_title(entry)
    if entry.status == "removed":
        return (
            f"<div class='table-embed'><div class='table-title'>{html.escape(title)}</div>"
            "<div class='table-removed'>TABLE REMOVED FOR CLARITY CHECK DOCUMENT END</div>"
            "</div>"
        )
    old_tbl = entry.old.matrix if entry.old is not None else []
    new_tbl = entry.new.matrix if entry.new is not None else []
    grid = _diff_table_cells(old_tbl, new_tbl)
    rows = ["<tr>" + "".join(_cell_html(o, n, st) for o, n, st in row) + "</tr>"
            for row in grid]
    return (
        f"<div class='table-embed'><div class='table-title'>{html.escape(title)}</div>"
        f"<table class='diff'>\n{''.join(rows)}\n</table></div>"
    )


def tables_diff_html(
    old_path: pathlib.Path,
    new_path: pathlib.Path,
    thr: float,
) -> str:
    tbls_old = load_tables(old_path)
    tbls_new = load_tables(new_path)
    tbl_rep = list(diff_tables(tbls_old, tbls_new, thr))

    sections = []
    for idx, grid, status in tbl_rep:
        if status == "removed":
            sections.append(
                f"<h3>Table {idx}</h3>"
                "<div class='table-removed'>TABLE REMOVED FOR CLARITY CHECK DOCUMENT END</div>"
            )
            continue
        rows = ["<tr>" + "".join(_cell_html(o, n, st) for o, n, st in row) + "</tr>"
                for row in grid]
        sections.append(f"<h3>Table {idx}</h3><table class='diff'>\n"
                        + "".join(rows) + "\n</table>")

    return (
        "<h1>Table Comparison</h1>\n" + _TABLE_CSS + "\n"
        + "".join(sections)
    )


# ╭──────────────────────── 3. Combined driver ───────────────────────────╮

def build_combined_report(
    old_path: pathlib.Path,
    new_path: pathlib.Path,
    out_html: pathlib.Path,
    tau: float = DEFAULT_SIM_TAU,
    thr: float = DEFAULT_TABLE_THR,
    min_words: int = DEFAULT_MIN_WORDS,
    debug: bool = False,
    api_key: Optional[str] = None,
    gpt_enrich: bool = False,
    max_gpt_items: int | None = 200,
    post_match: bool = True,
    post_match_window: int = DEFAULT_POST_MATCH_WINDOW,
    post_match_sim: float = DEFAULT_POST_MATCH_SIM,
    post_match_containment: float = DEFAULT_POST_MATCH_CONTAIN,
    post_match_same_page: bool = True,
    embed_tables: bool = True,
) -> None:
    """Internal helper that orchestrates the two diff sections and writes HTML."""

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(message)s" if debug else "%(message)s",
    )
    logger = logging.getLogger("doc-diff")

    # OpenAI key (client-scoped for thread safety)
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OpenAI API key: set OPENAI_API_KEY (or pass api_key=...)")
    client = OpenAI(api_key=key)

    # Build both sections
    logger.info("▶ AI sentence diff …")
    ai_html = sentence_diff_html(
        old_path,
        new_path,
        client,
        tau,
        min_words,
        logger,
        gpt_enrich=gpt_enrich,
        max_gpt_items=max_gpt_items,
        post_match=post_match,
        post_match_window=post_match_window,
        post_match_sim=post_match_sim,
        post_match_containment=post_match_containment,
        post_match_same_page=post_match_same_page,
        embed_tables=embed_tables,
        table_thr=thr,
    )

    tbl_html = ""
    if not embed_tables:
        logger.info("▶ Table diff …")
        tbl_html = "<hr style='margin:4rem 0;'>\n" + tables_diff_html(old_path, new_path, thr)

    # Compose single HTML doc
    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Document diff report</title>"
        "</head><body style='margin:2rem;'>\n"
        + ai_html
        + tbl_html
        + "\n</body></html>"
    )
    out_html.write_text(html_doc, encoding="utf-8")
    logger.info("✅ Combined report written → %s", out_html.resolve())


# ╭──────────────────────── 4. Public API ────────────────────────────────╮

def diff_documents(
    *,
    old: str | pathlib.Path,
    new: str | pathlib.Path,
    out: str | pathlib.Path = "combined_report.html",
    tau: float = DEFAULT_SIM_TAU,
    thr: float = DEFAULT_TABLE_THR,
    min_words: int = DEFAULT_MIN_WORDS,
    debug: bool = False,
    api_key: str | None = None,
    gpt_enrich: bool = False,
    max_gpt_items: int | None = 200,
    post_match: bool = True,
    post_match_window: int = DEFAULT_POST_MATCH_WINDOW,
    post_match_sim: float = DEFAULT_POST_MATCH_SIM,
    post_match_containment: float = DEFAULT_POST_MATCH_CONTAIN,
    post_match_same_page: bool = True,
    embed_tables: bool = True,
) -> pathlib.Path:
    """High-level wrapper that mirrors the old CLI but is now plainly callable.

    Parameters
    ----------
    old, new : str | Path
        File paths of the two OCR exports (JSON) to compare.
    out : str | Path, default "combined_report.html"
        Where to write the HTML diff.
    tau : float, default 0.78
        Sentence-alignment cosine similarity threshold.
    thr : float, default 0.25
        Table-matching cosine similarity threshold.
    min_words : int, default 4
        Sentences shorter than this are ignored.
    debug : bool, default False
        If True, enable verbose logging.
    api_key : str | None
        OpenAI API key. If None, falls back to the OPENAI_API_KEY env var.
    gpt_enrich : bool, default False
        If True, add an LLM summary for some modified sentences (token diff is always shown).
    max_gpt_items : int | None, default 200
        Maximum number of modified sentences to enrich with GPT (None = no limit).
    post_match : bool, default True
        If True, try to pair nearby added/deleted rows that appear to be the same sentence.
    post_match_window : int, default 3
        Max forward distance (in diff rows) to search for a pairing candidate.
    post_match_sim : float, default 0.80
        Similarity threshold for post-matching.
    post_match_containment : float, default 0.75
        Containment threshold for post-matching (old contained in new).
    post_match_same_page : bool, default True
        If True, only pair chunks that are on the same page (or adjacent page).
    embed_tables : bool, default True
        If True, embed table diffs inside the main table instead of a separate section.

    Returns
    -------
    pathlib.Path
        Absolute path of the generated HTML report.
    """

    old_path = pathlib.Path(old)
    new_path = pathlib.Path(new)
    out_path = pathlib.Path(out)

    build_combined_report(
        old_path=old_path,
        new_path=new_path,
        out_html=out_path,
        tau=tau,
        thr=thr,
        min_words=min_words,
        debug=debug,
        api_key=api_key,
        gpt_enrich=gpt_enrich,
        max_gpt_items=max_gpt_items,
        post_match=post_match,
        post_match_window=post_match_window,
        post_match_sim=post_match_sim,
        post_match_containment=post_match_containment,
        post_match_same_page=post_match_same_page,
        embed_tables=embed_tables,
    )
    return out_path.resolve()


__all__ = ["diff_documents"]
