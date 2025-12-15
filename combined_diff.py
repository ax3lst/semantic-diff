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

sha256 = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]


# ╭──────────────────────── 1. Paragraph-diff section ─────────────────────╮
@dataclass
class Chunk:
    id: str
    page: int
    order: int
    text: str
    norm: str
    hash: str
    emb: Optional[List[float]] = None
    embedding_failed: bool = False     # flagged if embedding API fails


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

_soft_hyphen = "\u00ad"
_join_hyphen_linebreak_rx = re.compile(r"(\w)-\s*\n\s*(\w)")
_space_before_punct_rx = re.compile(r"\s+([%.,;:!?\)\]])")


def normalize_text(text: str) -> str:
    text = text.replace(_soft_hyphen, "")
    text = _join_hyphen_linebreak_rx.sub(r"\1\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = _space_before_punct_rx.sub(r"\1", text)
    return text.strip()

def normalize_for_segmentation(text: str) -> str:
    text = text.replace(_soft_hyphen, "")
    text = _join_hyphen_linebreak_rx.sub(r"\1\2", text)
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
    pages: List[Tuple[int, str]],
    label: str,
    min_words: int,
    logger,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    dropped_short = 0
    oversize_split = 0
    order_by_page: dict[int, int] = defaultdict(int)

    for page_no, page_txt in pages:
        for sent in split_sentences(page_txt):
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
                    chunks.append(Chunk(cid, page_no, order_by_page[page_no], part, norm, sha256(norm)))
            else:
                order_by_page[page_no] += 1
                cid = f"{label}-p{page_no}-sent{order_by_page[page_no]}"
                chunks.append(Chunk(cid, page_no, order_by_page[page_no], sent, norm, sha256(norm)))

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


def _token_diff_two_col_html(old: str, new: str) -> tuple[str, str]:
    old_words, new_words = old.split(), new.split()
    sm = SequenceMatcher(None, old_words, new_words)
    out_old: list[str] = []
    out_new: list[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            txt = html.escape(" ".join(old_words[i1:i2]))
            if txt:
                out_old.append(txt)
                out_new.append(txt)
        elif op == "delete":
            txt = html.escape(" ".join(old_words[i1:i2]))
            if txt:
                out_old.append(f"<del>{txt}</del>")
        elif op == "insert":
            txt = html.escape(" ".join(new_words[j1:j2]))
            if txt:
                out_new.append(f"<ins>{txt}</ins>")
        elif op == "replace":
            old_txt = html.escape(" ".join(old_words[i1:i2]))
            new_txt = html.escape(" ".join(new_words[j1:j2]))
            if old_txt:
                out_old.append(f"<del>{old_txt}</del>")
            if new_txt:
                out_new.append(f"<ins>{new_txt}</ins>")
    return " ".join(x for x in out_old if x), " ".join(x for x in out_new if x)


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
  table.docdiff del{background:#ffb3b3;text-decoration:line-through;}
  table.docdiff ins{background:#b3ffb3;text-decoration:none;}
  table.docdiff details{margin-top:0.35rem;}
  table.docdiff pre{white-space:pre-wrap;margin:0.25rem 0 0 0;}
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
) -> str:
    old_di = _load_di_root(old_path)
    new_di = _load_di_root(new_path)
    old_pages = extract_pages_text(old_di, logger)
    new_pages = extract_pages_text(new_di, logger)
    logger.info("Pages – old: %d, new: %d", len(old_pages), len(new_pages))

    old_chunks = chunk_pages(old_pages, "v1", min_words, logger)
    new_chunks = chunk_pages(new_pages, "v2", min_words, logger)

    diffs = align_dp(client, old_chunks, new_chunks, tau, logger)
    if gpt_enrich:
        enrich_with_gpt(client, diffs, logger, max_items=max_gpt_items)

    parts: list[str] = [
        "<h1>Document Comparison (Old → New)</h1>\n",
        _SENTENCE_CSS,
        "<table class='docdiff'>\n",
        "<thead><tr><th style='width:50%;'>Old</th><th style='width:50%;'>New</th></tr></thead>\n",
        "<tbody>\n",
    ]

    for d in diffs:
        row_class = html.escape(d.kind)
        left = ""
        right = ""
        meta_left = ""
        meta_right = ""

        if d.old is not None:
            flag = " (embedding failed)" if d.old.embedding_failed else ""
            meta_left = f"{html.escape(d.old.id)}{flag}"
        if d.new is not None:
            flag = " (embedding failed)" if d.new.embedding_failed else ""
            sim = f" sim={d.sim:.3f}" if (d.kind == "modified" and d.sim is not None) else ""
            meta_right = f"{html.escape(d.new.id)}{flag}{sim}"

        if d.kind == "added":
            right = html.escape(d.new.text)
        elif d.kind == "deleted":
            left = html.escape(d.old.text)
        elif d.kind == "same":
            left = html.escape(d.old.text)
            right = html.escape(d.new.text)
        else:  # modified
            left, right = _token_diff_two_col_html(d.old.text, d.new.text)
            if d.details:
                right += (
                    "<details><summary>LLM summary</summary>"
                    f"<pre>{html.escape(d.details)}</pre>"
                    "</details>"
                )

        parts.append(
            f"<tr class='{row_class}'>"
            f"<td><span class='meta'>{meta_left}</span>{left}</td>"
            f"<td><span class='meta'>{meta_right}</span>{right}</td>"
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
    )

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
    )
    return out_path.resolve()


__all__ = ["diff_documents"]
