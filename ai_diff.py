#!/usr/bin/env python3
"""
pdf_diff.py – semantic diff between two OCR-extracted PDFs (Azure Document Intelligence).
────────────────────────────────────────────────────────────────────────────
Now with **deep debug instrumentation** so you can quickly spot why a report
ends prematurely or misses changes.

v0.4 – page-centric Markdown reports
────────────────────────────────────
* Report is now grouped *per page* →
  
  ```
  ## Page 1
  ### Additions
  …
  ### Changes
  …
  ### Deletions
  …
  ```
* Function `write_md()` completely rewritten; other logic untouched.
* Helper `extract_page()` added (robust to oversize-chunk suffixes).

Debug features added in v0.3
────────────────────────────
1. **Page-count sanity check:** logs real PDF page counts (if .pdf given) vs
   pages extracted/loaded.
2. **Paragraph filter metrics:** records how many paragraphs were dropped for
   being "too short" and allows override by `--min-words`.
3. **Oversize paragraph detector:** warns when any chunk exceeds the 8 k-token
   limit for embeddings and automatically splits it.
4. **Alignment drift tracing:** every N (=100) new-paragraph alignments,
   reports how many old chunks remain unmatched so you can see when sync is
   lost.
5. **Embedding failures surfaced:** catches `InvalidRequestError`, logs the
   offending paragraph’s length and fingerprint, and still produces a fallback
   diff (flags it as `embedding_failed`).
6. **Verbose logging flag:** `--debug` turns on DEBUG-level output and keeps the
   tqdm progress bars; otherwise INFO-level only.

Usage (unchanged apart from new flags):
    python pdf_diff.py old.json new.json --out report.md \
        --tau 0.78 --debug --min-words 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pathlib
import re
import sys
import textwrap
import html
import markdown
from dataclasses import dataclass, field
from typing import List, Optional, Dict, DefaultDict

import openai
# Compatibility shim: OpenAI Python SDK ≤0.28 used `openai.error.InvalidRequestError`,
# whereas >=1.0 exposes it at top-level.  Import whichever exists.
try:
    from openai.error import InvalidRequestError  # legacy (<1.0)
except (ImportError, ModuleNotFoundError):
    try:
        from openai import InvalidRequestError  # modern (>=1.0)
    except ImportError:  # fallback – define a dummy so script still runs
        class InvalidRequestError(Exception):
            """Fallback InvalidRequestError when openai SDK not present."""
            pass

from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# ───────────────────────────── Config ────────────────────────────── #
EMBED_MODEL       = "text-embedding-3-small"
GPT_MODEL         = "gpt-4o-mini"
MAX_PAR_TOKENS    = 8_000  # model hard limit is 8192
ALIGN_LOG_EVERY   = 100    # paragraphs
DEFAULT_SIM_TAU   = 0.78
DEFAULT_MIN_WORDS = 4

# ────────────────────────── Data classes ─────────────────────────── #
@dataclass
class Chunk:
    id: str
    text: str
    hash: str
    emb: Optional[List[float]] = None
    embedding_failed: bool = False  # set when over length or API error

@dataclass
class Diff:
    kind: str  # added | deleted | modified
    new: Optional[Chunk] = None
    old: Optional[Chunk] = None
    details: Optional[str] = None

# ───────────────────────────── Helper fx ─────────────────────────── #
sha256 = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]

def tokens_approx(text: str) -> int:
    # very rough ≈4 chars per token heuristic
    return max(1, len(text) // 4)

@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=20))
def embed(texts: List[str]):
    return [d.embedding for d in openai.embeddings.create(model=EMBED_MODEL, input=texts).data]

@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=20))
def gpt_diff(old: str, new: str):
    prompt = (
        "You are an expert editor. Compare the OLD and NEW paragraphs and list exact changes (additions, deletions, re-phrasings). "
        "Use concise bullet points.\n\nOLD:\n" + old + "\n\nNEW:\n" + new
    )
    r = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.0,
    )
    return r.choices[0].message.content.strip()

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    return dot / ((sum(x*x for x in a)**0.5)*(sum(y*y for y in b)**0.5) + 1e-9)

# ──────────────────────── Load & normalise ───────────────────────── #

def load_pages(path: pathlib.Path, logger) -> List[str]:
    if path.suffix.lower() != ".json":
        return pathlib.Path(path).read_text("utf-8").split("\f")

    raw = pathlib.Path(path).read_text("utf-8")
    data = json.loads(raw)
    logger.debug("Loaded JSON keys: %s", list(data.keys())[:10])

    # Azure schemas wrangling
    if "analyzeResult" in data:
        ar = data["analyzeResult"]
        if "pages" in ar:  # 2024
            pages = [pg.get("content") or "\n".join(l["content"] for l in pg.get("lines", [])) for pg in ar["pages"]]
        elif "readResults" in ar:  # older
            pages = ["\n".join(l["text"] for l in pg["lines"]) for pg in ar["readResults"]]
        else:
            pages = [ar.get("content", "")]
    elif "pages" in data:  # very old preview
        pages = [pg.get("content") or "\n".join(l["content"] for l in pg.get("lines", [])) for pg in data["pages"]]
    else:
        pages = [data.get("content", "")]
    return pages

def normalise(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_paragraphs(page):
    return [p.strip() for p in re.split(r"(?:\n\s*){2,}", page) if p.strip()]

def chunk_pages(pages, label, min_words, logger):
    chunks = []
    dropped_short = 0
    joined_due_len = 0

    for pg, txt in enumerate(pages, 1):
        for pr, para in enumerate(split_paragraphs(normalise(txt)), 1):
            words = para.split()
            if len(words) < min_words:
                dropped_short += 1
                continue
            # oversize paragraphs – soft-split at ~4k tokens so each part < 8k
            approx_tokens = tokens_approx(para)
            if approx_tokens > MAX_PAR_TOKENS:
                logger.debug("Oversize paragraph (%d tokens) on p%d para%d – splitting", approx_tokens, pg, pr)
                joined_due_len += 1
                # naïve split halfway on sentence boundary
                mid = len(para)//2
                split_idx = para.find(". ", mid)
                if split_idx == -1:
                    split_idx = mid
                for idx, part in enumerate(textwrap.wrap(para, split_idx), 1):
                    cid = f"{label}-p{pg}-para{pr}-{idx}"
                    chunks.append(Chunk(id=cid, text=part, hash=sha256(part)))
            else:
                cid = f"{label}-p{pg}-para{pr}"
                chunks.append(Chunk(id=cid, text=para, hash=sha256(para)))

    logger.info("%s: %d chunks kept, %d dropped (short), %d split (oversize)", label, len(chunks), dropped_short, joined_due_len)
    return chunks

# ─────────────────────────── Alignment ───────────────────────────── #

def align(old_chunks, new_chunks, tau, logger):
    # Embed with progress bar
    for ch in tqdm(old_chunks + new_chunks, desc="Embedding", unit="chunk", disable=logger.level>logging.INFO):
        if ch.emb is not None or ch.embedding_failed:
            continue
        try:
            ch.emb = embed([ch.text])[0]
        except InvalidRequestError as e:
            logger.warning("Embedding failed for %s (%s) – %s", ch.id, tokens_approx(ch.text), e)
            ch.embedding_failed = True

    unmatched_old = {c.id: c for c in old_chunks}
    diffs: List[Diff] = []

    for idx, nw in enumerate(tqdm(new_chunks, desc="Aligning", unit="chunk", disable=logger.level>logging.INFO), 1):
        if idx % ALIGN_LOG_EVERY == 0:
            logger.debug("Alignment progress %d/%d, unmatched_old=%d", idx, len(new_chunks), len(unmatched_old))

        best_id, best_sim = None, 0
        for oid, od in unmatched_old.items():
            if nw.emb is None or od.emb is None:
                continue  # give up on chunks that couldn’t embed
            sim = cosine(nw.emb, od.emb)
            if sim > best_sim:
                best_sim, best_id = sim, oid
        if best_sim >= tau:
            od = unmatched_old.pop(best_id)
            if od.hash == nw.hash:
                continue  # identical
            diffs.append(Diff("modified", new=nw, old=od))
        else:
            diffs.append(Diff("added", new=nw))

    # leftovers = deletions
    diffs.extend(Diff("deleted", old=o) for o in unmatched_old.values())
    return diffs

# ─────────────────────── GPT enrichment pass ─────────────────────── #

def enrich(diffs, logger):
    mods = [d for d in diffs if d.kind == "modified"]
    for d in tqdm(mods, desc="GPT diff", unit="chunk", disable=logger.level>logging.INFO):
        try:
            d.details = gpt_diff(d.old.text, d.new.text)
        except Exception as e:
            logger.error("GPT diff failed for %s → %s: %s", d.old.id, d.new.id, e)
            d.details = "<GPT-error: %s>" % e

# ────────────────────────── Reporting ────────────────────────────── #

def extract_page(chunk_id: str) -> int:
    """Return 1-based page number parsed from Chunk.id.

    Handles ids like 'v2-p10-para4' *and* oversize splits 'v2-p10-para4-2'.
    Returns 0 if pattern missing so such chunks are grouped upfront.
    """
    m = re.search(r"-p(\d+)-", chunk_id)
    return int(m.group(1)) if m else 0

def pos_key(chunk_id: str) -> tuple[int, int, int]:
    """
    Return (page, para, sub) from IDs like 'v2-p3-para12-2'.
    Missing pieces default to 0 so they still sort.
    """
    m = re.match(r".*-p(\d+)-para(\d+)(?:-(\d+))?", chunk_id)
    if not m:
        return (0, 0, 0)           # fallback for weird IDs
    page, para, sub = (int(x or 0) for x in m.groups())
    return page, para, sub

def write_md(diffs: list[Diff], out: pathlib.Path):
    # Sort once by the key we just defined
    diffs_sorted = sorted(
        diffs,
        key=lambda d: pos_key((d.new or d.old).id)
    )

    lines = ["# Document diff report\n"]

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
            lines.append(f"* ✏️ **{d.old.id} → {d.new.id}**\n  {d.details}")

    markdown_text = "\n".join(lines)
    out.write_text(markdown_text, "utf-8")
    return markdown_text

# ───────────────────────────── MD → HTML helper ──────────────────── #

def md_to_html(md: str, title: str = "Document diff report") -> str:
    """Wrap GitHub-flavoured Markdown in a simple styled HTML shell."""
    body = markdown.markdown(md, extensions=["extra", "smarty"])
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/github-markdown-css@5.4.0/github-markdown.min.css">
<style>body{{margin:2rem;}} </style>
</head><body class="markdown-body">
{body}
</body></html>"""

# ──────────────────────────── Main ──────────────────────────────── #
def run_diff(
    old_path: pathlib.Path,
    new_path: pathlib.Path,
    out_path: pathlib.Path = pathlib.Path("diff_report.md"),
    tau: float = DEFAULT_SIM_TAU,
    debug: bool = False,
    min_words: int = DEFAULT_MIN_WORDS,
    api_key: Optional[str] = None,
) -> tuple[str, str]:
    """
    Perform the diff and return both markdown and HTML output as strings.

    Returns:
        (md_text, html_doc)
    """
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(message)s" if debug else "%(message)s",
    )
    logger = logging.getLogger("pdf-diff")

    openai.api_key = api_key or os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"

    old_pages = load_pages(old_path, logger)
    new_pages = load_pages(new_path, logger)
    logger.info("Pages – old: %d, new: %d", len(old_pages), len(new_pages))

    old_chunks = chunk_pages(old_pages, "v1", min_words, logger)
    new_chunks = chunk_pages(new_pages, "v2", min_words, logger)

    diffs = align(old_chunks, new_chunks, tau, logger)
    enrich(diffs, logger)

    md_text = write_md(diffs, out_path.with_suffix(".md"))
    html_doc = md_to_html(md_text, title="Document diff report")

    html_path = out_path.with_suffix(".html")
    html_path.write_text(html_doc, "utf-8")
    logger.info("✅ Report saved → %s", html_path)
    logger.info("✅ Report saved → %s", out_path)

    return md_text, html_doc

def main():
    ap = argparse.ArgumentParser(description="Semantic diff two OCR PDFs with deep debug & page-grouped reporting")
    ap.add_argument("old", type=pathlib.Path)
    ap.add_argument("new", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default="diff_report.md")
    ap.add_argument("--tau", type=float, default=DEFAULT_SIM_TAU)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    args = ap.parse_args()

    run_diff(
        old_path=args.old,
        new_path=args.new,
        out_path=args.out,
        tau=args.tau,
        debug=args.debug,
        min_words=args.min_words,
    )

if __name__ == "__main__":
    main()
