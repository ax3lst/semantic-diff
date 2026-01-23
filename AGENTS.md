# Document Diff Approach

This repository compares two Azure Document Intelligence OCR JSON exports and
produces a combined HTML report. The diff is split into two pipelines:
sentence/paragraph diff (semantic + token) and table diff (cell-level).

## Inputs and Outputs
- Inputs: two JSON files from Azure Document Intelligence (classic
  analyzeResult or 2025-11-01 wrapper format).
- Output: a single HTML file containing a sentence diff table and a table diff
  section.
- Entry point: `diff_documents` in `combined_diff.py`.

## Document Ingestion
1. Load JSON and normalize to the inner "analyzeResult"-style root.
2. Extract page text:
   - Prefer `paragraphs`, excluding page headers/footers/page numbers and any
     paragraph that is part of a table cell.
   - Fallback to `pages[].lines`, `pages[].content`, or text spans if needed.
3. Normalize text to reduce OCR noise (soft hyphens, whitespace, line breaks,
   stray spaces before punctuation).

## Sentence Diff (Semantic + Token)
1. Sentence segmentation:
   - Split by blocks and line breaks, using spaCy sentencizer if available.
   - Preserve numbered headings (e.g., "3. TITLE") as single sentences.
2. Chunking:
   - Each sentence becomes a `Chunk` with page/order metadata.
   - Very short chunks are dropped; oversized chunks are split.
3. Embedding and alignment:
   - Embeddings are created via OpenAI `text-embedding-3-small`.
   - A dynamic-programming alignment keeps the old/new order intact.
   - Similarity threshold `tau` decides matches; unmatched items are added/
     deleted.
4. Token-level highlight:
   - For modified pairs, `SequenceMatcher` highlights insertions/deletions.
5. Optional LLM summary:
   - For modified pairs, a short GPT summary can be added (disabled by default).

## Table Diff (Cell-Level)
1. Table extraction:
   - Build a 2D matrix from each table's `cells` using row/column indices.
2. Table pairing:
   - Each table gets a signature string (normalized cell text).
   - SentenceTransformer embeddings + Hungarian matching pair old/new tables.
   - Similarity threshold `thr` decides matches; unmatched tables are added/
     removed.
3. Cell diff:
   - Each cell is compared after normalization; changes are highlighted in HTML.

## Tuning Knobs
- `tau` (default 0.78): sentence alignment threshold.
- `thr` (default 0.25): table pairing threshold.
- `min_words` (default 4): sentence length filter.
- `gpt_enrich` / `max_gpt_items`: optional LLM summaries for modified sentences.

## Notes and Failure Modes
- Requires OpenAI API key for the sentence diff embeddings.
- If embeddings fail, chunks are still reported as added/deleted.
- OCR structure variance can affect paragraph extraction; fallback paths are in
  place but may be less stable.
