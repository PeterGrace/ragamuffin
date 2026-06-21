# Task 2: Heading-Aware Markdown Chunking

**Date:** 2026-06-21

## Summary

Added heading-aware markdown chunking to `src/rag.rs` and routed `ingest_file`
to select a chunker by file extension.

## Changes

### `src/rag.rs`

- Added `is_heading(line: &str) -> bool` — private helper; returns true for ATX
  heading lines (after optional leading spaces, the line starts with `#`).
- Added `pub fn chunk_markdown(text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String>`
  — heading-aware splitter that accumulates lines into sections rooted at each
  heading; preamble (content before the first heading) becomes its own chunk;
  sections exceeding `chunk_words` are sub-split with the existing `chunk_text`.
- Added `pub fn chunk_for_path(path: &Path, text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String>`
  — dispatch function; routes `.md` / `.markdown` paths (case-insensitive) to
  `chunk_markdown`; all other extensions use `chunk_text`.
- Changed one line in `Rag::ingest_file`: replaced `chunk_text(&text, ...)` with
  `chunk_for_path(path, &text, ...)` so markdown files are chunked semantically.

### Tests added (`#[cfg(test)] mod tests`)

| Test | Purpose |
|------|---------|
| `chunk_markdown_splits_on_headings_keeps_sections_whole` | Two headings produce two chunks, each containing their heading and body |
| `chunk_markdown_preamble_is_its_own_chunk` | Content before the first heading is emitted as a separate chunk |
| `chunk_markdown_subsplits_oversized_section` | A section with 51 words and `chunk_words=20` yields more than one chunk |
| `chunk_markdown_empty_input_no_chunks` | Whitespace-only input returns an empty vec |
| `chunk_for_path_routes_by_extension` | `.md` path uses markdown chunker; `.txt` path uses word-window chunker |

## Quality Gates

- `cargo fmt --check`: clean
- `cargo clippy --all-targets -- -D warnings`: clean
- `cargo test --lib rag::`: 9/9 passed (4 pre-existing + 5 new)

## Commit

`69a04b6` — `feat: heading-aware markdown chunking; route ingest_file by extension`
