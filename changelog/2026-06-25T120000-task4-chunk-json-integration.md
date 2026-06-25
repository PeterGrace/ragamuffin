# Task 4: Assemble `chunk_json` and route JSON extensions

Date: 2026-06-25

## Summary

Integrated the Task 2 (parsing) and Task 3 (flattening) helpers into a
production `pub fn chunk_json`, routed JSON file extensions to it, and removed
the now-obsolete `#[cfg(test)]` gates from the five helpers that previously had
no production caller.

## Changes

- `src/chunk.rs`
  - Added `pub fn chunk_json(text, chunk_words, overlap_words) -> Vec<Chunk>`.
    Each record (array element, single value, or JSONL line) becomes one chunk:
    string fields flatten into the embedded text, scalar fields into metadata,
    and the original record is preserved verbatim under `raw`. Records whose
    text exceeds `chunk_words` are sub-split with `chunk_text`, each part tagged
    with `part`. Every chunk carries `record` and `source_kind: "json"`.
  - Routed `.json` / `.jsonl` / `.ndjson` to `chunk_json` in `chunk_for_path`.
  - Removed `#[cfg(test)]` from `parse_json_records`, `MAX_FLATTEN_DEPTH`,
    `join_key`, `compact_json`, and `flatten_value` (now reachable from the
    `chunk_json` production path).
  - Hardened the `flatten_value` depth-cap guard to be prefix-aware, so a
    collapsed top-level subtree no longer emits a leading `": "`.

## Behavior notes

- Empty input (whitespace-only) and a valid-but-empty JSON array (`[]`) yield no
  chunks. Non-JSON input falls back to plain-text chunking (`indexed_chunks`,
  metadata `{"chunk": i}`) with a `tracing::warn!`.
- A record with no string content embeds its compact JSON as the chunk text.

## Verification

- `cargo test`: 47 passed, 0 failed.
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo fmt --check`: clean.
