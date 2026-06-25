# Field-aware JSON ingestion — 2026-06-25

`ingest` now parses JSON, JSONL, and NDJSON by record instead of as raw text.

## Added
- `src/chunk.rs`: new module holding all chunking. Introduces `Chunk` (text +
  metadata); `chunk_for_path` now returns `Vec<Chunk>`. Moved `chunk_text` and
  `chunk_markdown` here unchanged.
- `chunk_json` (`src/chunk.rs`): `.json` / `.jsonl` / `.ndjson` ingest one chunk
  per record. A top-level array yields one record per element; a single object or
  scalar is one record; JSONL/NDJSON parse one record per line. Each record is
  flattened with dotted keys — string leaves and string arrays become embedded
  `key: value` text, scalar leaves become filterable metadata — and the original
  record is preserved verbatim under `raw`. Oversized records sub-split and tag
  `part`. A valid-but-empty array yields no chunks; malformed input falls back to
  plain-text chunking.

## Changed
- `Rag::ingest_file` (`src/rag.rs`) embeds each chunk's body and stores the
  chunk's own metadata instead of a fixed `{"chunk": i}`.

## Notes
- No store-format, search, or error-layer changes; metadata is already arbitrary
  JSON end-to-end. Text/markdown entries keep `{"chunk": i}`; JSON-derived
  entries carry `source_kind: "json"`. Re-ingesting is idempotent via the
  existing content-hash upsert.
