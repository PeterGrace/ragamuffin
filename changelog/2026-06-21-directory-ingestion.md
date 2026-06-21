# Directory ingestion — 2026-06-21

`ingest` now accepts a directory, not just a single file.

## Added
- `src/scan.rs`: recursive text-file discovery — content-sniff (NUL/UTF-8) to
  skip binaries, skip hidden entries, no symlink following, `--ext` allowlist,
  and a `--max-bytes` size cap. Deterministic (sorted) output.
- Heading-aware markdown chunking (`chunk_markdown` / `chunk_for_path` in
  `src/rag.rs`): `.md`/`.markdown` split on ATX headings into self-contained,
  heading-rooted chunks; oversized sections fall back to word-window splitting;
  other file types keep fixed-width chunking.
- `ingest <dir>`: single-process recursive ingest with an `indicatif` progress
  bar, per-file error resilience (warn + skip), and a summary line.

## Dependencies
- Added `walkdir` and `indicatif`.

## Notes
- No store-format or search changes. Re-ingesting a directory is idempotent via
  the existing content-hash upsert.
