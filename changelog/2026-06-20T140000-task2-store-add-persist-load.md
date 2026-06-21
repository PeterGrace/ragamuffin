# Task 2: Store add/persist/load

**Date:** 2026-06-20T14:00:00Z
**Branch:** mvp-implementation

## Summary

Implemented `src/store.rs` — the on-disk vector store with idempotent upsert keyed
by content hash. This is the persistence layer; search is deferred to Task 3.

## On-Disk Layout

```
<dir>/
    docs/<id>.txt        raw text, verbatim, one file per entry
    vectors.bin          contiguous little-endian f32, row-major, N * dim
    meta.json            JSON array of MetaRecord, row-aligned with vectors
    manifest.json        { "dim": <usize> }
```

## Key Design Decisions

- **Content-hash identity:** SHA-256 of text truncated to first 8 bytes (16 hex chars).
  Identical text upserts the same row rather than creating a duplicate.
- **Little-endian f32 encoding:** Uses `f32::to_le_bytes` / `f32::from_le_bytes`
  with `chunks_exact(4)` for portability without requiring `bytemuck`.
- **Atomic writes:** Every persist uses temp-file-then-rename so a crash cannot
  leave a torn file.
- **Parallel structures:** `vectors[i]` ⇄ `meta[i]` ⇄ `docs/<meta[i].id>.txt`
  invariant is maintained by all operations.
- **O(1) upsert:** `HashMap<Id, usize>` index maps content hash to row number,
  enabling in-place overwrite without scanning the full store.

## Public API

| Item | Description |
|------|-------------|
| `Store::open(dir, dim)` | Open/create store; validates dim against manifest |
| `Store::open_existing(dir)` | Open store reading dim from manifest (no embedder needed) |
| `Store::add(vector, text, source, metadata)` | Upsert entry; returns content-hash id |
| `Store::count()` | Number of stored entries |
| `Store::all()` | All metadata records in row order |
| `MetaRecord` | Row-aligned metadata struct |
| `Hit` | Search result struct (used in Task 3) |
| `Id` | Type alias for 16-hex-char content hash |

## Tests

Three unit tests covering the central invariants:

1. `add_then_reload_reproduces_entries` — persist + reload round-trip
2. `identical_text_does_not_duplicate` — idempotent upsert; last write wins for metadata
3. `wrong_dimension_is_rejected` — `StoreError::DimMismatch` on bad vector length

All tests pass: `cargo test --lib store::` → 3 passed, 0 failed.

## Known Issues / Forward-Looking Items

- `clippy -- -D warnings` on the binary target fails due to `use clap::Parser`
  in `src/main.rs` (Task 0 leftover; `cli.rs` is still a stub until Task 5).
  `cargo clippy --lib -- -D warnings` passes cleanly.
- `Hit` struct and `open_existing` are forward-looking items for Task 3/5;
  they are intentionally present and not annotated with `#[allow(dead_code)]`.
