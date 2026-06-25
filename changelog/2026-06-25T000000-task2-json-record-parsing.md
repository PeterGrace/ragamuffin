# Task 2: JSON Record Parsing

**Date:** 2026-06-25T00:00:00

## Summary

Added `parse_json_records` to `src/chunk.rs`. This private function turns raw
file text into a `Vec<serde_json::Value>` of records, applying a two-stage
strategy:

1. **Whole-file JSON** — tried first via `serde_json::from_str`. A top-level
   array yields one record per element; any other value (object or scalar)
   yields a single-element vec.
2. **JSONL fallback** — if whole-file parsing fails, each non-empty line is
   parsed independently and only successful parses are kept. Lines that fail are
   silently dropped.

Returns an empty vec for empty or non-JSON input.

## Tests Added (TDD)

Five unit tests written before the implementation:

| Test | Coverage |
|------|----------|
| `parse_json_records_array_yields_one_per_element` | JSON array splits per element |
| `parse_json_records_single_object_is_one_record` | JSON object yields 1 record |
| `parse_json_records_scalar_is_one_record` | Scalar (number, string) yields 1 record |
| `parse_json_records_jsonl_parses_each_line` | JSONL fallback with blank lines |
| `parse_json_records_malformed_is_empty` | Malformed and empty input yield empty vec |

## Implementation Notes

- `parse_json_records` is annotated `#[cfg(test)]` because it is currently only
  called from tests. Task 4 (`chunk_json`) will add a production call site and
  remove the `#[cfg(test)]` gate.
- No `#[allow(dead_code)]` was used; the `#[cfg(test)]` annotation is the
  idiomatic Rust approach to suppress the dead-code lint for test-only helpers.

## Files Changed

- `src/chunk.rs` — added `parse_json_records` function (24 lines) and 5 tests
  (30 lines).

## Validation

```
cargo fmt           -- ok
cargo clippy --all-targets -- -D warnings   -- ok, 0 warnings
cargo test          -- 33 passed, 0 failed
```

## Commit

`de7ce77` feat: parse JSON arrays, objects, scalars, and JSONL into records
