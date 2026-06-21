# Task 4: RAG Orchestration Layer

**Date:** 2026-06-20T00:00:00Z

## Summary

Implemented `src/rag.rs`, the orchestration layer that combines an `Embedder`
and a `Store` into the two operations callers want: remember and recall (§4.3).

## Changes

### `src/rag.rs` (new implementation)

- **`chunk_text(text, chunk_words, overlap_words) -> Vec<String>`** — public
  function that splits text into overlapping fixed-width word windows (§6.3).
  Overlapping windows ensure ideas straddling a chunk boundary remain
  retrievable from either side.

- **`Rag`** — public struct holding a `Store` and a `Box<dyn Embedder>`.
  Owning a boxed trait object allows tests to inject `FakeEmbedder` without a
  real model download.

- **`Rag::open(dir, embedder)`** — opens (or creates) a store under `dir`
  sized to the embedder's dimension.

- **`Rag::add_memory(text, source, metadata)`** — embeds a single text and
  upserts it into the store. Returns the content-hash `Id`.

- **`Rag::ingest_file(path, chunk_words, overlap_words)`** — reads a file,
  chunks it, batch-embeds the chunks, and stores each chunk with its index in
  metadata. Returns a `Vec<Id>`.

- **`Rag::search(query, k)`** — embeds the query string and returns the top-k
  hits from the store.

- **`Rag::count()`** and **`Rag::all()`** — thin delegation to the store,
  needed by the CLI layer (Task 5).

## Tests Added (TDD)

Three unit tests in `rag::tests`:

1. `chunk_text_overlaps_long_input` — verifies step size and boundary word
   positions for a 100-word sequence with `chunk_words=30, overlap_words=10`.
2. `chunk_text_short_input_is_one_chunk` — edge cases: short text returns a
   single chunk, whitespace-only returns empty.
3. `add_memory_then_search_roundtrips` — integration test using `FakeEmbedder`
   and `tempdir`; verifies that `search("rust memory", 1)` retrieves
   `"rust code is fast"` over `"the cat ate food"`.

## Test Results

```
running 10 tests
test embedder::fake::tests::rows_are_unit_length ... ok
test embedder::fake::tests::similar_text_is_closer_than_dissimilar ... ok
test rag::tests::chunk_text_short_input_is_one_chunk ... ok
test rag::tests::chunk_text_overlaps_long_input ... ok
test store::tests::empty_store_search_returns_empty ... ok
test store::tests::wrong_dimension_is_rejected ... ok
test store::tests::add_then_reload_reproduces_entries ... ok
test store::tests::identical_text_does_not_duplicate ... ok
test rag::tests::add_memory_then_search_roundtrips ... ok
test store::tests::search_ranks_closest_first_and_clamps_k ... ok

test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Quality Gates

- `cargo fmt --check` — clean (formatter ran and wrapped two long lines)
- `cargo clippy --lib -- -D warnings` — clean (no warnings)
- No `.unwrap()` in library code
- All public items have doc comments
