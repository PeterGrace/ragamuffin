# Task 1: Embedder Trait, `normalize`, and `FakeEmbedder`

**Date:** 2026-06-20T12:00:00

## Summary

Implements the embedder layer — the only component that maps language to
geometry. All produced vectors are L2-normalized (unit length) so the store
can treat cosine similarity as a plain dot product.

## Files Changed

- `src/embedder/mod.rs` — replaces placeholder with:
  - `Embedder` trait (`embed`, `dim`; `Send + Sync` for async sharing)
  - `normalize(v: &mut [f32])` — scales a slice to unit L2 length in-place;
    zero vectors are left as zeros (correct "no similarity" semantics)
  - Module declarations: `pub mod fastembed` (real impl, Task 6) and
    `#[cfg(test)] pub mod fake` (test-only, not compiled into production builds)

- `src/embedder/fake.rs` — `FakeEmbedder`:
  - Bag-of-words over a fixed 8-word vocabulary
  - Deterministic and offline — no model download, no network
  - Implements `Embedder` and `Default`
  - Two unit tests: `similar_text_is_closer_than_dissimilar` and
    `rows_are_unit_length`

- `src/embedder/fastembed.rs` — stub (single doc comment) so the module tree
  compiles; real implementation deferred to Task 6.

## Test Results

```
test embedder::fake::tests::rows_are_unit_length ... ok
test embedder::fake::tests::similar_text_is_closer_than_dissimilar ... ok

test result: ok. 2 passed; 0 failed
```

## Clippy

Clean in all embedder files. Pre-existing `unused import: clap::Parser`
warning in `src/main.rs` (Task 0 issue; Task 5 will resolve it) causes
`-D warnings` to fail for the binary target — not introduced by this task.

## Design Decisions

- `fake` module is `#[cfg(test)]`-gated so it is never compiled into
  production builds while remaining usable from any in-crate unit test.
- `normalize` is a public free function (not a method) so it can be reused by
  the real fastembed embedder in Task 6 without duplication.
- Zero-vector handling in `normalize` returns zeros (not an error) because
  "no vocabulary words found" is a valid degenerate case — its dot product
  with any other vector is 0, correctly signalling no similarity.
