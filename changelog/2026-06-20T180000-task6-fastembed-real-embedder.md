# Task 6: Real Local Embedder via fastembed

**Date:** 2026-06-20T18:00:00Z
**Branch:** `mvp-implementation`
**Commit:** cf9dcb3

## Summary

Replaced the Task 5 shim in `src/embedder/fastembed.rs` with a working
`FastEmbedder` backed by `fastembed` 5.8.0 (ONNX runtime, BGE-small-en-v1.5,
384 dimensions).

## API Adjustments

The task specification assumed `TextEmbedding::embed` takes `&self`, but in
fastembed 5.8.0 it takes `&mut self` (the ONNX session state is mutated during
inference). To satisfy the `Embedder` trait contract (`&self`) while sharing
the model behind an `Arc`, the `TextEmbedding` is wrapped in a `std::sync::Mutex`.
The lock is held only for the duration of each inference call; lock poisoning is
surfaced as `EmbedError::Model`.

The probe-based dimension detection at construction time (`model.embed(vec!["probe"], None)`)
also required `let mut model` before the model is moved into the `Mutex`.

No other API deviations from the task description were needed:
- `TextInitOptions::new(EmbeddingModel::BGESmallENV15)` - correct variant name
- `model.embed(vec_of_strs, None)` - correct signature
- `anyhow::Result` propagated via `.map_err(|e| EmbedError::Model(e.to_string()))`

## Files Changed

- `src/embedder/fastembed.rs` — complete replacement of 21-line shim with 61-line real impl

## Quality Gates

| Gate | Result |
|------|--------|
| `cargo build` | PASS |
| `cargo test --lib` | 11/11 PASS |
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -- -D warnings` | PASS |

## Smoke Test Results

Model was cached from a prior run; all three commands completed:

```
cargo run -- --store /tmp/ragstore_smoke add "ragamuffin stores memories as vectors on disk"
# stored 14c11cf6268ea8c5 (count=1)

cargo run -- --store /tmp/ragstore_smoke add "the quick brown fox jumps over the lazy dog"
# stored 05c6e08f1d9fdafa (count=2)

cargo run -- --store /tmp/ragstore_smoke search "how are memories persisted" -k 1
# [{"score":0.77758396,"id":"14c11cf6268ea8c5","text":"ragamuffin stores memories as vectors on disk","source":"manual","metadata":null}]
```

Store layout confirmed: `docs/`, `vectors.bin`, `meta.json`, `manifest.json`.
`manifest.json` contains `{"dim":384}`.
