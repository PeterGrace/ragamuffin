# Task 0: Project Foundation

**Date:** 2026-06-20

## Summary

Established the project foundation for ragamuffin, a local-first semantic memory backend exposed as a CLI and MCP server.

## Changes

### Cargo.toml

Replaced scaffold dependencies with the full dependency set:

- `anyhow` 1.0 — application-level error handling
- `clap` 4 (derive feature) — CLI argument parsing
- `dotenv` 0.15 — `.env` file loading
- `fastembed` 5 — embedding model support
- `rayon` 1 — CPU-bound parallelism
- `rmcp` 1.7.0 (server, macros, transport-io features) — MCP server SDK
- `schemars` 1.2.1 — JSON schema generation (also pulled in transitively by rmcp)
- `serde` 1 + `serde_json` 1 — serialization
- `sha2` 0.10 — content hashing
- `thiserror` 2 — library error types
- `tokio` 1 (rt-multi-thread, macros, sync, io-std) — async runtime
- `tracing` 0.1 + `tracing-subscriber` 0.3 — structured logging

Added `tempfile` 3 as a dev dependency for test fixtures.

Defined explicit `[lib]` and `[[bin]]` sections for the lib/bin split.

### .cargo/config.toml

Removed the `tokio_unstable` rustflag that was required by `console-subscriber`
(no longer a dependency).

### src/error.rs (new)

Three-layer error hierarchy using `thiserror`:

- `EmbedError` — embedding model failures
- `StoreError` — I/O, JSON, and vector dimension mismatches in the on-disk store
- `RagError` — wraps both of the above plus direct I/O, plus `EmptyEmbedding`

### src/lib.rs (new)

Public module tree: `cli`, `embedder`, `error`, `mcp`, `rag`, `store`.

### src/main.rs

Replaced tokio-console demo with a clean async entry point that:
- Loads `.env` via `dotenv`
- Initializes stderr-only tracing (so stdout stays clean for MCP stdio)
- Parses the CLI and dispatches to `ragamuffin::cli::run`

### Placeholder modules

All stubbed with doc comments pointing to the task that will implement them:
- `src/embedder/mod.rs` (Task 1)
- `src/store.rs` (Task 2)
- `src/rag.rs` (Task 4)
- `src/cli.rs` (Task 5) — includes a minimal `Cli` struct and `run` fn so `main.rs` links
- `src/mcp.rs` (Task 7)

## Version Notes

- `rmcp`: resolved to **1.7.0** (latest); the task description suggested 0.8, which does not exist. The feature set (`server`, `macros`, `transport-io`) is correct for 1.7.0.
- `schemars`: resolved to **1.2.1** (latest 1.x).

## Build Status

`cargo build` succeeds with one expected warning: `unused import: clap::Parser` in `src/main.rs`. This is intentional — the stub `Cli::parse()` does not use the trait; it will be replaced in Task 5 when real clap derive is wired up.
