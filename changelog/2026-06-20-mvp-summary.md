# ragamuffin MVP — 2026-06-20

Initial implementation of the local-first semantic memory backend.

## Added
- `Embedder` trait with a deterministic `FakeEmbedder` (tests) and a real
  `FastEmbedder` (fastembed / BGE-small, 384-dim, wrapped in a Mutex for
  `&self` inference).
- `Store`: idempotent content-hash upsert, little-endian `vectors.bin`,
  `meta.json`, per-entry `docs/*.txt`, atomic rewrites, brute-force parallel
  top-k search (rayon).
- `Rag`: overlapping word-window chunking, `add_memory`, `ingest_file`, `search`.
- CLI: `add`, `ingest`, `search` (JSON), `list` (offline JSON), `mcp`.
- MCP server (`rmcp` 1.7, stdio) exposing `search_memory` and `save_memory`.

## Notes
- The Anthropic chat harness from DESIGN.md is intentionally omitted; the host
  harness's model drives tool use via MCP.
- `vectors.bin` uses explicit little-endian f32 encoding (portable, no alignment
  constraints) instead of bytemuck.
