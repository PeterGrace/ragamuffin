# ragamuffin

A local-first, transparent semantic memory for LLM coding tools. Stores notes as
vectors plus plain text on disk and retrieves them by meaning. Exposed as an
offline CLI and an MCP server.

## Build

    cargo build --release

## CLI

    ragamuffin --store ./ragstore add "a fact worth remembering"
    ragamuffin --store ./ragstore ingest notes.txt --chunk-words 180 --overlap 40
    ragamuffin --store ./ragstore search "what did I note about X" -k 4
    ragamuffin --store ./ragstore list

`add`, `ingest`, and `search` use a local embedding model (BGE-small, downloaded
on first use into `.fastembed_cache/`). `list` is fully offline.

## MCP server

    ragamuffin --store ./ragstore mcp

Speaks the Model Context Protocol over stdio and exposes two tools,
`search_memory` and `save_memory`, for an LLM harness to call. The harness's own
model decides when to use them.

## Store layout

    ragstore/
        docs/<id>.txt     raw text, one file per entry
        vectors.bin       contiguous little-endian f32, N x dim
        meta.json         metadata, row-aligned with vectors.bin
        manifest.json     { "dim": 384 }

See `DESIGN.md` for the full design and `docs/superpowers/specs/` for the MVP spec.
