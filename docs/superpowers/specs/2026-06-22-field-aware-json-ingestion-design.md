# Field-Aware JSON Ingestion — Design

Date: 2026-06-22
Status: Approved (pending implementation plan)

## Problem

Today the ingest pipeline treats `.json` files as plain text. `chunk_for_path`
(`src/rag.rs`) only special-cases markdown; everything else, JSON included, goes
through the fixed-width word-window chunker `chunk_text`. The consequences:

- Embedding signal is wasted on JSON syntax (`{`, `"`, `,`).
- A word window can straddle two unrelated records, mixing them in one chunk.
- Field names and values are not retrievable or filterable as structure — the
  stored metadata is only `{"chunk": i}`.

We want JSON ingested record-by-record, with prose fields flattened into clean
embeddable text and scalar fields lifted into filterable metadata, while keeping
the original record verbatim.

## Goals

- Ingest a JSON array as one record (chunk) per element.
- Ingest a single JSON object (or scalar) as one record.
- Ingest JSONL / NDJSON (one JSON value per line) the same way.
- Flatten nested structure with dotted keys; route leaves to text vs. metadata
  by type.
- Always preserve the original record verbatim under `raw`.
- No new CLI flags — behavior is automatic by file extension.
- No regression for existing text/markdown ingestion or stored metadata.

## Non-Goals

- Configurable per-field text/metadata selection (no `--text-fields`).
- A JSON-pointer to a nested record array (e.g. `data.items[]`). Records are the
  top-level array elements or the top-level value only.
- Any change to the store, search, `Hit`, `list`, or error layers.

## Architecture

### Chunks carry metadata

The single structural change: a chunk is now text **plus** the metadata to store
with it.

```rust
/// A unit of text to embed plus the metadata to store alongside it.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub text: String,
    pub metadata: Value,
}
```

`chunk_for_path` returns `Vec<Chunk>` instead of `Vec<String>`. The text and
markdown paths wrap their existing `Vec<String>` output as
`Chunk { text, metadata: json!({ "chunk": i }) }`, so their stored metadata is
byte-for-byte what it is today. `chunk_text` and `chunk_markdown` keep returning
`Vec<String>` (pure splitters; their existing unit tests are untouched).

`ingest_file` (`src/rag.rs`) changes from hardcoding `{"chunk": i}` to using each
chunk's own metadata:

```rust
let chunks = chunk_for_path(path, &text, chunk_words, overlap_words);
if chunks.is_empty() {
    return Ok(Vec::new());
}
let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
let vectors = self.embedder.embed(&texts)?;
let source = path.to_string_lossy().to_string();
let mut ids = Vec::with_capacity(chunks.len());
for (chunk, vector) in chunks.iter().zip(vectors.iter()) {
    let id = self.store.add(vector, &chunk.text, &source, chunk.metadata.clone())?;
    ids.push(id);
}
```

The store already accepts arbitrary `serde_json::Value` metadata and already
returns it through `Hit` (search) and `all()` (list). No store/error changes.

### Module extraction

All chunking moves into a new `src/chunk.rs` module: the `Chunk` type,
`chunk_text`, `chunk_markdown`, `chunk_json`, `chunk_for_path`, and their tests.
`rag.rs` becomes pure orchestration and imports from `chunk`. This keeps each
file focused as the chunking logic grows by ~120 lines.

### Routing

`chunk_for_path` dispatches by lowercased extension:

- `md` / `markdown` -> `chunk_markdown`, wrapped with `{"chunk": i}`.
- `json` / `jsonl` / `ndjson` -> `chunk_json`.
- anything else -> `chunk_text`, wrapped with `{"chunk": i}`.

`chunk_for_path` stays infallible (`-> Vec<Chunk>`); JSON parse failure is
handled inside `chunk_json` by falling back to text (see below).

## The JSON chunker

`chunk_json(text, chunk_words, overlap_words) -> Vec<Chunk>`.

### Parse strategy

1. Parse the whole file as one `serde_json::Value`.
   - `Value::Array(items)` -> one record per element; record index = position.
   - any other `Value` (object or scalar) -> a single record, index 0.
2. If whole-file parse fails, try line-by-line (JSONL/NDJSON): parse each
   non-empty, non-whitespace line as a `Value`; collect the successes; record
   index = line position among parsed records.
3. If line-by-line also yields zero records, `tracing::warn!` and fall back to
   `chunk_text` (each chunk gets `{"chunk": i}`). A mislabeled or malformed file
   still ingests as raw text — no regression and no new error variant.

This makes both selected shapes work (array and single object) and transparently
handles JSONL regardless of extension.

### Per-record flattening (dotted-key, recursive)

For each record, recurse from the root with a dotted key prefix, building a list
of text lines and a metadata map:

| JSON leaf at `key`     | Destination                                        |
| ---------------------- | -------------------------------------------------- |
| string                 | text line `key: value`                             |
| array of strings       | text line `key: a, b, c` (one joined line)         |
| number / bool / null   | metadata entry `key` = value                       |
| nested object          | recurse (`author.name`, `author.team_size`)        |
| array of objects/mixed | recurse with indexed keys (`items.0.title`, ...)   |

The full original record is always also stored verbatim under `raw`.

### Chunk text and metadata assembly

- `text` = the collected text lines joined with `\n`.
- If a record produced no text lines (no string content), `text` = compact JSON
  of the record, so there is always something to embed and a stable content id.
- `metadata` base for a record:

```json
{
  "record": <index in file, 0 for a single value>,
  "source_kind": "json",
  "<flattened scalar dotted keys>": <value>,
  "raw": <original record verbatim>
}
```

### Oversize records

If a record's assembled `text` exceeds `chunk_words`, sub-split it with
`chunk_text(text, chunk_words, overlap_words)`. Each sub-chunk gets the record's
base metadata plus `"part": i`. `raw` repeats on every part so each chunk is
self-contained. Dedup is unaffected: each part's text hashes distinctly.

## Metadata schema (worked example)

Record:

```json
{"title":"Quarterly Report","year":2024,"published":true,
 "author":{"name":"Jane Doe","team_size":5},"tags":["finance","q3"]}
```

Embedded text:

```
title: Quarterly Report
author.name: Jane Doe
tags: finance, q3
```

Stored metadata:

```json
{
  "record": 0,
  "source_kind": "json",
  "year": 2024,
  "published": true,
  "author.team_size": 5,
  "raw": {
    "title": "Quarterly Report",
    "year": 2024,
    "published": true,
    "author": { "name": "Jane Doe", "team_size": 5 },
    "tags": ["finance", "q3"]
  }
}
```

Existing text/markdown entries keep `{"chunk": i}`, so `list`/`search` output is
backward compatible. `source_kind` distinguishes JSON-derived entries.

## Edge cases

- Empty array or empty/whitespace-only file -> no chunks (`Ok([])`), matching the
  existing convention.
- Record with no string fields -> embed compact JSON of the record.
- Top-level scalar (`"hello"`, `42`) -> one record; a string embeds directly, and
  `raw` keeps the value.
- Dotted-key collision (`{"a.b":1}` vs `{"a":{"b":1}}`) -> last write wins; rare,
  documented.
- Deep nesting -> a safety cap at depth 64 collapses deeper subtrees to a compact
  JSON string in the text, preventing stack overflow on adversarial input. This
  is a safety rail far beyond normal data, not a general depth limit.

## Testing

New `chunk_json` unit tests:

- top-level array -> N chunks, one per element, with ascending `record` indices.
- single object -> one chunk, `record` = 0.
- nested object flattens to dotted keys; string leaves in text, scalar leaves in
  metadata.
- array of strings joins into a single text line.
- `raw` preserves the original record verbatim.
- oversize record sub-splits and tags `part`.
- empty array -> empty result.
- malformed JSON (and non-JSONL) -> falls back to text chunking.
- JSONL / NDJSON input parses line-by-line into records.

Plus:

- `ingest_file` on a `.json` file stores per-record metadata (`source_kind`,
  `record`, `raw`).
- `chunk_for_path` routing covers `.json`, `.jsonl`, `.ndjson`.

Existing `chunk_text` / `chunk_markdown` tests are unchanged. `chunk_for_path`
tests update only for the new `Vec<Chunk>` return type.

## Files touched

- `src/chunk.rs` (new): `Chunk`, `chunk_text`, `chunk_markdown`, `chunk_json`,
  `chunk_for_path`, flattening helpers, and their tests.
- `src/rag.rs`: drop the moved chunkers; import from `chunk`; update `ingest_file`
  to embed `chunk.text` and store `chunk.metadata`.
- `src/lib.rs`: declare `pub mod chunk;` (alongside the existing `pub mod rag;`).
- `changelog/`: dated implementation note (per CLAUDE.md).
- `README.md` / `CHANGELOG.md`: document JSON field-aware ingestion.

No changes to `src/store.rs`, `src/error.rs`, `src/cli.rs`, or `src/mcp.rs`.
