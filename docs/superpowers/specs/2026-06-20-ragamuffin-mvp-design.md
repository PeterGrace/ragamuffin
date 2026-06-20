# ragamuffin MVP — Implementation Design

Date: 2026-06-20
Status: Approved for PoC MVP

This spec adapts the language-neutral `DESIGN.md` into a concrete Rust
implementation plan. Where this document and `DESIGN.md` disagree, this document
wins for the MVP (it deliberately scopes some things out). Section references
like "§7" point at `DESIGN.md`.

---

## 1. Scope decision: memory backend, not a chatbot

ragamuffin is a **local-first semantic memory that other LLM coding harnesses
call into** (Claude Code, opencode, zidane, Zed, ...). The host harness is
already an LLM agent with its own model and loop; ragamuffin does not run a
second one.

Consequences:

- **The §7 Anthropic chat harness is dropped.** No embedded conversation loop,
  no LLM SDK, no Anthropic-specific code, no API key required by ragamuffin
  itself.
- The "model-driven, just-in-time context" ideal (§2 #6) is still honored —
  the *host* harness's model decides when to call `search_memory` /
  `save_memory`, via MCP tools, instead of a model we embed.
- ragamuffin exposes its capability two ways over the *same* `RAG` core:
  1. **CLI subcommands** (`add`, `ingest`, `search`, `list`) — fully offline,
     JSON output, any harness can shell out.
  2. **MCP server** — `search_memory` and `save_memory` tools over stdio, the
     native protocol coding harnesses speak.

Everything in `DESIGN.md` §1–6 and §8 (the embedder/store/RAG core, on-disk
layout, algorithms, invariants) is implemented as written.

---

## 2. Technology choices

| Concern | Choice | Rationale |
|---|---|---|
| Real embedder | `fastembed` (BGE-small-en-v1.5, 384-dim) | Auto-downloads model; does tokenization + mean-pooling + L2-normalization, satisfying the §4.1 "rows must be unit length" contract. Lazy-loaded on first `embed`. Pulls in the `ort` native ONNX dependency (accepted cost of local embeddings). |
| Test embedder | hand-written bag-of-words over a fixed vocabulary, normalized | §10 — deterministic, offline, no model download. Makes the whole core testable without a network (ideal #9). |
| Vector search | hand-written dot product over `&[f32]`, parallelized with `rayon` | §6.2 brute force; unit vectors mean cosine similarity reduces to a dot product. No `ndarray`/BLAS dependency for one operation (ideal #8). |
| Content-hash IDs | `sha2` (SHA-256 truncated to 16 hex chars) | §6.4 idempotent upsert. |
| Vectors on disk | flat `f32` row-major `vectors.bin`, cast via `bytemuck` | §5 portable binary format. |
| Metadata | `serde` + `serde_json` → `meta.json` | §5; keeps the store human-readable (ideal #2). |
| CLI | `clap` (derive API) | §9 subcommand surface. |
| MCP server | `rmcp` (official Rust MCP SDK) over stdio transport | Native protocol for Claude Code / opencode / Zed. |
| App errors | `anyhow` (+ `.context()`) at the binary boundary | CLAUDE.md mandate. |
| Lib errors | `thiserror` enums per layer | CLAUDE.md mandate; no `.unwrap()` in lib paths. |

**Dependencies removed from the scaffold:** `ctrlc`, `lazy_static`,
`console-subscriber`, `tracing-log`, and the `tokio_unstable` rustflag — all
tokio-console scaffolding, unused by the MVP.

**Retained:** `tokio` (rmcp async runtime), `serde`/`serde_json`, `thiserror`,
`tracing`, `tracing-subscriber`, `dotenv`.

**Added:** `fastembed`, `rayon`, `sha2`, `bytemuck`, `clap`, `rmcp`, `anyhow`.

---

## 3. Module layout

Each module maps to one layer in §3 and depends only on the layer below it.

```
src/
    main.rs              # thin: parse CLI, init tracing, dispatch to a command handler
    lib.rs               # crate root; re-exports so the bin and integration tests share code
    error.rs             # thiserror enums: EmbedError, StoreError, RagError
    embedder/
        mod.rs           # the Embedder trait + dim()
        fastembed.rs     # real implementation (default path)
        fake.rs          # #[cfg(test)] deterministic bag-of-words embedder
    store.rs             # Store: add / search / all / count + persistence
    rag.rs               # RAG: add_memory / ingest_file / chunk_text / search
    cli.rs               # clap command structs + offline handlers
    mcp.rs               # rmcp server exposing search_memory + save_memory
```

Boundary decisions:

- `Embedder` is a trait; `RAG` holds `Box<dyn Embedder>` so tests inject the
  fake (ideal #9). The trait is the **only** thing that maps language to
  geometry (§4.1).
- `Store` never sees the embedder; it takes and compares `&[f32]` only (§4.2).
- `cli.rs` and `mcp.rs` are sibling thin adapters over the same `RAG` methods —
  MCP's two tools literally call `rag.search` and `rag.add_memory`. No logic is
  duplicated between them.

### 3.1 Embedder contract

```rust
pub trait Embedder {
    /// Embed a batch. Each returned row is L2-normalized (unit length).
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>;
    fn dim(&self) -> usize;
}
```

### 3.2 Store contract

```rust
impl Store {
    pub fn open(dir: &Path, dim: usize) -> Result<Store, StoreError>;
    pub fn add(&mut self, vector: &[f32], text: &str,
               source: &str, metadata: serde_json::Value) -> Result<Id, StoreError>;
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>, StoreError>;
    pub fn all(&self) -> &[MetaRecord];
    pub fn count(&self) -> usize;
}
```

### 3.3 RAG contract

```rust
impl Rag {
    pub fn open(dir: &Path, embedder: Box<dyn Embedder>) -> Result<Rag, RagError>;
    pub fn add_memory(&mut self, text: &str, source: &str,
                      metadata: serde_json::Value) -> Result<Id, RagError>;
    pub fn ingest_file(&mut self, path: &Path,
                       chunk_words: usize, overlap_words: usize) -> Result<Vec<Id>, RagError>;
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<Hit>, RagError>;
}
```

---

## 4. Data model, persistence & the central invariant

On-disk layout (§5), under a caller-chosen directory (default `./ragstore`):

```
<store_dir>/
    docs/<id>.txt        # raw source text for one entry, verbatim
    vectors.bin          # contiguous f32 row-major, N x dim
    meta.json            # ordered array of MetaRecord, row-aligned with vectors
    manifest.json        # { "dim": 384 } — records dim for consistency checks
```

`MetaRecord`: `{ id: String, source: String, metadata: Value, added_at: u64 }`.

**Central invariant (§5/§8):** `vectors.bin` row `i` ⇄ `meta[i]` ⇄
`docs/<meta[i].id>.txt`. Every mutation preserves this alignment.

In-memory Store state:
- `Vec<f32>` flat buffer (N×dim),
- `Vec<MetaRecord>`,
- `HashMap<Id, usize>` mapping id → row index for O(1) upsert lookup (§6.4).

**`add` flow:**
1. `id = sha256(text)[:16]`; write `docs/<id>.txt`.
2. If `id` already indexed → overwrite that row's vector slice and meta record
   in place. Else → append the vector rows and meta record; insert into the
   index.
3. Persist: rewrite `vectors.bin` and `meta.json` in full (§6.5) using
   write-to-temp-then-atomic-rename so a crash cannot leave a torn file.

**Load:** read `manifest.json` + `meta.json` + `vectors.bin`; rebuild the
`HashMap`. Reconstructing over an existing directory reproduces identical
entries, ordering, and search results (§8 reload fidelity).

**`search(query_vec, k)`:** empty store → `[]`; else `rayon` parallel
dot-product over every row, partial-sort the top-k, clamp `k` to N (§8). Each
returned `Hit { score, id, text, source, metadata }` reads its `text` from
`docs/<id>.txt` lazily, for the k hits only.

**Chunking (`chunk_text`, §6.3):** fixed-width overlapping word windows;
`step = max(1, chunk_words - overlap_words)`; input shorter than a window is a
single chunk; empty input yields no chunks.

**Edge cases enforced (§8):** empty-store search never errors; `k > N` returns
all; a query/add vector whose length ≠ stored `dim` → `StoreError::DimMismatch`;
adding identical text twice yields one entry updated in place (dedup).

---

## 5. CLI surface (§9)

Single binary, global `--store <dir>` (default `./ragstore`). All four are
fully offline.

| Command | Arguments | Effect |
|---|---|---|
| `add` | `<text>` `[--source]` | Store one already-compacted memory. |
| `ingest` | `<path>` `[--chunk-words] [--overlap]` | Chunk a text file, store each chunk. |
| `search` | `<query>` `[-k]` | Print top-k semantic matches (JSON). |
| `list` | — | List stored entries (id, source, metadata) as JSON. |

`search`/`list` emit JSON to stdout so a host harness can parse them.

---

## 6. MCP server

`ragamuffin mcp` (or a dedicated transport entry) starts an `rmcp` stdio server
exposing two tools over the same `Rag` instance:

- **`search_memory`** — input `{ query: String, k?: Integer (default 4) }`;
  returns the hits formatted as a numbered text list (source, score, stored
  text) so the host model can judge and cite what it got (§7.3).
- **`save_memory`** — input `{ text: String, source?: String (default "chat") }`;
  upserts via `rag.add_memory` and returns a confirmation string (new id +
  updated entry count).

Tool handlers catch errors and return them as tool-result **error text** rather
than failing the transport, so the host model can recover (§7.3).

---

## 7. Error handling

- `thiserror` enums per layer (`EmbedError`, `StoreError`, `RagError`),
  propagated with `?`.
- `main`/CLI handlers use `anyhow` with `.context()` for human-facing messages.
- No `.unwrap()` in library paths; `.expect()` only for documented invariant
  violations (CLAUDE.md).

---

## 8. Testing strategy (§10)

All tests run offline via the `FakeEmbedder` — no network, API key, or model
download.

- **Store:** add→reload reproduces entries (persistence); identical text does
  not duplicate (dedup); a query ranks the semantically closest fake-embedded
  entry first (ranking); empty-store search returns `[]`; `k > N` returns all.
- **RAG:** `chunk_text` produces multiple overlapping chunks for long input;
  ingest→search round-trips.
- **MCP:** drive the tool dispatcher with synthetic tool inputs (no transport) —
  assert `save_memory` writes and the entry is then retrievable via
  `search_memory`; assert a request doing both a search and a save executes
  both.

---

## 9. Build order

Mirrors §13, minus the chat harness, plus MCP:

1. Embedder trait + `FakeEmbedder` — unblocks offline testing immediately.
2. Store: add / persist / load — verify round-trip.
3. Store: search — verify ranking and k-clamping with the fake.
4. RAG: chunking, ingest, search — verify ingest→search round-trip.
5. CLI: add / ingest / search / list — a usable offline tool here.
6. Real `fastembed` embedder behind the same trait — swap in; fake-backed tests
   still pass.
7. MCP server: `search_memory` then `save_memory` — verify the dispatcher with
   synthetic tool calls.

By step 5 there is a working local semantic-search CLI; by step 7 it is a
memory backend any MCP-speaking harness can use.

---

## 10. Out of scope (MVP)

Per `DESIGN.md` §2 non-goals, plus this spec's scoping: no multi-user serving,
no auth, no approximate-nearest-neighbor index, no concurrent-writer handling
(last write wins), no embedded chat/LLM loop, no streaming. The §12 extension
points (auto-compaction, confirmation gate, smarter chunking, re-ranking,
metadata filtering, real vector index) are deferred.
