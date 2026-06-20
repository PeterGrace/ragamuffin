# ragamuffin MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first semantic memory backend — embed text, store vectors + raw text on disk, search by meaning — exposed as an offline CLI and an MCP server that LLM coding harnesses call into.

**Architecture:** Four layers, each depending only on the one below: `Embedder` (text→unit vector) and `Store` (vectors + text + metadata on disk, brute-force top-k) sit at the bottom; `Rag` combines them; `cli` and `mcp` are thin sibling adapters over the same `Rag`. The embedder is a trait so tests inject a deterministic fake and run fully offline.

**Tech Stack:** Rust 2021, `fastembed` (real embedder, BGE-small, 384-dim), `rayon` (parallel dot product), `sha2` (content-hash IDs), `clap` (CLI), `rmcp` (MCP server over stdio), `serde_json` (metadata), `thiserror`/`anyhow` (errors), `tempfile` (test dirs).

**Reference:** `docs/superpowers/specs/2026-06-20-ragamuffin-mvp-design.md` and `DESIGN.md`. Section refs like "§6.4" point at `DESIGN.md`.

**Implementation note on vectors file:** instead of `bytemuck` (named in the spec), we encode `vectors.bin` with explicit little-endian `f32::to_le_bytes` / `from_le_bytes`. This avoids byte-slice alignment panics on read and guarantees a portable, fixed endianness (§5 portability) with no extra dependency.

**Testing note:** all tests are in-crate `#[cfg(test)]` unit tests. This lets the `FakeEmbedder` stay `#[cfg(test)]`-gated yet be usable from every module's tests (including the MCP server's), with no separate integration-test crate.

---

## Task 0: Project foundation (Cargo.toml, lib skeleton, errors)

**Files:**
- Modify: `Cargo.toml`
- Modify: `.cargo/config.toml` (remove `tokio_unstable`)
- Create: `src/lib.rs`
- Create: `src/error.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Replace `Cargo.toml` dependencies**

```toml
[package]
name = "ragamuffin"
version = "0.1.0"
edition = "2021"
authors = ["Peter Grace <pete.grace@gmail.com>"]

[lib]
name = "ragamuffin"
path = "src/lib.rs"

[[bin]]
name = "ragamuffin"
path = "src/main.rs"

[dependencies]
anyhow = "1.0"
clap = { version = "4", features = ["derive"] }
dotenv = "0.15"
fastembed = "5"
rayon = "1"
rmcp = { version = "0.8", features = ["server", "macros", "transport-io"] }
schemars = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
thiserror = "2"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "sync", "io-std"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["fmt", "env-filter"] }

[dev-dependencies]
tempfile = "3"
```

Note: `rmcp` and `schemars` versions are coupled. After the first `cargo build` that touches `src/mcp.rs` (Task 7), if the `JsonSchema` derive fails to resolve, run `cargo tree -p schemars` and pin `schemars` to the version `rmcp` depends on. Use the latest `rmcp` `0.x` if `0.8` is unavailable.

- [ ] **Step 2: Remove the `tokio_unstable` rustflag**

Overwrite `.cargo/config.toml` with an empty build section (the console-subscriber scaffold is gone):

```toml
[build]
```

- [ ] **Step 3: Create `src/error.rs`**

```rust
//! Layered error types. Each layer has its own enum; higher layers wrap lower
//! ones with `#[from]` so `?` propagates cleanly (CLAUDE.md mandates thiserror).

use thiserror::Error;

/// Failure while turning text into vectors.
#[derive(Debug, Error)]
pub enum EmbedError {
    /// The underlying embedding model returned an error.
    #[error("embedding model failed: {0}")]
    Model(String),
}

/// Failure in the on-disk vector store.
#[derive(Debug, Error)]
pub enum StoreError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// A vector's length did not match the store's recorded dimension (§8).
    #[error("vector dimension mismatch: store has {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}

/// Failure in the RAG orchestration layer.
#[derive(Debug, Error)]
pub enum RagError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Embed(#[from] EmbedError),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    /// The embedder returned no vector for a single input.
    #[error("embedder returned no vector")]
    EmptyEmbedding,
}
```

- [ ] **Step 4: Create `src/lib.rs`**

```rust
//! ragamuffin: a local-first, transparent semantic memory backend.
//!
//! Layers (each depends only on the one below): [`embedder`] and [`store`] at
//! the bottom, [`rag`] combining them, and [`cli`] / [`mcp`] as thin adapters.

pub mod cli;
pub mod embedder;
pub mod error;
pub mod mcp;
pub mod rag;
pub mod store;
```

- [ ] **Step 5: Replace `src/main.rs`**

```rust
//! Binary entry point: parse the CLI and dispatch. All real logic lives in the
//! library crate so it can be unit-tested.

use clap::Parser;
use ragamuffin::cli::{run, Cli};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load a local .env if present (e.g. RUST_LOG); ignore if absent.
    let _ = dotenv::dotenv();
    init_tracing();
    let cli = Cli::parse();
    run(cli).await
}

/// Minimal stderr tracing so logs never pollute stdout (stdout carries JSON and
/// the MCP stdio protocol).
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}
```

- [ ] **Step 6: Create placeholder modules so the crate compiles**

Create each of these with a temporary empty body (later tasks fill them):

`src/embedder/mod.rs`:
```rust
//! Placeholder, implemented in Task 1.
```
`src/store.rs`:
```rust
//! Placeholder, implemented in Task 2.
```
`src/rag.rs`:
```rust
//! Placeholder, implemented in Task 4.
```
`src/cli.rs`:
```rust
//! Placeholder, implemented in Task 5.
use crate::error::RagError; // silence unused-import lints later; remove if needed

/// Temporary stub so `main.rs` links; replaced in Task 5.
pub struct Cli;
impl Cli {
    pub fn parse() -> Self { Cli }
}
pub async fn run(_cli: Cli) -> anyhow::Result<()> { Ok(()) }
```
`src/mcp.rs`:
```rust
//! Placeholder, implemented in Task 7.
```

(The `cli.rs` stub deliberately does not derive `Parser` yet; `main.rs` calls `Cli::parse()` and `run`, both of which the stub provides. Task 5 replaces it.)

- [ ] **Step 7: Verify it compiles**

Run: `cargo build`
Expected: builds with warnings only (unused imports). No errors.

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml Cargo.lock .cargo/config.toml src/
git commit -m "chore: project foundation, deps, and layered error types"
```

---

## Task 1: Embedder trait + FakeEmbedder

**Files:**
- Modify: `src/embedder/mod.rs`
- Create: `src/embedder/fake.rs`

- [ ] **Step 1: Write the failing test**

Create `src/embedder/fake.rs`:

```rust
//! A deterministic, offline embedder for tests (§10). It counts occurrences of
//! a tiny fixed vocabulary and L2-normalizes, so semantically similar text
//! (sharing vocabulary words) produces similar vectors — predictably, with no
//! model download or network.

use crate::embedder::{normalize, Embedder};
use crate::error::EmbedError;

/// Bag-of-words embedder over a fixed vocabulary.
pub struct FakeEmbedder {
    vocab: Vec<String>,
}

impl FakeEmbedder {
    /// Construct with a small fixed vocabulary.
    pub fn new() -> Self {
        let vocab = ["cat", "dog", "rust", "python", "memory", "vector", "food", "code"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        Self { vocab }
    }
}

impl Default for FakeEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for FakeEmbedder {
    fn dim(&self) -> usize {
        self.vocab.len()
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            let lower = text.to_lowercase();
            // `matches` counts non-overlapping occurrences of each vocab word.
            let mut v: Vec<f32> = self
                .vocab
                .iter()
                .map(|w| lower.matches(w.as_str()).count() as f32)
                .collect();
            normalize(&mut v);
            out.push(v);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similar_text_is_closer_than_dissimilar() {
        let e = FakeEmbedder::new();
        let vecs = e
            .embed(&[
                "rust code".to_string(),
                "rust memory".to_string(),
                "cat food".to_string(),
            ])
            .unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        // "rust code" is nearer "rust memory" (shared "rust") than "cat food".
        assert!(dot(&vecs[0], &vecs[1]) > dot(&vecs[0], &vecs[2]));
    }

    #[test]
    fn rows_are_unit_length() {
        let e = FakeEmbedder::new();
        let v = &e.embed(&["rust code".to_string()]).unwrap()[0];
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p ragamuffin embedder::fake`
Expected: FAIL to compile — `Embedder` and `normalize` are not yet defined in `src/embedder/mod.rs`.

- [ ] **Step 3: Implement `src/embedder/mod.rs`**

```rust
//! The embedder layer: the only component that maps language to geometry
//! (§4.1). Output rows MUST be unit length so the store can treat cosine
//! similarity as a plain dot product.

pub mod fastembed;
#[cfg(test)]
pub mod fake;

use crate::error::EmbedError;

/// Maps text to L2-normalized vectors. `Send + Sync` so the MCP server can hold
/// it behind a shared async lock.
pub trait Embedder: Send + Sync {
    /// Embed a batch. Each returned row is L2-normalized (unit length).
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// The dimensionality of every produced vector.
    fn dim(&self) -> usize;
}

/// Scale `v` in place to unit L2 length. A zero vector is left as zeros (its
/// dot product with anything is 0, which is the correct "no similarity").
///
/// Rust note: `iter_mut()` yields `&mut f32`, so `*x /= norm` writes back.
pub fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
```

Also create a stub `src/embedder/fastembed.rs` so the module tree compiles (the real impl lands in Task 6):

```rust
//! Real embedder, implemented in Task 6.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p ragamuffin embedder::fake`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/embedder/
git commit -m "feat: Embedder trait, normalize, and deterministic FakeEmbedder"
```

---

## Task 2: Store — add, persist, load

**Files:**
- Create: `src/store.rs` (replace placeholder)

- [ ] **Step 1: Write the failing test**

Create `src/store.rs` with the test module first (the impl follows in Step 3):

```rust
//! The store: persistence plus nearest-neighbor search. Knows nothing about
//! embedding models — it only receives and compares vectors (§4.2).
//!
//! Central invariant (§5): `vectors[i]` ⇄ `meta[i]` ⇄ `docs/<meta[i].id>.txt`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::StoreError;

/// 16-hex-char content hash identifying an entry (§6.4).
pub type Id = String;

/// One row of metadata, row-aligned with the vectors file and a docs file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaRecord {
    pub id: Id,
    pub source: String,
    pub metadata: Value,
    pub added_at: u64,
}

/// A single search result (§4.2).
#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub score: f32,
    pub id: Id,
    pub text: String,
    pub source: String,
    pub metadata: Value,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    dim: usize,
}

/// In-memory mirror of the on-disk store. Three parallel structures share one
/// ordering; `index` maps id → row for O(1) upsert (§6.4).
pub struct Store {
    dir: PathBuf,
    dim: usize,
    vectors: Vec<f32>, // flat, N * dim, row-major
    meta: Vec<MetaRecord>,
    index: HashMap<Id, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn v(dim: usize, fill: f32) -> Vec<f32> {
        vec![fill; dim]
    }

    #[test]
    fn add_then_reload_reproduces_entries() {
        let dir = tempdir().unwrap();
        let id = {
            let mut s = Store::open(dir.path(), 3).unwrap();
            let id = s
                .add(&v(3, 0.5), "hello world", "manual", serde_json::json!({"k": 1}))
                .unwrap();
            assert_eq!(s.count(), 1);
            id
        };
        // Fresh store over the same directory reproduces the entry (§8 reload).
        let s2 = Store::open(dir.path(), 3).unwrap();
        assert_eq!(s2.count(), 1);
        assert_eq!(s2.all()[0].id, id);
        assert_eq!(s2.all()[0].source, "manual");
    }

    #[test]
    fn identical_text_does_not_duplicate() {
        let dir = tempdir().unwrap();
        let mut s = Store::open(dir.path(), 3).unwrap();
        let a = s.add(&v(3, 0.1), "same text", "manual", Value::Null).unwrap();
        let b = s.add(&v(3, 0.9), "same text", "chat", Value::Null).unwrap();
        assert_eq!(a, b); // same content hash
        assert_eq!(s.count(), 1); // updated in place, not duplicated
        assert_eq!(s.all()[0].source, "chat"); // last write wins
    }

    #[test]
    fn wrong_dimension_is_rejected() {
        let dir = tempdir().unwrap();
        let mut s = Store::open(dir.path(), 3).unwrap();
        let err = s.add(&v(4, 0.1), "x", "manual", Value::Null).unwrap_err();
        assert!(matches!(err, StoreError::DimMismatch { .. }));
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p ragamuffin store::tests::add_then_reload_reproduces_entries`
Expected: FAIL to compile — `Store::open`, `add`, `count`, `all` are not defined.

- [ ] **Step 3: Implement the Store methods**

Add this `impl` block to `src/store.rs` (after the `Store` struct, before the `#[cfg(test)]` module):

```rust
impl Store {
    /// Open (creating if needed) a store of the given dimension. If a manifest
    /// already exists, its recorded dim must match (§8 dimension consistency).
    pub fn open(dir: &Path, dim: usize) -> Result<Store, StoreError> {
        fs::create_dir_all(dir.join("docs"))?;
        let manifest_path = dir.join("manifest.json");
        if manifest_path.exists() {
            let m: Manifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
            if m.dim != dim {
                return Err(StoreError::DimMismatch { expected: dim, got: m.dim });
            }
        }
        Self::load(dir, dim)
    }

    /// Open an existing store, reading its dimension from the manifest. Used by
    /// offline `list`, which has no embedder to ask for a dimension.
    pub fn open_existing(dir: &Path) -> Result<Store, StoreError> {
        let m: Manifest = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
        Self::load(dir, m.dim)
    }

    fn load(dir: &Path, dim: usize) -> Result<Store, StoreError> {
        let meta_path = dir.join("meta.json");
        let vectors_path = dir.join("vectors.bin");
        let meta: Vec<MetaRecord> = if meta_path.exists() {
            serde_json::from_slice(&fs::read(&meta_path)?)?
        } else {
            Vec::new()
        };
        let vectors: Vec<f32> = if vectors_path.exists() {
            // Read fixed little-endian f32s; `chunks_exact(4)` avoids any
            // alignment requirement on the raw byte buffer.
            fs::read(&vectors_path)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else {
            Vec::new()
        };
        let index = meta
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id.clone(), i))
            .collect();
        Ok(Store { dir: dir.to_path_buf(), dim, vectors, meta, index })
    }

    /// Upsert keyed by content hash of `text` (§6.4). Returns the entry id.
    pub fn add(
        &mut self,
        vector: &[f32],
        text: &str,
        source: &str,
        metadata: Value,
    ) -> Result<Id, StoreError> {
        if vector.len() != self.dim {
            return Err(StoreError::DimMismatch { expected: self.dim, got: vector.len() });
        }
        let id = content_id(text);
        fs::write(self.dir.join("docs").join(format!("{id}.txt")), text)?;
        let record = MetaRecord {
            id: id.clone(),
            source: source.to_string(),
            metadata,
            added_at: now_secs(),
        };
        if let Some(&row) = self.index.get(&id) {
            // Overwrite the existing row's vector slice and meta in place.
            self.vectors[row * self.dim..(row + 1) * self.dim].copy_from_slice(vector);
            self.meta[row] = record;
        } else {
            let row = self.meta.len();
            self.vectors.extend_from_slice(vector);
            self.meta.push(record);
            self.index.insert(id.clone(), row);
        }
        self.persist()?;
        Ok(id)
    }

    /// Number of stored entries.
    pub fn count(&self) -> usize {
        self.meta.len()
    }

    /// All metadata records, in row order.
    pub fn all(&self) -> &[MetaRecord] {
        &self.meta
    }

    /// Rewrite all on-disk files (§6.5). Each write is temp-file-then-rename so
    /// a crash cannot leave a torn file.
    fn persist(&self) -> Result<(), StoreError> {
        let mut bytes = Vec::with_capacity(self.vectors.len() * 4);
        for f in &self.vectors {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        write_atomic(&self.dir.join("vectors.bin"), &bytes)?;
        write_atomic(&self.dir.join("meta.json"), &serde_json::to_vec_pretty(&self.meta)?)?;
        write_atomic(
            &self.dir.join("manifest.json"),
            &serde_json::to_vec_pretty(&Manifest { dim: self.dim })?,
        )?;
        Ok(())
    }
}

/// SHA-256 of the text, truncated to 16 hex chars (64 bits) (§6.4).
fn content_id(text: &str) -> Id {
    let digest = Sha256::digest(text.as_bytes());
    // First 8 bytes -> 16 hex chars, without pulling in a hex crate.
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Seconds since the Unix epoch (0 if the clock is before the epoch).
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Write `bytes` to `path` atomically via a temp file + rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p ragamuffin store::`
Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/store.rs
git commit -m "feat: Store add/persist/load with idempotent content-hash upsert"
```

---

## Task 3: Store — search

**Files:**
- Modify: `src/store.rs`

- [ ] **Step 1: Write the failing test**

Add these tests inside the existing `#[cfg(test)] mod tests` in `src/store.rs`:

```rust
    #[test]
    fn empty_store_search_returns_empty() {
        let dir = tempdir().unwrap();
        let s = Store::open(dir.path(), 3).unwrap();
        assert!(s.search(&v(3, 1.0), 4).unwrap().is_empty());
    }

    #[test]
    fn search_ranks_closest_first_and_clamps_k() {
        let dir = tempdir().unwrap();
        let mut s = Store::open(dir.path(), 2).unwrap();
        // Unit vectors pointing in distinct directions.
        s.add(&[1.0, 0.0], "east", "manual", Value::Null).unwrap();
        s.add(&[0.0, 1.0], "north", "manual", Value::Null).unwrap();
        // Query closest to "east".
        let hits = s.search(&[0.9, 0.1], 4).unwrap(); // k > N -> clamps to 2
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].text, "east");
        assert!(hits[0].score > hits[1].score);
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p ragamuffin store::tests::search_ranks_closest_first_and_clamps_k`
Expected: FAIL to compile — `Store::search` is not defined.

- [ ] **Step 3: Implement `search`**

Add to the top of `src/store.rs` (with the other `use` lines):

```rust
use rayon::prelude::*;
```

Add this method inside the `impl Store` block:

```rust
    /// Top-k entries by descending cosine similarity (= dot product, since all
    /// vectors are unit length). Empty store -> empty list; k clamps to N (§8).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>, StoreError> {
        if query.len() != self.dim {
            return Err(StoreError::DimMismatch { expected: self.dim, got: query.len() });
        }
        let n = self.meta.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        // Parallel dot product over every row (§6.2). Rust note: `into_par_iter`
        // comes from the rayon prelude and turns the range into a parallel one.
        let mut scored: Vec<(f32, usize)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &self.vectors[i * self.dim..(i + 1) * self.dim];
                let dot = row.iter().zip(query).map(|(a, b)| a * b).sum::<f32>();
                (dot, i)
            })
            .collect();
        // Sort descending by score. `partial_cmp` because f32 is not `Ord`.
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = k.min(n);
        let mut hits = Vec::with_capacity(k);
        for &(score, i) in scored.iter().take(k) {
            let m = &self.meta[i];
            // Read raw text lazily, only for the k results we return (§4.4).
            let text = fs::read_to_string(self.dir.join("docs").join(format!("{}.txt", m.id)))?;
            hits.push(Hit {
                score,
                id: m.id.clone(),
                text,
                source: m.source.clone(),
                metadata: m.metadata.clone(),
            });
        }
        Ok(hits)
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p ragamuffin store::`
Expected: all store tests PASS (5 total).

- [ ] **Step 5: Commit**

```bash
git add src/store.rs
git commit -m "feat: brute-force parallel top-k search over the store"
```

---

## Task 4: RAG — chunking, add_memory, ingest, search

**Files:**
- Create: `src/rag.rs` (replace placeholder)

- [ ] **Step 1: Write the failing test for chunking**

Create `src/rag.rs`:

```rust
//! RAG orchestration: combines an [`Embedder`](crate::embedder::Embedder) and a
//! [`Store`](crate::store::Store) into the two operations callers want —
//! remember and recall (§4.3).

use std::path::Path;

use serde_json::Value;

use crate::embedder::Embedder;
use crate::error::RagError;
use crate::store::{Hit, Id, MetaRecord, Store};

/// Combines an embedder and a store. Owns a boxed embedder so tests can inject
/// the fake (ideal #9).
pub struct Rag {
    store: Store,
    embedder: Box<dyn Embedder>,
}

/// Split `text` into overlapping fixed-width word windows (§6.3) so an idea
/// straddling a boundary stays retrievable from either side.
pub fn chunk_text(text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    if words.len() <= chunk_words {
        return vec![words.join(" ")];
    }
    let step = chunk_words.saturating_sub(overlap_words).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + chunk_words).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end >= words.len() {
            break;
        }
        start += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::fake::FakeEmbedder;
    use tempfile::tempdir;

    #[test]
    fn chunk_text_overlaps_long_input() {
        let text = (0..100).map(|i| i.to_string()).collect::<Vec<_>>().join(" ");
        let chunks = chunk_text(&text, 30, 10); // step = 20
        assert!(chunks.len() > 1);
        // Overlap: chunk 0 ends with "29", chunk 1 starts at word index 20.
        assert!(chunks[0].split(' ').next_back().unwrap() == "29");
        assert!(chunks[1].split(' ').next().unwrap() == "20");
    }

    #[test]
    fn chunk_text_short_input_is_one_chunk() {
        assert_eq!(chunk_text("a b c", 30, 10), vec!["a b c".to_string()]);
        assert!(chunk_text("   ", 30, 10).is_empty());
    }

    #[test]
    fn add_memory_then_search_roundtrips() {
        let dir = tempdir().unwrap();
        let mut rag = Rag::open(dir.path(), Box::new(FakeEmbedder::new())).unwrap();
        rag.add_memory("rust code is fast", "manual", Value::Null).unwrap();
        rag.add_memory("the cat ate food", "manual", Value::Null).unwrap();
        let hits = rag.search("rust memory", 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "rust code is fast"); // shares "rust"
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p ragamuffin rag::tests::add_memory_then_search_roundtrips`
Expected: FAIL to compile — `Rag::open`, `add_memory`, `search` are not defined.

- [ ] **Step 3: Implement the `Rag` methods**

Add this `impl` block to `src/rag.rs` (after `chunk_text`, before the test module):

```rust
impl Rag {
    /// Open a store under `dir`, sized to the embedder's dimension.
    pub fn open(dir: &Path, embedder: Box<dyn Embedder>) -> Result<Rag, RagError> {
        let store = Store::open(dir, embedder.dim())?;
        Ok(Rag { store, embedder })
    }

    /// Embed once and upsert (§4.3).
    pub fn add_memory(&mut self, text: &str, source: &str, metadata: Value) -> Result<Id, RagError> {
        let vector = self.embed_one(text)?;
        Ok(self.store.add(&vector, text, source, metadata)?)
    }

    /// Read, chunk, embed the batch, and store each chunk (§4.3). The source is
    /// the file path; each chunk records its index in metadata.
    pub fn ingest_file(
        &mut self,
        path: &Path,
        chunk_words: usize,
        overlap_words: usize,
    ) -> Result<Vec<Id>, RagError> {
        let text = std::fs::read_to_string(path)?;
        let chunks = chunk_text(&text, chunk_words, overlap_words);
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = self.embedder.embed(&chunks)?;
        let source = path.to_string_lossy().to_string();
        let mut ids = Vec::with_capacity(chunks.len());
        for (i, (chunk, vector)) in chunks.iter().zip(vectors.iter()).enumerate() {
            let id = self.store.add(vector, chunk, &source, serde_json::json!({ "chunk": i }))?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Embed the query and return the top-k hits (§4.3).
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<Hit>, RagError> {
        let vector = self.embed_one(query)?;
        Ok(self.store.search(&vector, k)?)
    }

    /// Number of stored entries.
    pub fn count(&self) -> usize {
        self.store.count()
    }

    /// All metadata records.
    pub fn all(&self) -> &[MetaRecord] {
        self.store.all()
    }

    /// Embed a single string, returning its one vector.
    fn embed_one(&self, text: &str) -> Result<Vec<f32>, RagError> {
        let vectors = self.embedder.embed(&[text.to_string()])?;
        vectors.into_iter().next().ok_or(RagError::EmptyEmbedding)
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p ragamuffin rag::`
Expected: all three RAG tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag.rs
git commit -m "feat: RAG chunking, add_memory, ingest_file, and search"
```

---

## Task 5: CLI — add / ingest / search / list

**Files:**
- Create: `src/cli.rs` (replace placeholder)

- [ ] **Step 1: Write the failing test**

Replace `src/cli.rs` with the command definitions plus a test that exercises the offline handlers through a shared helper (no real model — the test injects a fake-backed `Rag` directly):

```rust
//! CLI surface (§9). A thin adapter over [`Rag`](crate::rag::Rag): parse args,
//! call one method, print. `search`/`list` emit JSON so host harnesses can
//! parse them. `add`/`ingest`/`search` use the real embedder; `list` is fully
//! offline and never loads a model.

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};

use crate::embedder::fastembed::FastEmbedder;
use crate::rag::Rag;
use crate::store::Store;

/// ragamuffin: local-first semantic memory.
#[derive(Parser)]
#[command(name = "ragamuffin", version, about = "Local-first semantic memory backend")]
pub struct Cli {
    /// Store directory.
    #[arg(long, global = true, default_value = "./ragstore")]
    pub store: PathBuf,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Store one already-compacted memory.
    Add {
        text: String,
        #[arg(long, default_value = "manual")]
        source: String,
    },
    /// Chunk a text file and store each chunk.
    Ingest {
        path: PathBuf,
        #[arg(long = "chunk-words", default_value_t = 180)]
        chunk_words: usize,
        #[arg(long = "overlap", default_value_t = 40)]
        overlap_words: usize,
    },
    /// Print the top-k semantic matches as JSON.
    Search {
        query: String,
        #[arg(short, default_value_t = 4)]
        k: usize,
    },
    /// List stored entries (id, source, metadata) as JSON.
    List,
    /// Run as an MCP server over stdio.
    Mcp,
}

/// Dispatch a parsed CLI to its handler.
pub async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Command::Add { text, source } => {
            let mut rag = open_rag(&cli.store)?;
            let id = rag.add_memory(&text, &source, serde_json::Value::Null)?;
            println!("stored {id} (count={})", rag.count());
        }
        Command::Ingest { path, chunk_words, overlap_words } => {
            let mut rag = open_rag(&cli.store)?;
            let ids = rag.ingest_file(&path, chunk_words, overlap_words)?;
            println!("stored {} chunks (count={})", ids.len(), rag.count());
        }
        Command::Search { query, k } => {
            let rag = open_rag(&cli.store)?;
            let hits = rag.search(&query, k)?;
            println!("{}", serde_json::to_string_pretty(&hits)?);
        }
        Command::List => {
            // Fully offline: read the store directly, no embedder/model load.
            if !cli.store.join("manifest.json").exists() {
                println!("[]");
                return Ok(());
            }
            let store = Store::open_existing(&cli.store)?;
            println!("{}", serde_json::to_string_pretty(store.all())?);
        }
        Command::Mcp => {
            crate::mcp::serve(&cli.store).await?;
        }
    }
    Ok(())
}

/// Build a RAG instance backed by the real local embedder.
fn open_rag(store: &Path) -> anyhow::Result<Rag> {
    let embedder = FastEmbedder::new().context("loading embedding model")?;
    Rag::open(store, Box::new(embedder)).context("opening store")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::fake::FakeEmbedder;
    use tempfile::tempdir;

    // The handlers above require the real model; here we test the same logic
    // path (add -> search -> list serialization) against a fake-backed Rag.
    #[test]
    fn add_search_list_roundtrip_offline() {
        let dir = tempdir().unwrap();
        let mut rag = Rag::open(dir.path(), Box::new(FakeEmbedder::new())).unwrap();
        rag.add_memory("rust code", "manual", serde_json::Value::Null).unwrap();
        drop(rag);

        // search serializes hits as JSON
        let rag = Rag::open(dir.path(), Box::new(FakeEmbedder::new())).unwrap();
        let hits = rag.search("rust", 4).unwrap();
        let json = serde_json::to_string(&hits).unwrap();
        assert!(json.contains("rust code"));

        // list reads the store offline via open_existing
        let store = Store::open_existing(dir.path()).unwrap();
        let listed = serde_json::to_string(store.all()).unwrap();
        assert!(listed.contains("\"source\":\"manual\""));
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p ragamuffin cli::tests::add_search_list_roundtrip_offline`
Expected: FAIL to compile — `FastEmbedder::new` and `crate::mcp::serve` are not implemented yet (placeholders).

- [ ] **Step 3: Add temporary shims so the crate compiles**

The CLI references `FastEmbedder::new` (Task 6) and `crate::mcp::serve` (Task 7). Add minimal shims so Task 5 compiles and its offline test runs; later tasks replace them.

`src/embedder/fastembed.rs` (temporary):
```rust
//! Real embedder — temporary shim; full impl in Task 6.

use crate::embedder::Embedder;
use crate::error::EmbedError;

pub struct FastEmbedder;

impl FastEmbedder {
    pub fn new() -> Result<Self, EmbedError> {
        Err(EmbedError::Model("not implemented until Task 6".into()))
    }
}

impl Embedder for FastEmbedder {
    fn dim(&self) -> usize {
        384
    }
    fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Err(EmbedError::Model("not implemented until Task 6".into()))
    }
}
```

`src/mcp.rs` (temporary):
```rust
//! MCP server — temporary shim; full impl in Task 7.

use std::path::Path;

pub async fn serve(_store: &Path) -> anyhow::Result<()> {
    anyhow::bail!("mcp not implemented until Task 7")
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p ragamuffin cli::tests::add_search_list_roundtrip_offline`
Expected: PASS. (The shims compile; the test uses the fake embedder, not `FastEmbedder`.)

- [ ] **Step 5: Verify the binary parses commands**

Run: `cargo run -- --help`
Expected: clap prints usage listing `add`, `ingest`, `search`, `list`, `mcp` and the global `--store`.

- [ ] **Step 6: Commit**

```bash
git add src/cli.rs src/embedder/fastembed.rs src/mcp.rs
git commit -m "feat: CLI surface (add/ingest/search/list/mcp) with offline list"
```

---

## Task 6: Real embedder (fastembed)

**Files:**
- Modify: `src/embedder/fastembed.rs` (replace the Task 5 shim)

- [ ] **Step 1: Implement the real `FastEmbedder`**

Replace `src/embedder/fastembed.rs` entirely:

```rust
//! Real, local embedder backed by `fastembed` (ONNX runtime). Downloads a small
//! sentence-transformer (BGE-small-en-v1.5, 384-dim) on first construction and
//! caches it. fastembed already mean-pools and L2-normalizes, satisfying the
//! unit-length contract (§4.1).

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use crate::embedder::Embedder;
use crate::error::EmbedError;

/// Wraps a loaded `fastembed` model.
pub struct FastEmbedder {
    model: TextEmbedding,
    dim: usize,
}

impl FastEmbedder {
    /// Load (downloading on first run) the BGE-small model and record its
    /// dimension by embedding a one-token probe.
    pub fn new() -> Result<Self, EmbedError> {
        let model = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::BGESmallENV15))
            .map_err(|e| EmbedError::Model(e.to_string()))?;
        let probe = model
            .embed(vec!["probe"], None)
            .map_err(|e| EmbedError::Model(e.to_string()))?;
        let dim = probe.first().map(|v| v.len()).unwrap_or(384);
        Ok(Self { model, dim })
    }
}

impl Embedder for FastEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        // fastembed wants `Vec<&str>`; borrow each owned String.
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.model.embed(refs, None).map_err(|e| EmbedError::Model(e.to_string()))
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build`
Expected: builds. fastembed pulls the `ort` ONNX runtime; the first build may download a prebuilt runtime. No code errors.

- [ ] **Step 3: Verify the existing offline tests still pass**

Run: `cargo test -p ragamuffin`
Expected: all unit tests PASS (the fake-backed tests are unaffected by the real embedder).

- [ ] **Step 4: Manual smoke test against the real model (requires network once)**

```bash
cargo run -- --store /tmp/ragstore add "ragamuffin stores memories as vectors on disk"
cargo run -- --store /tmp/ragstore add "the quick brown fox jumps over the lazy dog"
cargo run -- --store /tmp/ragstore search "how are memories persisted" -k 1
```
Expected: the first run downloads the model (logs to stderr); `search` prints a JSON array whose top hit is the "stores memories as vectors" entry with a `score` near the top. Inspect `/tmp/ragstore`: it contains `docs/*.txt`, `vectors.bin`, `meta.json`, `manifest.json`.

- [ ] **Step 5: Commit**

```bash
git add src/embedder/fastembed.rs
git commit -m "feat: real local embedder via fastembed (BGE-small, 384-dim)"
```

---

## Task 7: MCP server

**Files:**
- Modify: `src/mcp.rs` (replace the Task 5 shim)

- [ ] **Step 1: Write the failing test**

Replace `src/mcp.rs` entirely. (If a type/import path below is rejected by your exact `rmcp` version, cross-check it against the upstream `examples/servers/src/common/counter.rs` in the `modelcontextprotocol/rust-sdk` repo — the macro names and `Parameters`/`CallToolResult`/`ServerInfo` shapes are stable; only module paths drift between minor versions.)

```rust
//! MCP server exposing the memory as two tools over stdio (§6 of the spec /
//! §7.3 of DESIGN.md). The host harness's model decides when to call them.

use std::path::Path;
use std::sync::Arc;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler, ServiceExt};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::embedder::fastembed::FastEmbedder;
use crate::rag::Rag;
use crate::store::Hit;

fn default_k() -> usize {
    4
}
fn default_source() -> String {
    "chat".to_string()
}

/// Arguments for `search_memory`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchArgs {
    /// What to recall, described in natural language.
    pub query: String,
    /// Maximum number of notes to return.
    #[serde(default = "default_k")]
    pub k: usize,
}

/// Arguments for `save_memory`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SaveArgs {
    /// The self-contained note to remember.
    pub text: String,
    /// Where the note came from.
    #[serde(default = "default_source")]
    pub source: String,
}

/// MCP server holding the RAG behind an async lock (the harness may call tools
/// concurrently; writes must be serialized).
#[derive(Clone)]
pub struct MemoryServer {
    rag: Arc<Mutex<Rag>>,
    tool_router: ToolRouter<MemoryServer>,
}

#[tool_router]
impl MemoryServer {
    /// Wrap an open RAG as a server.
    pub fn new(rag: Rag) -> Self {
        Self {
            rag: Arc::new(Mutex::new(rag)),
            // `tool_router()` is generated by the `#[tool_router]` macro.
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Search the personal long-term memory by meaning. Returns the most semantically similar saved notes with their source and similarity score."
    )]
    async fn search_memory(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, McpError> {
        let rag = self.rag.lock().await;
        let hits = rag
            .search(&args.query, args.k)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format_hits(&hits))]))
    }

    #[tool(
        description = "Save a self-contained note to the personal long-term memory. Re-saving identical text updates the one existing entry rather than duplicating."
    )]
    async fn save_memory(
        &self,
        Parameters(args): Parameters<SaveArgs>,
    ) -> Result<CallToolResult, McpError> {
        let mut rag = self.rag.lock().await;
        let id = rag
            .add_memory(&args.text, &args.source, serde_json::Value::Null)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let count = rag.count();
        Ok(CallToolResult::success(vec![Content::text(format!(
            "saved {id} (count={count})"
        ))]))
    }
}

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Personal long-term memory. Use search_memory to recall past discussions or \
                 when context is insufficient; use save_memory when asked to remember something \
                 or when a durable fact or decision emerges. Make saved notes self-contained."
                    .to_string(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

/// Format hits as a numbered list so the host model can judge and cite them.
fn format_hits(hits: &[Hit]) -> String {
    if hits.is_empty() {
        return "No matching memories.".to_string();
    }
    hits.iter()
        .enumerate()
        .map(|(i, h)| {
            format!(
                "{}. [{} | score {:.3}] {}",
                i + 1,
                h.source,
                h.score,
                h.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Open a RAG under `store` and serve it over stdio until the client disconnects.
pub async fn serve(store: &Path) -> anyhow::Result<()> {
    let embedder = FastEmbedder::new()?;
    let rag = Rag::open(store, Box::new(embedder))?;
    let service = MemoryServer::new(rag).serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::fake::FakeEmbedder;
    use tempfile::tempdir;

    #[test]
    fn format_hits_numbers_and_cites() {
        let hits = vec![Hit {
            score: 0.9,
            id: "abc".into(),
            text: "rust note".into(),
            source: "chat".into(),
            metadata: serde_json::Value::Null,
        }];
        let out = format_hits(&hits);
        assert!(out.contains("1. [chat | score 0.900] rust note"));
    }

    #[tokio::test]
    async fn save_then_search_via_tools() {
        let dir = tempdir().unwrap();
        let rag = Rag::open(dir.path(), Box::new(FakeEmbedder::new())).unwrap();
        let server = MemoryServer::new(rag);

        // A synthetic save tool call writes the note.
        server
            .save_memory(Parameters(SaveArgs {
                text: "rust memory note".into(),
                source: "chat".into(),
            }))
            .await
            .unwrap();
        assert_eq!(server.rag.lock().await.count(), 1);

        // A synthetic search tool call retrieves it. Serialize the whole result
        // to JSON and substring-match, to avoid coupling to rmcp internals.
        let result = server
            .search_memory(Parameters(SearchArgs { query: "rust".into(), k: 4 }))
            .await
            .unwrap();
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("rust memory note"));
    }
}
```

- [ ] **Step 2: Run the test to verify it fails (then build to resolve API drift)**

Run: `cargo test -p ragamuffin mcp::`
Expected first: it may fail to compile. If errors mention `schemars`/`JsonSchema` version mismatch, run `cargo tree -p schemars`, then pin `schemars` in `Cargo.toml` to the version `rmcp` uses, and rebuild. If errors mention a moved import path (e.g. `ToolRouter`, `Parameters`, `McpError::internal_error`), open the upstream counter example referenced above and correct the path; the symbols exist, only the module path differs.

- [ ] **Step 3: Make the tests pass**

Apply any version/path corrections from Step 2 until:

Run: `cargo test -p ragamuffin mcp::`
Expected: `format_hits_numbers_and_cites` and `save_then_search_via_tools` PASS.

- [ ] **Step 4: Manual smoke test of the MCP server**

```bash
npx @modelcontextprotocol/inspector cargo run -- --store /tmp/ragstore mcp
```
Expected: the MCP Inspector connects over stdio and lists two tools, `search_memory` and `save_memory`, with their argument schemas. Calling `save_memory` then `search_memory` round-trips. (Requires Node/npx; skip if unavailable — the unit tests already cover the dispatch.)

- [ ] **Step 5: Commit**

```bash
git add src/mcp.rs Cargo.toml Cargo.lock
git commit -m "feat: MCP server exposing search_memory and save_memory over stdio"
```

---

## Task 8: Quality gate + documentation

**Files:**
- Create: `changelog/YYYY-MM-DD-HHMM-mvp.md`
- Create: `README.md`

- [ ] **Step 1: Format, lint, and test**

Run:
```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```
Expected: `fmt` makes no further changes; `clippy` reports zero warnings; all tests pass. Fix any clippy findings (e.g. remove the leftover unused import in `cli.rs` if present) before committing.

- [ ] **Step 2: Write the changelog entry**

CLAUDE.md requires a dated changelog doc per iteration. Create `changelog/<today>-<HHMM>-mvp.md` (use the real date/time, e.g. `2026-06-20-1530-mvp.md`):

```markdown
# ragamuffin MVP — <YYYY-MM-DD HH:MM>

Initial implementation of the local-first semantic memory backend.

## Added
- `Embedder` trait with a deterministic `FakeEmbedder` (tests) and a real
  `FastEmbedder` (fastembed / BGE-small, 384-dim).
- `Store`: idempotent content-hash upsert, little-endian `vectors.bin`,
  `meta.json`, per-entry `docs/*.txt`, atomic rewrites, brute-force parallel
  top-k search.
- `Rag`: overlapping word-window chunking, `add_memory`, `ingest_file`, `search`.
- CLI: `add`, `ingest`, `search` (JSON), `list` (offline JSON), `mcp`.
- MCP server (`rmcp`, stdio) exposing `search_memory` and `save_memory`.

## Notes
- The §7 Anthropic chat harness from DESIGN.md is intentionally omitted; the
  host harness's model drives tool use via MCP.
- `vectors.bin` uses explicit little-endian f32 encoding (portable, no
  alignment constraints) instead of bytemuck.
```

- [ ] **Step 3: Write a short `README.md`**

```markdown
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

`add`, `ingest`, and `search` use a local embedding model (downloaded on first
use). `list` is fully offline.

## MCP server

    ragamuffin --store ./ragstore mcp

Speaks the Model Context Protocol over stdio and exposes two tools,
`search_memory` and `save_memory`, for an LLM harness to call.

## Store layout

    ragstore/
        docs/<id>.txt     raw text, one file per entry
        vectors.bin       contiguous little-endian f32, N x dim
        meta.json         metadata, row-aligned with vectors.bin
        manifest.json     { "dim": 384 }

See `DESIGN.md` for the full design and `docs/superpowers/specs/` for the MVP spec.
```

- [ ] **Step 4: Final verification**

Run:
```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```
Expected: all three succeed with no output errors.

- [ ] **Step 5: Commit**

```bash
git add README.md changelog/
git commit -m "docs: MVP changelog and README"
```

---

## Self-Review Notes

**Spec coverage** — every spec section maps to a task:
- §1 scoping (no chat harness) → reflected throughout; MCP replaces it (Task 7).
- §2 tech choices → Task 0 (deps), Tasks 1/6 (embedders), Task 3 (rayon search), Task 2 (sha2 IDs, manual LE vectors).
- §3 module layout → Tasks 0–7 create exactly those files.
- §4 data model / invariants → Task 2 (`add`/persist/load, manifest dim), Task 3 (search edge cases).
- §5 CLI → Task 5.
- §6 MCP → Task 7.
- §7 errors → Task 0 (`error.rs`), enforced in every layer.
- §8 testing → fake embedder (Task 1) + tests in Tasks 2–7.
- §9 build order → Tasks 1–7 follow it.

**Known version risk:** `rmcp`/`schemars` coupling is handled explicitly in Task 7 Step 2 with a concrete remediation (pin `schemars` to `cargo tree` output; cross-check import paths against the upstream counter example). This is the one place the exact API may drift; the plan tells the engineer how to resolve it rather than assuming it compiles first try.

**Type consistency check:** `Embedder::{embed,dim}`, `Store::{open,open_existing,add,search,count,all}`, `Rag::{open,add_memory,ingest_file,search,count,all}`, `Hit`, `MetaRecord`, `Id`, and the MCP `SearchArgs`/`SaveArgs`/`MemoryServer` names are used identically across all tasks.
