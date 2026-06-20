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
