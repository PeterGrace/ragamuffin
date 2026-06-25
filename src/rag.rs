//! RAG orchestration: combines an [`Embedder`](crate::embedder::Embedder) and a
//! [`Store`](crate::store::Store) into the two operations callers want —
//! remember and recall (§4.3).

use std::path::Path;

use serde_json::Value;

use crate::chunk::chunk_for_path;
use crate::embedder::Embedder;
use crate::error::RagError;
use crate::store::{Hit, Id, MetaRecord, Store};

/// Combines an embedder and a store. Owns a boxed embedder so tests can inject
/// the fake (ideal #9).
pub struct Rag {
    store: Store,
    embedder: Box<dyn Embedder>,
}

impl Rag {
    /// Open a store under `dir`, sized to the embedder's dimension.
    pub fn open(dir: &Path, embedder: Box<dyn Embedder>) -> Result<Rag, RagError> {
        let store = Store::open(dir, embedder.dim())?;
        Ok(Rag { store, embedder })
    }

    /// Embed once and upsert (§4.3).
    pub fn add_memory(
        &mut self,
        text: &str,
        source: &str,
        metadata: Value,
    ) -> Result<Id, RagError> {
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
        let chunks = chunk_for_path(path, &text, chunk_words, overlap_words);
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        // Embed the chunk bodies in one batch; metadata travels with each chunk.
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let vectors = self.embedder.embed(&texts)?;
        let source = path.to_string_lossy().to_string();
        let mut ids = Vec::with_capacity(chunks.len());
        for (chunk, vector) in chunks.iter().zip(vectors.iter()) {
            let id = self
                .store
                .add(vector, &chunk.text, &source, chunk.metadata.clone())?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::fake::FakeEmbedder;
    use tempfile::tempdir;

    #[test]
    fn add_memory_then_search_roundtrips() {
        let dir = tempdir().unwrap();
        let mut rag = Rag::open(dir.path(), Box::new(FakeEmbedder::new())).unwrap();
        rag.add_memory("rust code is fast", "manual", Value::Null)
            .unwrap();
        rag.add_memory("the cat ate food", "manual", Value::Null)
            .unwrap();
        let hits = rag.search("rust memory", 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "rust code is fast"); // shares "rust"
    }

    #[test]
    fn ingest_json_file_stores_per_record_metadata() {
        let dir = tempdir().unwrap();
        let src = tempdir().unwrap();
        let json_path = src.path().join("records.json");
        std::fs::write(
            &json_path,
            r#"[{"title":"rust memory notes","year":2024},
                {"title":"the cat ate food","year":2023}]"#,
        )
        .unwrap();

        let mut rag = Rag::open(dir.path(), Box::new(FakeEmbedder::new())).unwrap();
        let ids = rag.ingest_file(&json_path, 180, 40).unwrap();
        assert_eq!(ids.len(), 2); // one chunk per record

        // Per-record metadata is stored: source_kind, record index, scalar year.
        let records = rag.all();
        assert!(records
            .iter()
            .any(|m| m.metadata["source_kind"] == serde_json::json!("json")
                && m.metadata["year"] == serde_json::json!(2024)));

        // A flattened string field is retrievable by semantic search.
        let hits = rag.search("rust", 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("rust memory notes"));
    }
}
