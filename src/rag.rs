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
    // Clamp to at least 1 to prevent infinite loops or empty chunks when
    // chunk_words = 0 is passed (e.g. via `--chunk-words 0`).
    let chunk_words = chunk_words.max(1);
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
        let chunks = chunk_text(&text, chunk_words, overlap_words);
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = self.embedder.embed(&chunks)?;
        let source = path.to_string_lossy().to_string();
        let mut ids = Vec::with_capacity(chunks.len());
        for (i, (chunk, vector)) in chunks.iter().zip(vectors.iter()).enumerate() {
            let id = self
                .store
                .add(vector, chunk, &source, serde_json::json!({ "chunk": i }))?;
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
    fn chunk_text_overlaps_long_input() {
        let text = (0..100)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" ");
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
    fn chunk_text_zero_width_is_clamped() {
        // chunk_words = 0 must not produce empty or infinite chunks.
        let chunks = chunk_text("a b c d e", 0, 0);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| !c.is_empty()));
    }

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
}
