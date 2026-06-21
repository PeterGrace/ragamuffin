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

/// Choose a chunker by file extension: markdown (`.md` / `.markdown`,
/// case-insensitive) gets heading-aware chunking; every other extension uses the
/// fixed-width word-window [`chunk_text`].
pub fn chunk_for_path(
    path: &Path,
    text: &str,
    chunk_words: usize,
    overlap_words: usize,
) -> Vec<String> {
    let is_md = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("md") || e.eq_ignore_ascii_case("markdown"))
        .unwrap_or(false);
    if is_md {
        chunk_markdown(text, chunk_words, overlap_words)
    } else {
        chunk_text(text, chunk_words, overlap_words)
    }
}

/// Split markdown into self-contained, heading-rooted chunks. A chunk is a
/// heading line plus its body up to the next heading of any level; content
/// before the first heading (preamble) becomes its own chunk. A section longer
/// than `chunk_words` is sub-split with [`chunk_text`] so no chunk is unbounded.
/// Empty or whitespace-only input yields no chunks. A heading-less document
/// behaves like [`chunk_text`].
pub fn chunk_markdown(text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String> {
    // Accumulate lines into sections, starting a new section at each heading
    // (unless the current section is still empty, e.g. the very first line).
    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        if is_heading(line) && !current.trim().is_empty() {
            sections.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        sections.push(current);
    }
    // Emit each non-empty section as a chunk, sub-splitting oversized ones.
    let mut chunks = Vec::new();
    for section in sections {
        let trimmed = section.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.split_whitespace().count() > chunk_words {
            chunks.extend(chunk_text(trimmed, chunk_words, overlap_words));
        } else {
            chunks.push(trimmed.to_string());
        }
    }
    chunks
}

/// True if `line` is an ATX markdown heading: after optional leading spaces it
/// starts with '#'.
fn is_heading(line: &str) -> bool {
    line.trim_start().starts_with('#')
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
    fn chunk_markdown_splits_on_headings_keeps_sections_whole() {
        let md = "# A\nalpha\n# B\nbeta gamma";
        let chunks = chunk_markdown(md, 100, 10);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("# A") && chunks[0].contains("alpha"));
        assert!(chunks[1].contains("# B") && chunks[1].contains("beta gamma"));
    }

    #[test]
    fn chunk_markdown_preamble_is_its_own_chunk() {
        let md = "intro line\n# Section\nbody";
        let chunks = chunk_markdown(md, 100, 10);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("intro line"));
        assert!(!chunks[0].contains("# Section"));
    }

    #[test]
    fn chunk_markdown_subsplits_oversized_section() {
        let big_body = (0..50).map(|i| i.to_string()).collect::<Vec<_>>().join(" ");
        let md = format!("# Big\n{big_body}"); // ~51 words in one section
        let chunks = chunk_markdown(&md, 20, 5); // exceeds 20 -> sub-split
        assert!(chunks.len() > 1);
    }

    #[test]
    fn chunk_markdown_empty_input_no_chunks() {
        assert!(chunk_markdown("   \n  ", 50, 10).is_empty());
    }

    #[test]
    fn chunk_for_path_routes_by_extension() {
        let md = chunk_for_path(Path::new("notes.md"), "# H\nbody here", 100, 10);
        assert_eq!(md.len(), 1);
        assert!(md[0].contains("# H"));
        // A non-markdown extension uses the fixed-window chunker.
        let txt = chunk_for_path(Path::new("notes.txt"), "a b c", 100, 10);
        assert_eq!(txt, vec!["a b c".to_string()]);
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
