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
