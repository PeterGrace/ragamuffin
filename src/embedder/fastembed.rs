//! Real, local embedder backed by `fastembed` (ONNX runtime). Downloads a small
//! sentence-transformer (BGE-small-en-v1.5, 384-dim) on first construction and
//! caches it. fastembed already mean-pools and L2-normalizes, satisfying the
//! unit-length contract (§4.1).
//!
//! **Interior mutability note:** `fastembed::TextEmbedding::embed` takes `&mut self`
//! (it manages internal ONNX session state), but the `Embedder` trait requires `&self`
//! so callers can share it behind an `Arc`. We wrap the model in a `Mutex` to allow
//! shared ownership while still obtaining an exclusive borrow for each inference call.

use std::sync::Mutex;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use crate::embedder::Embedder;
use crate::error::EmbedError;

/// Wraps a loaded `fastembed` model behind a `Mutex` for `&self` inference.
pub struct FastEmbedder {
    /// The underlying ONNX model; guarded because `embed` requires `&mut self`.
    model: Mutex<TextEmbedding>,
    dim: usize,
}

impl FastEmbedder {
    /// Load (downloading on first run) the BGE-small model and record its
    /// dimension by embedding a one-token probe.
    pub fn new() -> Result<Self, EmbedError> {
        let mut model = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::BGESmallENV15))
            .map_err(|e| EmbedError::Model(e.to_string()))?;
        let probe = model
            .embed(vec!["probe"], None)
            .map_err(|e| EmbedError::Model(e.to_string()))?;
        let dim = probe
            .first()
            .map(|v| v.len())
            .ok_or_else(|| EmbedError::Model("embedding probe produced no vector".to_string()))?;
        Ok(Self {
            model: Mutex::new(model),
            dim,
        })
    }
}

impl Embedder for FastEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        // fastembed wants `Vec<&str>`; borrow each owned String.
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        // Lock is held only for the duration of inference; poisoning means the
        // model is in an unknown state, so we surface it as a model error.
        let mut model = self
            .model
            .lock()
            .map_err(|e| EmbedError::Model(format!("embedder lock poisoned: {e}")))?;
        model
            .embed(refs, None)
            .map_err(|e| EmbedError::Model(e.to_string()))
    }
}
