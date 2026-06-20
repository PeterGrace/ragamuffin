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
