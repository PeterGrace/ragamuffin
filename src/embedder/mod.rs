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
