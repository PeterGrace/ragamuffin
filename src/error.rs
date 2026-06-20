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
    /// The on-disk vectors file and metadata disagree on row count, indicating
    /// a corrupt or partially-written store.
    #[error("corrupt store: {floats} vector floats is not {rows} rows x {dim} dims")]
    Corrupt {
        floats: usize,
        rows: usize,
        dim: usize,
    },
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
