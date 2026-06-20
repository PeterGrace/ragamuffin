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
