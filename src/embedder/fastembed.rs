//! Real, local embedder backed by `fastembed` (ONNX runtime). Downloads a small
//! sentence-transformer (BGE-small-en-v1.5, 384-dim) on first construction and
//! caches it. fastembed already mean-pools and L2-normalizes, satisfying the
//! unit-length contract (§4.1).
//!
//! **Cache location:** rather than letting fastembed drop a `.fastembed_cache`
//! directory into the current working directory, we pin the model cache to a
//! stable per-user path under `XDG_CACHE_HOME` (e.g. `~/.cache/ragamuffin` on
//! Linux). This keeps the multi-megabyte ONNX download out of project trees and
//! lets every invocation reuse the same download.
//!
//! **Interior mutability note:** `fastembed::TextEmbedding::embed` takes `&mut self`
//! (it manages internal ONNX session state), but the `Embedder` trait requires `&self`
//! so callers can share it behind an `Arc`. We wrap the model in a `Mutex` to allow
//! shared ownership while still obtaining an exclusive borrow for each inference call.

use std::path::PathBuf;
use std::sync::Mutex;

use directories::ProjectDirs;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use crate::embedder::Embedder;
use crate::error::EmbedError;

/// Wraps a loaded `fastembed` model behind a `Mutex` for `&self` inference.
pub struct FastEmbedder {
    /// The underlying ONNX model; guarded because `embed` requires `&mut self`.
    model: Mutex<TextEmbedding>,
    dim: usize,
}

/// Resolve the per-user model cache directory, honoring `XDG_CACHE_HOME`.
///
/// On Linux this yields `~/.cache/ragamuffin` (or `$XDG_CACHE_HOME/ragamuffin`);
/// the platform-appropriate equivalent is used on macOS and Windows. The
/// directory is created if it does not yet exist so fastembed can populate it.
///
/// # Errors
///
/// Returns [`EmbedError::Cache`] if no home/cache base directory can be
/// determined (e.g. a headless environment with no `$HOME`) or if the directory
/// cannot be created.
fn resolve_cache_dir() -> Result<PathBuf, EmbedError> {
    // `ProjectDirs` maps (qualifier, organization, application) onto each OS's
    // conventions. We only need the application segment, so the first two are
    // left empty; `cache_dir()` then returns `<base cache>/ragamuffin`.
    let project_dirs = ProjectDirs::from("", "", "ragamuffin").ok_or_else(|| {
        EmbedError::Cache("no valid home directory found for cache location".to_string())
    })?;
    let cache_dir = project_dirs.cache_dir().to_path_buf();
    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| EmbedError::Cache(format!("{}: {e}", cache_dir.display())))?;
    Ok(cache_dir)
}

impl FastEmbedder {
    /// Load (downloading on first run) the BGE-small model and record its
    /// dimension by embedding a one-token probe. The model is cached under a
    /// stable per-user directory (see [`resolve_cache_dir`]) rather than the
    /// current working directory.
    pub fn new() -> Result<Self, EmbedError> {
        let cache_dir = resolve_cache_dir()?;
        let init_options =
            TextInitOptions::new(EmbeddingModel::BGESmallENV15).with_cache_dir(cache_dir);
        let mut model =
            TextEmbedding::try_new(init_options).map_err(|e| EmbedError::Model(e.to_string()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// On Linux, `resolve_cache_dir` must honor `XDG_CACHE_HOME` and place the
    /// model cache in a `ragamuffin` subdirectory it has created. This test only
    /// runs on Linux because `ProjectDirs` ignores XDG variables elsewhere.
    #[cfg(target_os = "linux")]
    #[test]
    fn resolve_cache_dir_honors_xdg_cache_home() {
        let base = tempdir().expect("create temp dir");
        // SAFETY (2021 edition): setting an env var is safe here; this is the
        // only test that touches XDG_CACHE_HOME, so there is no data race.
        std::env::set_var("XDG_CACHE_HOME", base.path());

        let resolved = resolve_cache_dir().expect("resolve cache dir");

        assert_eq!(resolved, base.path().join("ragamuffin"));
        assert!(
            resolved.is_dir(),
            "cache directory should have been created"
        );

        std::env::remove_var("XDG_CACHE_HOME");
    }
}
