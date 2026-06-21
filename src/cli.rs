//! CLI surface (§9). A thin adapter over [`Rag`](crate::rag::Rag): parse args,
//! call one method, print. `search`/`list` emit JSON so host harnesses can
//! parse them. `add`/`ingest`/`search` use the real embedder; `list` is fully
//! offline and never loads a model.

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use tracing::warn;

use crate::embedder::fastembed::FastEmbedder;
use crate::rag::Rag;
use crate::scan::{self, ScanOpts};
use crate::store::Store;

/// ragamuffin: local-first semantic memory.
#[derive(Parser)]
#[command(
    name = "ragamuffin",
    version,
    about = "Local-first semantic memory backend"
)]
pub struct Cli {
    /// Store directory.
    #[arg(long, global = true, default_value = "./ragstore")]
    pub store: PathBuf,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Store one already-compacted memory.
    Add {
        text: String,
        #[arg(long, default_value = "manual")]
        source: String,
    },
    /// Ingest a text file, or every text-like file under a directory.
    Ingest {
        path: PathBuf,
        #[arg(long = "chunk-words", default_value_t = 180)]
        chunk_words: usize,
        #[arg(long = "overlap", default_value_t = 40)]
        overlap_words: usize,
        /// Directory ingest: comma-separated extension allowlist (e.g.
        /// "md,txt"). Omit to ingest any file that looks like text.
        #[arg(long)]
        ext: Option<String>,
        /// Directory ingest: skip files larger than this many bytes.
        #[arg(long = "max-bytes", default_value_t = 5_000_000)]
        max_bytes: u64,
    },
    /// Print the top-k semantic matches as JSON.
    Search {
        query: String,
        #[arg(short, default_value_t = 4)]
        k: usize,
    },
    /// List stored entries (id, source, metadata) as JSON.
    List,
    /// Run as an MCP server over stdio.
    Mcp,
}

/// Dispatch a parsed CLI to its handler.
pub async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Command::Add { text, source } => {
            let mut rag = open_rag(&cli.store)?;
            let id = rag.add_memory(&text, &source, serde_json::Value::Null)?;
            println!("stored {id} (count={})", rag.count());
        }
        Command::Ingest {
            path,
            chunk_words,
            overlap_words,
            ext,
            max_bytes,
        } => {
            let mut rag = open_rag(&cli.store)?;
            if path.is_dir() {
                let summary =
                    ingest_dir(&mut rag, &path, chunk_words, overlap_words, ext, max_bytes)?;
                println!(
                    "ingested {} files ({} chunks), skipped {}",
                    summary.files, summary.chunks, summary.skipped
                );
            } else {
                let ids = rag.ingest_file(&path, chunk_words, overlap_words)?;
                println!("stored {} chunks (count={})", ids.len(), rag.count());
            }
        }
        Command::Search { query, k } => {
            let rag = open_rag(&cli.store)?;
            let hits = rag.search(&query, k)?;
            println!("{}", serde_json::to_string_pretty(&hits)?);
        }
        Command::List => {
            // Fully offline: read the store directly, no embedder/model load.
            if !cli.store.join("manifest.json").exists() {
                println!("[]");
                return Ok(());
            }
            let store = Store::open_existing(&cli.store)?;
            println!("{}", serde_json::to_string_pretty(store.all())?);
        }
        Command::Mcp => {
            crate::mcp::serve(&cli.store).await?;
        }
    }
    Ok(())
}

/// Build a RAG instance backed by the real local embedder.
fn open_rag(store: &Path) -> anyhow::Result<Rag> {
    let embedder = FastEmbedder::new().context("loading embedding model")?;
    Rag::open(store, Box::new(embedder)).context("opening store")
}

/// Tally of a directory ingestion run.
struct IngestSummary {
    files: usize,
    chunks: usize,
    skipped: usize,
}

/// Parse a comma-separated extension list into lowercased, dot-stripped entries.
/// `None` (the flag omitted) means "no extension filter".
fn parse_exts(ext: Option<String>) -> Option<Vec<String>> {
    ext.map(|s| {
        s.split(',')
            .map(|e| e.trim().trim_start_matches('.').to_lowercase())
            .filter(|e| !e.is_empty())
            .collect()
    })
}

/// Walk `dir`, ingesting each text-like file with a progress bar. A file that
/// fails to read/chunk/embed is logged and counted as skipped; the run
/// continues (per-file resilience).
fn ingest_dir(
    rag: &mut Rag,
    dir: &Path,
    chunk_words: usize,
    overlap_words: usize,
    ext: Option<String>,
    max_bytes: u64,
) -> anyhow::Result<IngestSummary> {
    let opts = ScanOpts {
        exts: parse_exts(ext),
        max_bytes,
    };
    let files = scan::collect_text_files(dir, &opts).context("scanning directory")?;
    let bar = ProgressBar::new(files.len() as u64);
    bar.set_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} {msg}")
            .expect("static progress template is valid"),
    );
    let mut summary = IngestSummary {
        files: 0,
        chunks: 0,
        skipped: 0,
    };
    for path in &files {
        bar.set_message(path.display().to_string());
        match rag.ingest_file(path, chunk_words, overlap_words) {
            Ok(ids) => {
                summary.files += 1;
                summary.chunks += ids.len();
            }
            Err(e) => {
                warn!("skipping {}: {e}", path.display());
                summary.skipped += 1;
            }
        }
        bar.inc(1);
    }
    bar.finish_and_clear();
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::fake::FakeEmbedder;
    use tempfile::tempdir;

    // The handlers above require the real model; here we test the same logic
    // path (add -> search -> list serialization) against a fake-backed Rag.

    #[test]
    fn parse_exts_normalizes() {
        assert_eq!(
            parse_exts(Some(".MD, txt ,".to_string())),
            Some(vec!["md".to_string(), "txt".to_string()])
        );
        assert_eq!(parse_exts(None), None);
    }

    #[test]
    fn ingest_dir_roundtrip_offline() {
        let store_dir = tempdir().unwrap();
        let src = tempdir().unwrap();
        std::fs::write(src.path().join("a.md"), "# Rust\nrust code is fast").unwrap();
        std::fs::write(src.path().join("b.txt"), "the cat ate food").unwrap();
        std::fs::write(src.path().join("img.bin"), [0u8, 1, 2]).unwrap(); // binary

        let mut rag = Rag::open(store_dir.path(), Box::new(FakeEmbedder::new())).unwrap();
        let summary = ingest_dir(&mut rag, src.path(), 180, 40, None, 1_000_000).unwrap();
        assert_eq!(summary.files, 2); // a.md + b.txt; img.bin skipped as binary
        assert!(summary.chunks >= 2);

        // Content from an ingested file is searchable.
        let hits = rag.search("rust", 4).unwrap();
        assert!(hits.iter().any(|h| h.text.contains("rust code is fast")));

        // Re-ingesting the same directory is idempotent (no new entries).
        let before = rag.count();
        ingest_dir(&mut rag, src.path(), 180, 40, None, 1_000_000).unwrap();
        assert_eq!(rag.count(), before);
    }

    #[test]
    fn add_search_list_roundtrip_offline() {
        let dir = tempdir().unwrap();
        let mut rag = Rag::open(dir.path(), Box::new(FakeEmbedder::new())).unwrap();
        rag.add_memory("rust code", "manual", serde_json::Value::Null)
            .unwrap();
        drop(rag);

        // search serializes hits as JSON
        let rag = Rag::open(dir.path(), Box::new(FakeEmbedder::new())).unwrap();
        let hits = rag.search("rust", 4).unwrap();
        let json = serde_json::to_string(&hits).unwrap();
        assert!(json.contains("rust code"));

        // list reads the store offline via open_existing
        let store = Store::open_existing(dir.path()).unwrap();
        let listed = serde_json::to_string(store.all()).unwrap();
        assert!(listed.contains("\"source\":\"manual\""));
    }
}
