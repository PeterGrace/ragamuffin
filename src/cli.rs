//! CLI surface (§9). A thin adapter over [`Rag`](crate::rag::Rag): parse args,
//! call one method, print. `search`/`list` emit JSON so host harnesses can
//! parse them. `add`/`ingest`/`search` use the real embedder; `list` is fully
//! offline and never loads a model.

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};

use crate::embedder::fastembed::FastEmbedder;
use crate::rag::Rag;
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
    /// Chunk a text file and store each chunk.
    Ingest {
        path: PathBuf,
        #[arg(long = "chunk-words", default_value_t = 180)]
        chunk_words: usize,
        #[arg(long = "overlap", default_value_t = 40)]
        overlap_words: usize,
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
        } => {
            let mut rag = open_rag(&cli.store)?;
            let ids = rag.ingest_file(&path, chunk_words, overlap_words)?;
            println!("stored {} chunks (count={})", ids.len(), rag.count());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::fake::FakeEmbedder;
    use tempfile::tempdir;

    // The handlers above require the real model; here we test the same logic
    // path (add -> search -> list serialization) against a fake-backed Rag.
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
