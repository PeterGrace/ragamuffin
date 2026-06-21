//! MCP server exposing the memory as two tools over stdio. The host harness's
//! model decides when to call them.

use std::path::Path;
use std::sync::Arc;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler, ServiceExt};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::embedder::fastembed::FastEmbedder;
use crate::rag::Rag;
use crate::store::Hit;

fn default_k() -> usize {
    4
}
fn default_source() -> String {
    "chat".to_string()
}

/// Arguments for `search_memory`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchArgs {
    /// What to recall, described in natural language.
    pub query: String,
    /// Maximum number of notes to return.
    #[serde(default = "default_k")]
    pub k: usize,
}

/// Arguments for `save_memory`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SaveArgs {
    /// The self-contained note to remember.
    pub text: String,
    /// Where the note came from.
    #[serde(default = "default_source")]
    pub source: String,
}

/// MCP server holding the RAG behind an async lock (the harness may call tools
/// concurrently; writes must be serialized).
#[derive(Clone)]
pub struct MemoryServer {
    rag: Arc<Mutex<Rag>>,
    tool_router: ToolRouter<MemoryServer>,
}

#[tool_router]
impl MemoryServer {
    /// Wrap an open RAG as a server.
    pub fn new(rag: Rag) -> Self {
        Self {
            rag: Arc::new(Mutex::new(rag)),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Search the personal long-term memory by meaning. Returns the most semantically similar saved notes with their source and similarity score."
    )]
    async fn search_memory(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, McpError> {
        let rag = self.rag.lock().await;
        let hits = rag
            .search(&args.query, args.k)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format_hits(
            &hits,
        ))]))
    }

    #[tool(
        description = "Save a self-contained note to the personal long-term memory. Re-saving identical text updates the one existing entry rather than duplicating."
    )]
    async fn save_memory(
        &self,
        Parameters(args): Parameters<SaveArgs>,
    ) -> Result<CallToolResult, McpError> {
        let mut rag = self.rag.lock().await;
        let id = rag
            .add_memory(&args.text, &args.source, serde_json::Value::Null)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let count = rag.count();
        Ok(CallToolResult::success(vec![Content::text(format!(
            "saved {id} (count={count})"
        ))]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        // `ServerInfo` (alias for `InitializeResult`) is `#[non_exhaustive]`, so
        // it cannot be built with a struct literal from outside the crate. Start
        // from the default and mutate the fields we care about.
        let mut info = ServerInfo::default();
        info.instructions = Some(
            "Personal long-term memory. Use search_memory to recall past discussions or \
             when context is insufficient; use save_memory when asked to remember something \
             or when a durable fact or decision emerges. Make saved notes self-contained."
                .to_string(),
        );
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}

/// Format hits as a numbered list so the host model can judge and cite them.
fn format_hits(hits: &[Hit]) -> String {
    if hits.is_empty() {
        return "No matching memories.".to_string();
    }
    hits.iter()
        .enumerate()
        .map(|(i, h)| {
            format!(
                "{}. [{} | score {:.3}] {}",
                i + 1,
                h.source,
                h.score,
                h.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Open a RAG under `store` and serve it over stdio until the client disconnects.
pub async fn serve(store: &Path) -> anyhow::Result<()> {
    let embedder = FastEmbedder::new()?;
    let rag = Rag::open(store, Box::new(embedder))?;
    let service = MemoryServer::new(rag).serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::fake::FakeEmbedder;
    use tempfile::tempdir;

    #[test]
    fn format_hits_numbers_and_cites() {
        let hits = vec![Hit {
            score: 0.9,
            id: "abc".into(),
            text: "rust note".into(),
            source: "chat".into(),
            metadata: serde_json::Value::Null,
        }];
        let out = format_hits(&hits);
        assert!(out.contains("1. [chat | score 0.900] rust note"));
    }

    #[tokio::test]
    async fn save_then_search_via_tools() {
        let dir = tempdir().unwrap();
        let rag = Rag::open(dir.path(), Box::new(FakeEmbedder::new())).unwrap();
        let server = MemoryServer::new(rag);

        // A synthetic save tool call writes the note.
        server
            .save_memory(Parameters(SaveArgs {
                text: "rust memory note".into(),
                source: "chat".into(),
            }))
            .await
            .unwrap();
        assert_eq!(server.rag.lock().await.count(), 1);

        // A synthetic search tool call retrieves it. Serialize the whole result
        // to JSON and substring-match, to avoid coupling to rmcp internals.
        let result = server
            .search_memory(Parameters(SearchArgs {
                query: "rust".into(),
                k: 4,
            }))
            .await
            .unwrap();
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("rust memory note"));
    }
}
