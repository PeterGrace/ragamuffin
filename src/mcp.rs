//! MCP server — temporary shim; full impl in Task 7.

use std::path::Path;

pub async fn serve(_store: &Path) -> anyhow::Result<()> {
    anyhow::bail!("mcp not implemented until Task 7")
}
