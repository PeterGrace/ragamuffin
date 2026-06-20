//! Binary entry point: parse the CLI and dispatch. All real logic lives in the
//! library crate so it can be unit-tested.

use clap::Parser;
use ragamuffin::cli::{run, Cli};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load a local .env if present (e.g. RUST_LOG); ignore if absent.
    let _ = dotenv::dotenv();
    init_tracing();
    let cli = Cli::parse();
    run(cli).await
}

/// Minimal stderr tracing so logs never pollute stdout (stdout carries JSON and
/// the MCP stdio protocol).
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}
