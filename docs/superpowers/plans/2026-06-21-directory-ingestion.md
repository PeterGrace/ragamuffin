# Directory Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `ragamuffin ingest <path>` accept a directory, recursively ingesting every text-like file in one process with content-based file selection, heading-aware markdown chunking, a progress bar, and a summary.

**Architecture:** A new UI-free `scan` module does file discovery (recursive walk + content sniff + filters); `rag` gains chunking dispatch (markdown → heading-aware, else fixed-word); the `cli` `ingest` handler detects file-vs-directory and drives a progress loop with per-file error resilience. The store format and search path are unchanged.

**Tech Stack:** Rust 2021, `walkdir` (traversal), `indicatif` (progress bar), existing `fastembed`/`rayon`/`serde_json`. Tests run offline via the existing `FakeEmbedder`.

**Reference:** `docs/superpowers/specs/2026-06-21-directory-ingestion-design.md`. Section refs like "§6.3" point at `DESIGN.md`.

**Conventions (CLAUDE.md):** thiserror in lib, anyhow at the CLI boundary, no `.unwrap()`/`.expect()` in non-test code (a hardcoded-invariant `.expect()` is allowed with a descriptive message), 4-space indent, doc comments on public items, 100-char lines. **Run `cargo fmt` before every commit** so it passes `cargo fmt --check`. Add a dated changelog entry under `changelog/` per CLAUDE.md.

---

## Task 1: `scan` module — recursive text-file discovery

**Files:**
- Modify: `Cargo.toml` (add `walkdir`, `indicatif`)
- Modify: `src/lib.rs` (add `pub mod scan;`)
- Create: `src/scan.rs`

- [ ] **Step 1: Add dependencies**

In `Cargo.toml`, under `[dependencies]` (keep the existing entries; add these two, alphabetically `indicatif` after `fastembed` and `walkdir` at the end):

```toml
indicatif = "0.18"
```
```toml
walkdir = "2"
```

- [ ] **Step 2: Declare the module**

In `src/lib.rs`, add `pub mod scan;` so the module list reads (alphabetical, `scan` before `store`):

```rust
pub mod cli;
pub mod embedder;
pub mod error;
pub mod mcp;
pub mod rag;
pub mod scan;
pub mod store;
```

- [ ] **Step 3: Write the failing tests — create `src/scan.rs`**

```rust
//! File discovery for directory ingestion: recursively find text-like files,
//! skipping binaries, hidden entries, oversized files, and (optionally) files
//! outside an extension allowlist. Knows nothing about embeddings or the store.

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// How many leading bytes to inspect when sniffing whether a file is text.
const SNIFF_BYTES: usize = 8192;

/// Options controlling which files a directory walk yields.
#[derive(Debug, Clone)]
pub struct ScanOpts {
    /// Lowercased extensions without leading dots; `None` = no extension filter.
    pub exts: Option<Vec<String>>,
    /// Skip files larger than this many bytes.
    pub max_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn looks_like_text_accepts_utf8_rejects_binary() {
        assert!(looks_like_text(b"hello world"));
        assert!(looks_like_text("caf\u{e9}".as_bytes())); // multibyte UTF-8
        assert!(looks_like_text(b"")); // empty prefix counts as text
        assert!(!looks_like_text(b"abc\0def")); // NUL byte
        assert!(!looks_like_text(&[0xff, 0xfe, 0x41])); // invalid UTF-8
    }

    #[test]
    fn looks_like_text_allows_truncated_multibyte_tail() {
        // Valid UTF-8 then a cut-off multibyte lead byte (as if the 8 KiB sniff
        // sliced through a character). Must still count as text.
        let mut bytes = b"hello ".to_vec();
        bytes.push(0xe2); // first byte of a 3-byte char, truncated
        assert!(looks_like_text(&bytes));
    }

    #[test]
    fn collect_filters_hidden_binary_oversized() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("a.md"), "# Title\nbody").unwrap();
        fs::write(root.join("b.txt"), "plain text").unwrap();
        fs::create_dir(root.join("sub")).unwrap();
        fs::write(root.join("sub").join("c.md"), "nested").unwrap();
        fs::write(root.join("img.png"), [0u8, 1, 2, 3]).unwrap(); // binary (NUL)
        fs::write(root.join(".secret"), "hidden text").unwrap();
        fs::write(root.join("big.txt"), "x".repeat(100)).unwrap(); // oversized

        let opts = ScanOpts { exts: None, max_bytes: 50 };
        let found = collect_text_files(root, &opts).unwrap();
        let names: Vec<String> = found
            .iter()
            .map(|p| p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"))
            .collect();
        assert!(names.contains(&"a.md".to_string()));
        assert!(names.contains(&"b.txt".to_string()));
        assert!(names.contains(&"sub/c.md".to_string()));
        assert!(!names.iter().any(|n| n.contains("img.png"))); // binary skipped
        assert!(!names.iter().any(|n| n.contains("secret"))); // hidden skipped
        assert!(!names.iter().any(|n| n.contains("big.txt"))); // oversized skipped
    }

    #[test]
    fn collect_ext_filter_narrows_to_markdown() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("a.md"), "# md").unwrap();
        fs::write(root.join("b.txt"), "txt").unwrap();
        let opts = ScanOpts { exts: Some(vec!["md".to_string()]), max_bytes: 1_000_000 };
        let found = collect_text_files(root, &opts).unwrap();
        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with("a.md"));
    }

    #[test]
    fn collect_errors_on_missing_root() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("nope");
        let opts = ScanOpts { exts: None, max_bytes: 100 };
        assert!(collect_text_files(&missing, &opts).is_err());
    }
}
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cargo test --lib scan::`
Expected: FAIL to compile — `looks_like_text` and `collect_text_files` are not defined.

- [ ] **Step 5: Implement the functions**

Add to `src/scan.rs`, after the `ScanOpts` struct and before the `#[cfg(test)]` module:

```rust
/// Heuristic: does this byte prefix look like UTF-8 text? True when it contains
/// no NUL byte and is valid UTF-8. A prefix that is valid UTF-8 except for a
/// multibyte character truncated at the very end still counts as text (the 8 KiB
/// sniff may slice through a character). An empty prefix counts as text.
pub fn looks_like_text(prefix: &[u8]) -> bool {
    if prefix.contains(&0) {
        return false;
    }
    match std::str::from_utf8(prefix) {
        Ok(_) => true,
        // `error_len() == None` means the input ended mid-character — a
        // truncation artifact, not genuinely invalid bytes.
        Err(e) => e.error_len().is_none(),
    }
}

/// Recursively collect text-like files under `root`, sorted for deterministic
/// order. Skips hidden entries (name starts with '.'), does not follow symlinks,
/// skips files over `opts.max_bytes`, applies the extension filter when set, and
/// keeps only files whose first [`SNIFF_BYTES`] bytes [`looks_like_text`].
///
/// # Errors
///
/// Returns an error if `root` is not an existing directory. Per-entry I/O errors
/// during the walk are skipped, not propagated.
pub fn collect_text_files(root: &Path, opts: &ScanOpts) -> Result<Vec<PathBuf>, std::io::Error> {
    if !root.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("not a directory: {}", root.display()),
        ));
    }
    let mut out = Vec::new();
    // `filter_entry` prunes hidden directories (and their subtrees) too.
    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !is_hidden(e))
    {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue, // unreadable entry: skip, non-fatal
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if let Some(exts) = &opts.exts {
            match ext_lower(path) {
                Some(ext) if exts.iter().any(|e| *e == ext) => {}
                _ => continue,
            }
        }
        match entry.metadata() {
            Ok(m) if m.len() > opts.max_bytes => continue,
            Ok(_) => {}
            Err(_) => continue,
        }
        if is_text_file(path) {
            out.push(path.to_path_buf());
        }
    }
    out.sort();
    Ok(out)
}

/// True if a walk entry below the root has a name starting with '.'. The root
/// itself (depth 0) is never considered hidden, so scanning a path like
/// `./.config` still works when named explicitly.
fn is_hidden(entry: &walkdir::DirEntry) -> bool {
    entry.depth() > 0
        && entry
            .file_name()
            .to_str()
            .map(|s| s.starts_with('.'))
            .unwrap_or(false)
}

/// The file extension, lowercased, without a leading dot.
fn ext_lower(path: &Path) -> Option<String> {
    path.extension().and_then(|e| e.to_str()).map(str::to_lowercase)
}

/// Read up to [`SNIFF_BYTES`] from `path` and test whether it looks like text.
fn is_text_file(path: &Path) -> bool {
    use std::io::Read;
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut buf = [0u8; SNIFF_BYTES];
    match file.read(&mut buf) {
        Ok(n) => looks_like_text(&buf[..n]),
        Err(_) => false,
    }
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --lib scan::`
Expected: all five scan tests PASS.

- [ ] **Step 7: Format and commit**

```bash
cargo fmt
git add Cargo.toml Cargo.lock src/lib.rs src/scan.rs
git commit -m "feat: scan module for recursive text-like file discovery"
```

---

## Task 2: Heading-aware markdown chunking in `rag`

**Files:**
- Modify: `src/rag.rs`

- [ ] **Step 1: Write the failing tests**

Add these tests inside the existing `#[cfg(test)] mod tests` block in `src/rag.rs` (it already has `use super::*;` and imports `tempfile`/`FakeEmbedder`; `Path` is in scope via the module's top-level `use std::path::Path;`):

```rust
    #[test]
    fn chunk_markdown_splits_on_headings_keeps_sections_whole() {
        let md = "# A\nalpha\n# B\nbeta gamma";
        let chunks = chunk_markdown(md, 100, 10);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("# A") && chunks[0].contains("alpha"));
        assert!(chunks[1].contains("# B") && chunks[1].contains("beta gamma"));
    }

    #[test]
    fn chunk_markdown_preamble_is_its_own_chunk() {
        let md = "intro line\n# Section\nbody";
        let chunks = chunk_markdown(md, 100, 10);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("intro line"));
        assert!(!chunks[0].contains("# Section"));
    }

    #[test]
    fn chunk_markdown_subsplits_oversized_section() {
        let big_body = (0..50).map(|i| i.to_string()).collect::<Vec<_>>().join(" ");
        let md = format!("# Big\n{big_body}"); // ~51 words in one section
        let chunks = chunk_markdown(&md, 20, 5); // exceeds 20 -> sub-split
        assert!(chunks.len() > 1);
    }

    #[test]
    fn chunk_markdown_empty_input_no_chunks() {
        assert!(chunk_markdown("   \n  ", 50, 10).is_empty());
    }

    #[test]
    fn chunk_for_path_routes_by_extension() {
        let md = chunk_for_path(Path::new("notes.md"), "# H\nbody here", 100, 10);
        assert_eq!(md.len(), 1);
        assert!(md[0].contains("# H"));
        // A non-markdown extension uses the fixed-window chunker.
        let txt = chunk_for_path(Path::new("notes.txt"), "a b c", 100, 10);
        assert_eq!(txt, vec!["a b c".to_string()]);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib rag::tests::chunk_markdown_splits_on_headings_keeps_sections_whole`
Expected: FAIL to compile — `chunk_markdown` and `chunk_for_path` are not defined.

- [ ] **Step 3: Implement the chunkers**

Add these two public functions plus the private `is_heading` helper to `src/rag.rs`, immediately after the existing `chunk_text` function (before `impl Rag`):

```rust
/// Choose a chunker by file extension: markdown (`.md` / `.markdown`,
/// case-insensitive) gets heading-aware chunking; every other extension uses the
/// fixed-width word-window [`chunk_text`].
pub fn chunk_for_path(
    path: &Path,
    text: &str,
    chunk_words: usize,
    overlap_words: usize,
) -> Vec<String> {
    let is_md = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("md") || e.eq_ignore_ascii_case("markdown"))
        .unwrap_or(false);
    if is_md {
        chunk_markdown(text, chunk_words, overlap_words)
    } else {
        chunk_text(text, chunk_words, overlap_words)
    }
}

/// Split markdown into self-contained, heading-rooted chunks. A chunk is a
/// heading line plus its body up to the next heading of any level; content
/// before the first heading (preamble) becomes its own chunk. A section longer
/// than `chunk_words` is sub-split with [`chunk_text`] so no chunk is unbounded.
/// Empty or whitespace-only input yields no chunks. A heading-less document
/// behaves like [`chunk_text`].
pub fn chunk_markdown(text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String> {
    // Accumulate lines into sections, starting a new section at each heading
    // (unless the current section is still empty, e.g. the very first line).
    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        if is_heading(line) && !current.trim().is_empty() {
            sections.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        sections.push(current);
    }
    // Emit each non-empty section as a chunk, sub-splitting oversized ones.
    let mut chunks = Vec::new();
    for section in sections {
        let trimmed = section.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.split_whitespace().count() > chunk_words {
            chunks.extend(chunk_text(trimmed, chunk_words, overlap_words));
        } else {
            chunks.push(trimmed.to_string());
        }
    }
    chunks
}

/// True if `line` is an ATX markdown heading: after optional leading spaces it
/// starts with '#'.
fn is_heading(line: &str) -> bool {
    line.trim_start().starts_with('#')
}
```

- [ ] **Step 4: Route `ingest_file` through `chunk_for_path`**

In `src/rag.rs`, inside `ingest_file`, change the chunking line so markdown files get heading-aware chunking whether ingested singly or via a directory. Replace:

```rust
        let chunks = chunk_text(&text, chunk_words, overlap_words);
```

with:

```rust
        let chunks = chunk_for_path(path, &text, chunk_words, overlap_words);
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib rag::`
Expected: all rag tests PASS — the 4 pre-existing chunk tests plus the 5 new ones, and `add_memory_then_search_roundtrips` (which uses `add_memory`, unaffected).

- [ ] **Step 6: Format and commit**

```bash
cargo fmt
git add src/rag.rs
git commit -m "feat: heading-aware markdown chunking; route ingest_file by extension"
```

---

## Task 3: CLI directory ingestion

**Files:**
- Modify: `src/cli.rs`

- [ ] **Step 1: Add imports**

At the top of `src/cli.rs`, add these imports alongside the existing ones (the file already has `use std::path::{Path, PathBuf};`, `use anyhow::Context;`, `use clap::{Parser, Subcommand};`, and the `crate::` imports):

```rust
use indicatif::{ProgressBar, ProgressStyle};
use tracing::warn;

use crate::scan::{self, ScanOpts};
```

- [ ] **Step 2: Extend the `Ingest` command definition**

In the `Command` enum in `src/cli.rs`, replace the existing `Ingest` variant:

```rust
    /// Chunk a text file and store each chunk.
    Ingest {
        path: PathBuf,
        #[arg(long = "chunk-words", default_value_t = 180)]
        chunk_words: usize,
        #[arg(long = "overlap", default_value_t = 40)]
        overlap_words: usize,
    },
```

with this version (adds `--ext` and `--max-bytes`, and notes directory support):

```rust
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
```

- [ ] **Step 3: Update the `Ingest` handler**

In `run`, replace the existing `Command::Ingest { .. } => { .. }` arm:

```rust
        Command::Ingest {
            path,
            chunk_words,
            overlap_words,
        } => {
            let mut rag = open_rag(&cli.store)?;
            let ids = rag.ingest_file(&path, chunk_words, overlap_words)?;
            println!("stored {} chunks (count={})", ids.len(), rag.count());
        }
```

with this version (file-vs-directory dispatch):

```rust
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
```

- [ ] **Step 4: Add the helper functions**

Add to `src/cli.rs`, after the existing `open_rag` function (before the `#[cfg(test)]` module):

```rust
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
    let opts = ScanOpts { exts: parse_exts(ext), max_bytes };
    let files = scan::collect_text_files(dir, &opts).context("scanning directory")?;
    let bar = ProgressBar::new(files.len() as u64);
    bar.set_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} {msg}")
            .expect("static progress template is valid"),
    );
    let mut summary = IngestSummary { files: 0, chunks: 0, skipped: 0 };
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
```

- [ ] **Step 5: Add offline tests**

Add these tests inside the existing `#[cfg(test)] mod tests` block in `src/cli.rs` (it already has `use super::*;`, `use crate::embedder::fake::FakeEmbedder;`, and `use tempfile::tempdir;`):

```rust
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
```

- [ ] **Step 6: Build, then run the tests**

Run: `cargo build`
Expected: compiles. Then:

Run: `cargo test --lib cli::`
Expected: the pre-existing `add_search_list_roundtrip_offline` plus the two new tests PASS.

- [ ] **Step 7: Verify the CLI help shows the new flags**

Run: `cargo run -- ingest --help`
Expected: usage lists `--chunk-words`, `--overlap`, `--ext`, and `--max-bytes`, and the about text mentions a directory.

- [ ] **Step 8: Format, lint, and commit**

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
git add src/cli.rs
git commit -m "feat: directory ingestion in the ingest CLI with progress and summary"
```
Expected: clippy clean.

---

## Task 4: Docs and final quality gate

**Files:**
- Modify: `README.md`
- Create: `changelog/2026-06-21-directory-ingestion.md`

- [ ] **Step 1: Update the README CLI section**

In `README.md`, replace the single ingest example line:

```
    ragamuffin --store ./ragstore ingest notes.txt --chunk-words 180 --overlap 40
```

with both file and directory examples:

```
    ragamuffin --store ./ragstore ingest notes.txt --chunk-words 180 --overlap 40
    ragamuffin --store ./ragstore ingest ./notes --ext md,txt --max-bytes 5000000
```

Then, immediately after the paragraph that begins "`add`, `ingest`, and `search` use a local embedding model", add:

```markdown

`ingest` accepts a file or a directory. Given a directory it recurses, skipping
hidden entries and anything that does not look like UTF-8 text (binaries are
detected by content, not just extension). Use `--ext` to narrow by extension and
`--max-bytes` to cap file size. Markdown files are chunked on headings so each
chunk stays self-contained; other files use fixed-width word windows. Re-running
`ingest` is idempotent.
```

- [ ] **Step 2: Write the changelog entry**

Create `changelog/2026-06-21-directory-ingestion.md`:

```markdown
# Directory ingestion — 2026-06-21

`ingest` now accepts a directory, not just a single file.

## Added
- `src/scan.rs`: recursive text-file discovery — content-sniff (NUL/UTF-8) to
  skip binaries, skip hidden entries, no symlink following, `--ext` allowlist,
  and a `--max-bytes` size cap. Deterministic (sorted) output.
- Heading-aware markdown chunking (`chunk_markdown` / `chunk_for_path` in
  `src/rag.rs`): `.md`/`.markdown` split on ATX headings into self-contained,
  heading-rooted chunks; oversized sections fall back to word-window splitting;
  other file types keep fixed-width chunking.
- `ingest <dir>`: single-process recursive ingest with an `indicatif` progress
  bar, per-file error resilience (warn + skip), and a summary line.

## Dependencies
- Added `walkdir` and `indicatif`.

## Notes
- No store-format or search changes. Re-ingesting a directory is idempotent via
  the existing content-hash upsert.
```

- [ ] **Step 3: Final verification**

Run:
```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```
Expected: all three succeed. Test count is the MVP's 15 plus the new tests (5 scan + 5 rag chunking + 2 cli = 12), i.e. 27 lib tests.

- [ ] **Step 4: Commit**

```bash
git add README.md changelog/2026-06-21-directory-ingestion.md
git commit -m "docs: document directory ingestion in README and changelog"
```

---

## Self-Review Notes

**Spec coverage** — every spec section maps to a task:
- §2 CLI surface (`--ext`, `--max-bytes`, summary, file-or-dir) → Task 3.
- §3.1 `scan` module (`ScanOpts`, `collect_text_files`, `looks_like_text`) → Task 1.
- §3.2 chunking dispatch (`chunk_for_path`, `chunk_markdown`, `ingest_file` switch) → Task 2.
- §3.3 CLI handler (dispatch, progress, summary, `--ext` parse) → Task 3.
- §4 data flow → Tasks 2+3 (ingest_file routes chunking; CLI loops scan results).
- §5 heading-aware chunking algorithm → Task 2 `chunk_markdown`.
- §6 error handling (scan returns io::Error for bad root; CLI warn+skip per file) → Tasks 1+3.
- §7 testing (scan, chunking, dir round-trip) → Tasks 1, 2, 3.
- Recursion/hidden/symlink/size defaults → Task 1 `collect_text_files`/`is_hidden`.

**Type/name consistency:** `ScanOpts { exts: Option<Vec<String>>, max_bytes: u64 }`, `scan::collect_text_files`, `scan::looks_like_text`, `rag::chunk_for_path`, `rag::chunk_markdown`, and the CLI `ingest_dir`/`parse_exts`/`IngestSummary { files, chunks, skipped }` are used identically across tasks. `ingest_file`'s signature is unchanged (only its internal chunking call changed), so all existing callers and tests still compile.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; commands have expected output.
