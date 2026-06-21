# Directory Ingestion — Design Specification

Date: 2026-06-21
Status: Approved for implementation

Adds the ability to seed ragamuffin from a directory of text-like files in a
single process (one model load, batched per file), with content-based file
selection and heading-aware chunking for markdown. Builds on the shipped MVP
(see `docs/superpowers/specs/2026-06-20-ragamuffin-mvp-design.md`). Section
references like "§6.3" point at `DESIGN.md`.

---

## 1. Goal & motivation

Today `ingest` accepts a single file. Seeding a directory means a shell loop, and
each `ragamuffin` invocation reloads the embedding model — slow at scale and with
no cross-file batching. This feature lets `ingest` accept a directory, walking it
recursively in one process and embedding each file's chunks in batches.

Scope is a single, focused extension. No change to the store format, the search
path, or the MCP server.

---

## 2. CLI surface

Extend the existing `ingest` command to accept a **file or a directory** — no new
subcommand.

```
ragamuffin --store ./ragstore ingest <path> \
    [--ext md,txt,log,json] [--max-bytes 5000000] \
    [--chunk-words 180] [--overlap 40]
```

- `<path>` is a file → behaves as before, except markdown now gets heading-aware
  chunking (§5).
- `<path>` is a directory → recursive walk, content-sniff each file, ingest the
  text-like ones, showing an `indicatif` progress bar (CLAUDE.md mandates a
  progress bar for long-running operations).
- `--ext <list>` (optional): comma-separated extension allowlist that *narrows*
  the content-sniff result. Matching is case-insensitive, leading dots optional
  (`md` and `.md` both work). Omitted → ingest anything that sniffs as text.
- `--max-bytes <n>` (default `5000000`): files larger than this are skipped.
- `--chunk-words` (default 180) and `--overlap` (default 40) are unchanged.

The directory run ends with a summary line:
`ingested <N> files (<M> chunks), skipped <K>`.

Defaults and behaviors:
- **Recursive**, skipping any entry whose file name starts with `.` (skips
  `.git`, `.fastembed_cache`, dotfiles).
- **Symlinks are not followed** (avoids cycles).
- **Per-file resilience:** a file that cannot be read or whose embedding fails is
  logged with `tracing::warn!` and counted as skipped; the run continues.
- **Idempotent:** re-ingesting a directory updates entries in place via the
  existing content-hash upsert (§6.4); safe to re-seed.

---

## 3. Architecture

Three focused units that preserve the MVP layering — the CLI is a thin adapter,
and `Rag` stays free of filesystem-walking and UI concerns.

```
src/
    scan.rs   (new)     file discovery: walk + content sniff + filters
    rag.rs    (extend)  chunking dispatch (markdown vs fixed-word)
    cli.rs    (extend)  `ingest` handler: file-vs-dir, progress, summary
    lib.rs    (extend)  add `pub mod scan;`
```

New dependencies: `walkdir` (recursive traversal), `indicatif` (progress bar).

### 3.1 `scan` module (new)

No knowledge of embeddings, the store, or the CLI. Pure discovery, unit-testable.

```rust
/// Options controlling which files a directory walk yields.
pub struct ScanOpts {
    /// Lowercased extensions without leading dots; None = no extension filter.
    pub exts: Option<Vec<String>>,
    /// Skip files larger than this many bytes.
    pub max_bytes: u64,
}

/// Recursively collect text-like files under `root`.
///
/// Skips: hidden entries (name starts with '.'), symlinks (not followed),
/// files over `max_bytes`, files failing the extension filter (when set), and
/// files that do not sniff as text. Returns paths in a deterministic
/// (sorted) order.
pub fn collect_text_files(root: &Path, opts: &ScanOpts) -> Result<Vec<PathBuf>, std::io::Error>;

/// Heuristic: does this byte prefix look like UTF-8 text (no NUL byte and valid
/// UTF-8)? Reads at most the first 8 KiB at the call site.
pub fn looks_like_text(prefix: &[u8]) -> bool;
```

`collect_text_files` opens each candidate, reads up to the first 8 KiB, and keeps
it only if `looks_like_text` returns true. Results are sorted for deterministic
ordering (and stable tests).

### 3.2 `rag` module (extend)

Add chunking dispatch; `ingest_file` switches from `chunk_text` to
`chunk_for_path` so markdown files get heading-aware treatment whether ingested
singly or as part of a directory.

```rust
/// Choose a chunker by file extension: markdown -> heading-aware, else the
/// existing fixed-width word-window chunker.
pub fn chunk_for_path(path: &Path, text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String>;

/// Split markdown into self-contained, heading-rooted chunks (§5).
pub fn chunk_markdown(text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String>;
```

`.md` and `.markdown` (case-insensitive) route to `chunk_markdown`; everything
else routes to the existing `chunk_text`.

### 3.3 `cli` module (extend)

The `ingest` handler:
1. Open the `Rag` (loads the real embedder once).
2. If `path` is a file → `rag.ingest_file(path, chunk_words, overlap)`, print
   `stored <M> chunks (count=<total>)` (unchanged behavior).
3. If `path` is a directory:
   - Build `ScanOpts` from `--ext` / `--max-bytes`; `files =
     scan::collect_text_files(path, &opts)?`.
   - Create an `indicatif` progress bar of length `files.len()` with a
     context-sensitive message.
   - For each file: `rag.ingest_file(...)`; on `Ok(ids)` add `ids.len()` to the
     chunk count and increment the file count; on `Err(e)` log
     `tracing::warn!` and increment the skipped count. Tick the bar.
   - Finish the bar and print `ingested <N> files (<M> chunks), skipped <K>`.

The `--ext` string is parsed into `Option<Vec<String>>` (lowercased, dots
stripped) in the CLI before constructing `ScanOpts`.

---

## 4. Data flow

```
ingest <dir>
  -> scan::collect_text_files(dir, opts)        # walk + sniff + filter -> [paths]
  -> for path in paths (progress bar):
       rag.ingest_file(path, chunk_words, overlap)
         -> read file
         -> chunk_for_path(path, text, ...)      # md -> chunk_markdown, else chunk_text
         -> embedder.embed(chunks)               # batched per file
         -> store.add(...) per chunk             # content-hash upsert
  -> print summary
```

No change to the store's on-disk format or the search path. The `source` recorded
for each chunk remains the file path (as passed to `ingest_file`), and each chunk
keeps its `{"chunk": i}` metadata.

---

## 5. Heading-aware markdown chunking

`chunk_markdown(text, chunk_words, overlap_words)`:

1. Scan lines; a line is a heading if, after trimming leading spaces, it starts
   with `#` (ATX heading). Split the document into sections, each beginning at a
   heading line and running until the next heading of **any** level.
2. Any content before the first heading (preamble) is its own section.
3. Each section is one chunk, so it carries its heading and is self-contained
   (design ideal #7 — a retrieved note must stand alone).
4. If a section's word count exceeds `chunk_words`, sub-split it with the existing
   overlapping word-window splitter (`chunk_text`) so large sections stay
   retrievable; the contract remains `text -> Vec<String>`.
5. Empty or whitespace-only input → no chunks. A document with no headings → the
   whole text handled as a single preamble section (then word-window sub-split if
   oversized), i.e. equivalent to `chunk_text` for heading-less markdown.

Sub-splitting reuses `chunk_text`, so there is no duplicated windowing logic.

---

## 6. Error handling

- `scan::collect_text_files` returns `std::io::Error` only for a failure to read
  the root directory itself; per-entry I/O errors during the walk are treated as
  "skip this entry" (logged at `debug`), not fatal.
- The CLI directory loop never aborts on a single bad file: read/chunk/embed
  failures from `ingest_file` are `tracing::warn!`-logged and counted as skipped.
- All library errors continue to flow through the existing `thiserror` enums
  (`RagError`, `StoreError`, `EmbedError`); the CLI boundary uses `anyhow` with
  `.context()`. No `.unwrap()`/`.expect()` in non-test code.

---

## 7. Testing (all offline via FakeEmbedder)

- **scan:**
  - `looks_like_text` accepts valid UTF-8, rejects a buffer containing a NUL
    byte and rejects invalid UTF-8.
  - `collect_text_files` over a temp dir containing a `.md`, a `.txt`, a nested
    subdir file, a fake binary (`.png` with NUL bytes), a hidden `.secret`, and
    an oversized file returns exactly the expected text files, sorted, excluding
    the binary, hidden, and oversized ones.
  - `--ext` narrowing: with `exts = Some(["md"])`, only markdown is returned.
- **chunking:**
  - `chunk_markdown` splits on headings, keeps a small section whole, sub-splits
    an oversized section, treats preamble as its own chunk, and returns no chunks
    for empty input.
  - `chunk_for_path` routes `.md` to heading-aware and `.txt` to fixed-word.
- **directory ingest round-trip:** create a temp dir of a few mixed files, ingest
  each (the CLI loop's logic) via a fake-backed `Rag`, then `search` returns
  content originating from more than one file; re-ingesting the directory leaves
  the entry count unchanged (idempotent).

None require a network, an API key, or a model download.

---

## 8. Out of scope

- Following symlinks; `.gitignore`-aware filtering; per-file-type parsing (e.g.
  extracting prose from JSON/CSV structurally) — files are treated as plain text.
- Parallel file ingestion (the embedder is already batched per file and guarded
  by a single mutex; per-file parallelism is unnecessary at personal scale).
- Watching a directory for changes / incremental re-sync. Re-running `ingest` is
  the re-sync mechanism, relying on idempotent upsert.
- Deleting store entries for files removed from the directory.
