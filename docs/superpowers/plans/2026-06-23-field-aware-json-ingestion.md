# Field-Aware JSON Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest JSON arrays, single objects, and JSONL record-by-record — flattening string fields into embeddable text and scalar fields into filterable metadata, with the original record preserved under `raw`.

**Architecture:** Extract chunking out of `src/rag.rs` into a new `src/chunk.rs`. A chunk becomes text plus metadata (`Chunk` struct). `chunk_for_path` routes `.json`/`.jsonl`/`.ndjson` to a new `chunk_json`, which parses records, flattens each with dotted keys, and emits one chunk per record. The store, search, list, and error layers are unchanged — `metadata` is already an arbitrary `serde_json::Value` flowing end-to-end.

**Tech Stack:** Rust, `serde_json` (already a dependency), `tracing` (already a dependency), `cargo test`.

Reference spec: `docs/superpowers/specs/2026-06-22-field-aware-json-ingestion-design.md`

---

## File Structure

- **Create `src/chunk.rs`** — all chunking: the `Chunk` type, `chunk_text`, `chunk_markdown`, `is_heading`, `chunk_for_path`, plus (added in later tasks) `parse_json_records`, `flatten_value`, `join_key`, `chunk_json`, and their unit tests.
- **Modify `src/lib.rs`** — declare `pub mod chunk;`.
- **Modify `src/rag.rs`** — remove the moved chunkers; import from `chunk`; update `ingest_file` to embed `chunk.text` and store `chunk.metadata`.
- **Create `changelog/2026-06-23-json-field-aware-ingestion.md`** — implementation note.
- **Modify `README.md`** — document the feature (no root `CHANGELOG.md` exists; the repo tracks history in `changelog/`).

No changes to `src/store.rs`, `src/error.rs`, `src/cli.rs`, `src/mcp.rs`.

---

## Task 1: Extract chunking into `src/chunk.rs` with the `Chunk` type

This is a refactor: move the existing chunkers verbatim, introduce `Chunk`, and change `chunk_for_path` to return `Vec<Chunk>`. Verified by the existing test suite staying green.

**Files:**
- Create: `src/chunk.rs`
- Modify: `src/lib.rs` (module list)
- Modify: `src/rag.rs` (remove chunkers, import from `chunk`, update `ingest_file`)

- [ ] **Step 1: Create `src/chunk.rs` with the moved code, `Chunk`, and the new `chunk_for_path`**

Create `src/chunk.rs` with exactly this content:

```rust
//! Text chunking: splitting documents into the units that get embedded and
//! stored. Each chunker is pure (no I/O); [`chunk_for_path`] routes by file
//! extension and attaches per-chunk metadata.

use std::path::Path;

use serde_json::{json, Value};

/// A unit of text to embed plus the metadata to store alongside it.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    /// The text that gets embedded and stored as the document body.
    pub text: String,
    /// Arbitrary JSON metadata stored row-aligned with the chunk.
    pub metadata: Value,
}

/// Wrap plain text chunks with positional `{"chunk": i}` metadata, preserving
/// the metadata shape used before chunks carried their own.
fn indexed_chunks(texts: Vec<String>) -> Vec<Chunk> {
    texts
        .into_iter()
        .enumerate()
        .map(|(i, text)| Chunk {
            text,
            metadata: json!({ "chunk": i }),
        })
        .collect()
}

/// Split `text` into overlapping fixed-width word windows (§6.3) so an idea
/// straddling a boundary stays retrievable from either side.
pub fn chunk_text(text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String> {
    // Clamp to at least 1 to prevent infinite loops or empty chunks when
    // chunk_words = 0 is passed (e.g. via `--chunk-words 0`).
    let chunk_words = chunk_words.max(1);
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    if words.len() <= chunk_words {
        return vec![words.join(" ")];
    }
    let step = chunk_words.saturating_sub(overlap_words).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + chunk_words).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end >= words.len() {
            break;
        }
        start += step;
    }
    chunks
}

/// Choose a chunker by file extension: markdown (`.md` / `.markdown`,
/// case-insensitive) gets heading-aware chunking; JSON (`.json` / `.jsonl` /
/// `.ndjson`) gets field-aware chunking; every other extension uses the
/// fixed-width word-window [`chunk_text`]. Returns chunks carrying their own
/// metadata.
pub fn chunk_for_path(
    path: &Path,
    text: &str,
    chunk_words: usize,
    overlap_words: usize,
) -> Vec<Chunk> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("md") | Some("markdown") => {
            indexed_chunks(chunk_markdown(text, chunk_words, overlap_words))
        }
        _ => indexed_chunks(chunk_text(text, chunk_words, overlap_words)),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_overlaps_long_input() {
        let text = (0..100)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunk_text(&text, 30, 10); // step = 20
        assert!(chunks.len() > 1);
        // Overlap: chunk 0 ends with "29", chunk 1 starts at word index 20.
        assert!(chunks[0].split(' ').next_back().unwrap() == "29");
        assert!(chunks[1].split(' ').next().unwrap() == "20");
    }

    #[test]
    fn chunk_text_short_input_is_one_chunk() {
        assert_eq!(chunk_text("a b c", 30, 10), vec!["a b c".to_string()]);
        assert!(chunk_text("   ", 30, 10).is_empty());
    }

    #[test]
    fn chunk_text_zero_width_is_clamped() {
        // chunk_words = 0 must not produce empty or infinite chunks.
        let chunks = chunk_text("a b c d e", 0, 0);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| !c.is_empty()));
    }

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
        assert!(md[0].text.contains("# H"));
        assert_eq!(md[0].metadata, json!({ "chunk": 0 }));
        // A non-markdown extension uses the fixed-window chunker.
        let txt = chunk_for_path(Path::new("notes.txt"), "a b c", 100, 10);
        assert_eq!(txt.len(), 1);
        assert_eq!(txt[0].text, "a b c");
        assert_eq!(txt[0].metadata, json!({ "chunk": 0 }));
    }
}
```

- [ ] **Step 2: Declare the module in `src/lib.rs`**

Add `pub mod chunk;` to the module list in `src/lib.rs`, keeping alphabetical order:

```rust
pub mod chunk;
pub mod cli;
pub mod embedder;
pub mod error;
pub mod mcp;
pub mod rag;
pub mod scan;
pub mod store;
```

- [ ] **Step 3: Remove the moved code from `src/rag.rs` and import from `chunk`**

In `src/rag.rs`, delete `chunk_text`, `chunk_for_path`, `chunk_markdown`, `is_heading` and their preceding doc comments (the block currently spanning the `chunk_text` doc comment through the end of `is_heading`). Also delete the four moved tests from the `#[cfg(test)] mod tests` block: `chunk_text_overlaps_long_input`, `chunk_text_short_input_is_one_chunk`, `chunk_text_zero_width_is_clamped`, `chunk_markdown_splits_on_headings_keeps_sections_whole`, `chunk_markdown_preamble_is_its_own_chunk`, `chunk_markdown_subsplits_oversized_section`, `chunk_markdown_empty_input_no_chunks`, and `chunk_for_path_routes_by_extension`.

Update the imports near the top of `src/rag.rs`. Replace:

```rust
use std::path::Path;

use serde_json::Value;

use crate::embedder::Embedder;
use crate::error::RagError;
use crate::store::{Hit, Id, MetaRecord, Store};
```

with:

```rust
use std::path::Path;

use serde_json::Value;

use crate::chunk::chunk_for_path;
use crate::embedder::Embedder;
use crate::error::RagError;
use crate::store::{Hit, Id, MetaRecord, Store};
```

- [ ] **Step 4: Update `ingest_file` to use per-chunk metadata**

In `src/rag.rs`, replace the body of `ingest_file` (currently lines ~131-152). The new body:

```rust
    pub fn ingest_file(
        &mut self,
        path: &Path,
        chunk_words: usize,
        overlap_words: usize,
    ) -> Result<Vec<Id>, RagError> {
        let text = std::fs::read_to_string(path)?;
        let chunks = chunk_for_path(path, &text, chunk_words, overlap_words);
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        // Embed the chunk bodies in one batch; metadata travels with each chunk.
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let vectors = self.embedder.embed(&texts)?;
        let source = path.to_string_lossy().to_string();
        let mut ids = Vec::with_capacity(chunks.len());
        for (chunk, vector) in chunks.iter().zip(vectors.iter()) {
            let id = self
                .store
                .add(vector, &chunk.text, &source, chunk.metadata.clone())?;
            ids.push(id);
        }
        Ok(ids)
    }
```

- [ ] **Step 5: Format, lint, and run the full suite**

Run:
```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```
Expected: clippy clean, all tests PASS (the moved chunk tests now run from `src/chunk.rs`; `rag.rs` orchestration tests still pass; `cli.rs` `ingest_dir` test still passes).

- [ ] **Step 6: Commit**

```bash
git add src/chunk.rs src/lib.rs src/rag.rs
git commit -m "refactor: extract chunking into chunk.rs; chunks carry metadata"
```

---

## Task 2: Parse JSON into records

Add `parse_json_records`: whole-file JSON first (array -> per element, other -> single record), JSONL fallback (one value per non-empty line), empty otherwise.

**Files:**
- Modify: `src/chunk.rs`
- Test: `src/chunk.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/chunk.rs`:

```rust
    #[test]
    fn parse_json_records_array_yields_one_per_element() {
        let records = parse_json_records(r#"[{"a":1},{"a":2},{"a":3}]"#);
        assert_eq!(records.len(), 3);
        assert_eq!(records[0], json!({"a": 1}));
        assert_eq!(records[2], json!({"a": 3}));
    }

    #[test]
    fn parse_json_records_single_object_is_one_record() {
        let records = parse_json_records(r#"{"a":1,"b":2}"#);
        assert_eq!(records, vec![json!({"a": 1, "b": 2})]);
    }

    #[test]
    fn parse_json_records_scalar_is_one_record() {
        assert_eq!(parse_json_records("42"), vec![json!(42)]);
        assert_eq!(parse_json_records(r#""hello""#), vec![json!("hello")]);
    }

    #[test]
    fn parse_json_records_jsonl_parses_each_line() {
        let text = "{\"a\":1}\n\n{\"a\":2}\n";
        let records = parse_json_records(text);
        assert_eq!(records, vec![json!({"a": 1}), json!({"a": 2})]);
    }

    #[test]
    fn parse_json_records_malformed_is_empty() {
        assert!(parse_json_records("not json at all {").is_empty());
        assert!(parse_json_records("").is_empty());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib chunk::tests::parse_json_records`
Expected: FAIL — `cannot find function parse_json_records`.

- [ ] **Step 3: Implement `parse_json_records`**

Add to `src/chunk.rs`, above the `#[cfg(test)]` module:

```rust
/// Parse `text` into JSON records. Whole-file JSON is tried first: an array
/// yields one record per element, any other value (object or scalar) yields a
/// single record. If whole-file parsing fails, fall back to JSONL — each
/// non-empty line parsed independently, keeping only the lines that parse.
/// Returns an empty vec when nothing parses (empty or non-JSON input).
fn parse_json_records(text: &str) -> Vec<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        return match value {
            Value::Array(items) => items,
            other => vec![other],
        };
    }
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .collect()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib chunk::tests::parse_json_records`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/chunk.rs
git commit -m "feat: parse JSON arrays, objects, scalars, and JSONL into records"
```

---

## Task 3: Flatten a record into text lines and scalar metadata

Add `join_key` and `flatten_value`: dotted-key recursion routing string leaves and string arrays to text lines, scalars to a metadata map, with a depth-64 safety cap.

**Files:**
- Modify: `src/chunk.rs`
- Test: `src/chunk.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/chunk.rs`:

```rust
    #[test]
    fn flatten_value_routes_strings_to_text_and_scalars_to_meta() {
        let value = json!({
            "title": "Quarterly Report",
            "year": 2024,
            "published": true,
            "author": {"name": "Jane Doe", "team_size": 5},
            "tags": ["finance", "q3"]
        });
        let mut lines = Vec::new();
        let mut meta = serde_json::Map::new();
        flatten_value("", &value, 0, &mut lines, &mut meta);

        // String leaves and string arrays become text lines.
        assert!(lines.contains(&"title: Quarterly Report".to_string()));
        assert!(lines.contains(&"author.name: Jane Doe".to_string()));
        assert!(lines.contains(&"tags: finance, q3".to_string()));
        // Scalar leaves (including nested) go to metadata under dotted keys.
        assert_eq!(meta.get("year"), Some(&json!(2024)));
        assert_eq!(meta.get("published"), Some(&json!(true)));
        assert_eq!(meta.get("author.team_size"), Some(&json!(5)));
        // Strings are not duplicated into metadata.
        assert!(meta.get("title").is_none());
    }

    #[test]
    fn flatten_value_top_level_string_has_no_key_prefix() {
        let mut lines = Vec::new();
        let mut meta = serde_json::Map::new();
        flatten_value("", &json!("hello world"), 0, &mut lines, &mut meta);
        assert_eq!(lines, vec!["hello world".to_string()]);
        assert!(meta.is_empty());
    }

    #[test]
    fn flatten_value_array_of_objects_uses_indexed_keys() {
        let value = json!({"items": [{"title": "A"}, {"title": "B"}]});
        let mut lines = Vec::new();
        let mut meta = serde_json::Map::new();
        flatten_value("", &value, 0, &mut lines, &mut meta);
        assert!(lines.contains(&"items.0.title: A".to_string()));
        assert!(lines.contains(&"items.1.title: B".to_string()));
    }

    #[test]
    fn flatten_value_caps_recursion_depth() {
        // Build a structure deeper than the cap; it must not overflow the stack
        // and the deepest content collapses into a single line.
        let mut value = json!("leaf");
        for _ in 0..100 {
            value = json!({ "n": value });
        }
        let mut lines = Vec::new();
        let mut meta = serde_json::Map::new();
        flatten_value("", &value, 0, &mut lines, &mut meta);
        assert!(!lines.is_empty());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib chunk::tests::flatten_value`
Expected: FAIL — `cannot find function flatten_value`.

- [ ] **Step 3: Implement `join_key`, `flatten_value`, and the depth cap**

Add to `src/chunk.rs`, above the `#[cfg(test)]` module:

```rust
/// Maximum nesting depth before a subtree is collapsed to a compact JSON
/// string. A safety rail against stack overflow on adversarial input — far
/// beyond the depth of normal records, so it never triggers in practice.
const MAX_FLATTEN_DEPTH: usize = 64;

/// Join a dotted key path: `""` + `k` -> `k`; `prefix` + `k` -> `prefix.k`.
fn join_key(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

/// Recursively flatten `value` under the dotted-key `prefix`. String leaves and
/// all-string arrays become `key: value` text lines (pushed to `lines`); number,
/// bool, and null leaves are inserted into `meta` under their dotted key. Nested
/// objects and non-string arrays recurse (arrays with indexed keys). At
/// [`MAX_FLATTEN_DEPTH`] the remaining subtree is emitted as one compact-JSON
/// text line instead of recursing further.
fn flatten_value(
    prefix: &str,
    value: &Value,
    depth: usize,
    lines: &mut Vec<String>,
    meta: &mut serde_json::Map<String, Value>,
) {
    if depth >= MAX_FLATTEN_DEPTH {
        lines.push(format!("{prefix}: {}", compact_json(value)));
        return;
    }
    match value {
        Value::String(s) => {
            if prefix.is_empty() {
                lines.push(s.clone());
            } else {
                lines.push(format!("{prefix}: {s}"));
            }
        }
        Value::Number(_) | Value::Bool(_) | Value::Null => {
            // A bare top-level scalar has no key; it is preserved via `raw`.
            if !prefix.is_empty() {
                meta.insert(prefix.to_string(), value.clone());
            }
        }
        Value::Array(items) => {
            if !items.is_empty() && items.iter().all(Value::is_string) {
                let joined = items
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
                if prefix.is_empty() {
                    lines.push(joined);
                } else {
                    lines.push(format!("{prefix}: {joined}"));
                }
            } else {
                for (i, item) in items.iter().enumerate() {
                    let key = join_key(prefix, &i.to_string());
                    flatten_value(&key, item, depth + 1, lines, meta);
                }
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                let key = join_key(prefix, k);
                flatten_value(&key, v, depth + 1, lines, meta);
            }
        }
    }
}

/// Serialize `value` to a compact one-line JSON string. Serialization of an
/// in-memory `Value` is infallible in practice; on the impossible error path we
/// fall back to an empty string rather than panic.
fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib chunk::tests::flatten_value`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/chunk.rs
git commit -m "feat: dotted-key recursive flattening of JSON records"
```

---

## Task 4: Assemble `chunk_json` and route JSON extensions

Combine parsing + flattening into per-record `Chunk`s (with `record`, `source_kind`, `raw`, oversize `part`), and route `.json`/`.jsonl`/`.ndjson` to it.

**Files:**
- Modify: `src/chunk.rs` (`chunk_json`, routing, imports)
- Test: `src/chunk.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/chunk.rs`:

```rust
    #[test]
    fn chunk_json_array_yields_one_chunk_per_record() {
        let text = r#"[{"title":"Alpha"},{"title":"Beta"}]"#;
        let chunks = chunk_json(text, 180, 40);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "title: Alpha");
        assert_eq!(chunks[0].metadata["record"], json!(0));
        assert_eq!(chunks[1].metadata["record"], json!(1));
        assert_eq!(chunks[0].metadata["source_kind"], json!("json"));
    }

    #[test]
    fn chunk_json_single_object_is_one_record_with_raw_and_scalars() {
        let text = r#"{"title":"Quarterly Report","year":2024,
            "author":{"name":"Jane Doe","team_size":5},"tags":["finance","q3"]}"#;
        let chunks = chunk_json(text, 180, 40);
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert!(c.text.contains("title: Quarterly Report"));
        assert!(c.text.contains("author.name: Jane Doe"));
        assert!(c.text.contains("tags: finance, q3"));
        assert_eq!(c.metadata["record"], json!(0));
        assert_eq!(c.metadata["year"], json!(2024));
        assert_eq!(c.metadata["author.team_size"], json!(5));
        // The original record is preserved verbatim under `raw`.
        assert_eq!(c.metadata["raw"]["author"]["name"], json!("Jane Doe"));
    }

    #[test]
    fn chunk_json_record_without_strings_embeds_compact_json() {
        let chunks = chunk_json(r#"[{"count":7}]"#, 180, 40);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, r#"{"count":7}"#);
        assert_eq!(chunks[0].metadata["count"], json!(7));
    }

    #[test]
    fn chunk_json_oversize_record_subsplits_with_part() {
        // A single record whose text exceeds chunk_words must sub-split.
        let body = (0..50).map(|i| i.to_string()).collect::<Vec<_>>().join(" ");
        let text = format!(r#"[{{"note":"{body}"}}]"#);
        let chunks = chunk_json(&text, 20, 5);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].metadata["part"], json!(0));
        assert_eq!(chunks[1].metadata["part"], json!(1));
        // `raw` is repeated on every part so each chunk is self-contained.
        assert!(chunks[0].metadata["raw"]["note"].is_string());
        assert!(chunks[1].metadata["raw"]["note"].is_string());
    }

    #[test]
    fn chunk_json_empty_array_yields_no_chunks() {
        assert!(chunk_json("[]", 180, 40).is_empty());
        assert!(chunk_json("   ", 180, 40).is_empty());
    }

    #[test]
    fn chunk_json_malformed_falls_back_to_text() {
        let chunks = chunk_json("this is not json", 180, 40);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "this is not json");
        assert_eq!(chunks[0].metadata, json!({ "chunk": 0 }));
    }

    #[test]
    fn chunk_json_jsonl_parses_each_line() {
        let text = "{\"title\":\"A\"}\n{\"title\":\"B\"}";
        let chunks = chunk_json(text, 180, 40);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "title: A");
        assert_eq!(chunks[1].text, "title: B");
    }

    #[test]
    fn chunk_for_path_routes_json_extensions() {
        for name in ["data.json", "data.jsonl", "data.ndjson"] {
            let chunks = chunk_for_path(Path::new(name), r#"{"title":"X"}"#, 180, 40);
            assert_eq!(chunks.len(), 1, "routing failed for {name}");
            assert_eq!(chunks[0].metadata["source_kind"], json!("json"));
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib chunk::tests::chunk_json`
Expected: FAIL — `cannot find function chunk_json`.

- [ ] **Step 3: Implement `chunk_json`**

Add to `src/chunk.rs`, above the `#[cfg(test)]` module:

```rust
/// Field-aware chunking for JSON / JSONL input. Each record (array element,
/// single value, or JSONL line) becomes one chunk: string fields flatten into
/// the embedded text, scalar fields into metadata, and the original record is
/// kept verbatim under `raw`. A record whose text exceeds `chunk_words` is
/// sub-split with [`chunk_text`], each part tagged `part`. Empty input yields no
/// chunks; non-JSON input falls back to plain-text chunking (with a warning).
pub fn chunk_json(text: &str, chunk_words: usize, overlap_words: usize) -> Vec<Chunk> {
    let records = parse_json_records(text);
    if records.is_empty() {
        if text.trim().is_empty() {
            return Vec::new();
        }
        tracing::warn!("content did not parse as JSON or JSONL; ingesting as plain text");
        return indexed_chunks(chunk_text(text, chunk_words, overlap_words));
    }

    let mut chunks = Vec::with_capacity(records.len());
    for (record, value) in records.into_iter().enumerate() {
        let mut lines = Vec::new();
        let mut base = serde_json::Map::new();
        flatten_value("", &value, 0, &mut lines, &mut base);

        // A record with no string content still needs embeddable text.
        let body = if lines.is_empty() {
            compact_json(&value)
        } else {
            lines.join("\n")
        };

        base.insert("record".to_string(), json!(record));
        base.insert("source_kind".to_string(), json!("json"));
        base.insert("raw".to_string(), value);

        if body.split_whitespace().count() > chunk_words {
            for (part, sub) in chunk_text(&body, chunk_words, overlap_words)
                .into_iter()
                .enumerate()
            {
                let mut metadata = base.clone();
                metadata.insert("part".to_string(), json!(part));
                chunks.push(Chunk {
                    text: sub,
                    metadata: Value::Object(metadata),
                });
            }
        } else {
            chunks.push(Chunk {
                text: body,
                metadata: Value::Object(base),
            });
        }
    }
    chunks
}
```

- [ ] **Step 4: Route JSON extensions in `chunk_for_path`**

In `src/chunk.rs`, update the `match` in `chunk_for_path` to add the JSON arm:

```rust
    match ext.as_deref() {
        Some("md") | Some("markdown") => {
            indexed_chunks(chunk_markdown(text, chunk_words, overlap_words))
        }
        Some("json") | Some("jsonl") | Some("ndjson") => {
            chunk_json(text, chunk_words, overlap_words)
        }
        _ => indexed_chunks(chunk_text(text, chunk_words, overlap_words)),
    }
```

- [ ] **Step 5: Format, lint, run the JSON tests, then the full suite**

Run:
```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test --lib chunk::tests
cargo test
```
Expected: clippy clean; `chunk::tests` all PASS (including the 8 new ones); full suite PASS.

- [ ] **Step 6: Commit**

```bash
git add src/chunk.rs
git commit -m "feat: field-aware chunk_json and route .json/.jsonl/.ndjson"
```

---

## Task 5: End-to-end ingest of a JSON file

Verify `ingest_file` ingests a `.json` file with the fake embedder, storing per-record metadata, and that the records are searchable.

**Files:**
- Test: `src/rag.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/rag.rs`:

```rust
    #[test]
    fn ingest_json_file_stores_per_record_metadata() {
        let dir = tempdir().unwrap();
        let src = tempdir().unwrap();
        let json_path = src.path().join("records.json");
        std::fs::write(
            &json_path,
            r#"[{"title":"rust memory notes","year":2024},
                {"title":"the cat ate food","year":2023}]"#,
        )
        .unwrap();

        let mut rag = Rag::open(dir.path(), Box::new(FakeEmbedder::new())).unwrap();
        let ids = rag.ingest_file(&json_path, 180, 40).unwrap();
        assert_eq!(ids.len(), 2); // one chunk per record

        // Per-record metadata is stored: source_kind, record index, scalar year.
        let records = rag.all();
        assert!(records
            .iter()
            .any(|m| m.metadata["source_kind"] == serde_json::json!("json")
                && m.metadata["year"] == serde_json::json!(2024)));

        // A flattened string field is retrievable by semantic search.
        let hits = rag.search("rust", 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("rust memory notes"));
    }
```

- [ ] **Step 2: Run the test to verify it fails (or compiles and passes)**

Run: `cargo test --lib rag::tests::ingest_json_file_stores_per_record_metadata`
Expected: PASS — the implementation already exists (Tasks 1-4). This test guards the integration. If it FAILS, fix the implementation before continuing. (TDD note: this is a characterization test over already-built behavior; a green run here is the success condition.)

- [ ] **Step 3: Commit**

```bash
git add src/rag.rs
git commit -m "test: end-to-end JSON file ingestion stores record metadata"
```

---

## Task 6: Documentation

Add the changelog entry and update README/CHANGELOG.

**Files:**
- Create: `changelog/2026-06-23-json-field-aware-ingestion.md`
- Modify: `README.md`

- [ ] **Step 1: Write the changelog entry**

Create `changelog/2026-06-23-json-field-aware-ingestion.md`:

```markdown
# Field-aware JSON ingestion — 2026-06-23

`ingest` now parses JSON, JSONL, and NDJSON by record instead of as raw text.

## Added
- `src/chunk.rs`: new module holding all chunking. Introduces `Chunk` (text +
  metadata); `chunk_for_path` now returns `Vec<Chunk>`. Moved `chunk_text` and
  `chunk_markdown` here unchanged.
- `chunk_json` (`src/chunk.rs`): `.json` / `.jsonl` / `.ndjson` ingest one chunk
  per record. A top-level array yields one record per element; a single object or
  scalar is one record; JSONL/NDJSON parse one record per line. Each record is
  flattened with dotted keys — string leaves and string arrays become embedded
  `key: value` text, scalar leaves become filterable metadata — and the original
  record is preserved verbatim under `raw`. Oversized records sub-split and tag
  `part`. Malformed input falls back to plain-text chunking.

## Changed
- `Rag::ingest_file` (`src/rag.rs`) embeds each chunk's body and stores the
  chunk's own metadata instead of a fixed `{"chunk": i}`.

## Notes
- No store-format, search, or error-layer changes; metadata is already arbitrary
  JSON end-to-end. Text/markdown entries keep `{"chunk": i}`; JSON-derived
  entries carry `source_kind: "json"`. Re-ingesting is idempotent via the
  existing content-hash upsert.
```

- [ ] **Step 2: Update the README**

In `README.md`, the `ingest` paragraph (lines 22-27) ends with a sentence about Markdown chunking. Replace this exact text:

```markdown
`--max-bytes` to cap file size. Markdown files are chunked on headings so each
chunk stays self-contained; other files use fixed-width word windows. Re-running
`ingest` is idempotent.
```

with:

```markdown
`--max-bytes` to cap file size. Markdown files are chunked on headings so each
chunk stays self-contained; JSON files (`.json`, `.jsonl`, `.ndjson`) are
ingested field-aware — one entry per record, with string fields flattened into
the embedded text, scalar fields stored as filterable metadata, and the original
record preserved under `raw`; other files use fixed-width word windows. Re-running
`ingest` is idempotent.
```

- [ ] **Step 3: Verify the docs build and the suite is green**

Run:
```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```
Expected: all PASS, no warnings.

- [ ] **Step 4: Commit**

```bash
git add changelog/2026-06-23-json-field-aware-ingestion.md README.md
git commit -m "docs: document field-aware JSON ingestion"
```

---

## Final Verification

- [ ] Run the full gate from `.claude/CLAUDE.md`:
```bash
cargo test
cargo build
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```
Expected: all green, no warnings.

- [ ] Manual smoke test (optional, requires the real model):
```bash
printf '[{"title":"rust is fast","year":2024},{"title":"cats nap","year":2023}]' > /tmp/rec.json
cargo run -- --store /tmp/ragstore ingest /tmp/rec.json
cargo run -- --store /tmp/ragstore search "rust" -k 1
```
Expected: `ingest` reports 2 chunks; `search` returns the "rust is fast" record with `source_kind: "json"` and `year: 2024` in its metadata.

---

## Notes on Conventions

- No `justfile` in this repo — use `cargo` directly (`cargo test`, `cargo clippy --all-targets -- -D warnings`, `cargo fmt`).
- `serde_json` and `tracing` are already dependencies; no `Cargo.toml` changes needed.
- Per `.claude/CLAUDE.md`: doc comments on all public items, `thiserror` errors (none added here), no `.unwrap()` in library code (tests may use it), 4-space indent, 100-char lines.
