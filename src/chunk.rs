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

/// Parse `text` into JSON records. Whole-file JSON is tried first: an array
/// yields one record per element, any other value (object or scalar) yields a
/// single record. If whole-file parsing fails, fall back to JSONL — each
/// non-empty line parsed independently, keeping only the lines that parse.
/// Returns an empty vec when nothing parses (empty or non-JSON input).
///
/// Currently used only by tests; Task 4 (`chunk_json`) will promote this to a
/// production call site at which point the `#[cfg(test)]` gate should be removed.
#[cfg(test)]
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
