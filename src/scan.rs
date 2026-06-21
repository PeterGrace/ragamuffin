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
                Some(ext) if exts.contains(&ext) => {}
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
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
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

        let opts = ScanOpts {
            exts: None,
            max_bytes: 50,
        };
        let found = collect_text_files(root, &opts).unwrap();
        let names: Vec<String> = found
            .iter()
            .map(|p| {
                p.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
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
        let opts = ScanOpts {
            exts: Some(vec!["md".to_string()]),
            max_bytes: 1_000_000,
        };
        let found = collect_text_files(root, &opts).unwrap();
        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with("a.md"));
    }

    #[test]
    fn collect_errors_on_missing_root() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("nope");
        let opts = ScanOpts {
            exts: None,
            max_bytes: 100,
        };
        assert!(collect_text_files(&missing, &opts).is_err());
    }
}
