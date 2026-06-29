# Fix cross-compiled release builds failing on OpenSSL

**Date:** 2026-06-29T23:04:16Z

## Problem

The `Release` workflow's two Linux jobs (`x86_64-unknown-linux-gnu` and
`aarch64-unknown-linux-gnu`, both built with `cross`) failed at the
`Build (cross)` step with:

```
Could not find openssl via pkg-config
The system library `openssl` required by crate `openssl-sys` was not found.
```

`cross` build containers do not ship `libssl-dev`, and `openssl-sys` was being
pulled into the dependency graph transitively. Nothing in the project uses
OpenSSL directly; it was only there as a TLS backend for downloading artifacts.

## Root cause

`fastembed`'s default features pulled OpenSSL in via two paths:

1. **Runtime/target link** — `hf-hub-native-tls` (default) → `native-tls` →
   `openssl-sys`, compiled for the *target* (the hard case, especially aarch64).
   Used to download model files from HuggingFace.
2. **Build-time host** — `ort`'s `download-binaries` → `ort-sys` build script
   uses `ureq` with `native-tls` hardcoded to download the ONNX Runtime binary.
   This build script runs on the *host* inside the cross container.

## Fix

- **`Cargo.toml`** — switch `fastembed` off default features and select the
  rustls TLS backend, eliminating OpenSSL from the target/runtime link entirely:

  ```toml
  fastembed = { version = "5", default-features = false, features = [
      "ort-download-binaries",
      "hf-hub-rustls-tls",
      "image-models",
  ] }
  ```

  rustls is pure Rust, so the released binary links no system OpenSSL.

- **`Cross.toml`** (new) — `ort-sys`'s build script still needs host-side
  OpenSSL headers (its `ureq` dependency hardcodes `native-tls` upstream with no
  rustls option). A `pre-build` step installs `libssl-dev` and `pkg-config` into
  the cross container for both Linux targets. Because the build script runs on
  the host, host (amd64) OpenSSL suffices for both target architectures.

## Verification

- `cargo tree -i openssl-sys` confirms the only remaining OpenSSL path is the
  `ort-sys` build-dependency `ureq`; the runtime/target path is gone.
- `cargo tree -i rustls` confirms `hf-hub` now uses rustls.
- `cargo build --release` succeeds locally with the new feature set.

Functionality is unchanged: rustls vs native-tls only swaps the TLS
implementation used to fetch models and the ONNX Runtime binary.
