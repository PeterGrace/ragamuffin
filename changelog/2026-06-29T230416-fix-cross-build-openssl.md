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

- **`Cargo.toml` + `build.rs`** (new) — `ort-sys`'s build script still links
  `openssl-sys` for the build host (its `ureq` dependency hardcodes `native-tls`
  upstream with no rustls option). The default `cross` images are based on
  Ubuntu Xenial, which only ships OpenSSL 1.0.2 — too old for modern
  `openssl-sys` (`This crate is only compatible with OpenSSL 1.1.0, 1.1.1, 3.x,
  or 4.x`). Rather than depend on the image's system OpenSSL, we vendor it:
  declare `openssl-sys` with the `vendored` feature as a build dependency, which
  pulls in `openssl-src` and compiles OpenSSL 3.x from source for the build
  host. Because `openssl-sys` is only reached through `ort-sys`'s build script
  (the build dependency graph), the feature is enabled from *our* build
  dependencies so it unifies across that graph; the no-op `build.rs` makes the
  build dependency active.

- **`Cross.toml`** (new) — vendored OpenSSL compiles from source, which needs
  `perl` and `make`; a `pre-build` step installs them into the cross container
  for both Linux targets. No system OpenSSL (`libssl-dev`) is required.

## Verification

- `cargo tree -i openssl-sys` confirms the only remaining OpenSSL path is the
  `ort-sys` build-dependency `ureq`; the runtime/target path is gone.
- `cargo tree -i rustls` confirms `hf-hub` now uses rustls.
- `cargo build --release` succeeds locally with the new feature set.

Functionality is unchanged: rustls vs native-tls only swaps the TLS
implementation used to fetch models and the ONNX Runtime binary.

## Follow-up: drop `cross` entirely in favour of native runners

After OpenSSL was resolved, the `cross`-built Linux jobs hit a final linker
error: `ort` statically links `libonnxruntime.a`, but inside the `cross`
container the linker could not resolve `-lonnxruntime` (mangled search path /
link mode). The same `cargo build --release` links the static ONNX Runtime
cleanly on a native host.

`cross` was never actually required here:

- `x86_64-unknown-linux-gnu` runs on an already-x86_64 `ubuntu-latest` runner.
- `aarch64-unknown-linux-gnu` can build natively on GitHub's free arm64 Linux
  runner, `ubuntu-24.04-arm` (available for public repositories).

`.github/workflows/release.yaml` was simplified to build every target natively
(removing the `use_cross` matrix flag, the `Install cross` step, and the split
native/cross build steps). `Cross.toml` was deleted as dead configuration. The
vendored-OpenSSL build dependency is retained: it is target-agnostic, keeps the
build hermetic regardless of runner image, and removes any dependence on the
runner's system OpenSSL.
