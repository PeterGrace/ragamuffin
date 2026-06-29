# Add Windows release targets

**Date:** 2026-06-29T23:34:57Z

## Investigation

Adding Windows to the release matrix raised two questions: whether the ONNX
Runtime dependency (`ort`) supports Windows, and whether the project's
vendored-OpenSSL workaround would break there.

- **ONNX Runtime:** `ort-sys`'s `dist.txt` manifest lists prebuilt binaries for
  both `x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc` (the CPU `none`
  variant). ONNX is not a blocker on Windows.
- **OpenSSL:** `native-tls` only depends on `openssl-sys` on
  `cfg(not(any(target_os = "windows", target_vendor = "apple")))`. On Windows it
  uses SChannel and on macOS Security.framework. The only thing that would have
  forced OpenSSL on Windows was this project's *own* unconditional vendored
  `openssl-sys` build dependency, which would have required Perl and NASM to
  compile from source on the Windows runner.

## Changes

- **`Cargo.toml`** — scoped the vendored `openssl-sys` build dependency to
  `cfg(target_os = "linux")`. OpenSSL is only pulled in on Linux, so Windows and
  macOS no longer attempt to compile it (also trims the needless OpenSSL build
  from the macOS jobs). Verified `cargo tree`/`cargo build` on Linux still
  vendor OpenSSL as before.

- **`.github/workflows/release.yaml`** — added two native Windows build targets:
  - `x86_64-pc-windows-msvc` on `windows-latest`
  - `aarch64-pc-windows-msvc` on `windows-11-arm`

  The "Rename binary" step now copies `ragamuffin.exe` (with its extension) when
  present, so Windows assets stay runnable. The `softprops/action-gh-release`
  `files: ragamuffin-*` glob already matches the new `.exe` assets.

## Remaining risk

The native-Rust and ONNX/TLS pieces are confirmed Windows-compatible. The C/C++
dependencies (`onig_sys`, `esaxx_rs`, `ring`) compile via the MSVC toolchain
present on the Windows runners; this is standard but is only fully validated by
a release CI run.
