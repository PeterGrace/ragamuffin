# Release CI Workflow

**Date:** 2026-06-29T18:50:24

## Summary

Added `.github/workflows/release.yaml`, a tag-triggered GitHub Actions workflow
that builds release binaries across a target matrix and uploads them to a
GitHub Release. Adapted from the `codeowner-gen` reference workflow.

## Behavior

- **Trigger:** `push` of a tag matching `v*.*.*` (e.g. `v0.1.0`).
- **Permissions:** `contents: write` so the release can be created/updated.
- **Build matrix:**
  - `x86_64-unknown-linux-gnu` (cross)
  - `aarch64-unknown-linux-gnu` (cross)
  - `x86_64-apple-darwin` (native, macOS runner)
  - `aarch64-apple-darwin` (native, macOS runner)
- Linux targets use `cross` (Docker-based), macOS targets build natively.
- macOS binaries are ad-hoc signed (`codesign --sign -`) so they run without a
  Gatekeeper quarantine prompt.
- Built binaries are renamed to `ragamuffin-<target>` and uploaded via
  `softprops/action-gh-release@v2`, which collects all `ragamuffin-*` assets.

## Differences from the `codeowner-gen` reference

- Binary name changed to `ragamuffin`; no Windows-specific `.exe` handling.
- Dropped the `x86_64-pc-windows-gnu` target and added `aarch64-unknown-linux-gnu`.
- Added explicit `permissions: contents: write` and `fail-fast: false`.

## Caveats

- `ragamuffin` depends on `fastembed`, which pulls in `ort`/ONNX Runtime. ONNX
  Runtime cross-compilation under `cross` (and even native macOS builds) can
  require additional setup that pure-Rust crates do not. The first tagged build
  should be watched; if a target fails to link ONNX Runtime, that matrix entry
  may need a target-specific `ort` feature, a prebuilt runtime, or removal.
- The Windows target was intentionally omitted for this reason; it can be added
  back once ONNX Runtime packaging for Windows is verified.

## Usage

```sh
git tag v0.1.0
git push origin v0.1.0
```
