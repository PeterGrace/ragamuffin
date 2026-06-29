//! Build script.
//!
//! Intentionally a no-op. Its sole purpose is to make the `openssl-sys`
//! build dependency (declared with the `vendored` feature in `Cargo.toml`)
//! active, so OpenSSL is compiled from source for the build host rather than
//! linked from the system. See the `[build-dependencies]` note in `Cargo.toml`.
fn main() {}
