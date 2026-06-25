# Task 3: Dotted-Key Recursive Flattening of JSON Records

**Date:** 2026-06-25T01:00:00

## Summary

Added per-record flattening helpers to `src/chunk.rs`. These helpers convert a
`serde_json::Value` into two orthogonal representations consumed by the JSON
chunking pipeline (Task 4):

- **text lines** — string leaves and all-string arrays formatted as
  `key: value` (or bare value at the top level), for embedding;
- **metadata map** — scalar leaves (number, bool, null) stored under dotted
  keys, for structured filtering.

## New items (all `#[cfg(test)]`-gated until Task 4 promotes them)

| Item | Kind | Purpose |
|------|------|---------|
| `MAX_FLATTEN_DEPTH` | `const usize` | Cap recursion at 64 levels; collapses deeper subtrees to compact JSON to prevent stack overflow on adversarial input. |
| `join_key(prefix, key)` | `fn` | Constructs dotted key paths (`""` + `k` → `k`; `p` + `k` → `p.k`). |
| `compact_json(value)` | `fn` | Serialises a `Value` to a one-line JSON string; falls back to `""` on the infallible-in-practice error path. |
| `flatten_value(prefix, value, depth, lines, meta)` | `fn` | Recursive dispatcher: strings → `lines`; scalars → `meta`; all-string arrays → single joined line; mixed arrays → indexed recursion; objects → keyed recursion. |

## Tests added

Four new unit tests inside `chunk::tests`:

1. `flatten_value_routes_strings_to_text_and_scalars_to_meta` — exercises the
   full routing: string leaves, nested string, all-string array → lines; number,
   bool, nested number → meta; strings not duplicated in meta.
2. `flatten_value_top_level_string_has_no_key_prefix` — bare string at the root
   emits without a `key:` prefix.
3. `flatten_value_array_of_objects_uses_indexed_keys` — mixed-type array of
   objects gets `items.0.title`, `items.1.title` keys.
4. `flatten_value_caps_recursion_depth` — 100-level deep structure completes
   without stack overflow and produces at least one line.

## Dead-code notes

All four items (`MAX_FLATTEN_DEPTH`, `join_key`, `compact_json`, `flatten_value`)
are gated with `#[cfg(test)]` because they have no non-test caller yet. This
mirrors the approach used for `parse_json_records`. Task 4 will remove these
gates when `chunk_json` is implemented.

## Quality checks

- `cargo fmt` — clean
- `cargo clippy --all-targets -- -D warnings` — 0 warnings, 0 errors
- `cargo test` — 37/37 passed
