# ragamuffin — Design Specification

A language-neutral description of a local-first retrieval-augmented memory with
two-way LLM tool calling. This document captures *what* the system is and *why*
it is shaped the way it is, so it can be faithfully reimplemented in any
language. It avoids language-specific syntax; pseudocode signatures describe the
contracts, and concrete details (file formats, the agentic loop) are spelled out
precisely enough to port without ambiguity.

---

## 1. Purpose & Mental Model

The system gives an LLM a persistent, personal long-term memory that lives
entirely on the user's machine. It does two things:

- **Remember:** compact a piece of text (a summary, a decision, a fact) into a
  vector and store it alongside its raw text.
- **Recall:** given a query, find the stored entries whose meaning is closest and
  return their text — to a CLI user, or to an LLM mid-conversation.

The recall step matches by **meaning, not keywords**. Every piece of text is
turned into a point in a high-dimensional space (a vector) such that texts about
similar things land near each other. Searching is then "find the nearest stored
points to this query point." Nothing in the storage layer understands language;
the understanding is entirely in the model that produces the vectors. The store
is pure geometry.

The headline capability is that the LLM itself decides, during a live
conversation, when to read from and write to this memory — via tool calls — so
it behaves like memory rather than a manual lookup.

---

## 2. Design Ideals

These are the principles the implementation should preserve across the port.
They are listed in priority order; when two pull against each other, prefer the
earlier one.

1. **Local-first and private.** Embedding and storage happen on the user's
   machine with no data leaving it. A hosted embedding API is an *allowed
   fallback* for portability, but the default path should keep data local.

2. **Transparency over magic.** The "vector database" is a readable matrix plus
   JSON plus plain-text files, not an opaque engine. A user should be able to
   open the store directory and understand exactly what is there. This is a
   teaching tool as much as a utility.

3. **Narrow, swappable interfaces.** Each layer (embedder, store, orchestration,
   conversation harness) depends only on the small contract of the layer below.
   Replacing the brute-force store with a real vector database, or the local
   embedder with a hosted one, must not ripple upward.

4. **Source text and index are physically separate.** The vector index exists
   only to *locate* entries; the raw human-readable text is stored on its own and
   is what gets fed back to the model. Keeping them apart makes both the concept
   and the storage obvious.

5. **Idempotent writes.** Storing the same text twice updates one entry rather
   than creating duplicates. Identity is derived from content.

6. **Model-driven, just-in-time context.** Retrieval and saving are driven by the
   model through tools, not hard-coded by the harness. The harness provides
   capabilities; the model chooses when to use them.

7. **Self-contained memories.** A stored note is retrieved out of its original
   context, so it must carry enough detail to stand alone. Compaction is the act
   of making a note self-sufficient, not merely short.

8. **Scale-appropriate simplicity.** Brute-force similarity search over a few
   tens of thousands of vectors is correct and fast enough for a personal memory.
   Do not add an approximate-nearest-neighbor index until the data demands it.

9. **Minimal dependencies and offline-testable.** The core logic must be testable
   without a network, an API key, or a model download, by injecting a fake
   deterministic embedder.

### Non-goals

These are deliberately out of scope; a reimplementation need not handle them, and
should resist the temptation to:

- Multi-user serving, authentication, or access control.
- Millions of vectors / approximate search / sharding.
- Concurrent writers (the store assumes a single writer; last write wins).
- Crash-consistency guarantees beyond "rewrite the files on each change."
- Streaming token output (the harness collects each turn, then prints it).

---

## 3. System Architecture

Four layers, each depending only on the one below it. Dependencies point
downward; nothing lower knows about anything higher.

```
        +-------------------------------------------------+
        |  CLI  /  Conversation Harness                   |   user-facing
        |  (commands; the tool-calling loop)              |
        +-------------------------------------------------+
                         |  uses
                         v
        +-------------------------------------------------+
        |  RAG orchestration                              |   put-in / get-out
        |  (chunking, ingest, add_memory, search)         |
        +------------------+------------------------------+
              |  uses       \  uses
              v              v
        +-----------+   +-----------------------------------+
        | Embedder  |   | Store                             |
        | text ->   |   | vectors + raw text + metadata,    |
        | vector    |   | persisted to a directory;         |
        |           |   | nearest-neighbor search           |
        +-----------+   +-----------------------------------+
```

Key consequence of this shape: the conversation harness never touches vectors or
files directly. It calls `RAG.search` and `RAG.add_memory`. The store never
calls the embedder. This is what makes the embedder and the store independently
replaceable.

---

## 4. Component Contracts

Pseudocode signatures only. `Matrix[N][D]` is an N-by-D array of 32-bit floats;
`Vector[D]` is one row. "Hit" is a small record returned from search.

### 4.1 Embedder

The only component that maps language to geometry.

```
Embedder
    embed(texts: List[String]) -> Matrix[len(texts)][dim]
        # Each output row is L2-normalized (unit length).
    property dim: Integer
```

Contract notes:
- Output rows **must** be unit length. This is what lets the store treat cosine
  similarity as a plain dot product.
- `embed` takes a list and returns a matrix so callers can embed a batch in one
  call (important for ingest performance).
- The model should be loaded lazily (on first `embed`) so that non-embedding
  operations like listing the store stay fast.
- Two implementations are expected: the real one (a local or hosted model) and a
  **fake deterministic** one used in tests.

### 4.2 Store

The "database": persistence plus nearest-neighbor search. Knows nothing about
embeddings models — it only receives and compares vectors.

```
Store(directory: Path)
    add(vector: Vector[dim], text: String,
        source: String, metadata: Map) -> Id
        # Upsert keyed by content hash of `text`. Returns the entry id.
    search(query_vector: Vector[dim], k: Integer) -> List[Hit]
        # Top-k by descending cosine similarity. Empty store -> empty list.
    all() -> List[MetaRecord]
    count() -> Integer

Hit { score: Float, id: Id, text: String, source: String, metadata: Map }
```

### 4.3 RAG orchestration

Combines an embedder and a store into the two operations callers care about.

```
RAG(directory: Path, embedder: Embedder)
    add_memory(text: String, source = "manual", metadata = {}) -> Id
        # embed once, store
    ingest_file(path: Path, chunk_words = 180, overlap_words = 40) -> List[Id]
        # read -> chunk -> embed batch -> store each chunk
    search(query: String, k = 4) -> List[Hit]
        # embed query -> store.search
```

### 4.4 Conversation Harness

The interactive loop that exposes memory to the LLM as tools. See §7.

```
run_chat(rag: RAG, model: String)
    # REPL: each user turn runs the agentic tool loop to completion.
```

---

## 5. Data Model & On-Disk Layout

Everything for one memory store lives under a single directory the caller
chooses. Default suggestion: `./ragstore`.

```
<store_dir>/
    docs/
        <id>.txt        # raw source text for one entry, verbatim
        <id>.txt
        ...
    vectors.<bin>       # contiguous Matrix[N][dim] of float32, row-major
    meta.json           # ordered array of MetaRecord, row-aligned with vectors
```

`MetaRecord`:

```
{
    "id":       String,   # 16-hex-char content hash (see §6.4)
    "source":   String,   # e.g. "manual", "chat", or a filename
    "metadata": Map,      # arbitrary JSON (e.g. {"chunk": 3})
    "added_at": Number    # unix timestamp
}
```

**The central invariant:** `meta[i]` describes the entry whose vector is row `i`
of the vectors file, and whose text is `docs/<meta[i].id>.txt`. Three parallel
structures, one shared ordering. Every operation must preserve this alignment.

Format choices for the port:
- **Vectors:** a flat binary file of raw float32 in row-major order is the most
  portable and compact. Store `N` and `dim` (either in a header, or derive `N`
  from file size / `dim`, with `dim` recorded in `meta` or a small manifest).
  Alternatively use the language's native ndarray serialization.
- **Metadata:** JSON is fine at this scale and keeps the store human-readable
  (ideal #2).
- **Raw text:** one file per entry keeps the source-data store trivially
  inspectable and decouples it from the index (ideal #4).

---

## 6. Core Algorithms

### 6.1 Embedding & normalization

Run text through the model; normalize each output vector to unit length
(divide by its L2 norm). Normalization is done once, at the embedder boundary, so
no other code has to think about magnitudes.

### 6.2 Similarity search

Cosine similarity between vectors `a` and `b` is `(a · b) / (|a| · |b|)`. Because
all stored vectors and the query are unit length, this reduces to the dot product
`a · b`.

```
search(query_vector, k):
    if store is empty: return []
    sims = matrix_vector_dot(vectors, query_vector)   # length N
    take indices of the k largest sims
    return those entries as Hits, each with score = its sim
```

A full pass over N vectors per query is intentional (ideal #8). For N in the
thousands-to-tens-of-thousands, this is sub-millisecond with a contiguous float32
buffer and a BLAS-backed dot product.

### 6.3 Chunking

Long documents are split into overlapping windows so that ideas straddling a
boundary remain retrievable from either side.

```
chunk_text(text, chunk_words, overlap_words):
    words = split text on whitespace
    if len(words) <= chunk_words: return [text] (or [] if empty)
    step = max(1, chunk_words - overlap_words)
    chunks = []
    for start in 0, step, 2*step, ...:
        window = words[start : start + chunk_words]
        if window is empty: break
        chunks.append(join(window, " "))
        if start + chunk_words >= len(words): break
    return chunks
```

This is deliberately simple. A natural upgrade is to split on sentence or heading
boundaries instead of a fixed word count; the contract (`text -> List[chunk]`)
stays the same.

### 6.4 Idempotent upsert

Entry identity is a hash of the text (e.g. SHA-1 or SHA-256, truncated to 16 hex
chars). On `add`:

```
id = hash(text)[:16]
write docs/<id>.txt
if id already in meta:
    replace that row's vector and its meta record in place
else:
    append the vector as a new row, append the meta record
persist
```

Truncating a cryptographic hash to 16 hex chars (64 bits) makes accidental
collisions negligible for a personal store. Using the content hash as the id is
what delivers idempotency (ideal #5): re-ingesting an unchanged document is a
no-op-equivalent update rather than duplication.

### 6.5 Persistence

The simple, portable approach: after any mutation, rewrite `meta.json` and the
vectors file in full, and the one changed `docs/<id>.txt`. At personal scale the
rewrite cost is irrelevant.

Robustness considerations to weigh during the port (all optional, none required
to match the reference behavior): write to a temp file and atomically rename to
avoid a torn file on crash; take a directory lock if multiple processes might
write; treat concurrent writers as last-write-wins (the default assumption).

---

## 7. The Conversation Harness (Tool-Calling Loop)

This is the heart of the "active conversation" capability and the part most worth
getting exactly right. The harness hands the model two tools and lets it choose
when to use them. The general pattern is provider-independent; the concrete field
names below are for the Anthropic Messages API and should be mapped to whatever
LLM SDK you target.

### 7.1 The agentic loop

Maintain a running `messages` history of alternating user/assistant turns. For
each line of user input:

```
append { role: "user", content: user_input } to messages
loop:
    response = model.create(system = SYSTEM_PROMPT,
                            tools   = [SEARCH_TOOL, SAVE_TOOL],
                            messages = messages)
    append { role: "assistant", content: response.content } to messages  # verbatim
    if response indicates tool use:
        results = []
        for each tool-call block in response.content:
            output_text = execute_tool(rag, block)      # local, see 7.3
            results.append(tool_result(block.id, output_text))
        append { role: "user", content: results } to messages
        continue                                         # let the model use them
    else:
        print the assistant's text
        break
```

Critical details:
- The assistant turn is appended **verbatim**, including any tool-call blocks,
  before the results are sent back. The model must see its own call to interpret
  the result.
- All tool calls in a single turn are executed and their results returned
  together as one user message. The model may both search and save in one turn.
- The loop only exits when the model returns a turn with no tool calls.

### 7.2 Provider-specific shapes (Anthropic Messages API)

- **Tool definition:** `{ name, description, input_schema }`, where
  `input_schema` is a JSON Schema object (`type: "object"`, `properties`,
  `required`).
- **Response content** is a list of blocks. Each block has a `type`; the relevant
  ones are `"text"` (with `.text`) and `"tool_use"` (with `.id`, `.name`,
  `.input`).
- **Signal to run tools:** the response's `stop_reason` equals `"tool_use"`.
- **Returning a result:** a block `{ type: "tool_result", tool_use_id, content }`
  inside a `user`-role message. `tool_use_id` must match the `id` of the
  originating `tool_use` block.
- **Model id** is a string (this design was validated against `claude-opus-4-8`).
  The API key is read from the environment.

Other providers expose the same concepts under different names (function/tool
definitions with JSON-Schema parameters, a finish reason indicating tool calls,
and tool outputs keyed back to a call id). Only the field names and SDK calls
change; the loop in §7.1 does not.

### 7.3 The two tool contracts

**search_memory** (read):

```
name: "search_memory"
input: { query: String, k?: Integer (default 4) }
behavior: hits = rag.search(query, k); return hits formatted as text
```

A reasonable text format for the result is a numbered list, each item showing the
source, the similarity score, and the stored text, so the model can judge and
cite what it got.

**save_memory** (write):

```
name: "save_memory"
input: { text: String, source?: String (default "chat") }
behavior: id = rag.add_memory(text, source); return a confirmation string
          (e.g. the new id and the updated entry count)
```

`execute_tool` is a small dispatcher keyed on `block.name`; an unknown name
returns an error string rather than throwing, so the model can recover.

### 7.4 System prompt intent

The system prompt should tell the model that it has a personal long-term memory
with a read tool and a write tool; to **search** when the user refers to past
discussions or when current context is insufficient; to **save** when asked to
remember something or when the conversation yields a durable fact or decision;
to make saved notes self-contained (ideal #7); and to say which retrieved note an
answer came from so the user can trust the source. Steering tool use is done
through this prompt, not in code.

### 7.5 Transparency

When a tool runs, print a short line to the user indicating what happened (the
query that was searched, or a preview of the text that was saved). The memory
should never be modified invisibly.

---

## 8. Invariants & Edge Cases

A reimplementation is correct when it upholds these:

- **Row alignment.** `vectors[i]` ⇄ `meta[i]` ⇄ `docs/<meta[i].id>.txt`, always.
- **Unit vectors.** Every stored vector and every query vector is L2-normalized
  and has the same `dim`.
- **Dimension consistency.** Querying with a vector from a different embedding
  model/dim than what is stored is a usage error; the dot product would be
  meaningless. Record `dim` so this can be checked.
- **Empty store.** `search` returns an empty list; it must not error.
- **k clamping.** If `k` exceeds the number of stored entries, return all of them.
- **Dedup.** Adding identical text twice yields one entry, updated in place.
- **Reload fidelity.** Constructing a fresh store over an existing directory
  reproduces the same entries, ordering, and search results.

---

## 9. CLI Surface

A single executable with subcommands. A global `--store <dir>` selects the store
directory (default `./ragstore`).

| Command | Arguments | Effect |
|---|---|---|
| `add` | `<text>` `[--source]` | Store one already-compacted memory. |
| `ingest` | `<path>` `[--chunk-words] [--overlap]` | Chunk a text file and store each chunk. |
| `search` | `<query>` `[-k]` | Print the top-k semantic matches. |
| `list` | — | List stored entries (id, source, metadata). |
| `chat` | `[--model]` | Start the tool-calling conversation. |

`add`, `ingest`, `search`, and `list` are fully offline. `chat` requires an LLM
API key in the environment.

---

## 10. Testing Strategy

The injectable embedder (ideal #9) is what makes the core testable without a
model. Use a **fake deterministic embedder** — for example, a bag-of-words count
over a tiny fixed vocabulary, normalized — so that "similar" text produces
"similar" vectors predictably.

Layers to cover:

- **Store:** add then reload reproduces entries (persistence); identical text does
  not duplicate (dedup); a query ranks the semantically closest fake-embedded
  entry first (ranking); empty-store search returns empty.
- **RAG:** chunking produces multiple overlapping chunks for long input; ingest
  then search round-trips.
- **Harness:** feed the dispatcher synthetic tool-call objects (no network) and
  assert that `save_memory` actually writes, that the saved entry is then
  retrievable via `search_memory`, and that a turn containing both a search and a
  save executes both.

None of these require a network, an API key, or a model download.

---

## 11. Porting Notes

Mapping the design onto another language ecosystem. Treat specific library names
as starting points to verify, not commitments — pick the current, maintained
option in your target language.

- **Embeddings (local path).** Most ecosystems can run a small
  sentence-transformer-style model through an ONNX runtime or native ML bindings.
  Confirm what is current and well-maintained for your language. Whatever you
  pick, normalize outputs to unit length at the boundary.
- **Embeddings (portable fallback).** Call a hosted embeddings HTTP endpoint and
  normalize the result. This is the simplest cross-language route but trades away
  the local-first ideal, so make it opt-in rather than the default.
- **Vector math.** Any ndarray/linear-algebra library, or even a hand-written dot
  product over a contiguous float32 buffer. Brute-force top-k is trivial to write
  directly.
- **Persistence.** A flat binary file of float32 for vectors plus a JSON sidecar
  for metadata ports cleanly everywhere; memory-map the vectors file if N grows.
  Keep raw text as individual files.
- **Identity.** Any stable hash (SHA-1/SHA-256) truncated to ~16 hex chars.
- **LLM SDK.** Use the provider's official SDK and reproduce the loop in §7.1,
  mapping the field names per §7.2.
- **CLI.** Use the language's standard argument-parsing library; the subcommand
  surface in §9 is the target.

If you outgrow brute-force search, the *only* component that changes is the Store
(§4.2): swap it for a real vector database or an approximate-nearest-neighbor
index behind the same `add` / `search` contract. Nothing above it moves.

---

## 12. Extension Points

- **Auto-compaction.** Have the model distill and `save_memory` a summary at the
  end of a session, instead of (or in addition to) on explicit request.
- **Confirmation gate.** Require user approval before a `save_memory` call
  executes, so the store only grows with vetted notes.
- **Smarter chunking.** Split on sentence or heading boundaries; optionally store
  a short generated summary per chunk and embed that.
- **Re-ranking.** Retrieve a larger candidate set, then re-score the top
  candidates with a cross-encoder or the LLM before returning.
- **Metadata filtering.** Filter by `source`, date, or tags before (or after) the
  vector search to scope retrieval.
- **Real vector index.** Replace the Store internals with FAISS / sqlite-vec /
  Chroma / LanceDB or equivalent when scale demands it.

---

## 13. Suggested Build Order

Implement and test incrementally; each step is verifiable before the next.

1. **Embedder interface + fake implementation.** Lets everything below be tested
   offline immediately.
2. **Store: add / persist / load.** Verify a round-trip reproduces entries.
3. **Store: search.** Verify ranking and k-clamping with the fake embedder.
4. **RAG: chunking, ingest, search.** Verify ingest→search round-trips.
5. **CLI: add / ingest / search / list.** A usable offline tool at this point.
6. **Real embedder** behind the same interface. Swap it in; the tests still pass
   with the fake.
7. **Harness: search_memory loop.** The read-only conversation.
8. **Harness: save_memory.** Two-way memory; verify the dispatcher with synthetic
   tool calls.

By step 5 you have a working local semantic-search CLI; by step 8 you have the
full read/write conversational memory.
