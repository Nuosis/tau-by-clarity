# Memory ↔ Active Compression — Lane Split

Tau has two context-management systems. They overlap on tool output and are
routinely confused. This document is the canonical map of which system owns
which concern. **Code that violates this map is a bug.**

## TL;DR

| Concern | Owner | Storage | Lifetime | Read primitive |
|---|---|---|---|---|
| In-flight compression of `toolResult` payloads | **active compression** | `~/.tau/agent/ccr.db` | single session, transient (TTL ~30 min default) | `ccr_retrieve([CCR:handle], query)` |
| Durable per-tool-call log | **memory** (tool_log_memory) | `<cwd>/.tau/memory/memory.db` | cross-session, project-local | `memory.tool_log_lookup(tool_call_id)` |
| Exact conversation log per session | **memory** (conversation_memory) | `<cwd>/.tau/memory/memory.db` | cross-session, project-local | `memory.summarize_expand(summary_id)` |
| Atomic curated facts (decisions, constraints, etc.) | **memory** (semantic_memory) | `<cwd>/.tau/memory/memory.db` | cross-session, project-local | passive auto-injection at the `transform_context` seam |
| File-fact staleness (tau-specific) | **memory** (semantic_memory + content_hash) | `<cwd>/.tau/memory/memory.db` | durable, re-checked on `invalidate_stale` | n/a — managed by `MemoryStore.invalidate_stale()` |

## The cross-link

Both systems store the tool output. They MUST NOT be the same record, and they
MUST be addressable from each other. The bridge is `tool_call_id`:

1. `agent_session._after_tool_call` runs after every tool execution.
2. It pins `tool_name` and `tool_call_id` on the active-compression chokepoint
   via `set_current_compression_tool_context(tool_name, tool_call_id)`.
3. It then writes the full original body to `tool_log_memory` with that
   `tool_call_id` as the primary key.
4. The next outbound call to the model fires the chokepoint. The active
   compression chokepoint reads the contextvars and wraps the CCR store with
   `_CCRWithContext(store, tool_name, tool_call_id)`, so every `ccr.put(...)`
   call made deep inside the compressor carries the same id.
5. The CCR row is stored under its content-hash handle, but the row's
   `tool_call_id` and `tool_name` columns are populated, so a `[CCR:handle]`
   marker in the compressed output is addressable from a `tool_call_id` and
   vice versa.

## Write order (locked, do not reorder)

1. `agent_session._after_tool_call` — contextvar set, then `_record_tool_log`
   (durable row first), then extension hooks.
2. Next outbound model call — active compression chokepoint fires, CCR caches
   the *original* body, returns the compressed form to the model.
3. The model sees `[CCR:handle] compressed body`. A pointer to the durable
   record in `tool_log_memory` is reachable via the shared `tool_call_id`.

## Read paths (and which to use)

- **In this session, need the exact body of a recent tool call?**
  → `ccr_retrieve([CCR:handle], query)` (scoped query, never full-dump).
- **Across sessions, need what `tool_call_id` X produced?**
  → `memory.tool_log_lookup(tool_call_id)`.
- **Need a curated fact the model might have forgotten?**
  → it's already in the passive recall block injected at `_transform_context`
  (the `CustomMessage(custom_type="memory_recall")` tail block).
- **Need the full text of a compacted conversation slice?**
  → `memory.summarize_expand(summary_id)`.

## Lane violations (what NOT to do)

- **Do not write to CCR from the memory write path.** The CCR cache is owned
  by the active-compression chokepoint. Memory writes its own table.
- **Do not write to `tool_log_memory` from the active-compression chokepoint.**
  The chokepoint is in-flight; durability is the memory store's job.
- **Do not let memory recall inject full tool outputs into the prompt.** Recall
  injects *atomic* facts; the durable tool output is for
  `memory.tool_log_lookup`, not for passive recall.
- **Do not have the agent call `ccr_retrieve` without a `query`.** Query-less
  retrieval is a back-door to the full payload dump and defeats CCR's purpose.
  The `_VALID_CCR_HANDLE_RE` + required `query` param is a structural
  enforcement; do not relax it.

## Why this is not a circular dependency

CCR is a *transient* in-flight cache keyed by content hash. `tool_log_memory`
is a *durable* per-tool-call log keyed by `tool_call_id`. They live in
different databases (CCR in `~/.tau/agent/ccr.db`, memory in
`<cwd>/.tau/memory/memory.db`). They have different lifecycles (CCR has a TTL
and is process-scoped for expansion state; memory is durable and
project-scoped). They serve different retrieval primitives (`ccr_retrieve` for
scoped BM25, `memory.tool_log_lookup` for an exact tool_call_id match). The
cross-link is metadata, not data: same `tool_call_id` field in both, no shared
storage.
