# Context & Memory Management — Design Thinking

Status: design + offline/Tier-1 evidence in hand; **not yet wired into `pi-py`**.
Direction (decided): **replace pi-py's native compaction with a proper memory system
modeled on Claire's production implementation** (atomic extraction + hybrid local
recall — see §8 "Reference architecture"). Still validating the compression policy that
feeds it.
Scope: how `pi-py` assembles per-call context today, and the planned **"active
compression"** strategy for managing it.

This document captures both the plan (next section) and the reasoning that produced
it (§§1–10). Read the TL;DR for *what we're building*; read the rest for *why*, and
for the dead-ends we deliberately rejected.

---

## TL;DR — the current plan (v1: "active compression")

Decided by the evidence in §9. If you only read one section, read this.

- **Compress by POSITION, not per-turn relevance.** Keep a **head budget** (anchor:
  task/spec/early decisions) and a **tail budget** (recent turns) verbatim; compress
  the middle. Compression is **persistent** — compress once and hold it stable so the
  prompt **prefix stays byte-stable and the model's prompt cache stays warm**. As the
  frontier advances, turns aging past the tail budget are compressed in **batches**,
  not every turn.
- **Compression pairs with ATOMIC MEMORY EXTRACTION, not block-stubbing.** As a block
  ages past the tail budget, **extract atomic memory units** from it (facts, entities,
  decisions, summaries — Oracle's typed memory) and keep the full text retrievable. The
  recoverable signal lives in the atomic memories, *not* in a compressed raw chunk. (A
  keyword cue may stay as an in-context breadcrumb, but it is not the recovery
  substrate.)
- **Recovery is AUTOMATIC, harness-driven, and SEMANTIC over atomic memories — NOT
  model-driven, NOT over raw blocks.** Each turn the harness retrieves the atomic
  memories most relevant to the prompt (embedding cosine) and injects them at the tail
  (cache-safe append). Do **not** rely on a model-called `expand` tool: Tier-1 showed
  the model won't call it (0/6, even `gpt-5.4-mini` instructed to). This is Claire's /
  Oracle's production pattern.
- **Embeddings: yes, but LOCAL and only at ATOMIC granularity.** NIAH at 160k tokens
  (§9) showed semantic retrieval over *coarse 300-word blocks* is **worse** than lexical
  (the needle is diluted in one vector); over an *atomic* unit it catches the paraphrase
  query (cos 0.72) that lexical structurally cannot. So embeddings earn their keep
  exactly where Claire applies them — over small curated units. Run them **locally**
  (Ollama `nomic-embed-text`); never send customer content to a cloud embedder
  (Anthropic has no embeddings endpoint regardless).
- **Flag-gated; lossless ⇒ safe to A/B in prod.** Toggle on/off. Because full text is
  retained, population A/B (or shadow replay) verifies effectiveness post-hoc. Default
  ON once validated (kill-switch flag retained).

**Seam:** `transform_context()` in `core/agent_session.py` (currently an identity stub).
**Substrate:** an atomic-memory store + extractor (this is the Claire/Oracle pattern —
the open design work is how pi-py extracts atomic memories as content ages).

**Open questions:** (1) the **extractor** — what atomic units pi-py pulls from aging
context and how (the curator step); (2) end-to-end non-inferiority of compressed-context
**with atomic-memory semantic recovery wired in** (every eval so far tested the wrong
unit — coarse blocks — or the abandoned model-driven path).

**Rejected along the way** (kept below as reasoning, each with its reason): a high/low-
recency **mode router** (§6 — emergent, not configured); **per-turn relevance
re-shaping** / **pivot-triggered** re-shaping (cache-hostile; position approximates it);
a multi-tier **fidelity ramp** (unvalidated + needed an LLM); **model-driven `expand`**
(Tier-1: model won't call it); **recovery over raw compressed blocks** (§9 NIAH: wrong
granularity — needle diluted; recover atomic memories instead).

---

## 1. The ground truth: the model is stateless

Every model call is independent. We send one flat payload —
`system_prompt + messages[] + tools[]` (`Context` in
`packages/agent/src/pi_agent/types.py`) — the model runs one forward pass over it,
emits a response, and **forgets everything**. The next call re-sends the whole
(grown) payload. The "conversation" is an illusion the harness maintains by
re-sending the transcript.

Consequences that drive the rest of this doc:

- **The context window is a capacity ceiling, not a memory.** It is the maximum
  size of *one* payload (input **and** output share it), set at training time
  (positional range + the KV cache the serving hardware can hold). It is not a
  store the model keeps between calls.
- **The harness owns 100% of context management.** The model cannot prune,
  re-rank, or "ignore the noise." It attends to whatever is on the plate. Garbage
  in the window = garbage attended to. Whatever selection happens, happened in our
  code *before* the call.
- **Models do cache — but it's compute state, not knowledge.** During a forward
  pass the model computes key/value tensors per token (the KV cache). Prompt
  caching persists that KV cache for an exact token *prefix* across calls (short
  TTL) so an unchanged prefix isn't recomputed. It is keyed to the exact tokens:
  change one token and the cache is invalid from that point forward. It saves
  recomputation on tokens you re-send anyway; it does not "remember" anything.

## 2. Why "smart" context management is not free

The KV cache creates a perverse incentive:

- **Append-only context is cache-cheap.** A stable prefix hits the cache every
  turn; you pay almost nothing to carry history.
- **Intelligent management is cache-expensive.** Reordering, pruning a middle
  message, re-ranking by relevance, or injecting a retrieved fact mid-prompt
  **invalidates the cache from the edit point forward — every turn.** The clever
  pruning that saved input-billing tokens can cost *more* in reprocessing than it
  saved.

So "dumb append-and-compact" is cheap *because* it's dumb. Compaction is the one
tolerated disruption precisely because it fires *rarely* (at a threshold), not
every turn.

Other reasons harnesses default to append-only:

- **Recency is a strong baseline.** For coding work the most relevant context
  usually *is* the most recent. A relevance selector must consistently beat
  recency to be worth it, and when it guesses wrong it drops the one thing the
  model needed — a failure mode recency never has.
- **Determinism.** A linear transcript is replayable and auditable. Dynamic
  selection is non-deterministic: same session, different slice, different answer —
  miserable to debug and eval.
- **Big windows made laziness cheap.** Once windows hit 200k–1M, the pressure to
  be clever dropped; throwing tokens at the window is cheap and safe.

The frontier answer to "do it smarter" is mostly **move selection into the
model's hands**: on-demand `read`s (the filesystem is the store), sub-agents that
absorb a noisy sub-task and return only the conclusion, explicit `/compact`. The
agent selects on purpose instead of the harness guessing.

## 3. What pi-py does today

- **Cold store:** `.pi-py/agent/sessions/*.jsonl` — append-only tree of entries
  (`message`, `compaction`, `branch_summary`, ...) linked by `parentId`. Durable,
  episodic, per-session.
- **Hot assembly:** `build_session_context()` → `convert_to_llm()` →
  `Context{system_prompt, messages, tools}`. The whole assembled session, growing
  each turn, until compaction fires.
- **Window management = compaction**, not retrieval
  (`packages/coding-agent/src/pi_coding_agent/core/compaction/`). Two triggers:
  overflow (immediate retry) and threshold (~70%, proactive). It summarizes old
  turns into one block and keeps recent ~20k tokens.
  - Defaults: `reserveTokens: 16384`, `keepRecentTokens: 20000`.
- **Selection criterion is recency only.** Keep recent, lossily summarize the
  rest. Once compacted, the detail is gone from the working set forever (still in
  the JSONL, but nothing reads it back in). There is no relevance-based recall.

This is well-matched to coding and should be the default. The opportunity is to
recognize the cases where recency is the *wrong* selector.

## 4. The core question: does recency predict relevance?

The decision between strategies reduces to one question, asked **per session**:

> Does "most recent" reliably predict "most relevant"?

- **Yes → cache mode.** Append-only + threshold compaction + JIT file reads. The
  cache pays off big; curation would mostly just risk dropping something for
  little gain. (Coding, iterative edits.)
- **No → lean mode.** Relevance is dispersed back through history (a constraint
  from turn 3, a decision at turn 20). Recency is a weak selector, so curated
  retrieval earns its cost. (Planning, speccing, research, long debugging
  archaeology.)

"Slow/high-value vs fast/cheap" is a useful proxy but not the root cause. The root
cause is recency-vs-relevance. Three forces all point the same way:

1. **Relevance.** Dispersed relevance breaks the recency baseline.
2. **Economics.** Slow-turn = few calls, so an expensive (cache-miss + retrieval)
   call is affordable. Fast-turn = hundreds of calls, so cache efficiency
   compounds — a miss per turn is brutal.
3. **Cost of rot is asymmetric.** A stale token in a coding session wastes budget.
   A rotted *premise* in a spec propagates into the whole artifact. Worth paying
   more to avoid rot exactly where relevance is dispersed.

## 5. It is not binary

Two refinements that keep us off both poles:

- **Hybrid: stable prefix + curated tail.** Keep `system_prompt` + core tools +
  recent verbatim turns as a **stable, cacheable prefix**, and confine retrieval
  churn to a **block at the tail**. Cache hits on the big stable front; pay the
  miss only on the small retrieved section. This is the right shape for lean mode —
  *not* Oracle's "rebuild the whole context every turn" (costed for a latency-free
  research assistant, not for us).
- **"Less wizardry" for coding ≠ none.** Coding still needs management, just a
  different kind: **model-driven JIT** — the filesystem is the memory store, the
  model reads what it needs this turn and lets it fall out after; sub-agents absorb
  noise and return conclusions. Lean context through tool design, without a harness
  retriever and without touching the cacheable prefix.

> §§4–5 frame this as two *modes* (cache vs lean) chosen by a router. §6 supersedes
> that: there is one relevance metric and the modes are emergent, not configured.
> Read §§4–5 as the reasoning that led there, not the conclusion.

## 6. The one metric: relevance to the current prompt (shape is emergent)

The earlier drafts of this doc reached for a *mode router* — detect "high vs low
recency", switch between a barbell strategy and a fidelity-gradient strategy. That
was the wrong altitude. There is one metric and one policy; the rest is emergent.

**The metric is relevance of each block to the current prompt.** Not position, not
recency — those are at most modifiers. Recency only ever mattered as a *proxy* for
relevance; measure relevance directly and the proxy retires.

### The policy: two stages

1. **Eligibility.** A block is eligible to compress when it is **low-relevance to
   the current prompt** *and* total context is over the floor (§7). High-relevance
   blocks are **never eligible** — full fidelity, wherever they sit.
2. **Aggressiveness.** Among *eligible* blocks, the further back a block sits, the
   lower the fidelity you take it to (verbatim → light → heavy → breadcrumb).

That's it. High relevance is protected absolutely; low relevance is demoted, harder
with age.

### Shape is an output, not an input

You never choose "barbell" or "wedge". You score by relevance, protect the relevant,
demote the rest by age — and *whatever silhouette falls out, falls out*:

- Relevance clustered at task-start + recent (the usual case) → a **barbell** emerges.
- Relevance genuinely dispersed → something **lumpier** emerges.
- Relevance smoothly decaying with age → a **wedge / long hair** emerges.

The wedge-vs-barbell debate dissolves: they were silhouettes of one policy, not
rival strategies. Likewise the **regime router dissolves** — there is no
"high/low-recency mode" to detect and switch. High-recency sessions have relevance
concentrated recent, so little is eligible and the context looks like plain append;
low-recency sessions have scattered relevance, so the demotion bites the irrelevant
middle. **The regime is emergent behaviour of the one rule, not a configured mode.**

## 7. Why it's safe, and what it costs: eviction + recall = paging

Relevance-scored demotion is exactly an **OS paging policy**: relevance is the
eviction policy (which pages leave RAM), and **recall is the disk** — demoted content
still lives in the cold store and is paged back when needed. This reframing carries
three load-bearing consequences.

### The losslessness comes from recall, not from clever compression

"Compress the right stuff losslessly" is true *only as good as recall is.* The
compression decides what to evict from the hot working set; **recall is what makes
eviction non-destructive.** Tier-0 (§9) proved the contrapositive empirically:
relevance-demotion *without* a recall layer drops a dispersed mid-conversation needle
60–76 % of the time. So the policy is inseparable from recall — never ship the
eviction half alone.

### Relevance is time-varying — so don't evict to zero, leave a breadcrumb

A block can be low-relevance *now* and become the needle 30 turns later (the "return
to NBNA" pivot: irrelevant during the invoicing turns, critical at the pivot). The
policy *will* correctly demote it now, and you are safe **only because recall
resurrects it** at the pivot. Two rules follow:

- **Demote to a breadcrumb, not to nothing.** The lowest fidelity tier is a
  relevance-tagged stub ("…NBNA migration constraint here…"), not deletion. That
  residue is the **retrieval cue** that tells recall to fire; evict to nothing and
  recall fires blind on the prompt alone.
- **Protect-high only helps for relevance you can see *now*.** Future-needles look
  irrelevant today, so "high-relevance is never touched" does not save them — recall
  does. Budget for recall accordingly; it is the safety net, not an enhancement.

### The cache discipline → why v1 compresses by POSITION, not per-turn relevance

Scoring is free and continuous. **Re-shaping (recompressing) is not** — it mutates
the prefix, invalidates the cache, and forces reprocessing. An earlier draft tried to
contain this by re-shaping "only at relevance pivots." **v1 drops the pivot machinery
entirely** and compresses by *position* instead, because position is cache-stable by
construction and Lost-in-the-Middle says the middle is the right thing to thin anyway:

- Protect a **head budget** and **tail budget** verbatim; compress the middle.
- Compression is **persistent** — compress once, hold it. The prefix only changes when
  the frontier advances enough to age a turn past the tail budget, handled in
  **batches** (every K turns / N tokens), so cache breaks are rare and bounded.
- **No pivot detection.** Position decides *what* is compressed; relevance is used only
  on the **recovery** path (which compressed block to page back), which appends at the
  tail and never mutates the prefix.

This still bounds cache cost in both regimes, but without per-turn scoring or a pivot
detector: high-recency work keeps everything in the head/tail budgets (looks like
append); long low-recency dialogue compresses an aging middle in batches.

> Deferred upgrade: per-turn relevance scoring + pivot-triggered re-shaping (the
> superseded approach above) could in principle out-select position. It is deferred
> because position + automatic recovery already clears ~98% (§9) at far lower
> complexity and zero cache risk. Revisit only if evals show position leaving value on
> the table.

### Two thresholds: the ceiling (measured) and the floor (cost)

Compression is bounded by **two** sizes, both derived from one calibrated number —
`~128k` was a placeholder for both.

**Ceiling C — the measured reliable working-set size.** The §9 coding eval showed
gpt-5.4-mini **fails to recall a verbatim fact buried in 55k tokens of dense content**
(lost-in-the-middle / context rot). So:

> **C = the largest context in which the model still recalls a worst-depth needle at
> ≥ ~95%, on realistic dense content.** Compress to keep the working set **≤ C**.

**Floor F — below it, don't compress at all.** Two reasons, same direction: below C the
model finds everything (no recall benefit), and compression's overhead (curator +
embedding + retrieval + cache invalidation) exceeds its savings while a stable small
prefix stays cache-warm. So:

> **F ≈ 0.5·C** (start compressing with headroom as you approach C), floored by an
> absolute cost-minimum (never run the machinery below ~20–30k — any model is fine and
> cheap there). **Below F:** append + cache-warm, do nothing. **F→C:** compress
> progressively to stay under C. **Above C:** danger zone, must compress.

Context size manages **effectiveness *and* cost**: below F more context is both
effective (model handles it) and cheap (cache-warm), so compressing trades a free lunch
for overhead and risk. The cost gate above F is still a break-even (act when
`tokens-reclaimed × cost-per-turn × turns-remaining > cache-reprocessing cost`;
per-location — editing the middle reprocesses everything after it, the tail little).

Consequences:
- **Model-specific.** Each model/tier has its own reliable size (Context Rot, §10,
  measured 18 models — they vary widely). Calibrate per model; recalibrate on switch.
- **Couples context-management to model selection.** A model with weak middle-recall
  needs *more aggressive* compression — "which model" and "how hard to compress" are one
  decision. A weak-recall model can be *more expensive* to run well than a stronger one.
- **Calibrate via a NIAH sweep** (`evals/niah_calibrate.py`): size **× depth**,
  realistic dense distractors, threshold on accuracy. The number is a heuristic *upper
  bound* — it measures recall, not multi-fact reasoning, which may degrade earlier.

**Calibration results (depth 0.5, n=1 — indicative, not precise):**

| MiniMax-M3 (1M window) | reliable | fails by |
| --- | --- | --- |
| clean prose | ~500k | 700k |
| dense agent content | ≥250k (corpus-capped) | — |

Two findings dominate: **(1) content density, not raw size, sets the reliable point** —
on clean prose models hold far longer than on dense agent content (the §9 55k dense
failure was density, not length). **(2) Even a 1M-window model degrades before the
window fills** — M3 reliably uses ~half its window on clean prose, less on dense. So a
model's *advertised* context window is **not** its usable working set; the compression
target is the *measured* reliable size on **realistic dense** content (per model, per
content type). (Sweep was n=1/cell — indicative; a precise number needs N needles ×
multiple depths. gpt-5.4-mini was dropped from comparison — not a model we'd run; for
the record it collapsed on dense content even at 10k.)

### Measuring relevance (per-block score, cheapest first)

The relevance score blends, in rising cost (the Generative Agents formula is the
starting point — §10):

1. **Last-reference recency (free).** When was this block last touched — read,
   re-mentioned, depended on? Untouched for N turns and not in the protected set =
   decaying.
2. **Size-to-reference ratio (free).** Big and never referenced again = prime
   demote target. Verbose tool outputs score worst — correctly.
3. **Lexical / embedding similarity to the current prompt (cheap).** The direct
   relevance signal; also what flags a buried region the new prompt points back to.

### Recall placement — the standing cache trap

Recall injection fights the cache **unless it lives at the tail.** Inject fresh
recall after the system prompt but before the transcript and you invalidate the
whole transcript cache every turn. Rule: **stable prefix = system prompt +
(persistently-compressed) transcript; churning tail = recovered blocks + new turn.**
Recovered full-text and the new turn go at the bottom.

## 8. Where this plugs into pi-py

One strategy (the TL;DR plan), three seams:

- **`transform_context()`** (`core/agent_session.py`, currently the identity stub) —
  the assembly seam. Each turn it: (1) **position-compresses** — protect head/tail
  budgets, batch-compress turns that have aged past the tail budget into keyword
  cue + ref (persistent; no per-turn re-scoring, no pivot detection); (2) runs
  **automatic recovery** — score compressed blocks' cues against the current prompt
  (lexical in v1) and **append the full text of matches at the tail**; (3) appends the
  new turn. The prefix stays byte-stable between compression batches → cache warm.
- **A recall layer — the non-negotiable other half, and it must be HARNESS-DRIVEN.**
  Demotion is only lossless because the harness *automatically* re-injects a compressed
  block when its cue matches the prompt. **Do not rely on a model-called `expand`
  tool** — Tier-1 showed the model won't call it in real flow (0/6, even
  `gpt-5.4-mini` instructed to): it can't tell it's missing what it can no longer see,
  and confabulates instead. An `expand` tool may exist as a *fallback*, but it is never
  the primary recovery path. (This is exactly Oracle's "retrieval is **programmatic**,
  not agent-triggered" rule — see §10.)
- **Reversible store** — every compression keeps the original retrievable by id (à la
  the Oracle pattern, *not* pi's current lossy compaction). Standalone upgrade over
  today's compaction; worth landing first.

What this is **not**: a "cache mode vs lean mode" toggle, and **not** model-driven
expand. The append-like behaviour in coding and the middle-compression in long dialogue
both *emerge* from position-compression — nothing selects them.

### Reference architecture (ADOPT): Claire's memory system replaces pi-py compaction

**Decision:** pi-py's native compaction is **replaced** by a proper memory system
**modeled on Claire's production implementation** — canonical
`/Users/marcusswift/python/clarify_voice`, doc `docs/agent-memory-system.md`. Claire is
the existence proof the pattern works; pi-py adapts it for a **local, single-user coding
agent**. Build against this map, not from scratch:

| Claire (canonical) | role | pi-py adaptation |
| --- | --- | --- |
| `agent_semantic_memory` / `SemanticMemoryRow` — pgvector(1536) + HNSW cosine, typed (preference/workflow/entity/knowledge/toolbox), scope keys, `key` dedup, `active`, audit metadata (`agent_memory/db_models.py`) | atomic durable memories | SQLite + **LOCAL** embeddings (Ollama `nomic-embed-text`, 768-d); same typed rows; scope simplifies to project/session/cwd (no tenant/workspace/user) |
| `agent_conversation_memory` / `agent_summary_memory` / `agent_tool_log_memory` — exact turns; summaries (full text retained via `summary_id`); offloaded tool outputs | transcript + compaction substrate | adopt all three — **this is what replaces native compaction**: summarize-and-retain (not discard), offload verbose tool dumps |
| `TierMemoryCuratorAgent` (`voice_harness/memory_commit.py`) — LLM curator, post-turn + post-session: evidence packet → atomic candidates → commit decision (auto/review/reject + confidence) → **second grounding-verification pass** + structural guards (`source_event_ids` must be real); **assistant output is NOT eligible evidence** | the **extractor / writer-of-record** (our open "extractor" question) | same LLM-curator pattern; extract coding atoms (decisions, constraints, file/API facts, tool-result facts); local tier model |
| retrieval: **hybrid `max(lexical, vector cosine)`**, scope-filtered, top-5 (+top-8 preferences), `min_score≈0.05`, ~1200-tok budget (`agent_memory/postgres_manager.py`, `context.py`) | recall into context | **adopt hybrid as-is** |
| three writer paths: programmatic curator; agent `memory.propose` (audit only); **NO direct agent write** | safety boundary | keep verbatim — model proposes, curator commits; never let the agent write durable memory directly |
| three reader paths: passive auto-injection (per turn, budgeted); agent `memory.lookup_scoped` (on-demand, scope-gated); procedural advisory | recall surfaces | passive auto-injection at the `transform_context` seam + a scope-gated lookup tool |

**What Claire confirms from our evals:**
- **Granularity** — Claire embeds *atomic curated units* (title + content), never raw
  chunks → no dilution. Exactly the §9 NIAH lesson, in production.
- **Hybrid retrieval `max(lexical, vector)`** settles our lexical-vs-semantic question:
  **use both** — lexical catches rare-term/literal/IDs, vector catches paraphrase.
- **Programmatic, not model-driven** — the curator writes; the agent only *proposes*.
  Matches the Tier-1 "model won't self-recover" result.

**Status:** the *memory architecture* is settled (adopt Claire). Still being validated
(don't block on it): the **compression/position policy** that feeds it (§§6–9) and an
end-to-end non-inferiority eval at atomic granularity. We are using the docs + Claire as
the model while that validation continues.

### Full memory system (project-local) — compression is one consumer

The §8 Claire-mapping table is the component blueprint; this is the *whole system* pi-py
builds, with the deltas that are pi-py-specific. **Active compression (§§6–7) is one
consumer of the store, not the system.** The system = **store + scope + curator +
lifecycle + retrieval**, and the defining choice is **project-local**:

- **Storage / persistence.** `./.pi-py/memory/memory.db` in the **cwd** — SQLite + local
  768-d embeddings (§8 table; local per §10 Oracle-divergence note). Tables as in §8
  (semantic / conversation / summary / tool_log). Cross-session persistence per project
  is the continuity pi lacks today (§3).
- **Scope = poisoning boundary.** Project-local storage means an agent in repo A cannot
  read/write repo B's store — that *is* the anti-poisoning mechanism. Within a project,
  scope by session / task / file-module + a project-global lane. Every memory carries a
  **provenance tag** (which session wrote it) → read-side trust filter; the curator's
  grounding-verification (§8, assistant output ineligible) is the write-side gate.
  **Open decision:** `./.pi-py/memory/` **gitignored** (per-clone, fully isolated —
  recommended) vs **committed** (team/CI share, but memories mix). Default isolate +
  explicit export.
- **Formation = the curator** (§8 writer-of-record). pi-py triggers add post-tool-result
  and post-decision to Claire's post-turn/session. Coding atomic taxonomy: decisions,
  constraints, file/API facts, task-state, error→fix, preferences. Dedup via canonical
  keys + supersession.
- **Lifecycle + the coding-specific hazard.** States active/superseded/archived/deleted.
  **New vs Claire: file-fact staleness.** Code facts rot when files change — tie
  file-scoped memories to a **content hash / mtime** and invalidate on change, or the
  store self-poisons with stale signatures (worse than cross-agent poisoning).
- **Retrieval** — exactly §9's verified path: hybrid `max(lexical, semantic)`, scoped,
  breadcrumb-aware (never evict to zero), top-k recovery as the bonus, injected at the
  tail. Passive auto-inject + scope-gated `memory.lookup`.

So the build order is store → curator → retrieval (the proven core), with active
compression layered on top as the working-context consumer that pages atomic memories in.

### Implementation plan (build phases — each gated by an eval before the next)

Build the *proven core first* (store → retrieval → curator), compression last. Reuse the
existing `evals/` harnesses — do not rebuild them. Each phase ships behind the same flag,
default off until its gate passes.

- **P0 — Store & scaffolding.** `core/memory/` package; SQLite at **`./.pi-py/memory/
  memory.db`** (cwd-rooted §8); typed tables (§8 Claire map); scope + provenance model;
  local embedding provider (Ollama `nomic-embed-text` 768-d) + deterministic test
  fallback. *Gate:* schema + write/read round-trip tests.
- **P1 — Retrieval (read path) first.** Hybrid `max(lexical, semantic)`, scoped,
  breadcrumb-aware, top-k + budget (§9 verified path); passive tail-injection via
  `transform_context`. *Gate:* re-run `evals/coding_recall_ab.py` against the **live
  store** (not the proxy) → reproduce §9's 4/4 on M3.
- **P2 — Curator (write path).** LLM curator post-turn/session/tool-result; grounding
  verify + structural guards; `memory.propose` + scope-gated `memory.lookup` tools
  (hardened: input_schema + retry-hinting errors + tracing); **no direct agent write**.
  *Gate:* TDD + add curator to the BASIC eval runner with curator-specific requirements
  (atom-extraction precision, assistant-output-ineligible, grounding) — never excluded.
- **P3 — Lifecycle + staleness.** active/superseded/archived/deleted; **file-fact
  content-hash invalidation** (the pi-py-specific hazard). *Gate:* stale-file
  invalidation test.
- **P4 — Active compression consumer.** Position-compress middle above floor **F** to
  breadcrumb+ref; recover atomic memories into the lean tail; cache discipline (§7);
  floor/ceiling per-model config from calibration (`evals/niah_calibrate.py`). *Gate:*
  cache-stability + recall on the live path.
- **P5 — Replace native compaction.** Swap the compaction path; flag default-on once
  validated, kill-switch retained; lossless ⇒ prod-A/B-able (TL;DR).
- **P6 — Acceptance.** Plug pi-py into Pier as a Harbor agent; **deep-swe micro run**
  (`--n-tasks 10 --sample-seed 0`) baseline vs memory-augmented on M3 → fast pass@1
  estimate; full 113 once the micro delta looks right.

**Curator atomic taxonomy (DECIDED):** decisions · constraints · file/API facts ·
task-state · error→fix · preferences. (Canonical-keyed where the unit has identity,
e.g. `decision:db_choice`, `fileapi:config/net.py`; superseded on change.)

**Open decision before P0:** gitignore-vs-commit the store (§8). **Process:** building
this is substantial code — work on an authorized branch only (no branch without explicit
go-ahead).
## 9. What to build first — measure before you ship

The cheapest end-to-end slice that proves the idea, changing **no** behavior:

1. Instrument **relevance-to-prompt** per block and the **recall-miss counter** (model
   reaches for demoted content). Log them.
2. Run across real sessions and just *look* at the numbers.
3. Confirm the policy demotes the right blocks (low-relevance, aging) and that recall
   would have to fire to keep the rest lossless.

Only once the eviction *and* recall halves are validated together do we wire the
policy into `transform_context`. Eviction earns the right to change behaviour after the
logs prove recall catches what it drops — observe the signal is real first.

### Tier-0 harness (built) — `evals/tier0_context_replay.py`

Offline counterfactual replay, **zero LLM calls**. For each user turn it reassembles
context two ways — `baseline` (full append, today's behavior) and `candidate` (the §7
fidelity gradient: anchor + frontier verbatim, cold/low-relevance/bulky middle demoted
to reversible stubs, Generative-Agents scorer) — and reports three numbers:

- **token delta** — how many fewer tokens the candidate sends.
- **prefix stability** — cache-hit proxy (fraction of the prompt byte-identical to the
  previous turn). Surfaces the *cache cost* of re-compression directly.
- **needle preservation** — of facts a later turn actually re-references, what fraction
  survived (the compaction-recall-miss, inverted). `--needle` plants a controlled
  early-fact / late-query pair (Chroma context-rot style).

Status of the first run:

- **Plumbing proven** on real local sessions. They are 1–2 turns — no *middle* — so
  the gradient is correctly a no-op there (anchor + frontier cover everything). Not a
  win, just a clean pass; confirms the floor gate and U-shape behave.
- **Mechanism proven** on a synthetic planning fixture
  (`evals/fixtures/synthetic_planning.jsonl`: early hard constraint, bulky
  never-referenced middle, late turn that re-invokes the constraint): **17–31% token
  savings, needle preserved 100%, and candidate prefix stability measurably below
  baseline (~0.53 vs ~0.66)** — i.e. the predicted tradeoff (cheaper, lossless on the
  needle, *less cacheable*) shows up in the numbers. That last fact is the empirical
  basis for the regime split: pay it for slow planning turns, not fast coding turns.

**Real-data run (clarify Langfuse, 21 agent traces).** `evals/langfuse_fetch.py`
pulls the long agent-run traces from the clarity Langfuse ClickHouse (each trace's
most-accumulated message array → a session fixture) and Tier-0 replays them with
`--per-message-turns`. Result across 21 production traces (41–200 messages each):

- **33% aggregate token savings** (12.9–56.9% per trace) from compressing the cold
  middle.
- **But prefix stability collapses: baseline ~0.89–0.97 → candidate ~0.28 avg
  (as low as 0.06).** The gradient saves a third of the tokens *while destroying
  cacheability.*

This is the **regime split confirmed on production data**, not theory: these are
high-recency agent tool-loops (frequent turns, near-perfect append-only cacheability),
so the gradient's token win is paid back — and then some — in cache-miss reprocessing.
Conclusion holds: **for this regime, do NOT run the continuous gradient — keep append +
rare discrete compaction (barbell).**

**Low-recency regime (Claire chat, 14 sessions, `evals/clarify_chat_fetch.py`).**
clarify Langfuse stores one trace per session, so the human-dialogue regime lives in
the clarify Postgres `agent_chat_messages` (voice transcripts are reconstructable from
`agent_events` `stt.final`/`agent.response` but noisy; chat is clean). These are genuine
multi-turn dialogues (up to 170 turns) with **topic pivots and cross-conversation
recall** ("return to NBNA — report what you recall"). Tier-0 result:

- **Only ~14% aggregate token savings** — *less* than the agent loops. Human turns are
  short and information-dense; there is little bulky cruft to compress. Cache stability
  drops here too (baseline ~0.53–0.95 → candidate ~0.20–0.56).

**The critical finding (mid-conversation needle).** With the needle correctly planted in
the *compressible middle* (not the anchor): **baseline preserves it 100%; the gradient
preserves it only 24–41%.** The naive score-and-stub gradient **drops dispersed middle
relevance 60–76% of the time** — precisely the thing the low-recency regime exists to
protect. So Tier-0's verdict on the gradient is *negative in both regimes as currently
specced*: cache-destroying in the high-recency regime, and lossy-on-recall in the
low-recency regime.

This sharpens the design rather than refuting it: the fidelity gradient is **only safe
paired with reversible expand** (§7/MemGPT) — the demoted block must be retrievable and
the model (or a query-time re-score) must pull it back when the prompt turns relevant.
Tier-0 measures only the *static working set*, so it cannot see the expand path; proving
the gradient net-positive therefore **requires Tier-1** (cheap-model A/B with a pairwise
non-inferiority judge + an expand tool), then production model on the contested subset.
Token counts are char/4 approximations — fine for deltas, not billing.

### Recovery eval — do we need embeddings? (`evals/recovery_eval.py`)

The "active compression" recovery question — *can lexical recovery (no embeddings)
surface a compressed block when it becomes relevant?* — tested stdlib-only (no
embedding model is installed; embeddings would mean network + a key + shipping content
out). Position-compress the middle to a deterministic TF-IDF **keyword cue**; recovery
signal = overlap of the query terms with that cue. Two tests:

- **A — controlled needle.** Literal query (shares terms) → **recovered**; paraphrase
  query (zero shared terms) → **missed at any cue size.** That paraphrase miss is the
  entire embedding-shaped gap, shown without embeddings.
- **B — real cross-reference recall vs cue size:**

  | cue size | agent trace (big tool-dumps) | Claire chat (small dense) |
  | --- | --- | --- |
  | 12 | 43 % | 60 % |
  | 30 | 80 % | 94 % |
  | 60 | 87 % | 98 % |
  | 120 | 98 % | — |

**Findings.** (1) Lexical recovery reaches **~98 %** of genuine cross-references on both
regimes with an adequately-sized cue — the early low number was *cue starvation*, not a
method limit. (2) **Cue size is the knob**, and it self-aligns with where compression
pays: chat plateaus at ~30–60 terms (but barely compresses anyway); huge tool-dumps need
~120 terms yet still compress ~16× at 98 % recall. (3) The residual ~2 % is the
paraphrase case — embeddings' *entire* marginal value here.

**Decision: v1 ships WITHOUT embeddings.** Deterministic keyword cue (sized to block:
~30 chat / ~120 tool-dump) + lexical recovery clears ~98 %. Embeddings stay a measured,
**local-only** upgrade (Anthropic has no embeddings endpoint; never ship Claire content
to a cloud embedder). **Caveat:** Test B's term-overlap ground truth cannot see
*pure-paraphrase* cross-references, so their real-world frequency is unmeasured — that is
the one thing Tier-1's LLM judge must quantify before trusting the ~2 % figure.

### Tier-1 A/B — model-driven `expand` fails; recovery must be automatic (`evals/tier1_recovery_ab.py`)

Live A/B on the low-recency Claire chat: full context vs active-compressed-with-an-
`expand`-tool, pairwise non-inferiority judge. Runs on a chosen model (local Ollama
`qwen3:30b`, or OpenAI). Result was the same across **two very different models**:

- `qwen3:30b`: 1/4 non-inferior, **0/4 used expand**.
- `gpt-5.4-mini` (frontier tool-caller, hardened tool def, system prompt *mandating*
  expand): 2/6 non-inferior, **0/6 used expand**.

A smoke test confirmed `gpt-5.4-mini` *will* call `expand` on a pointed question about a
single obviously-relevant stub — so this is not a tool-definition defect. In **real
multi-block flow it never fires**: the model can't tell it's missing what it can no
longer see, so it confabulates a fluent answer and silently drops the compressed
specifics (names, numbers, the exact flow steps). The compressed arm degrades exactly
as Tier-0 predicted *when recovery doesn't fire*.

**Decision — recovery must be HARNESS-DRIVEN (automatic), not model-driven.** Tier-0
said automatic lexical recovery ≈ 98 %; Tier-1 says model-driven `expand` ≈ 0 % in
realistic flow. This is precisely Oracle's "retrieval is **programmatic**, not
agent-triggered" rule (§10): we tested the agent-triggered path, it failed as Oracle
predicts, and v1 adopts the programmatic path.

A first **harness auto-recovery** arm (lexical top-6 over compressed *conversation
blocks*) tied model-expand at 1/6 — which led to the next test:

### NIAH at scale — recovery is a GRANULARITY problem (`evals/niah_recovery.py`)

Plant a needle, compress its block, and ask whether **lexical** vs **semantic** (local
`nomic-embed-text`) recovery surfaces it — in a **160k-token** corpus (Moby Dick), above
the compression floor, with a **literal** and a **paraphrase** query.

| needle representation | cos(paraphrase) | cos(literal) | retrieval result |
| --- | --- | --- | --- |
| diluted in a 300-word block | 0.53 | 0.48 | lexical gets literal @ rank 1; **semantic worse (rank 131)**; both miss paraphrase |
| **atomic** (sentence alone) | **0.72** | **0.89** | semantic recovers it, **including paraphrase** (lexical: 0) |

**The finding: granularity, not lexical-vs-semantic, is the lever.** A short needle
diluted in a coarse 300-word block can't be represented by one embedding vector, so
semantic retrieval is *worse* than lexical there. At **atomic** granularity semantic
catches the paraphrase case lexical structurally cannot.

**This reconciles everything — and explains why Claire's recall works.** Claire embeds
**atomic curated memories** (facts/entities/summaries), so its granularity is right and
semantic recall succeeds in production. Every recovery eval here that *failed* did so on
**coarse raw chunks** — the wrong unit. So the plan corrects: **recovery operates over
atomic extracted memories, semantically (local embeddings), not over raw compressed
blocks.** Compression of aged content and atomic-memory *extraction* are the same step.
**Open:** the extractor (what units, extracted how) and an end-to-end eval at the right
granularity.

### Coding-agent recall — VERIFIED end-to-end (`evals/coding_recall_ab.py`)

The decisive test, in the actual target domain. Plant a coding fact (`config/net.py:
MAX_RECONNECT_ATTEMPTS = 7741`) as its **own atomic unit** in a **real `devin_agent`
trace**, compress the middle, recover via **hybrid `max(lexical, local-semantic)`**,
and have the model answer. Two traces (55k & 12k tok), literal + paraphrase queries:

| arm | result |
| --- | --- |
| full context (verbatim) | **fails both traces** — needle present but lost-in-the-middle |
| no-recovery (middle stubbed) | **fails both** — needle absent |
| **hybrid recall (lean context)** | **correct in 3/4** query-arms; needle retrieved **#1 of 25–29** in all 4 |

**This verifies recall for coding** — not theorized: atomic-unit + hybrid recall
retrieves the fact at rank 1 (literal *and* paraphrase) and the model answers from a
**lean** recovered context, beating both dump-everything and drop-the-fact. Two
preconditions proved **load-bearing** (and they're why earlier proxies failed):

1. **Atomic granularity** — the fact is its own unit, not diluted in a chunk (rank 1,
   not 131).
2. **Lean recovered context** — recovered atomic facts at the tail, *not* head+tail+all-
   stubs (which re-creates lost-in-the-middle; that was the bug, not recall). This is
   why the **curator** that produces small atomic units is load-bearing, not optional.

**Honest residue:** the one miss (trace-1 paraphrase) had the needle retrieved #1 but
answer-synthesis lost it among semantic distractors — a *precision* issue, not
retrieval, and narrower for coding (recall keys are mostly exact tokens — IDs, paths,
identifiers — where literal recovery is clean).

### MiniMax-M3 replication + low-recency papers (matrix completed)

**Coding A/B on MiniMax-M3** (the model we'd actually run): hybrid recall **4/4**
(2 traces × literal+paraphrase), needle rank #1. Two contrasts vs gpt-5.4-mini:
- **M3 full-context = correct at 55k where gpt-5.4-mini failed** — the calibration's
  model-reliable-size gap, reproduced in the A/B. Recovery is *more* necessary for
  weak-recall models; M3 at 55k didn't even need it.
- **no-recovery LITERAL = correct** — the keyword-cue stub retains the distinctive token
  (`7741`), so the breadcrumb alone answers exact-token queries; paraphrase needs full
  recovery.

**Low-recency regime — stitched arXiv papers** (4 disjoint papers, real head/tail,
camouflaged paper-claim needle in the compressed middle, M3): hybrid **4/4**. The
decisive finding came from a bug: a "lean" render that *dropped* the stubs lost the
paraphrase needle (which ranked **#12**, not top-8 — a retrieval-benchmark needle has
~11 semantic competitors in a corpus *of retrieval papers*). Keeping the **breadcrumb
stub** (with `84.3`/`RUBICON-7` in its cue) recovered the answer.

**Two load-bearing conclusions, now evidenced:**
1. **Never evict to zero — the breadcrumb stub is the safety net.** In topically-dense
   corpora, top-k semantic recovery *alone* misses (needle ranks #12); the cue's
   distinctive tokens are what save it. Top-k full-text recovery is the *bonus* for
   paraphrase, not the floor.
2. **Reliable working-set size is the dominant tuning constant**, and it's per-model:
   M3 tolerates large dense working sets (recovery rarely needed below ~250k); a weak
   model needs aggressive compression early. Model choice ⇔ compression aggressiveness.

Matrix complete (coding + low-recency, on M3). Caveats unchanged: n is small, judging
is exact-substring, calibration was n=1/cell — indicative, not statistically tight.

### Acceptance benchmark (post-build): deep-swe

The evals above verify the recall *mechanism*. The end-to-end question — *does the memory
harness outperform a simple one?* — needs a task-success metric, and **deep-swe**
(`github.com/datacurve-ai/deep-swe`) fits: **113 long-horizon SWE tasks** (TS/Go/Python/
JS/Rust), **test-based pass/fail**, run via the **Pier** runner (`--agent`/`--model`).
Plan: plug pi-py in as a Harbor/Pier agent and run **baseline pi-py vs memory-augmented
pi-py, same model (M3), same tasks → pass@1 delta.** Long-horizon = the realistic
context accumulation our micro-evals lacked; pass/fail = objective (no judge). This is
the **acceptance test run after the system is built** (113 long tasks × 2 harnesses × M3
= real compute), and the guard against the harness *degrading* vs a simple baseline.

**Estimate first with a micro run.** A full 113-task pass is slow/expensive; Pier takes a
deterministic subset — `pier run -p deep-swe/tasks --n-tasks 10 --sample-seed 0` (or a
single `deep-swe/tasks/<task-id>`). So the loop is: **micro run (≈10 tasks, fixed seed)
baseline vs memory-augmented for a fast pass@1 estimate → full 113 only once the micro
delta looks right.**

## 10. Prior work — this design is a recombination, not a new invention

Two layers in the literature; keep them distinct:

- **Serving/KV layer** (token-level, inside the model server — *we do not control
  this through the API*): StreamingLLM, H2O, cache-aware eviction. They validate the
  *shape* but are not things pi-py implements.
- **Harness/application layer** (what tokens we put in the prompt — *our layer*):
  MemGPT, LLMLingua, Generative Agents, RECOMP, compaction.

### Confirms the design

- **Lost in the Middle** (Liu et al., TACL 2024). Performance is highest when
  relevant info is at the **beginning or end**, degrades in the middle, even for
  long-context models. Empirical basis for the fidelity **U** (§7).
- **Context Rot** (Chroma, Jul 2025). Degradation begins **well before the window
  fills** (a 200k model degrades at ~50k); *how* info is presented matters more than
  whether it's present. → keep context tight even when it fits; the break-even gate
  must weight **quality**, not just cost. Public replication toolkit.
- **Generative Agents** (Park et al., UIST 2023). The per-block meter already exists
  as a formula: `score = α_recency·recency + α_importance·importance +
  α_relevance·relevance`; recency = exp-decay since last access, relevance = embedding
  cosine to the query, min-max normalized to [0,1]. **Use as the literal starting
  scorer for §7.**
- **LLMLingua / LongLLMLingua** (Microsoft, EMNLP'23 / ACL'24). Graduated compression
  via a small cheap model; up to 20x with minimal loss; LongLLMLingua is
  **question-aware** and targets middle-loss. Concrete tool for the light/heavy
  fidelity tiers; "question-aware" = the user prompt is the relevance query (§7).
- **MemGPT** (Packer et al., 2023). OS-paging: main context (RAM) vs recall/archival
  (disk). This *is* our reversible-ref store. **Caveat:** MemGPT pages in/out via
  *model* function calls — the model-driven path Tier-1 (§9) found unreliable in real
  flow. We keep the paging architecture but drive recovery programmatically (next).
- **Oracle Agent Memory** (workshop, §11). The decisive rule: **retrieval is
  programmatic, not agent-triggered** — "critical memory behaviour must not depend on
  the model remembering to call a tool." The harness retrieves and injects relevant
  memory *every turn*; only optional extras (web search, expanding an *already-surfaced*
  reference) are agent-triggered. Tier-1's 0/6 `expand` rate is this rule proven: the
  agent-triggered path fails, so v1's recovery is **programmatic/automatic**, exactly as
  Oracle prescribes. Oracle uses *semantic* (embedding) retrieval; v1 uses lexical — the
  one place we diverge, and the locus of the open paraphrase-tail question.

### Changes the design

- **StreamingLLM attention sinks.** The first tokens are *mechanical* attention sinks
  the model needs regardless of meaning — a KV-layer artifact, *not* evidence that
  our semantic anchor should be "first N tokens." Pin the anchor because the scorer
  says it's relevant, not because it's at the front.
- **Cache-aware eviction** ("Not All Tokens Are Worth Caching", KVFlow, 2025–26).
  Naive LRU "discards KV shortly before reuse"; isolation costs 8–38.9% TTFT. The
  cache-cost term is a *miss-timing* penalty, which is why §7's gate is per-location
  and biased toward editing near the tail.

### Net

The pieces are each independently validated — the U-shape (Lost in the Middle),
reversible paging (MemGPT), programmatic retrieval (Oracle), graduated compression
(LLMLingua). v1 is their **recombination**, not a new invention: **position-based
lossless compression** (head/tail protected, middle → keyword cue + ref) with
**programmatic/automatic lexical recovery** — cache-stable by construction, embeddings-
free, and Oracle-shaped (programmatic, not model-driven). The genuinely unproven parts —
what the *remaining* evals must earn — are narrow: (1) the **paraphrase tail** (how often
recovery needs share no terms, which would force local embeddings — Oracle's semantic
choice); and (2) **end-to-end non-inferiority with auto-recovery wired in** (every eval
so far tested either the static working set or the abandoned model-driven path). Those
two, not the architecture, are the risk.

References: Lost in the Middle `arxiv.org/abs/2307.03172` · Context Rot
`research.trychroma.com/context-rot` · Generative Agents `arxiv.org/abs/2304.03442`
· MemGPT `arxiv.org/abs/2310.08560` · StreamingLLM `arxiv.org/abs/2309.17453` ·
LLMLingua `github.com/microsoft/LLMLingua`.

## 11. Reference

The typed-memory / context-engineering vocabulary here draws on the Oracle Agent
Memory workshop notes at `/Users/marcusswift/AI_Learning/agent-memory-by-oracle`.
Adopt its core discipline — *memory is not context; context is a working set
selected from memory; harness owns state integrity, model owns local strategy;
summaries are reversible references* — but **do not** copy its unconditional
per-turn context rebuild: it was costed for a latency-free research assistant, and
it fights both prompt caching and (for any voice/interactive use) latency.
