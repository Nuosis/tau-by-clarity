# Context & Memory Management — Design Thinking

Status: design note / not yet implemented
Scope: how `pi-py` assembles per-call context today, and a proposed signal-routed
strategy for managing it more intelligently.

This document captures a line of reasoning, not a finished spec. It exists so the
"why" survives — the mechanics below constrain every memory decision we make, and
they are easy to forget once the code looks like it "just works."

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

## 6. Route by signal: the recency-relevance meter

Recency-relevance is **measurable from data we already have** — the model tells us
what was relevant every turn; we just read the signal.

### Signals, cheapest first

**Tier 1 — free, no extra LLM calls:**

- **Tool-read reach.** When the model reads a file (or expands an old block), how
  far back does it reach? Reaching into early turns / old artifacts = low
  recency-relevance. Staying on recently-touched files = high. The model literally
  pointing at where the needle is — the best free signal.
- **Back-reference distance.** Per turn, distance (turns/tokens) from now to the
  oldest thing the turn depended on (re-touched file paths, reappearing
  entities/IDs, tool-call targets). Rising oldest-dependency distance = dispersing
  relevance.
- **Session shape.** Length, turn count, branch/revisit activity. Crude but free.
- **Compaction-recall miss (the killer signal for pi-py).** If the model reaches
  for something already compacted away, recency mode just dropped what it needed.
  A direct, ground-truth meter reading.

**Tier 2 — costs something; use only to *confirm* a flip:**

- Embedding similarity between the current turn and old turns: if the
  highest-similarity content is consistently old, relevance is dispersed. Cheap
  embedding pass, not a full call.
- A tiny LLM judge ("did this turn depend on anything older than the last N
  turns?"). Accurate but costs a call — run it to confirm a flip the cheap signals
  already flagged, not every turn.

### The sharp edge: a latched switch with hysteresis, not a continuous knob

Two failure modes the naive version hits:

1. **Flapping.** A continuous meter that flips mode turn-to-turn is
   self-defeating: *the cost of flipping is the cache invalidation we're trying to
   avoid.* A jittery meter causes the harm it measures. Use a **latched state with
   hysteresis**: high threshold to *enter* lean mode, a lower threshold to *leave*
   it, plus a minimum dwell time. Flip rarely and deliberately.
2. **The truest signal reads true only after the damage.** "Reached for compacted
   content" tells us recency was wrong *after* we already dropped the needle. So
   the meter can't be purely reactive: let the **early** signals (read-reach,
   back-ref distance) trigger the flip *before* compaction fires — switch to
   reversible-summary / curated mode when the session starts smelling like
   archaeology, so we stop doing lossy drops before we lose the needle. The
   compaction-recall-miss count then becomes the **eval** signal ("did the meter
   flip in time?"), not the primary trigger.

### Shape

```
start in CACHE mode (append + threshold compaction + JIT reads)   ← right default
  │
  │  flip to LEAN only after early signals (read-reach / back-ref
  │  distance) cross a HIGH threshold for K turns
  ▼
LEAN mode (stable prefix + curated tail + reversible summaries;
           stop lossy compaction)
  │
  │  leave only after sustained low signal + minimum dwell time
  ▼
back to CACHE mode
```

One-directional defaulting (cache is home), latched transitions, hysteresis on
both edges. The meter's output is a **mode decision**, not a continuously dialed
knob — the cost asymmetry of switching forces us to commit and stay until the
evidence pays for one more switch.

## 7. Refinement: a relevance-shaped fidelity gradient

The latched two-mode switch (§6) is the conservative version. The fuller design
drives one continuous knob — a per-region *fidelity* level — from one measurement, a
relevance profile over the transcript. No arbitrary size cutoff and no binary
keep/trim; the regime (§6) only decides whether that gradient is applied
continuously or as rare discrete trims.

### No arbitrary cutoff — a break-even gate, assessed continuously

Drop the idea of a fixed 128k threshold as a *rule*. **Assessing** relevance needs
no threshold — the cheap signals are free and the middle's decay is self-evident
every turn. But **assessing is not acting.** Acting (trim/compress) mutates the
prefix → invalidates the cache → reprocessing. So the gate is a break-even:

> Act when (tokens reclaimed × cost-per-turn-to-carry × turns-remaining)
> > (cache-reprocessing cost of editing at that point).

This is principled, not arbitrary, and it *reproduces* a soft threshold on its own:
early in a session there's nothing worth reclaiming and the cache cost dominates, so
the test fails and nothing happens. Two consequences:

- **Decouple the loops.** Assessment runs continuously and free; action is gated by
  cost. Keep a cheap constant floor (~128k) only as a "don't bother computing
  break-even below this" shortcut — a floor, not a rule.
- **The gate is per-location.** Trimming the middle reprocesses everything after it
  (expensive); trimming near the tail reprocesses little (cheap). Break-even is not
  one number — it's cheaper to act the closer to the tail the cut is.

### Relevance is a profile, and the score sets *fidelity*, not survival

"Scan from the front, keep the relevant prefix, compact the rest" assumes relevance
is a contiguous block at the head. It isn't. And keep/trim is too blunt — a **binary**
decision (verbatim | gone) is what produces a barbell with a hole in the middle.

Generalize to a **graduated, reversible fidelity gradient**. The relevance score
picks a *fidelity level*, not survival:

- frontier → **verbatim**
- aging near-mid → **light compression** (keep structure, drop verbosity)
- far-mid → **heavy summary**
- deep / cold → **ref code only** (ID + one line)

Everything stays referenceable; nothing is lost; a buried region that becomes
relevant is one `expand` away. The gap disappears — a continuous strand thinning
toward the back ("long hair") instead of a barbell.

**Fidelity is a U, not a monotonic decay.** Highest at both the **anchor** (original
task/spec/constraints/decisions — small, almost always relevant) and the
**frontier** (recent turns, where exact code/tool-output/user-words live and where
lossy summary does the most damage, *and* which is cheap to keep because it's small).
Compression thins toward the **cold middle** — the back-slope behind the frontier —
never the live edge. So: keep the frontier verbatim; let regions compress only as
they *age into* the middle.

### Two shapes = the two regimes (same meter chooses)

The fidelity gradient and the barbell are not competitors. They are the two regimes
from §6, because **continuous re-compression is cache-hostile**: re-representing each
turn at lower fidelity as it ages mutates the prefix *every turn*, so the cache never
warms.

- **Cache mode (fast/cheap turns, coding):** verbatim append + *occasional discrete*
  compaction. The gradient's smooth savings aren't worth thrashing the cache when
  turns are frequent and cheap. → **barbell, trimmed rarely.**
- **Lean mode (slow/expensive turns, research/planning):** few calls, cache barely
  matters. → **continuous fidelity gradient ("long hair").**

Long hair *is* the lean-mode representation; barbell-trimmed-rarely *is* the
cache-mode representation. The §6/§7 recency meter is what picks between them.

### Measuring where relevance drops off (per-block score, cheapest first)

1. **Last-reference recency (free).** When was this block last touched — read,
   re-mentioned, depended on? Not referenced in N turns and not in the anchor set =
   decaying. This *is* "relevance dropping off," measured per block.
2. **Size-to-reference ratio (free).** Big and never referenced again = prime
   compress target. Verbose tool outputs score worst here — correctly.
3. **Embedding similarity to the active frontier (cheap).** Each old block vs.
   (last turn or two + current memory query). Low + old = compress harder. Catches
   the buried-region-becomes-relevant case.

Below the floor, compute nothing. Above it, score every region, set each region's
fidelity from its score, and act only where break-even is met — each demoted region
leaves a retrievable ref behind.

### The trigger is the user prompt

Don't sample continuously. **Re-evaluate the relevance profile at each user turn** —
the only moment relevance can discontinuously jump. A pivot is detectable right
there: compare the new prompt's similarity to the recent frontier vs. to older
regions. New prompt closer to something 40 turns back than to the last 5 turns →
recency collapsed, *and* you've identified which buried region to resurrect. One
measurement yields both "trim aggressively" and "here's what to keep." This is why
"let a long coding session ride, then hard-prune the instant the user pivots" works:
the pivot *is* the prompt, and the prompt *is* the new relevance query.

### Cache trap to bake in now

"Memory recall + writes every turn" quietly fights the cache **unless the recall
block lives at the tail.** Inject fresh recall after the system prompt but before
the transcript and you invalidate the whole transcript cache every turn — full
prefill in the exact zone you wanted free. Rule: **stable prefix = system prompt +
transcript; churning tail = memory recall + new turn.** Recall goes at the bottom.

## 8. Where this plugs into pi-py

- **`transform_context()`** (`core/agent_session.py`, currently the identity stub)
  is the seam. Context assembly becomes a **pluggable strategy** selected here.
  - **Default (cache mode):** current behavior + *occasional discrete* compaction —
    a stable verbatim prefix, trimmed rarely (barbell). Keep it; resist making it
    cleverer.
  - **Opt-in (lean mode):** the continuous fidelity gradient (§7) — per-region
    verbatim → light → heavy → ref, all reversible (keep the original retrievable, à
    la the Oracle pattern — *not* pi's current lossy compaction), plus tool-output
    offloading. Cache-hostile by design, which is fine here: few, expensive turns.
- **Reversible summaries** are a concrete upgrade over today's lossy compaction
  even before routing exists: store summary + retrievable original + a
  `summary_id`, and let the model expand on demand.

## 9. What to build first — measure before you route

The cheapest end-to-end slice that proves the idea, changing **no** behavior:

1. Instrument **read-reach** and the **compaction-recall-miss counter**. Log them.
2. Run across real sessions and just *look* at the two numbers.
3. Verify coding sessions cluster high-recency and planning/research sessions
   cluster dispersed — i.e. the meter actually separates the two regimes.

Only if the signal separates cleanly do we wire it to the latched flip off
`transform_context`. The meter earns the right to drive the switch after the logs
prove it discriminates — observe the signal is real before letting it change
behavior.

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

**Honest data gap:** real signal needs sessions with genuine middles. Next data: point
the same harness at (a) richer local coding sessions for the high-recency regime and
(b) clarify Langfuse traces for the conversational/low-recency regime. Tiers 1–2
(cheap-model A/B with a pairwise non-inferiority judge, then production model on the
contested subset) come after Tier-0 separates the regimes on real data. Token counts
are char/4 approximations — fine for deltas, not billing.

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
  (disk), model pages content in/out via function calls. This *is* reversible-ref +
  model-triggered `expand`.

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

Barbell / gradient / anchor / scorer are each independently validated. The
**less-charted** part — and what the evals (§9, §11) must earn — is the **regime
router**: a recency meter choosing barbell vs gradient *on cache economics*. No prior
work does exactly that adaptive, cache-aware switch. That is the contribution and the
risk.

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
