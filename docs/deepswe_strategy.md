# Why Frontier Coding-Agent Failure Is an Engineering Problem, Not a Model Problem

## Thesis

Frontier coding-agent benchmarks expose a class of failures that scale poorly with model
size. The failures are not about reasoning power — they are about *coverage*,
*grounding*, *precision*, *state*, and *verification*. Each of these is an engineering
problem with engineering solutions: deterministic or DAG-based scaffolding, not
bigger models.

On the DeepSWE benchmark, top frontier models cluster between 12% and 70% pass rate,
and the failure distribution across models is *invariant in model size* above a
threshold. The pattern that distinguishes a 70%-pass-rate model from a 12%-pass-rate
model is not raw reasoning. It is process discipline. Process discipline is a property
of the scaffold, not the model.

This document states the five failure modes, the three-prong solution, and the
validation path. It is a strategy document. It does not reference any particular
implementation.

## The Five Failure Modes

### 1. Spec decomposition

A task spec describes a feature in prose. The model must extract every behavior the
spec implies. Dense specs pack 30+ behaviors into 15 lines with no examples, no API
surface, no error strings.

The dominant failure is *missed requirement*: the model implements the obvious
branch and forgets to mirror. The "sync vs async" pattern is canonical — when a spec
says "support both sync and async," the model implements one and misses the other.
The error is not in *how* the code is written; it is in *which* code is written.

This is coverage, not comprehension. The model knows how to write the code; it does
not know how to enumerate the requirements.

### 2. Codebase grounding

The model must place the new feature in an unfamiliar codebase. The repository has
its own module layout, type system, idioms, and patterns. The model has to find the
right place to plug in.

The failure is the model building most of the right structure but missing a path. A
worked example: a task asked for cancellation support in a JS engine. The model
built cancellation handles, parent/child cancellation, metadata on queued jobs. It
missed that `Promise.then()` callbacks are also queued jobs, and those callbacks
were still being enqueued without the cancellation handle. The result: after
cancelling, callbacks created inside a cancelled script could still run.

The model has the *capability* to find the right place. It does not have the
*search* to find it. The failure is grounded in the model not knowing what to look
for.

### 3. Precision

Each behavior in the spec must be exactly right. The verifier does not give partial
credit. Missing 1 of 30 behaviors is the same score as missing 30 of 30.

A representative task: 37 f2p behaviors and 451 p2p tests that must stay green. The
model has to satisfy 488 tests while breaking none. The difficulty is not writing
the code; it is *enumerating all the contracts*.

The direction of precedence is in the spec ("subcommand config key takes precedence
over parent"), not in the obvious reading. The format of quoted strings is in the
spec ("values in double quotes preserve spaces"), not in the obvious behavior. The
model has to read the spec, enumerate the contracts, and satisfy each one.

### 4. Long-horizon state management

Frontier coding-agent runs last 88-134 steps on average. The model has to remember
what it discovered, what it decided, and what it has not yet done, across 100+ tool
calls that fill the context window.

The failure mode is gradual context drift: early decisions are forgotten, in-progress
work is abandoned, the model re-derives what it already knew. The work is not lost —
it is in the context — but the model treats later context as authoritative and
contradicts earlier decisions without realizing.

State management is a property of the agent loop, not the model. A model that
re-reads a structured scratchpad on every turn does not have this problem. A model
that relies on context to remember does.

### 5. Failure to verify

The strongest models self-test unprompted, on 80%+ of runs. The weakest models skip
verification, on 18%+ of runs. The behavior is bimodal: a model either tests its own
work or it does not.

The "test" here is not running existing tests — that is a low bar. The test is
*writing new tests* in the repository's own framework, exercising the new behavior
the model just implemented. This is what separates the strongest models from the
rest.

Verification is a *behavior*, not a *capability*. A model that has scaffolding which
forces verification before commit will verify. A model that has scaffolding which
does not force it will not. The same model behaves differently under different
scaffolding.

## Why This Is Engineering, Not Model Power

The five failure modes are not correlated with model size above a threshold. The
70%-pass-rate model and the 12%-pass-rate model have comparable raw reasoning. The
difference is in their *process*:

- The 70% model decomposes the spec more thoroughly.
- The 70% model searches the codebase more efficiently.
- The 70% model writes more property tests for itself.
- The 70% model self-tests before committing.

Each of these is a *behavior the scaffold can substitute for*. If the scaffold
decomposes the spec, the model does not need to. If the scaffold points the model at
the right code, the model does not need to search. If the scaffold forces property
tests, the model does not need to self-motivate.

A larger model is more *capable* in the abstract, but capability is not the
bottleneck. The bottleneck is *behavior under long-horizon work*. A 35B model with
the right scaffold is more effective than a 70B model without it, because the
scaffold closes the process gap.

This is the central claim: the failure is engineering. The solution is engineering.

## The Three-Prong Solution

The five failure modes reduce to three orthogonal solution prongs. Each prong is
largely deterministic or DAG-based, and the active model is not the primary actor in
any of them.

### Prong 1: Spec decomposition as a structured pipeline

The spec is the source of truth for what success means. The solution is a
*structured pipeline* that extracts atomic, testable behaviors from prose.

The pipeline has six nodes:

1. **Tokenize and split into sentences.** Deterministic.
2. **Classify each sentence** as behavior, constraint, non-behavioral, or meta. A
   small model call, narrow in scope, verifiable by content.
3. **Normalize to atomic imperatives.** Each behavior becomes a structured tuple:
   actor, action, object, condition. The model parses, the format constrains, the
   wrong parses are detectable.
4. **Deduplicate and detect conflicts.** Two sentences that state the same
   requirement in different words collapse to one. Two requirements that contradict
   are surfaced.
5. **Anchor to the f2p test list.** Each requirement tries to match an f2p test by
   name, keyword, and proximity. Unanchored requirements and orphan f2p tests are
   surfaced for human review.
6. **Flag ambiguities.** A requirement with multiple plausible parses is flagged for
   clarification.

The f2p test list is the *validation harness* for the pipeline, not the oracle. The
pipeline produces requirements; the f2p says "you got 28/30, here are the 2 you
missed, here are the 2 f2p tests that don't match any requirement." The pipeline's
accuracy is measurable: how many requirements are anchored, how many f2p tests are
covered, how many ambiguities are flagged.

**The critical discipline**: the model under test is not the model in the pipeline.
The pipeline is a separate, controlled process. A 7B doing the spec extraction might
miss 5 of 30 requirements — the pipeline surfaces that as "5 unanchored
requirements" — and the human reviews. The 7B's failure mode is the pipeline's
diagnostic, not the pipeline's silent acceptance.

The f2p-as-cheat-sheet is a real temptation. If the pipeline sees the f2p test list
before producing the checklist, the checklist becomes a derivative of the test list
rather than the prompt. The pipeline must produce the checklist from the prompt
*without* looking at the f2p, then validate the checklist against the f2p. The
f2p is the held-out test set, not a training label.

The validation round-trip:

- *Forward*: extract checklist from prompt. Compare against f2p. Each item in the
  checklist either maps to an f2p test or is marked "unanchored." Each f2p test
  either has a corresponding checklist item or is marked "orphan."
- *Coverage score*: |anchored requirements| / |total requirements|. A high score
  means the pipeline captured the spec.
- *Inverse coverage score*: |covered f2p tests| / |total f2p tests|. A high score
  means the pipeline captured what the verifier checks.

These two scores together tell you whether the pipeline is producing the right
checklist. They do not require the pipeline to use the f2p as a hint. They are the
pipeline's held-out test.

### Prong 2: Repo graph and pattern cloud

The codebase is the surface the new feature plugs into. The solution is two
complementary structures.

**The repo graph** is a deterministic map of the codebase: files, symbols (classes,
functions, methods, exported constants), imports, and test cases. The edges include
containment (file contains symbol), dependency (file imports file), call (symbol
references symbol), and the load-bearing edge: *test case references symbol*. This
last edge is gold — it gives the reverse index "which tests cover this file"
deterministically, from the AST.

The graph is built once per task, cached, and queried through a small tool surface:
surface (public API of a module), cover (tests that cover a file), blast_radius
(files transitively imported by those tests), callers, recent_touches, imports,
imported_by. The model does not walk the graph — it asks questions. The graph is
the bones; the model fills in the flesh.

**The pattern cloud** is a pre-curated database of "how features are implemented."
Each pattern is a structured record: name, problem, structural signature (AST
shape), test signature (property test shape), repos where it appears, and a
co-occurrence graph (which patterns tend to appear with which). Built once from a
corpus of public repositories, with curation by a human reviewer.

The pattern cloud is not a runtime model call. It is a learned corpus. The runtime
retrieval is fast: AST match filters candidates, text similarity ranks them, the
model gets the top 2-3 patterns with structural templates and test signatures.

The two structures compose: the repo graph tells the model *where* to plug in, the
pattern cloud tells it *how*. The pattern cloud's test signatures are the property
tests that drive the convergence loop.

### Prong 3: Memory management

Long-horizon work requires state that survives context resets. The solution is a
persistent scratchpad with a fixed structure.

The scratchpad has five sections, each with a canonical key:

- **Spec** — the full atomic requirement list, read-only, never compressed.
- **Plan** — the current implementation plan, structured by spec item.
- **Decisions** — append-only log, never re-decided.
- **Open questions** — current ambiguities, with resolution status.
- **Coverage** — the per-spec-item state: pending, in-progress, done, blocked.

The read path is *full re-injection* on every turn, not query-relevant subset. The
model wakes up after a context reset with the full spec and the current state, not
a summary. The cost is token overhead; the benefit is the model never has to
remember what it wrote in a previous turn.

The write path is structured: each section has a canonical key, the model writes to
specific keys, the harness knows where to read from. Free-form prose is rejected at
the schema layer.

The compression path is *selective*: the spec section is non-compressible, the plan
and coverage can be compressed, the decision log can be summarized, the open
questions can be dropped. The harness knows the original spec length and refuses
to inject a compressed version with fewer items.

The decision log is immutable. When the model is about to make a decision that
contradicts a logged one, the harness surfaces the conflict before the model
commits to the new path. This catches the "I forgot I decided X at step 12"
failure mode.

## How the Three Prongs Compose

The three prongs are not independent. They compose into a single convergence loop:

1. The **spec decomposition** produces the list of behaviors the model must satisfy.
2. The **repo graph** identifies where each behavior plugs in.
3. The **pattern cloud** provides the implementation template and the property test
   for each behavior.
4. The **memory layer** holds the state across iterations.
5. The **test feedback** is the convergence signal: write code → run property test →
   see specific counterexample → fix → re-run.

The convergence loop is the load-bearing insight. A model that writes bad code can
iterate to correct code, *if the test failure is diagnostic*. A property test gives
a counterexample: "source 1 = X, source 2 = Y, expected X, got Y." That is a one-shot
fix. A snapshot test gives only "expected: ..., got: ..." — a weaker signal. The
pattern cloud's test signatures are property tests by construction.

A 35B model with property tests converges in 3-5 iterations. A 35B model without
property tests does not converge regardless of iterations, because each iteration is
a coin flip. The size of the model is not the lever; the quality of the test is.

This is why the failure is engineering. The model is the implementer; the scaffold is
the spec, the map, the patterns, the tests, the memory. A small model with a rich
scaffold outperforms a large model with a poor one, on the failure modes that
benchmark the gap.

## The Validation Path

The scaffold is not self-validating. The validation is a measurable property: how
often does the model satisfy the f2p tests it was given?

The validation experiment is straightforward:

1. Pick 5-10 representative tasks across the corpus (one short-spec, one long-spec,
   one multi-language, one codebase-grounding-heavy, one precision-heavy).
2. Run the model with no scaffold against the subset. Record the pass rate and the
   failure mode distribution.
3. Run the model with each scaffold component added incrementally. Record the
   deltas.

The success criterion is not "the model passes all tasks." It is: "the model passes
more tasks with the scaffold than without, and the failure mode distribution shifts
from 'missed requirement' to something more specific." A scaffold that does not
move the score but does surface *where* the model is failing has diagnostic value
even without performance value.

The corpus is itself the held-out test set. The f2p tests are not used to train
the spec-decomposition pipeline; they are used to *validate* the pipeline's output.
This is a legitimate round-trip: the pipeline produces requirements, the f2p tests
are the label, the alignment between them is the score.

## The Risks

The strategy has three known risks.

**Risk 1: pattern identification is lossy.** The pattern cloud's runtime retrieval
uses AST-shape matching and text similarity. Both are lossy. A pattern that is
structurally similar to the new feature but semantically different will be
suggested. The fallback is the model's judgment — and the model might trust the
pattern too much. Mitigation: the pattern cloud's confidence score is exposed; the
model sees "this is a low-confidence match" and treats it as a starting point, not
a prescription.

**Risk 2: the test-feedback loop is not free.** Each iteration is a tool call, a
test run, a model inference. The cost compounds. For a 113-task benchmark, the
total iteration budget is finite. If the model's first attempt at 30 behaviors is
bad, the iteration budget is exhausted before convergence. Mitigation: the
spec-decomposition pipeline produces a *plan* that the model follows. The first
attempt is not a free exploration; it is the plan's structured implementation.

**Risk 3: the memory layer can fail silently.** A bad compression that drops spec
items does not produce an error. The model wakes up with a 15-item checklist when
the original was 30, and the missing 15 are gone without trace. Mitigation: the
harness knows the original spec length and refuses compression that violates it.
The check is structural, not advisory.

## The Claim, Restated

Frontier coding-agent benchmarks measure the wrong thing if the goal is to
differentiate model capability. They measure the *combined capability of model +
scaffold*, and the scaffold is doing most of the work. The model's contribution is
bounded by its ability to follow the scaffold's plan, write code from a template,
and iterate on test feedback. None of these require frontier model size.

The engineering work is to build the scaffold well. The model work is to find a
small model that is reliable at the loop, not a large model that is capable in the
abstract. The recipe is: a 35B local model, a four-skill scaffold (spec
decomposition, repo graph, pattern cloud, memory), property tests from the pattern
cloud, and a convergence loop that closes on the test signal.

The failure is engineering. The solution is engineering.
