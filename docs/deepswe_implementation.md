# From Strategy to Implementation: The DeepSWE Capability Roadmap

This document is the companion to `docs/deepswe_strategy.md`. The strategy doc
states the diagnosis (five failure modes) and the prescription (three solution
prongs plus the test-feedback convergence loop). This doc maps the prescription
to the implementation state and identifies what to strengthen or build.

The implementation reviewed is the Devin agent and the DevFlow engine. Two
pieces in particular that the prior review undercounted: a spec subagent
(consumes spec documents, extracts scopes for the scope-to-idea workflow) and
convergence infrastructure at two levels — inner loops with success criteria
inside the agentic nodes, and an outer loop via the `/goal` command that
persists across turns to meet a stated goal.

## What Is Already in Place

### Spec decomposition substrate: the spec subagent

The spec subagent is the load-bearing piece of Prong 1. It consumes and
generates spec documents and extracts requirements into *scopes* — discrete
requirement units that the DevFlow scope-to-idea workflow can then process.
This is the substrate the strategy doc called for as "a structured pipeline
that extracts atomic, testable behaviors from prose." The subagent provides
the artifact boundary; what remains is the six-node pipeline shape and the
held-out f2p-anchor validation.

The DevFlow primitives tie-in is intentional and correct: DevFlow is the
implementation harness through which the model writes code, runs tests, and
commits. The Planner's job is to convert scopes into DevFlow primitives
(idea, story, error_story, quick_action). The spec subagent's job is to
convert spec text into scopes. These are two distinct decomposition stages,
each with its own artifact contract.

### Repo graph: the repo map module

The repo map module is the load-bearing piece of Prong 2a. It produces a
structured code map with: established patterns (ten kinds: file_pattern,
route_pattern, service_pattern, model_pattern, test_pattern, ui_pattern,
deploy_pattern, configuration_pattern, data_pattern, workflow_pattern),
workflows, libraries/frameworks, domain models, integration points, and
API/auth/storage/test/deployment surfaces. AST-based import scanning across
Python, TypeScript, and JavaScript, with project-root detection,
internal-vs-external classification, and standard-library filtering. The
test surfaces are extracted by path-pattern and import-pattern, producing a
path-import reverse index.

### Pattern library: project-local

The repo map extracts patterns from the target project. This is a useful
subset of the cross-project pattern cloud the strategy calls for, but it is
not the cloud. It answers "this project uses pattern X" but not "the
canonical implementation of pattern X looks like this in 47 other repos."
The local pattern library is a starting point; the cross-project
curation is the missing piece.

### Memory: Tau's memory module + Devin's wiring

Tau's `pi_coding_agent.core.memory` module is intact and used: `MemoryStore`
(SQLite, project-local, hybrid lexical+semantic), `build_recall_block`
(token-budgeted, tail-injected), `Curator` (grounded evidence, structural
guards), `WorkingContextConfig` (per-model floor/ceiling from NIAH
calibration), and active compression as a separate concern. Devin wires
into this via `.tau/memory/`. The infrastructure primitives are in place.

### Subagent pattern with typed artifacts

Devin, Explorer, Planner, Queuer, Runner, Doctor, Provisioner — each a
full subagent with its own `OBJECTIVES.md`, `SYSTEM.md`, settings,
extensions. The Explorer → Planner → Queuer handoff uses typed artifacts
(`ExplorerOutput` → `PlannerOutput` → queued primitive). This is the
"typed input/output artifacts" discipline the strategy calls for, applied
to the DevFlow domain. The spec subagent will plug into this discipline
once its artifact contract is defined.

### Inner loops with success criteria in agentic nodes

Many of the agentic nodes in the DevFlow DAGs have an inner loop with
success criteria: the node retries until the success criteria are met,
rather than firing once and reporting. This is the per-node convergence
discipline: each node has its own definition of "done," and the node
exits only when that definition is satisfied. The strategy doc's
convergence-loop insight is partially satisfied at this level: the
model iterates within a node on the node's success criteria.

### Outer loop via the `/goal` command

Devin exposes a `/goal` command that persists across turns to meet a
stated goal. The user (or the parent process) states a goal, and Devin
continues iterating until the goal is met or the budget is exhausted.
This is the strategy doc's outer convergence loop, expressed at the
session level. The interaction with the inner loops is the open
question: the inner loops run within a node, the outer loop runs
across nodes and turns. Both need to converge on the same signal.

### Implementation harness: DevFlow DAGs

`implementation/dag.py` is 8,673 lines, the largest module. It owns the
story-execution flow: claim story, set up local environment, run
implementation, run green-gate, persist results. `process/dag.py` runs
the process DAG. `recovery/dag.py` runs the post-queue-failure
recovery flow with explicit `RecoveryDiagnosisArtifact`,
`RemediationPlanArtifact`, `ReenqueueArtifact`, and
`SystemicPatternArtifact` types — a real "the model notices a failure,
re-plans, re-executes" loop, scoped to queue failures. The DAG framework
(datalumina-genai) gives the harness Node / Workflow / AgentNode /
RouterNode primitives used pervasively.

## What Needs Strengthening

These are real pieces that are not yet at strategy quality.

### The spec subagent needs the six-node pipeline shape and the held-out f2p-anchor

The spec subagent consumes spec documents and produces scopes. To be at
strategy quality, the subagent's pipeline needs the six-node shape from
the strategy doc:

1. Tokenize and split into sentences.
2. Classify each sentence as behavior, constraint, non-behavioral, or meta.
3. Normalize to atomic imperatives (actor, action, object, condition).
4. Deduplicate and detect conflicts.
5. Anchor to a held-out f2p test list (where it exists) — each
   requirement maps to an f2p test, or is marked unanchored.
6. Flag ambiguities.

The f2p-anchor step is the discipline that prevents "the model
hallucinates the spec, the scaffold trusts it." Without the held-out
anchor, the subagent's output is the model's interpretation with no
external validation. The f2p-as-validation-harness is the round-trip
that catches hallucinations: extract scopes without looking at the f2p,
then validate that each scope corresponds to an f2p test (or to a code
location, for behaviors the f2p doesn't directly cover).

The validation is the scaffold's self-check. The subagent should expose
its anchoring coverage as a first-class metric: how many scopes
anchored, how many unanchored, how many orphan f2p tests.

### The repo map needs a symbol-level reverse index

The current test-to-code reverse index is path-import based. A test file
is identified by path pattern (`test_`, `_test.`, `.spec.`, `tests/`) and
by imports of test frameworks. The index says "this test file exists"
and "this source file has tests nearby." It does not say "this test
function exercises this source function."

The symbol-level reverse index is the load-bearing primitive for
targeted test selection. Given "I changed `command/command.ts`," the
scaffold needs to know which tests reference which symbols in that
file, not just which test files import the file. The construction is
mechanical — AST call-graph extraction on test code, building a
`test_function → symbol` map — but it is a discrete piece of work
that the current repo map does not do.

The strategy's "targeted test selection" primitive depends on this
index. Without it, the model either runs the full test suite (slow,
often infeasible for repos with 16,000+ tests) or runs a
heuristic-derived subset (noisy, misses regressions).

### The memory infrastructure needs the policy layer verified

The memory infrastructure primitives are in place. The policy layer
that turns "the model can store and recall things" into "the model
has a five-section scratchpad with structural guarantees" has not
been verified in the implementation. The five sections: Spec (full
requirement list, read-only, never compressed), Plan (current
implementation plan, structured by spec item), Decisions (append-only
log, never re-decided), Open Questions (current ambiguities with
resolution status), Coverage (per-spec-item state: pending,
in-progress, done, blocked).

The non-compressible spec guarantee is the load-bearing check. The
harness knows the original spec length and refuses a compressed
version that drops items. Without this guarantee, compression can
silently lose half the spec. The model wakes up with a 15-item
checklist when the original was 30, and the missing 15 are gone
without trace. This is the silent-failure mode flagged in the
strategy doc's risks.

The decision log immutability is the second check. When the model is
about to make a decision that contradicts a logged one, the harness
surfaces the conflict. Without this, the "I forgot I decided X at
step 12" failure mode is uncaught.

These are one-day builds on top of the existing memory module. The
hard part of the memory layer is done; the policy layer is the
remaining work.

### The test gate needs to require property tests, not just existing tests

The green gate runs the existing test suite and gates on pass/fail. The
strategy's fourth prong is different: the model writes *new* property
tests in the project's framework, and the gate runs those. The
property tests assert the right behavior, not just that the code
runs.

The discipline this changes: a green gate that passes only on the
existing test suite is a "completion oracle" gate — the code runs,
the existing tests pass, the gate clears. A green gate that requires
*new* property tests is a "property oracle" gate — the code
implements the right behavior, the property tests pass, the gate
clears. The model that passes the first gate might still implement
the wrong behavior; the model that passes the second gate has
demonstrated the right behavior.

The property-test templating is the cross-project pattern cloud's
contribution. Without the cloud, the property tests are ad-hoc; with
the cloud, they are templated per pattern.

## What Needs to Be Built

These pieces do not exist yet.

### Cross-project pattern cloud

The pattern cloud is a one-time corpus curation project. Each pattern
is a structured record: name, problem, structural signature (AST
shape), test signature (property test shape), repos where it
appears, and a co-occurrence graph (which patterns tend to appear
with which). Built from a corpus of public repositories, with
curation by a human reviewer.

The runtime use: given a new task, the scopes are matched against
the pattern signatures in the cloud. The cloud returns the
top-matching patterns with structural templates in the target
project's language and the property-test template for each. The model
mirrors the template, fills in the specifics, and the property test
becomes the green gate for that behavior.

The cloud is not a runtime model call. It is a learned corpus. The
runtime retrieval is fast: AST match filters candidates, text
similarity ranks them, the model gets the top 2-3 patterns. The
hard work is one-time curation; the runtime is cheap.

### Property-test templating

The cross-project pattern cloud's test signatures become property-test
templates. For a "precedence chain" pattern, the template is
"for any two source layers, the higher-priority one wins, regardless
of which keys are defined in which layer." For a "type coercion at
boundary" pattern, the template is "for any value matching the
boundary type, the coerced value matches the inner type's contract."

The templates are in the project's test framework (pytest, deno test,
go test). The model fills in the template; the harness generates the
test file. The green gate runs the generated test alongside the
existing tests. The model cannot pass the gate without a property
test for each implemented behavior.

### Convergence loop on test feedback, integrated with `/goal`

The inner loops and the `/goal` outer loop are the substrate. The
convergence loop the strategy calls for is a specific signal: the
model iterates on test feedback until the property tests pass. The
inner loops may already cover some of this at the per-node level; the
open question is how the inner loops and the outer loop compose
around the test feedback signal.

The integration work: route test failures from the green gate into
the model's input on the next iteration, with the failing test's
property and the model's current diff. The model iterates; the
property test converges; the green gate clears; the node exits; the
outer loop checks the goal and either iterates to the next node or
declares done.

This is the test-feedback convergence loop. The components are
there; the routing is the missing piece.

### Non-compressible spec compression guarantee

A structural check: the harness records the original spec length (in
scopes) at subagent output time, and refuses to inject a compressed
memory version that has fewer items. The check is a hard refusal,
not an advisory — the model's request to compress the spec fails
with a specific error, and the model must use the full spec.

The check belongs in the memory policy layer, alongside the
structured scratchpad template. It is small but load-bearing
because it is the only thing that prevents the silent-failure mode
where the spec is compressed away.

## Build Order to DeepSWE

The build is sequenced so each step is independently validatable.

**Step 1 — Memory policy layer.** The memory module is in place. The
policy layer adds the five-section scratchpad template, the
non-compressible spec guarantee, the immutable decision log, and the
full re-injection per turn. One-day build. Validated by: a Devin
session that takes a spec, decomposes it, writes a plan, makes
decisions, compresses the working context, and re-injects correctly
on a context reset. Failure modes caught: spec items missing from
re-injection, decisions silently re-decided.

**Step 2 — Spec subagent pipeline + f2p-anchor.** The spec subagent is
being built by the user's colleague. The pipeline shape (six nodes)
and the f2p-anchor validation land on top. The DeepSWE eval is the
ideal validator: extract scopes from each of the 113 tasks' spec
texts without looking at the f2p, then measure anchor coverage
(scopes anchored to f2p / total scopes) and inverse anchor coverage
(f2p covered by scopes / total f2p). The target: anchor coverage
above 80%, inverse coverage above 90%. Failure modes caught:
hallucinated scopes, missed scopes, scope-f2p mismatches.

**Step 3 — Cross-project pattern cloud.** The corpus curation. Start
with the 91 repos DeepSWE draws from. Cluster similar AST shapes
across repos; name the clusters; record the structural and test
signatures. Co-occurrence graph from file-level co-presence. Human
review of each pattern. Validated by: pick a held-out repo, extract
its patterns, compare against the cloud's signatures, measure
agreement.

**Step 4 — Property-test templating from the cloud.** For each pattern
in the cloud, write the property-test template. The template includes
the test shape (property vs. example, fixtures, assertions) and the
per-language implementation (pytest, deno test, go test). Validated
by: pick a DeepSWE task, generate the property tests for each scope,
run them against a reference solution, verify all pass.

**Step 5 — Symbol-level reverse index in the repo map.** AST
call-graph extraction on test code, building the `test_function →
symbol` map. Validated by: pick a DeepSWE task, change one symbol,
verify the index surfaces the right tests as candidates for
targeted run.

**Step 6 — Convergence loop integration with `/goal`.** Route the
green gate's test failures into the model's input on the next
iteration. Wire the inner loops to the outer loop. The
implementation DAG's failure path becomes a feedback channel. The
`/goal` command's iteration becomes the convergence driver.
Validated by: pick a DeepSWE task, run the model with the
convergence loop, count iterations to green, compare against a
non-converging baseline.

**Step 7 — DeepSWE end-to-end run.** All six pieces integrated. Run
the 113-task benchmark on a 35B local model. Compare against the
leaderboard. The success criterion is not "win the leaderboard"; it
is "pass more tasks with the scaffold than without, and the failure
mode distribution shifts from 'missed requirement' to a more
specific class."

## Risks and Unknowns

**The pattern cloud is the largest unknown.** The corpus curation is
a one-time project, but its quality determines the quality of the
property-test templates and the pattern suggestions. A cloud that
misses the "precedence chain" pattern will produce a scaffold that
misses cliffy-style tasks. A cloud that captures it will produce
property tests that converge.

**The convergence loop is unverified at the integration level.** The
inner loops and the `/goal` outer loop are individually present. The
question is whether they compose into the test-feedback convergence
loop. If the inner loop's success criteria don't include
property-test pass, the inner loop converges on the wrong signal.
This is a coordination problem, not a missing-piece problem.

**The DeepSWE eval infrastructure has open integration work.** The
adapter exists (`evals/deep_swe/pi_py_agent.py`), but the headless
tau invocation, model wiring inside the sandbox, and embedding
fallback in the no-Ollama environment are the open issues. These
are not blocking the scaffold work; they are blocking the validation
of the scaffold against the full 113-task benchmark. A
representative-subset run can validate the scaffold before the full
benchmark is operational.

**The strategy doc's "engineering, not model" claim depends on the
scaffold.** If the cross-project pattern cloud is wrong, if the
property-test templates are wrong, if the convergence loop
converges on the wrong signal, the claim fails. The scaffold is the
load-bearing engineering. A 35B with a wrong scaffold is no better
than a 7B with no scaffold.

## The Claim, Restated with the Implementation State

The strategy doc argues that frontier coding-agent failure is
engineering, not model. The implementation state supports the claim
with caveats: the foundation is real and substantial (repo map, DAG
framework, memory infrastructure, subagent pattern, inner loops,
`/goal` outer loop, spec subagent under construction), but the
scaffold is incomplete in the specific pieces the strategy calls
out — cross-project pattern cloud, property-test templating,
convergence loop integration, non-compressible spec guarantee,
f2p-anchor validation.

The next year of work is engineering, not model. The model is the
implementer; the scaffold is the spec, the map, the patterns, the
tests, the memory. The recipe is: a 35B local model, a six-piece
scaffold, the spec subagent's six-node pipeline, and the
convergence loop wired through `/goal`. The build order is sequenced
for independent validation. The failure is engineering. The solution
is engineering.
