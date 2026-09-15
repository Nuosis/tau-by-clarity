# Router completion review

Router-enabled sessions review terminal text candidates at `turn_end` using the
router owner's configured **max** provider, model and reasoning effort. There is
no hard-coded reviewer model. Fixed-model sessions and ordinary tool turns are
unchanged.

## Intention and placeholder (0.58.9)

Before the first working invocation for a user request, configured Max establishes
an `Intention`: outcome, completion evidence and authorization scope. The reviewer
receives compressed context and CCR retrieval during this stage too. Intention is
stored separately from conversation messages in `tau.intention.completed` session
entries and restored from the selected session branch. User steering or queued
follow-ups trigger a revision when that input reaches the working context.
Reviewer-generated continuations do not trigger revisions.

The input editor displays the outcome as **placeholder text only** while its text
buffer is empty. Typing hides it; clearing the buffer reveals it again. It never
enters submission, history, undo, or working-model input through the renderer.
An `intention_changed` event updates the TUI without modifying the editor buffer.

End review treats intention as its completion target. `accept` requires both
`intention_met=true` and `answer_sound=true`; the host rejects inconsistent verdicts.
An unmet target or unsound answer leads to actionable worker continuation. The
reviewer cannot redefine intention in its end-review output. User corrections may
revise it, and cancellation remains available. Persistence does not expand tool
permissions or user authorization. Completion review has a bounded protocol:
decide immediately or make one successful CCR retrieval and then decide.

Verification includes production TUI rendering with a captured terminal transport,
editor typing/empty-Enter/history behavior, correction via steering and follow-up,
session reopen, stable intention across worker continuations, cancellation in both
review stages and invalid-acceptance contracts. Controlled provider tests establish
orchestration, not semantic model quality.

Live configured-Max sandbox evidence is retained at
`/Users/marcusswift/outputs/tau-intention-20260914/session.json`: Max established
intention once, rejected a premature ending, then accepted after native edit/read
tools changed and verified a temporary file. Three Max calls, 4,904 tokens;
four scripted worker responses. This proves live review-driven continuation with
file effects, not autonomous worker planning or broad answer quality.

A second replay at `tau-intention-20260914/answer-quality/session.json` additionally
proposed an incorrect answer after the file work was complete. Max returned
`intention_met=true, answer_sound=false`, requested only an answer correction,
then accepted the corrected report. One intention, three ending reviews, five
scripted worker responses. Both live replays passed. The source checks passed
89 focused tests plus two existing turn-end hook tests.

The reviewer receives the transformed working context (persisted CCR compression,
context extensions and recalled memory), current instructions, worker capability
descriptions and candidate answer. It is a separate native Agent with only
`ccr_retrieve` and `submit_review`. Retrieval uses the existing same-process CCR
store and focused-query search; no full-history expansion or worker mutation
tools are exposed to the reviewer.

An `accept` verdict permits the existing output-finalization/Stop flow. A
`continue` verdict queues actionable requirements for the original worker with
its tools intact. Queued user input takes precedence. Invalid/missing decisions
surface an error rather than silent approval. Cancellation drains the review's
provider producer as well as its consumer. A completion review must call a review
tool on its first generation; after one successful CCR retrieval the next request
is forced to `submit_review`. A provider, schema, retrieval, or tool error ends the
review immediately and is not sent back to the reviewer for another attempt.

Terminal review errors are shown even when the candidate was already rendered,
are written to stderr in print mode, and are persisted after the candidate so a
reopened session cannot make an unapproved candidate look approved.

Session custom entries `tau.turn_review.started/completed/failed` retain selection,
verdict, CCR calls, duration and review response usage/transcript. The input packet
is not duplicated in the completion event. Existing `/stats` working-response
shares do not include auxiliary reviewer calls.

This is a continuation gate, not a streaming presentation gate: an initial draft
can already be visible before review. The existing cosmetic finalizer runs after
semantic acceptance and is not recursively reviewed.

## Verification, 2026-09-14

Controlled-provider functional tests exercise native router orchestration, actual
file edits after rejection, acceptance, Max/effort selection, fixed-model bypass,
native HTTP tool transport, actual CCR retrieval, compressed-context and custom
memory handoff, missing verdict errors, and cancellation cleanup. They are not
model-quality evals.

Two isolated live review cases used the locally configured Max, which resolved
to `openai/gpt-6-astra`, medium. A false healthy-deployment answer was rejected
after retrieving hidden HTTP 503/startup-failure evidence from a temporary CCR
database; a correct `2 + 2 = 4` answer was accepted. The first case recovered from
an invalid prefixed CCR handle before successful retrieval. The tool schema now
explicitly specifies bare 12-hex handles. First run: 4 provider calls, 2,125 total
reported tokens (433 + 534 + 749 + 409), 10.65s rejection / 3.31s acceptance.
These narrow cases establish live retrieval and verdict behavior, not broad
review quality, economic savings, or a live rejected-worker continuation. That
historical run predates the bounded protocol: an invalid CCR handle now ends and
reports the review instead of asking the reviewer to recover.
