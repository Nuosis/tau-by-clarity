# Router completion review

Router-enabled sessions review terminal text candidates at `turn_end` using the
router owner's configured **max** provider, model and reasoning effort. There is
no hard-coded reviewer model. Fixed-model sessions and ordinary tool turns are
unchanged.

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
provider producer as well as its consumer. No new retry/turn-count budget exists.

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
review quality, economic savings, or a live rejected-worker continuation.
