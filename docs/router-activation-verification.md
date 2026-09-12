# Router activation verification — 0.58.1

## Claim and test route

`/model router` activates metadata-driven model selection and displays `router on`
in place of the footer's model/thinking fields. Selecting a concrete model
restores fixed-model operation and the normal footer. Configuration failure must
leave the previous mode intact; malformed generated metadata must not execute
requested tools.

The human verification protocol was to configure four tiers in an isolated
models file, select the router through the model command, observe the footer,
run a fixture task, inspect provider/model/effort plus persisted fixture state,
and then select a concrete model and repeat the footer/provider observation.
Expected fixture values were not fed back as tool results; native tools produced
those results. Provider fixtures prescribe metadata only for orchestration
checks; classifier quality is not inferred from them.

## Evidence

- 47 focused tests cover routing policy, the command/menu, activation failure,
  HTTP provider serialization and parsing, native read/edit execution, a
  default → ultra-light → max handoff, reset to default on a new user prompt,
  switch back to fixed operation, nested optional tool arguments, malformed
  responses, usage retention on error, and existing hook/configuration paths.
- The local HTTP fixture exercises Tau's provider adapter and OpenAI SDK. It
  supplies prescribed response metadata and is not a paid-model eval.
- A live task went through `/model router` and `AgentSession.prompt`: Sol high
  requested a read and emitted metadata; Luna low received the file contents and
  returned the correct `Quiet Harbor` / `23` values. The file bytes were unchanged.
  Two provider invocations used 4,726 total tokens; no paid judge or retry.
  Artifacts: `/tmp/tau-router-activation-live/20260912T004806/` (review, calls,
  request/response traces, normalized messages, results). The fixture exposed
  only a guarded native read tool.
- A terminal session using the real interactive mode displayed
  `default | thinking: off`, then `router on` after `/model router`, then
  `light | thinking: off` after selecting `router-fixture/light`.
  Capture: `/tmp/tau-router-ui/terminal-capture.json`. This used a sandbox session
  and no model invocation. An initial CLI-launch attempt selected an unrelated
  startup default and was exited; it was not credited as footer proof.

Pre-run review used the ongoing local-review authorization recorded in
`/Users/marcusswift/.codex/memories/tau-routing-contract.md`. Post-run inspection
confirmed the selected provider IDs/efforts, actual read result, unchanged file,
and accurate final text. The initial HTTP fixture omitted the argument-completion
stream event and failed in provider parsing; adding that normal event completed
the fixture. No provider-parser production change was made for that fixture.

## Limits

One live read/report task establishes that this path executes and switches models;
it does not establish general classification accuracy, task-quality parity,
dollar savings, or subscription-quota savings. The opt-in matrix remains the
experimental sketch. API eligibility is checked on activation; broad tool-schema
and multi-provider compatibility are not claimed beyond the documented adapters.

The next experiment is a small held-out routed-versus-fixed workflow comparison
using this implementation, measuring completed work and all execution consumption.
