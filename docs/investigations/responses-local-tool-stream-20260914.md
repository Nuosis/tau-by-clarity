# Responses tool completion crash in local routing experiment

Observed: four local Tau routing trials stopped after one provider request, before changing any task file, with `cannot access local variable parse_result where it is not associated with a value`. Captured llama.cpp SSE includes tool argument deltas and output_item.done, without function_call_arguments.done.

Hypotheses and evidence:
1. Invalid provider arguments: null (valid arguments) holds. The captured completed item contains valid JSON handle/query arguments and normal completion usage.
2. Tau requires an intermediate event to initialize final parse metadata: null (final item parses independently) falsified. Existing malformed-final-arguments test fails with the same UnboundLocalError; 3 other focused tests pass before repair.

Repair: parse final raw arguments into parse_streaming_json_result at output_item.done, deriving both values and repair metadata from that same result. Do not retain metadata from a previous call. Extend existing final-item test with valid JSON as a positive control.

Verification: 5 focused tests pass in source and installed home environment. Replay of actual captured SSE succeeds, retains exact handle/query, marks no repair, and preserves input859/output39. This is parser functionality evidence, not a router quality pass. Local home/global installations updated with backups; no package release.

Evidence: ~/.codex/evidence/tau-efficiency-followup-20260914/routing-workloads-v3/settings-all-large/http-01-response.sse and parser-replay.json. All v3 failures preserved and excluded from routing economics comparisons; v4 reruns after the parser repair.
