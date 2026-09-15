# Reviewer compression bypass — September 14, 2026

Problem: transformed worker content retained CCR compression, but reviewer JSON serialization included toolResult.details, which retained uncompressed output in details.truncation.content. The ordinary tool evidence contract is content plus tool identifiers and error status; details is runtime/UI metadata.

Null hypothesis: compressed tool content cannot be reintroduced through metadata in the reviewer request. Two provider-boundary regression cases (dictionary and typed worker messages) failed before the change because details remained in the serialized payload. Both pass after excluding only top-level toolResult.details. The worker and persisted messages are not mutated. This change applies to the shared intention/review payload builder and preserves configured Max selection and verdict semantics.

Human verification protocol: persist a compressed tool result whose details retains its original log, run the session transform and terminal review, inspect the payload at the provider boundary, retrieve the emitted handle from the isolated CCR database, and compare the worker messages before/after. Assertions verify byte-for-byte compressed content, identifiers, error status, custom memory, capabilities, candidate reference, retrieved original evidence and worker nonmutation. Provider responses are scripted at the stream boundary; CCR storage, compression, session orchestration and tools execute locally. These tests do not assess model judgment quality.

Validation: 12 focused turn-review tests passed. The two added regression variants failed before the fix. Independent read-only review identified missing content-equality/retrieval/nonmutation assertions; these were added and passed.

Recorded-session replay: 258 messages from 4b0bc18c-7d35-429e-bd34-cc118822f757, including 138 tool results, ran through review_turn with a scripted provider. JSON payload fell from 1,670,724 to 1,124,147 characters: 546,577 removed (32.71%). All other message fields remained equal and worker context stayed unchanged. Replay used a temporary session copy and empty working instructions/tools; it is not a reconstruction of the historical HTTP request or a billed-token estimate. Artifacts: ~/.codex/evidence/tau-efficiency-experiment-20260914/{replay_projection.py,projection-replay.json}.

Remaining limits: a 1.12M-character reviewer payload remains large. This repair does not demonstrate cache hit rates, total router savings, long-running review quality, or cure repeated reassessments. No provider requests were needed for this repair.

Operational memory: tau-subscription-transport.md identifies the home installed runtime; tau-routing-contract.md identifies the global launcher and home dispatch. Local module patching preserves existing subscription transport repairs. Restart existing Tau processes to load updated code.
