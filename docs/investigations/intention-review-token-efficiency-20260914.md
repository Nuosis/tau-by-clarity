# Initial intention and Astra review token efficiency

Initial intention now uses the router's configured ultra-light selection (Marcus's installed mapping: Luna low). Existing-intention revisions continue to use Max. Completion review continues to use Max (Astra medium); worker routing is unchanged.

The shared submit tool description previously told intention-setting calls to cite evidence like a reviewer. In six Luna low/high trials, all preserved outcome/scope, but five put request citations in completion_evidence rather than observable completion checks. The intention-specific tool description now asks for future completion checks. Both low and high met the complete contract on three cases: scoped edit, read-only request, and superseding user correction. Low consumed1925input+441output versus high1925+597, equivalent to .022855 versus .027535 published-rate credits (17%lower). These are small fixtures, not general effectiveness certification; correction tested prior user text, not a persisted intention object.

Review remains Astra medium. Empty transport annotations are removed without changing tool arguments, business null/false values, evidence text, CCR handles, or references. Rationales are requested in at most30words, with evidence references rather than repeated evidence text and concise actionable follow-ups. The rationale limit is a prompt instruction, not a hard output-token cap.

Four paired review cases (verified, wrong saved value, unauthorized additional change, missing saved-file verification) produced all8correct decisions. Current7139input+646output versus lean6288+474:11.9%lessinput,26.6%lessoutput,16.5%lower estimated credits (2.59225→2.16450). This tests metadata cleanup and concise output together; it does not isolate their individual effects. Original source evidence and nonempty parse/repair diagnostics remain available.

Native AgentSession follow-through: initial Luna low626input/112output; final Astra medium1540/88. Six provider calls, correct persisted settings, accepted review, all6remoteusage rows equal captured terminal usage. Estimated .997762credits versus earlier1.465345 (31.9%lower observed). Cache hits and worker outputs differed, so this is not a causal savings guarantee.

Auxiliary events now carry their selected tier. /stats consumes that tier, with Max fallback for legacy records lacking it; new ultra-light intentions are no longer mislabeled Max.

Validation:21focused source checks pass, including actual local file edit/review orchestration with controlled provider responses, initialultra-light and laterMaxselection, persisted tier records, preservedbusinessnulls/false values/error diagnostics, CCRretrieval and failure accounting. These checks are not model-quality evidence. Two known pre-existing midturnuser-correction tests were excluded; the later sequential request path is explicitly covered.

Live model evidence used saved subscription OAuth and Tau's provider boundary, with frozen native request packets for role comparisons.20comparison calls+6nativefollow-through calls were recorded; every remote usage row matched. Rate-card estimates use https://learn.chatgpt.com/docs/pricing; they are not account debits or dollar invoices. No paid API fallback. No hard output cap claimed on subscription transport.

Artifacts under ~/.codex/evidence/tau-subscription-instrumentation-20260914:
- tau-intention-review-f49274fcd2f3, tracea6fc18ea23aad7e3b4b08203f08c8b9e
- tau-intention-contract-022fee2e711b, trace71bd41143d60d2778e32030a57a35f83
- tau-native-routing-94f20103603b, tracec3727ed395afbd31b71b8a2f0caf79d1
- intention-review-comparison.json

Independent review confirmed6/6correct revised Luna intentions,8/8correct Astradecisons, raw/remote accounting, and preservation of review selection. Operational credential/runtime paths follow tau-subscription-transport.md, tau-cost-efficiency-lessons.md and clarity-internal-secrets.md. Restart existing Tau processes to load local runtime updates.
