# Tau cost-efficiency investigation: local experiments through OpenAI subscription validation

**Evidence cutoff: September 14, 2026.** This report consolidates the experiments and implementation through commit `80056195f72e95e8ce9860cf975d9126120ff9f4`. It supersedes earlier summaries where their conclusions conflict with the scope corrections below; original artifacts remain unchanged.

## Questions and present answers

Marcus reported unexpectedly rapid consumption and asked three questions. Those reported invoices motivated the investigation; these tests do not reconcile or attribute the historical invoices.

| Question | What testing established | What remains unestablished |
|---|---|---|
| Does active compression reduce consumption? | Yes on tested inputs: local newly processed input fell 86.5%; OpenAI input for a native-compressed fixture fell 50.2%. Compression can cause a cold transition before the new representation becomes reusable. | A universal net saving after every retrieval, rewrite, retry and model response; production-scale savings. |
| Does routing help, and does switching destroy cache? | Switching does not inevitably prevent reuse on return: demonstrated locally and on the OpenAI subscription route. Configured routing consumed less than an all-Sol routed workflow in one task, but substantially more than plain fixed Sol. | General router savings, the earlier 80% estimate, or reliable cache reuse across the configured tiers. |
| Can review be token-efficient? | Focused source evidence sharply reduced review input locally and on OpenAI. A later, incremental Astra-medium optimization reduced estimated review credits 16.5% across four paired cases, with all decisions correct. | Universal review accuracy, optimal evidence selection, or a guaranteed cache benefit. |

**Current implementation:** initial intention uses the configured ultra-light tier, currently Luna low. Existing-intention revisions and completion review remain Max; **review remains Astra medium**. Worker routing tiers are unchanged.

## Measurement rules

The original investigation concerns **token usage and cost, not task effectiveness**. Count failed calls, retries, intention, review, retrieval and worker calls. Early work incorrectly made task success a prerequisite for cost analysis; that gate is superseded. The later question about whether Luna can establish intention explicitly added a narrow effectiveness test for that role. It does not retroactively invalidate earlier consumption measurements.

Keep these measures separate:

- **Serialized characters** measure payload size, not model tokens.
- **Provider input tokens** include cached input in the raw OpenAI Responses usage field. Tau's normalized `usage.input` instead excludes `cache_read`.
- **Newly processed input** is input minus reported cached input where the backend provides those counters.
- **Output tokens** include reasoning when the provider reports reasoning as an output subset; do not add it twice.
- **Cache-write tokens** are a reported category, not permission to invent a surcharge. Unknown counters are not zero.
- **Credit estimates** apply a recorded rate card to usage. They are not measured account debits, subscription-quota changes or dollar invoices.

For the subscription comparisons with zero reported writes, the estimate was:

`credits = [(input − cached) × input_rate + cached × cached_rate + output × output_rate] / 1,000,000`

The [published subscription rate card](https://learn.chatgpt.com/docs/pricing), checked during testing, supplied these standard credits per million tokens:

| Model | Uncached input | Cached input | Output |
|---|---:|---:|---:|
| Luna | 5 | 0.5 | 30 |
| Sol | 100 | 10 | 500 |
| Astra | 250 | 25 | 1,250 |

These are the rates used for this historical report, not a promise of future pricing. Local-model counts were not converted into OpenAI dollar charges. Experimental call counts below are per phase; do not sum every report or telemetry event into a grand total because some reports reuse earlier controls and captures.

## 1. Local cache and compression mechanics

The first phase used a 128-GiB Mac, separate resident llama.cpp servers with Qwen3-0.6B and Qwen3-1.7B weights, and approximately 10.5K-token synthetic contexts. There were **190 instrumented cache requests**, with stored request hashes and counter identities verified. Actual Tau review tests also used local Qwen3.6-35B through LM Studio/MLX.

### Compression and cache transitions

| Test | Measured result | Interpretation |
|---|---|---|
| Three matched four-request raw/compressed sequences | Newly processed input **31,644 → 4,268**, down **86.5%** | Native compression reduced processing work in this fixture. |
| Warm raw → compressed | First changed request processed about **1,326** tokens; next unchanged compressed request processed **1** | Representation changes have an immediate cost; stable compressed representations can subsequently reuse cache. |
| Compressed → original raw again | About **10,450** newly processed tokens each time | Restoring old text was not free in this slot/cache configuration. |
| Owned server restart | Warm **10,481 cached + 1 new** became **0 cached + 10,482 new**; next repeat warmed again | Restarting a cache-owning process differs from switching between resident models. |

### Model switching and review layout

All **six return-to-model requests** reused **10,493 of 10,494** input tokens. The first request to a different resident model incurred cold work. A stable five-request schedule processed about **10,483** new tokens, versus about **20,958** for the two-model schedule. This established retained model-local caches plus additional cold work; it did not exercise Tau's semantic routing policy.

| Review layout, same evidence/candidate sequence | Newly processed tokens |
|---|---:|
| One JSON message, changing material at the tail | 31,758 |
| Separate appended messages | 31,701 |
| Changing version field before the evidence | 94,492 |

Early-prefix rewrites required **2.98×** the processing. These measurements did **not** establish that JSON serialization itself destroys cache reuse. Timing runs contaminated by overlapping inference were excluded from timing conclusions; an isolated rerun supplied the reported timing evidence.

Source: [local experiment report](/Users/marcusswift/.codex/evidence/tau-local-cache-20260914/REPORT.md), [verified summary](/Users/marcusswift/.codex/evidence/tau-local-cache-20260914/final-summary.json).

## 2. Local review evidence, retrieval overhead and repairs

The intention-aware Tau `review_turn` path was exercised with a frozen candidate pair on the local 35B model.

| Review input | Calls | Reported input | Reported total |
|---|---:|---:|---:|
| Full source | 3 | 35,296 | 36,885 |
| Focused original CCR excerpt | 3 | 4,942 | 7,340 |
| Compressed source with retrieval, after the evidence-delivery repair | 5 | 13,916 | 16,749 |

Focused original evidence reduced input **86.0%** and total tokens **80.1%**. The excerpt came from a predetermined `audit_marker` query; the test did not prove autonomous query selection. Including retrieval and generation, the repaired compressed path used **54.6% fewer total tokens** than full source, while making two additional calls.

MLX did not expose usable cache counters for these reviewer sequences. Their input totals are **not** newly processed-input or cache-write measurements.

Two observed problems were relevant to interpreting the runs:

1. A retrieved `deployment_http_status=503` and adjacent timestamp were wrongly transformed into a PII placeholder in the outgoing request. The narrow status/date redaction repair preserved the evidence; **26 focused checks** passed. This does not mean one repair explains every changed model judgment.
2. The local Responses parser crashed on a stream without an intermediate `arguments.done` event. The repair passed **5 focused checks and captured-SSE replay**.

Those failures remain part of the record; their consumed tokens were not discarded. Correctness checks showed full and focused candidates judged correctly, but that was not a prerequisite for retaining the cost measurements.

Sources: [cost-only disposition](/Users/marcusswift/.codex/evidence/tau-efficiency-followup-20260914/COST-ONLY.md), [redaction repair](ccr-status-redaction-20260914.md), [stream repair](responses-local-tool-stream-20260914.md).

## 3. Local native-router workloads and the stronger-model detour

Actual Tau routing was then tested on file-edit and invoice-summing tasks, using a 4B model for all tiers versus 1.7B for ultra-light/light and 4B for default/max.

| Aggregate across two tasks | All-large | Routed |
|---|---:|---:|
| Calls | 19 | 23 |
| Input including cache | 51,313 | 61,556 |
| Cached input | 15,101 | 25,211 |
| Newly processed input | 36,212 | 36,345 |
| Output | 2,236 | 3,042 |

Routing added **20.0% input**, **0.37% newly processed input**, and **36.0% output**. Larger-model calls stayed at 19; four smaller-model calls were added. Larger-model input fell only about 7%. On the invoice case alone, new input rose **6,190 → 9,228**, or **49.1%**.

These are valid consumption records despite task failures. Initial cache warmth was uncontrolled, and the local backend ignored object-shaped named tool choice. Those facts limit attribution to model selection; they do not erase the counts. Earlier harness/parser variants remain archived separately and are not silently pooled with this comparison.

The measured break-even condition was retained rather than inventing prices. With `LI/LC/LO` denoting large-model per-token uncached/cached/output rates and `SI/SC/SO` the smaller-model rates, routed consumption is cheaper for this ledger only if:

`7086×SI + 6753×SC + 1019×SO < 6953×LI − 3357×LC + 213×LO`

A subsequent stronger-model detour used Qwen3.6-35B-A3B GGUF and captured **39 calls** across six trials. All six saved artifacts were correct, but the invoice runs omitted requested post-write verification. An effectiveness gate blocked the next routing comparison; **that gate was subsequently rejected as outside the cost investigation**.

The preserved cost comparison was:

| Strong-model configuration, two tasks | Calls | Input | Cached | New input | Aggregate output | Total |
|---|---:|---:|---:|---:|---:|---:|
| Corrected singleton tool choice, reasoning off | 12 | 32,221 | 17,632 | 14,589 | 2,316 | 34,537 |
| Same path, reasoning on with 512-token thinking budget | 15 | 41,458 | 20,206 | 21,252 | 4,704 | 46,162 |

The latter used **33.7% more total tokens**. Output was aggregate generation, not a separately measured reasoning category. This was not a routed-versus-fixed result or a general claim about reasoning efficiency.

Sources: [routing ledger](/Users/marcusswift/.codex/evidence/tau-efficiency-followup-20260914/routing-ledger.json), [corrected cost accounting](/Users/marcusswift/.codex/evidence/tau-efficiency-followup-20260914/cost-only-accounting.json), [strong-model report—historical effectiveness gate superseded](/Users/marcusswift/.codex/evidence/tau-strong-baseline-20260914/REPORT.md).

## 4. Applying the local lessons and establishing the subscription route

Tau review payloads were changed to keep stable material first, omit duplicated runtime metadata, abbreviate long raw tool output with a CCR handle, and supply focused excerpts while retaining retrievable originals. Already-compressed blocks were not eagerly expanded. Two captured payload replays shrank **23,005 → 3,289** and **23,014 → 3,298 characters**, approximately **85.7%**. This was an offline character measurement, not a billed-token result.

Per-invocation intention/review usage recording and configured-rate snapshots were added. Failed/cancelled calls with missing usage remain unpriced; completion records do not double-count invocation records. A pending parallel-tool cancellation change was also completed. **55 focused checks** supported this implementation stage.

The separate subscription transport repair corrected catalog-dependent fallback: saved OAuth now selects the subscription transport for the configured Luna/Sol/Astra IDs, and an unavailable subscription credential does not silently become a paid API request. **5 transport checks** and three tiny live calls (**75 total tokens**) established routing/access, not historical invoice reconciliation.

A direct paid-API cache pilot had produced **one HTTP429 `organization_spend_limit_exceeded` rejection**, no completed response and no usage. It supplied no cache evidence. The spend limit was not raised to continue the investigation.

Sources: [implementation report](tau-cost-lessons-implementation-20260914.md), [subscription transport verification](subscription-routing-20260914.md), [blocked API pilot](/Users/marcusswift/.codex/evidence/tau-efficiency-experiment-20260914/pilot-result.md).

## 5. OpenAI subscription probes with Langfuse reconciliation

The experiment connected Tau's native publisher to the Clarity-gated Langfuse ingress using the established credential workflow. Secrets were not included in experiment artifacts. A small subscription canary reported **18 input + 5 output tokens**.

An important instrumentation limitation was observed: Tau's normal subscription `on_response` hook receives the HTTP response before streamed terminal usage. The experiment therefore published supplemental flat terminal-usage events through the native publisher and fetched them back for comparison. This is not proof that every normal Tau session automatically publishes a complete cost ledger. Raw captures retain details that Tau's normalized parser omits, including reasoning attribution and the raw cache-write field; deep native serialization can stringify nested attribution.

The first **21-call** probe used **71,392 input + 105 output = 71,497 tokens**. All 21 remote usage rows matched local rows.

| Paired payload | Input tokens per call | Result |
|---|---:|---|
| Raw versus native compressed | 4,049 → 2,017 | **50.2% less input** |
| Full versus native focused review payload | 4,234 → 668 | **84.2% less input** |

Output was five tokens per request. These were payload-consumption probes, not complete reviewer or router workflows. Initial repeated/switching requests all reported zero cache reads; that did not establish global cache failure.

Source: [21-call report](/Users/marcusswift/.codex/evidence/tau-subscription-instrumentation-20260914/tau-openai-cost-e828e21d619d/REPORT.md). Trace: `a407477d08cfeb2abbe6f87124491d65`.

## 6. Resolving what model switching does—and does not—prove

Delayed controls and model-return probes established:

- GPT-5.5 reused **5,888 of 6,352 tokens** on an unchanged delayed repeat.
- A native all-Sol workload reported a **2,048-token** hit.
- **GPT-5.5 → Sol → GPT-5.5** reused **5,888 of 6,352 input tokens on return**. All three captured request bodies were identical except `model`, including the cache key and low reasoning setting. This was a model-only probe, not the configured Sol-high worker policy.
- A separate eight-call Sol-high/Luna-low/Astra-medium sequence with 6,351-token identical prefixes and 15-second delays reported zero reads throughout, including unchanged repeats. The inconsistent reuse is unresolved.

The return result disproves the blanket claim that switching inevitably prevents reuse of a prior model's cache. It does not prove cross-model sharing, uninterrupted residency on a particular server, or guaranteed reuse for the configured tiers. The first call in the return sequence was itself a miss; an earlier control also used that prefix.

The subscription endpoint rejected `prompt_cache_options` with HTTP400. Explicit input breakpoints were also rejected for Luna and Astra. Although [API documentation describes these controls](https://developers.openai.com/api/docs/guides/prompt-caching), that did not make them supported on the tested subscription route. Three rejected attempts had **unknown usage**, not assumed zero cost.

The routing/cache follow-up recorded **31 completed calls, 118,510 reported tokens, and three rejected attempts without usage**. This phase includes the native comparisons below; do not count them again.

Source: [routing/cache evidence](/Users/marcusswift/.codex/evidence/tau-subscription-instrumentation-20260914/ROUTING-CACHE-ANSWERS.md). Return-cache trace: `275143b02950c95daad0f66e5b22f9b4`.

## 7. Native OpenAI routing economics: overhead versus model allocation

A small settings-file task was run through native Tau three ways. Every call was counted. The configured router included intention and review; plain fixed mode did not have that machinery.

| Mode | Calls | Input | Cached subset | Output | Estimated credits |
|---|---:|---:|---:|---:|---:|
| Plain fixed Sol high | 4 | 2,216 | 0 | 128 | 0.285600 |
| Configured router | 6 | 13,189 | 0 | 1,504 | 1.465345 |
| Router/review machinery, all tiers Sol high | 6 | 13,009 | 2,048 | 1,517 | 1.875080 |

Configured routing used **21.85% fewer estimated credits than the all-Sol routed run**, but **5.13× plain fixed Sol**. These are single-run observations: model/effort allocation, generated output and cache warmth differed. They are not an isolated causal coefficient or a general savings forecast.

The two Astra calls—initial intention and terminal review—accounted for **0.978750 credits, 66.79%** of configured-router consumption. The first routed worker request carried **2,502 input tokens**, versus **421** for plain Sol. Added workflow/context/output expense was measurable without assuming a destroyed cache.

All 16 native-call usage rows matched captured terminal responses and Langfuse rows. Traces: `049a34127df7bb4547cd96953f6c9add` and `0f792f254f87c3b7bbdabb25e856a84e`. [Machine-readable comparison](/Users/marcusswift/.codex/evidence/tau-subscription-instrumentation-20260914/native-comparison.json).

## 8. Initial intention on Luna; review efficiency with Astra held fixed

Marcus then explicitly asked whether Luna low/high could establish initial intention, and whether review could use fewer tokens. **Changing the review model was not authorized.** The tests and installed change in this stage kept review at Astra medium.

### Intention contract and low/high comparison

In the original six Luna trials, both efforts preserved requested action/scope, but five `completion_evidence` fields cited the request rather than specifying observable completion checks. This was a separately reported contract defect, not a hidden retroactive scoring rule. The shared submission-tool description contained reviewer-oriented citation instructions.

Only the intention tool description was changed to request future observable completion checks. Both efforts then met the frozen full contract on **3/3 cases**: scoped edit, read-only request, and a correction superseding earlier user text. Independent semantic review confirmed all six. This does not certify all intentions or live revision of a persisted intention object.

| Luna intention effort, three cases | Input | Output including reasoning | Estimated credits |
|---|---:|---:|---:|
| Low | 1,925 | 441 | 0.022855 |
| High | 1,925 | 597 | 0.027535 |

Low used **17.0% fewer estimated credits**, with no observed decision advantage from high in these cases.

### Incremental Astra review optimization

The candidate removed empty transport fields and requested a concise rationale, preserving evidence text, tool arguments, business null/false values, nonempty errors, CCR handles and references. Astra medium reviewed verified completion, a wrong saved value, an unauthorized extra change, and missing required verification.

| Astra-medium review, four cases | Input | Output | Estimated credits |
|---|---:|---:|---:|
| Current payload/output instruction | 7,139 | 646 | 2.592250 |
| Lean payload + concise rationale | 6,288 | 474 | 2.164500 |

All **8/8 decisions** were correct. Input fell **11.9%**, output **26.6%**, and estimated credits **16.5%**. This combined two changes; their individual effects were not isolated. These savings are incremental to prior focused-review work and must not be added to the earlier 84–86% reductions. The 30-word rationale instruction is not a hard output-token limit.

### Native follow-through and installed behavior

A native six-call run confirmed **Luna low initial intention** (626 input/112 output) and **Astra medium completion review** (1,540/88), with the file changed, verified, and accepted. All six remote usage rows matched. Its estimate was **0.997762 credits**, **31.9% below** the earlier configured-router run; output/cache behavior differed, so the percentage remains observational. It was still above the earlier plain-Sol estimate.

The source change and local home/global runtimes now:

- Use ultra-light only when establishing intention without an existing intention; later intention revisions remain Max.
- Keep completion review on Max/Astra medium and preserve worker tier configuration.
- Use the intention-specific completion-check description and the Astra review-token cleanup.
- Record the selected auxiliary tier so `/stats` does not mislabel ultra-light startup as Max; legacy events retain a Max fallback.

**21 focused checks passed against source and installed home runtime.** Controlled-provider orchestration checks verified initial ultra-light, later sequential Max, all reviews Max, persisted tier records, CCR retrieval and evidence preservation. Those tests are plumbing evidence, not model-effectiveness evidence. Two known pre-existing mid-turn correction tests were excluded; mid-turn steering is outside this verification. Eight installed module copies matched source hashes. Existing processes require restart to load them; these were local updates, not a PyPI publication.

Sources: [implementation and validation](intention-review-token-efficiency-20260914.md), [comparison data](/Users/marcusswift/.codex/evidence/tau-subscription-instrumentation-20260914/intention-review-comparison.json). Role tests comprised 20 calls; native follow-through added six, all reconciled with Langfuse. Traces: `a6fc18ea23aad7e3b4b08203f08c8b9e`, `71bd41143d60d2778e32030a57a35f83`, `c3727ed395afbd31b71b8a2f0caf79d1`.

## Implementation record

| Commit | Change supported by the investigation |
|---|---|
| `aa2207dc5987f96a67a1de7b181bb93611afcf2f` | Preserve status/date evidence through PII handling. |
| `d363bb2a72aef3617b85cd4822741ee9879ae985` | Handle local Responses streams without intermediate argument-completion events. |
| `12d91e593cda8c8592717a6b10ecd6bac21987f4` | Focus review evidence and record per-call usage/rate snapshots. |
| `8f458b439419c79bbd2602fc2f3ab368edf251db` | Cancel and drain parallel tool tasks when their parent is cancelled. |
| `80056195f72e95e8ce9860cf975d9126120ff9f4` | Lightweight initial intention; Astra review-token cleanup; correct auxiliary tier reporting. |

## Limits and next evidence to collect

1. **No validated 80% router saving.** Repeat matched workloads of different sizes and include all auxiliary/retry costs before claiming a default net benefit.
2. **Cache variability remains unresolved.** Reuse is possible after switching, but current-tier repeats are not consistently cache hits. Latency alone cannot resolve this.
3. **No historical invoice reconciliation or subscription account-debit measurement.** Published-rate arithmetic does not supply either.
4. **No large-context/long-running guarantee.** The controlled results do not establish behavior for the original very long sessions or contexts near model limits.
5. **No hard subscription output cap.** The tested Tau transport omitted `max_output_tokens`; experiments used request/context/time bounds and observed-usage stops. A prompt length instruction is not an enforced token budget.
6. **Instrumentation still has limits.** The experiments reconciled supplemental terminal-usage events; complete automatic raw-usage publication and retained attribution in ordinary sessions remain separate work.
7. **Intention/review effectiveness is narrowly tested.** Extend cases without replacing the cost ledger with an effectiveness gate. Keep review on Astra medium.

The next cost study should compare the installed configuration with fixed-model execution across several small and larger tasks, preserving the same evidence accounting and counting every dispatched call. That will test whether the changes produce net savings beyond the single native follow-through.

## Provenance

Codex root executed the experiments and implementation. `/root/cache_pilot_review` performed read-only protocol, accounting and semantic reviews where recorded; it was not the executor. This consolidation used the existing raw ledgers/reports and local memories `tau-cost-efficiency-lessons.md`, `tau-subscription-transport.md`, `tau-routing-contract.md` and `clarity-internal-secrets.md`. Writing this report launched no additional model experiments.
