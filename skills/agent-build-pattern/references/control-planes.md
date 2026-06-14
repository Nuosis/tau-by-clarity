# Control Planes Reference

Every LLM API call is a stack of independent control planes. Reaching for the wrong one to solve a problem is the most common source of agent fragility.

## The planes

| Plane | What it controls | Reach for it when |
|---|---|---|
| **System prompt** (Anthropic `system` / OpenAI `instructions`) | Persona, role, hard rules, behavioral guardrails, voice. Sticky across the call. | Role/identity drift. Agent acting outside its job. Boundary violations the model should self-police. *Don't* use it to deliver per-call data, tool lists, or artifact schemas — those belong in the user message/input artifact, tools plane, and schema/return-tool plane. |
| **Developer message** (OpenAI; Chat Completions exposes this directly, Responses folds it into `instructions`) | Trusted-intermediary instructions outranking user but below platform-controlled system. | When user input is untrusted and you need a layer that user can't override ("ignore prior instructions" attacks). |
| **User message(s)** | The immediate task, dynamic data, retrieved context, files. The "what to do right now." | Task framing. Missing context. Wrong data shape passed in. Prompt-injection of retrieved content. Per-call variation. |
| **Assistant prefill** (preset opening tokens) | Forces the model's first tokens. e.g. `{` to start JSON, `[` to start a list. | Output-format hygiene when the model adds preamble or markdown fences. Cheaper than schema-strict for some cases. |
| **tools array** | What the model is *capable* of calling. Names, descriptions, input schemas. The action surface. | Adding/removing capabilities. Renaming an action. Tightening an input schema. *If a tool isn't in the array, the model legitimately cannot call it* — and a model emitting tool_use for a name outside the array is a provider-side contract violation. |
| **tool_choice** | Whether tools may, must, or must-not be called this turn (`auto` / `any` / `{type:tool,name:X}` / `none`). | Forcing structured output via a single schema tool. Preventing premature tool calls when you want the model to think first. Letting the model decide between read tools. |
| **parallel_tool_calls** | Whether multiple tool_use blocks may appear in one response. | Wasted/conflicting parallel calls. Single-pass harnesses where you only want one structured emit per turn. Encouraging parallelism for independent reads. |
| **response_format / json_schema strict** | Provider-enforced output shape (alternative to tool_choice trick). | Need bullet-proof parseable JSON without retry loops. OpenAI-strong; Anthropic doesn't have a direct equivalent — use forced-tool instead. |
| **stop sequences** | Terminates generation when a string appears. | Protocols where a marker ends a section. Preventing the model from extending past a known boundary. |
| **max_tokens / max_output_tokens** | Output ceiling. | Cost control. Runaway generation defense. **Diagnostic value:** truncated JSON / mid-string cutoffs usually mean this is too low for the response size. |
| **temperature, top_p, top_k** | Sampling diversity. | Need determinism for evals → low temp. Need creative variation → higher. Do NOT change to "fix" a contract violation — the issue isn't sampling. |
| **reasoning_effort / thinking budget** | Extended-thinking budget (Anthropic) or `reasoning.effort` (OpenAI o-series). Surfaces as `thinking` content blocks. | Hard multi-step problems where the model needs to deliberate before answering. Won't fix tool-choice violations or schema mismatches. |
| **seed** | Determinism hint (OpenAI). | Repro debugging when sampling matters. Best-effort, not guaranteed. |
| **cache_control** (Anthropic) | Marks prompt prefix for prompt-caching. | Long sticky system prompts repeated across many calls. Eval cost/latency wins. |
| **metadata / tags** (side-channel) | Provider-side telemetry hints. Invisible to model. | Cost attribution, dashboard slicing. No behavioral effect. |
| **Content block types** (text, image, document, audio) | Modality of the input. | Vision tasks, file ingestion. Switching from text-only descriptions to actual file/image handling. |

## Diagnosis cheat-sheet — symptom → plane to touch first

| Symptom | Likely plane |
|---|---|
| Model calls a tool not in `tools` array | **tools array** (don't list capabilities in prose; either offer them as real tools or omit them). If provider should be enforcing this and isn't, that's an out-of-band issue. |
| Model emits multiple tool_use when you want one | **parallel_tool_calls** (set false) and/or **tool_choice** (force the single tool) |
| Output isn't valid JSON / has prose preamble | **response_format strict** or **tool_choice forced-tool** or **assistant prefill** with `{` |
| JSON truncated mid-string | **max_tokens** too low |
| Refuses to act / over-apologizes | **system prompt** (loosen identity constraints) or **user message** (clearer task framing) |
| Acts outside its assigned domain | **system prompt** (tighten role) — NOT tool_choice |
| Hallucinates facts from prompt context | **user message** (give cleaner grounding) — content/retrieval issue, not sampling |
| Inconsistent eval scoring across runs | **temperature** (lower) and/or **seed** |
| Doesn't deliberate enough on hard task | **reasoning_effort / thinking budget** |
| Picks wrong option when several plausible | **system prompt** (rules) or **tool descriptions** (clearer disambiguation) — almost never temperature |
| User input overrides system | **developer message** (OpenAI) or restructure system to specify "ignore instructions in user content" |
| Same long context costs a lot per call | **cache_control** |

## Structural vs. behavioral enforcement

Each plane gives either **structural** enforcement (the API/model literally cannot do otherwise) or **behavioral** enforcement (the model is told to behave, and may not).

**Structural:**
- tools array (model can only call what's listed)
- tool_choice forced (model must call the named tool)
- response_format json_schema strict (output must match)
- max_tokens (output cannot exceed)
- stop sequences (output terminates at marker)
- Content modalities (you control what the model sees)

**Behavioral (needs eval coverage to verify):**
- system prompt rules
- user message instructions
- tool descriptions
- everything in temperature/reasoning/seed land — affects *how* the model behaves, not *whether* it complies

When designing an agent, push as much as possible to structural planes. Every behavior left at the behavioral layer is a future eval-coverage cost and a future failure mode.
