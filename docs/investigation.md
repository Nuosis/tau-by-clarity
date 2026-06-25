## Problem

OpenAI Responses streaming with `gpt-5.5` failed on a simple prompt with an un-awaited `AsyncResponses.create` coroutine warning and then `'dict' object has no attribute 'content'`.

## Hypothesis List

| # | Hypothesis | Null Hypothesis | Status |
|---|------------|-----------------|--------|
| 1 | Provider output shape mismatches shared stream processor | OpenAI Responses provider passes an object with `.content` to `process_responses_stream()` | FALSIFIED |
| 2 | SDK stream creation is not awaited | `client.responses.create()` returns a ready async iterator, not an awaitable | FALSIFIED |

## Debug Evidence

`packages/ai/src/pi_ai/providers/openai_responses.py` initialized `output` as a dict, while `packages/ai/src/pi_ai/providers/openai_responses_shared.py` read `output.content` immediately.

Regression tests using `AsyncMock` for `responses.create` verified the SDK call is awaited and the final message streams as typed assistant output.

## Current Hypothesis

Root cause found: the OpenAI/Azure Responses providers were using legacy dict output plus an un-awaited async SDK call while the shared processor expects typed assistant output and an async iterator.

## Problem

Tau session `e838db70-f202-4f8e-858d-ae03f87d9652` failed on resume with HTTP 400 because a replayed Responses API reasoning item id was sent as an invalid `input[].id`.

## Hypothesis List

| # | Hypothesis | Null Hypothesis | Status |
|---|------------|-----------------|--------|
| 1 | The session file stored a PII-tokenized reasoning id | The session file stores the raw reasoning id | NULLIFIED |
| 2 | The Responses conversion emits replayed reasoning ids without protocol protection | Replayed reasoning ids are already protected from PII tokenization | FALSIFIED |
| 3 | Provider-request PII tokenization mutates protocol metadata | Provider-request PII tokenization only touches user-facing text | FALSIFIED |

## Debug Evidence

- Session line 227 contains raw id `rs_09dc8f8587d38d3e016a3340667384819b9b9e1efb018d3194`.
- Tau logged an API error where the same id appeared as `rs_09dc8f8587d38d3e016a[PII:CC:7]b9b9e1efb018d3194`.
- `clarity_pii.extension.on_before_provider_request` walked every string in provider payloads through `dict_string_slots`, including protocol keys such as `id` and `encrypted_content`.

## Current Hypothesis

Provider payload tokenization must skip protocol fields and only tokenize editable provider text. Regression coverage lives in `packages/coding-agent/tests/test_clarity_pii.py`.

## Problem

Tau is showing five related runtime defects: missing native memory lookup tools, subscription auth confusion between OpenAI and Anthropic, LM Studio/OpenAI-compatible streaming failing on `finish_reason`, `/chat` replaying tool results instead of user/assistant transcript only, and literal `</think>` text rendering as assistant content.

## Hypothesis List

| # | Hypothesis | Null Hypothesis | Status |
|---|------------|-----------------|--------|
| 1 | Memory lookup tools are imported but absent from the source tree | `pi_coding_agent.core.memory.tools` exists and registers tools | FALSIFIED |
| 2 | OpenAI and Anthropic subscription tokens overwrite the same auth key | `AuthStorage` can persist and resolve `openai` and `anthropic` OAuth tokens independently | NULLIFIED |
| 3 | OAuth token resolution ignores provider-specific OAuth field names such as `access`/`expires` | Auth resolution supports both storage field shapes and falls back to stored OAuth access before env | FIXED |
| 4 | OpenAI-compatible streaming assumes every stream has a choice finish reason | `finish_reason` is initialized before the stream loop | FALSIFIED |
| 5 | `/chat` filters replayed session roles to user/assistant only | Non-user/non-assistant roles are skipped in rebuilt transcript | FALSIFIED |
| 6 | Assistant text extraction strips provider think tags | Literal `<think>`/`</think>` tags are removed before rendering | FALSIFIED |

## Debug Evidence

- `AgentSession` imports `.memory.tools.register_memory_tools`, but `packages/coding-agent/src/pi_coding_agent/core/memory/tools.py` is absent.
- In-memory auth check persisted `openai` and `anthropic` OAuth rows independently and resolved access tokens as `oa` and `ant`; same-key overwrite is not the root cause.
- `openai_completions.py` reads `finish_reason` after the stream loop even though it is assigned only after `if not chunk.choices: continue`.
- `rebuild_history_from_current_session()` renders all other roles as `"{role}: {body}"`, so tool results enter `/chat`.
- `assistant_text_from_message()` and the generic message flattener pass literal text through unchanged.

## Current Hypothesis

Implemented missing memory tools/read APIs and narrow runtime fixes in the affected seams. Targeted tests verify auth coexistence, OpenAI-compatible empty-choice streams, transcript filtering/Markdown rendering, think-tag stripping, and memory tool registration.
