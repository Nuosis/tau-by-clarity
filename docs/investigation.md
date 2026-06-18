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
