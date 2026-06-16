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
