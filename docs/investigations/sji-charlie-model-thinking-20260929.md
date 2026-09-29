# SJI Charlie model and thinking selection, 2026-09-29

## Problem

Charlie rejected `/model openai/gpt-6-astra medium` and `/model openai/gpt-6-astra` as unknown models, although Marcus expected to select that model and its thinking level.

## Null-hypothesis checks

| Hypothesis | Null test and observed output | Result |
| --- | --- | --- |
| RPC parser treats the whole argument as a model ID | The screenshot's first error was `Unknown model: openai/gpt-6-astra medium`. | Falsified: the parser did not separate thinking level. |
| Astra is already available to Tau | Production `get_available_models` returned four models, `gpt6 []`, `gpt55 [('openai', 'gpt-5.5')]`. | Falsified: the live catalog has no Astra entry. |
| Missing OpenAI authentication alone explains the rejection | The same production catalog returned OpenAI `gpt-5.5`; encrypted auth storage exists. | Null holds: auth is present for OpenAI, so the missing model entry remains causal. |
| A custom OpenAI model entry cannot be loaded by installed Tau | An isolated `PI_CODING_AGENT_DIR` with Astra metadata returned `('openai', 'gpt-6-astra', 'openai-codex-responses')` without touching live config. | Null holds: Tau accepts the custom entry. |

The initial isolated registry probe used `PI_AGENT_DIR`, which this Tau config does not use for the model registry. It read the global registry and was inconclusive; the corrected probe used `PI_CODING_AGENT_DIR`.

Official OpenAI model documentation: https://developers.openai.com/api/docs/models/gpt-6-astra (model ID and supported low, medium, high, xhigh, max efforts).
