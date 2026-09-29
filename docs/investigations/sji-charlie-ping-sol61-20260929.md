# SJI Charlie ping failure after GPT-6.1 Sol selection

The authenticated SJI chat stored two owner `ping` messages at 22:15 and 22:16 UTC, but Charlie logged `Charlie finished the turn without a response.` The earlier Sol canary had used Tau `--print --no-tools` in an isolated project, so it had not exercised Charlie's RPC prompt with project hooks and active tools.

## Evidence and hypothesis sequence

| Hypothesis | Null check | Observed result |
| --- | --- | --- |
| The SJI chat bridge did not deliver ping | Read production chat rows and Charlie journal | Both `ping` rows were stored and Tau warm RPC sessions started; delivery was intact. |
| The model could not answer a prompt | Replay a captured failed `context-envelope.md` through the installed Tau RPC process | Tau emitted an assistant error: HTTP 400, `input[0].content[1].type` was `text`, but the Responses API required `input_text`. |
| The per-turn hook created the invalid block | Inspect the installed `pi_agent.hooks._append_to_last_user_message` and replay after changing only its Responses `input` block type | The request moved past content validation and returned `I’m here.` with tools disabled. |
| Active tools caused the remaining failure | Replay the same saved ping context with active tools | Tau emitted HTTP 400: `tools[8].name` did not match `^[a-zA-Z0-9_-]+$`. The Codex Responses transport bypassed the existing OpenAI tool name mapping. |

The repair appends `input_text` to Responses `input` messages and applies the existing collision-safe tool name mapping in the Codex Responses transport, including reverse mapping for tool responses. No prompt or SJI chat route changed.

## Verification

- Focused payload checks: `3 passed` for Responses hook block type, non-Responses block preservation, and provider-safe Codex tool request name.
- Authenticated SJI WebSocket `message: ping` on `agent-direct-ping-repair-a516ea67ac--u6d19a75a2da9` completed with `I’m here. Ready when you are.` Production transcript stored one user `ping` row and that agent reply. `/health` returned 200; Charlie chat and supervisor services were active.
- An existing unrelated test in `test_openai_codex_responses.py` still exposes `UnboundLocalError: parse_result` in `process_responses_stream` when a model emits a function call. It predates this repair and was not used as evidence that ping passed. A live tool invocation was outside this ping acceptance case.
