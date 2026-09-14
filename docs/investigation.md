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

---

## Claire voice post-tool timeout — 2026-07-15

### Problem

The production `claire_ea` voice session on `/google-voice/session2` received
“ask for Anastasia, do you see it?”, completed three read-only action lookups,
then produced neither a final transcript nor audio. The Tau 0.56.13 trace
contained `tau.provider_request` for the post-tool turn but no
`tau.provider_response`.

### Observable success

For every Anthropic-compatible provider turn, including MiniMax, Tau emits one
raw response envelope containing the SDK stream events and final SDK message.
If iteration or parsing fails, it still emits the events received before the
failure. The exact Claire production replay must then show the initial and
post-tool provider responses, successful read-only tool results, persisted
`transcript_out`, and non-empty audio output.

### Hypothesis list

| # | Hypothesis | Null hypothesis | Status |
|---|------------|-----------------|--------|
| 1 | Tau's Anthropic provider never invokes `on_response` | The adapter invokes the configured callback for every provider turn | CONFIRMED; repaired in 0.56.14 candidate |
| 2 | The 90-second gap was provider unavailability | The exact captured MiniMax request can start and complete promptly | FALSIFIED by a direct 3.478-second streamed replay |
| 3 | The tool results lacked the requested record | The read tools returned Anastasia records before the final turn | FALSIFIED by production `voice_events` and tool-result readback |

### Current hypothesis

Tau 0.56.13 wires `AgentSession._on_provider_response` into
`SimpleStreamOptions`, but `pi_ai.providers.anthropic.stream_simple` never calls
it. This is a provider-adapter control-plane defect. Repair the adapter callback
contract first, then replay the unchanged production scenario and use the newly
visible raw response to isolate any remaining tool-choice or stream-lifecycle
failure.

### Repair evidence

- Contract tests first failed with zero callbacks on both completed and broken
  streams, then passed after the adapter change. A third test proves closing a
  partially consumed async stream still emits the response received so far.
- The adjacent provider/stream slice passed 46 tests from workspace source.
- The built 0.56.14 wheel passed its response-instrumentation and stream tests
  from a clean installed environment (9 tests).
- A disposable container using the Claire server's real MiniMax credential
  loaded the candidate wheel directly, completed a MiniMax-M3 turn in 1.273
  seconds, and captured one serializable envelope with nine raw events and a
  final SDK message.

This evidence proves the provider callback contract. It does not yet prove the
Claire voice workflow; that requires publishing, deploying, and replaying the
exact production route and utterance.

---

## Tau startup runtime directories — 2026-07-16

### Problem

A cloned Tau agent can start without restoring the ignored
`.tau/agent/sessions` and `.tau/memory` runtime directories that exist in an
already-used agent such as Devin's Planner.

### Observable success

Starting Tau in a clean agent checkout recreates `.tau/agent/sessions` and a
schema-initialized `.tau/memory/memory.db`. `--no-session` still restores the
required directory layout but does not persist a session JSONL file.

### Hypothesis list

| # | Hypothesis | Null hypothesis | Status |
|---|------------|-----------------|--------|
| 1 | Normal startup creates sessions but only `--init` creates memory | A normal startup creates both runtime stores | FALSIFIED by isolated startup |
| 2 | In-memory/no-session startup can omit the whole agent directory | Startup restores required directories independently of session persistence | FALSIFIED by isolated startup |
| 3 | A cwd-rooted repair is sufficient for Devin subagents | Devin's explicit agent root and subprocess cwd resolve to the same directory | FALSIFIED by Devin's launch environment and an adversarial replay |

### Debug evidence

An isolated `0.56.19` startup using the real `--list-models --offline` path
created `.tau/agent/sessions/<session-id>.jsonl` and `.tau/settings.json`, but
no `.tau/memory`. `SessionManager.in_memory()` performs no project-directory
creation, while `ensure_memory_store()` is only called from `scaffold_project()`
on the explicit `--init` path.

Devin launches Planner with `TAU_CODING_AGENT_DIR=<planner>/.tau` while the
subprocess cwd remains the target project. An initial cwd-only repair passed its
test but left Planner's `.tau` untouched in that real topology. The regression
was replaced with a distinct-agent-root/distinct-target-project startup whose
extension records the runtime layout visible during its first activation.

### Current hypothesis

Root cause confirmed: runtime-directory restoration had no startup owner, and
session routing ignored the explicit active agent root. Startup now resolves
the explicit PI/TAU agent directory first (falling back to `<cwd>/.tau`),
initializes sessions and memory before extension loading, and routes persistent
session JSONL files into that active agent's sessions directory. The focused
Devin-topology regression passes for persistent and `--no-session` launches.

---

## Read-only agent definition with external sessions — 2026-07-16

### Problem

`agent charlie` launches the selected production persona from the read-only
`/opt/agents` bind mount and supplies a writable external `--session-dir`.
Tau 0.56.21 nevertheless tries to create `.tau/agent/sessions` below the
persona source before launch and exits with `PermissionError`.

### Observable success

Tau starts from a read-only agent definition when `--session-dir` points at a
writable external location, creates that external directory when needed, and
does not attempt to create `.tau/agent` inside the read-only source tree. The
literal `agent charlie` EA route must then reach the interactive TUI.

### Hypothesis list

| # | Hypothesis | Null hypothesis | Status |
|---|------------|-----------------|--------|
| 1 | The Charlie bind mount is unwritable to the container user | The selected persona's `.tau` directory is writable to `appuser` | FALSIFIED by `test -w` and the read-only mount readback |
| 2 | The launcher omits an external runtime path | The launcher passes a writable `--session-dir` | FALSIFIED by the literal launcher command and writable-volume readback |
| 3 | Startup ignores the parsed session override during runtime restoration | Startup routes initial directory creation through `first_pass.session_dir` | FALSIFIED by the deployed traceback and non-interactive reproduction |

### Debug evidence

The deployed container runs as `uid=1000(appuser)`. `/opt/agents` is mounted
read-only and `/data/project-agents` is mounted read-write. The launcher passes
`--session-dir /data/project-agents/.charlie-tui-home/sessions`, but the
traceback shows `ensure_agent_runtime_directories(os.getcwd())` attempting
`/opt/agents/charlie/ea/.tau/agent/sessions` before the parsed override is used.

### Current hypothesis

Root cause confirmed: startup runtime restoration does not receive the already
parsed explicit session directory. It must treat that directory as the session
runtime owner and leave the agent definition tree untouched.

---

## OpenAI subscription callback race — 2026-08-12

### Problem

Selecting OpenAI subscription login opens Safari, but the successful OpenAI
authorization redirects to `http://localhost:1455/auth/callback` while nothing
is listening, so Safari reports that it cannot open the page.

### Observable success

Tau binds the registered OpenAI callback address before opening the browser,
accepts the real HTTP callback, validates its state, and proceeds to exchange
the authorization code.

### Hypothesis list

| # | Hypothesis | Null hypothesis | Status |
|---|------------|-----------------|--------|
| 1 | Tau opens the browser before starting its callback listener | Port 1455 is accepting connections when Tau invokes `on_auth` | FALSIFIED by socket observation |
| 2 | The registered callback path is wrong | Tau's authorization request does not use OpenAI's registered `http://localhost:1455/auth/callback` URI | NULLIFIED by the emitted authorization URL and source |
| 3 | The installed Tau distribution lacks the callback-server dependency | `aiohttp` imports in the isolated `uv tool` environment | FALSIFIED by installed-interpreter `ModuleNotFoundError` |

### Debug evidence

During the real `login_openai_codex()` call order, a socket connection attempted
inside `on_auth` returned macOS error 61 (`connection refused`). The installed
Tau 0.56.27 and the workspace source both call `callbacks.on_auth(...)` before
entering `_wait_for_callback_code()`, which owns server startup.

After rebuilding 0.56.31 from the workspace, its isolated tool interpreter
raised `ModuleNotFoundError: No module named 'aiohttp'`. The workspace lock had
`aiohttp` only as a transitive dependency of the pinned Google SDK; a fresh
tool resolution selected dependencies that did not install it. Tau therefore
entered its manual-input fallback and never bound port 1455.

### Current hypothesis

Two root causes are confirmed: the installed distribution does not directly
declare its callback-server dependency, and when that dependency is present,
browser launch precedes listener startup. The repair must directly require
`aiohttp` and make listener readiness the prerequisite for invoking `on_auth`,
while retaining the exact registered callback URI.

### Repair evidence

- The focused OAuth file passes 30 tests from workspace source. The new
  regression makes a real HTTP request to the fixed callback route and failed
  with `ConnectionRefusedError` before the lifecycle repair.
- A fresh `uv tool` install directly resolved `aiohttp 3.14.3` from Tau's
  package metadata.
- The repaired installed build completed the literal Tau TUI route
  `/login` → OpenAI → Subscription against OpenAI. Safari rendered
  “Authorization complete,” Tau reported “Subscription login stored for
  OpenAI,” and an encrypted-auth readback confirmed a current OpenAI Codex
  credential with access token, refresh token, account ID, and future expiry.

---

## OpenRouter thinking level mismatch — 2026-08-12

### Problem

In the Interviewer agent, `/model` selecting OpenRouter, `standard`, and
`gpt-5.6-luna` reports that `xhigh` was applied, but the live footer and saved
agent setting show `high`. The `/thinking` cycle also omits `xhigh` for GPT-5.6
and omits `adaptive` for models that use adaptive thinking.

### Observable success

The existing OpenRouter tier configuration resolves `gpt-5.6-luna` as a
reasoning model, model selection leaves the live session at `xhigh`, and the
confirmation and persisted default report the effective level. `/thinking`
offers `xhigh` for GPT-5.6 and `adaptive` for Claude 4.6 adaptive-thinking
models.

### Hypothesis list

| # | Hypothesis | Null hypothesis | Status |
|---|------------|-----------------|--------|
| 1 | The saved tier does not contain `xhigh` | The active models file stores OpenRouter standard with `thinkingLevel: xhigh` | NULLIFIED by `~/.tau/agent/models.json` readback |
| 2 | The OpenRouter model resolves as non-reasoning and clamps `xhigh` | The runtime registry resolves `openrouter/gpt-5.6-luna` with `reasoning=True` | CONFIRMED; repaired by tier-derived reasoning capability |
| 3 | The confirmation reads the live effective thinking state | The confirmation uses `session.thinking_level` after applying the requested level | CONFIRMED; repaired to read back live session state |
| 4 | GPT-5.6 and adaptive Claude models are represented in `/thinking` capability checks | The capability helpers recognize GPT-5.6 `xhigh` and Claude 4.6 adaptive thinking | CONFIRMED; repaired capability lists and cycling |
| 5 | The OpenAI-compatible adapter preserves `xhigh` in the provider request | The outbound OpenRouter payload contains `reasoning_effort: xhigh` | CONFIRMED; original payload regression observed `high`, repaired payload and live request observed `xhigh` |

### Debug evidence

The active `~/.tau/agent/models.json` contains OpenRouter standard mapped to
`gpt-5.6-luna` with `thinkingLevel: xhigh`, while a registry instantiated in
the Interviewer environment resolves that exact model as
`api=openai-completions, reasoning=False`. The active Interviewer setting is
therefore `defaultThinkingLevel: high` even though the tier remains `xhigh`.

### Root cause and repair evidence

The compatible-provider registry discards the tier's reasoning declaration
when its `models` array is empty. Independently, GPT-5.6 is absent from the
`supports_xhigh` capability check, adaptive Claude models have no selectable
adaptive level, and the TUI confirmation reports the requested rather than
effective state. The OpenAI-compatible adapter also unconditionally downgrades
`xhigh` to `high` in its outbound payload.

The focused source slice passes 34 tests covering model capabilities, registry
resolution, `/thinking` cycling, model selection, effective-state persistence,
and OpenAI-compatible request construction. An isolated composed run using the
real `~/.tau/agent/models.json` and encrypted OpenRouter credential resolved
`openrouter/gpt-5.6-luna` as reasoning-capable and applied/persisted `xhigh`.
Finally, the public PyPI `0.56.33` wheel was installed into a clean environment
without a local `direct_url`, then sent a live OpenRouter request with
`reasoning_effort: xhigh`; OpenRouter returned exactly `OK`, stop reason `stop`,
no error, and 16 total tokens. Interviewer was pinned, locked, and synced to
that same public `0.56.33` build, and its literal `tau update` route resolved
`tau-by-clarity==0.56.33`.

# Tau instrumentation turn-finalization failure (2026-08-22)

## Problem

An instrumented Claire/Tau turn delivered its tool events but exited with
`TypeError: flush() got an unexpected keyword argument 'timeout_seconds'`
instead of reaching a normal assistant terminal response.

## Hypothesis list

| # | Hypothesis | Null hypothesis | Status |
|---|---|---|---|
| 1 | The eval invoked Tau with an invalid instrumentation setting. | The same error reproduces by calling the installed instrumentation boundary directly. | Null falsified: direct installed call reproduced the exact TypeError. |
| 2 | The caller and instrumentation module disagree on the flush contract. | The imported `flush` accepts the timeout argument used by `AgentSession`. | Null falsified: runtime signature was `()` while `AgentSession` passed `timeout_seconds`. |
| 3 | A duplicate definition displaced the intended implementation. | Only one `flush` definition exists and it uses the active pending-task set. | Null falsified: two definitions existed; the surviving one had no timeout, while the first referred to nonexistent `_pending_tasks`. |

## Debug evidence

- Production Gate 3 captured two delivered `tau.tool_call` and two
  `tau.tool_result` observations, followed by the exact TypeError at turn end.
- Direct runtime inspection reported `module_flush_signature () -> None` and a
  minimal call reproduced the TypeError.
- The focused pre-fix test
  `packages/coding-agent/tests/test_clarity_instrumentation.py` failed because
  `emit()` also did not return its scheduled task, confirming the same merge
  had split the pending-task contract.

## Repair

Keep one timeout-aware `flush`, point it at the actual `_pending` set, and
return the scheduled task from `emit`. This restores the single contract used
by both the runtime and the existing focused test without changing event
payloads, provider behavior, or sink routing.

The first release build used `0.56.33+voicehook4`; PyPI rejected it before
accepting any file because public indexes do not allow local version suffixes.
The release identifier was therefore corrected to the next public patch,
`0.56.34`.

---

## Anthropic dotted tool-name rejection — 2026-08-22

### Problem

Claire's Anthropic subscription accepted `claude-opus-4-8` with high thinking,
but rejected the request before inference with HTTP 400 because Tau sent the
registered tool name `world.topics` directly. Anthropic requires every tool
name to match `^[a-zA-Z0-9_-]{1,128}$`.

### Evidence and decision

- The production request failed at `tools.5.custom.name`; Claire's ordered
  debug registry showed index 5 was `world.topics`.
- Disabling active compression removed `ccr_retrieve`, shifted the same dotted
  tool to index 4, and produced the same validation error. Disabling PII did
  not remove or rename the invalid tool.
- The same subscription, model, and thinking level returned `OK` when Tau was
  limited to five regex-safe tools, isolating authentication and model access
  from the tool-schema failure.

Repair classification: existing-capability bug in the Anthropic provider
adapter. The adapter now owns a deterministic, collision-free bidirectional
mapping between Tau's internal names and Anthropic-safe names; histories and
tool definitions share the outbound mapping, and streamed tool calls are
restored to the registered internal name before dispatch. The provider copy of
each input schema also omits only root-level `oneOf`, `allOf`, and `anyOf`,
which Anthropic rejects before inference; Claire's original schema remains the
runtime validator. Claire's tool registry, compression, PII, prompts, and
extension dispatch remain unchanged.

### Verification

Four focused contract tests cover regex and length enforcement, deliberate
alias collision, history/definition consistency, root-schema transport, and an
SDK-boundary dotted tool round trip. The broader affected
Anthropic/OAuth/stream slice passed 92 tests; three unrelated tests failed
identically on the untouched `0.56.34` baseline and were excluded from the
focused gate. A wheel-isolated Claire canary then completed against the full
tool registry using the Anthropic subscription, Opus/high thinking, and default
compression/PII. Canary run `80bdfe822261` dispatched the provider alias back
to `world.topics`, completed the read-only tool call, and exited zero.

---

## Tau session slowdown investigation — 2026-09-14

### Problem

Tau session `d90a86c8-d39d-4308-bccd-68f0eb58e473` continues to move, but Terminal input is delayed or appears ignored and rendering is very slow.

### Hypothesis list

| # | Hypothesis | Null hypothesis | Status |
|---|---|---|---|
| 1 | Mac-wide memory pressure | Free memory, swap, compressor, and pageout metrics are abnormal | NULLIFIED |
| 2 | PID 21098 Python heap leak | Its physical footprint/RSS grows materially or exceeds peer Tau sessions | NULLIFIED |
| 3 | Persisted context is pathologically large | Its transcript/message sizes are materially larger than peers | NULLIFIED |
| 4 | Terminal rendering contributes | Terminal.app is idle and not drawing text | FALSIFIED; contributor |
| 5 | A child tool is blocked on interactive input | PID 21098 has no child waiting on `ttys002` input | FALSIFIED; primary cause |

### Debug evidence

- The target session maps to PID 21098 on `ttys002`; its persisted transcript is `.tau/agent/sessions/d90a86c8-d39d-4308-bccd-68f0eb58e473.jsonl`.
- The host has 128 GiB RAM; the observed snapshot reported 96% free memory, zero swap I/O, zero compressor pages, and zero pageouts.
- PID 21098 remained near 283 MiB RSS and 219 MiB physical footprint, comparable to the other local Tau sessions.
- The target transcript was 610,295 bytes across 116 records and 232,470 message characters; peer sessions were equal or larger. Its largest persisted tool result was 21,257 characters.
- Terminal.app PID 611 used roughly 10–25% CPU during sampling; `sample` showed CoreText glyph layout/drawing. The TUI requests renders for streamed updates and tool/loader state changes.
- At 08:04:11, PID 21098 spawned PID 23901, `/bin/bash ./deploy.sh`, with stdin `/dev/ttys002` and stdout/stderr piped back to Tau. It remained alive more than 12 minutes with no `ssh`, Docker, curl, sleep, or other child process.
- `/Users/marcusswift/python/Claire/deploy.sh` reaches `read -r commit_message` when staged changes exist and no commit message argument is supplied. The Claire worktree had staged changes in `app/worker/sms_dispatch.py` and `tests/worker/test_sms_dispatch_addressing.py`, so the script waits for that input before remote deployment.
- The target's network sockets were `CLOSE_WAIT` and its transcript stopped changing during observation, consistent with a local subprocess wait rather than an active provider stream.
- Target debug events recorded `memory_not_attached` for curation; this does not indicate a memory leak or host pressure.

### Current hypothesis

The session is blocked in `deploy.sh`'s interactive commit-message `read`, while Tau's busy-state loader and Terminal.app continue rendering. That hidden prompt explains ignored input; shared Terminal rendering explains the secondary visual lag. There is no evidence of system-wide memory pressure or a per-session memory leak.

### Safe recovery

If this deployment is intended, provide the intended commit message to the waiting prompt, then verify the commit, remote deployment, and health check. If it is not intended, cancel the current Tau operation using its normal session control and verify PID 23901 exits; do not send a commit message or interrupt blindly while deployment intent is unknown.

### Repair

Both Python bash execution paths now pass `stdin=asyncio.subprocess.DEVNULL`.
The focused regression gate passes 26 tests, including a `read` command through
each path that returns `stdin-eof` instead of waiting for Terminal input. Shell
heredocs remain usable because they provide explicit input redirection.
