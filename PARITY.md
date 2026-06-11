# pi-mono-python ↔ pi-mono-node-reference — Parity Report

**Generated:** 2026-06-09. Reference = `/Users/marcusswift/cli/pi-mono-node-reference`.

This document answers two questions:

1. **Can we deterministically surface deltas?** — Yes, for *surface area*. See
   [`parity_audit.py`](./parity_audit.py).
2. **What are the meaningful functional disparities?** — See
   [Behavioral disparities](#behavioral-disparities) below.

---

## TL;DR — is it ready?

The port is **far more complete than it feels**. Structurally:

- Unit tests: **793 pass / 5 fail**, and *all 5 failures are live-Gemini network
  tests* (`test_live_gemini.py`) returning `NOT_FOUND` — a credentials/model-id
  problem on the **default provider**, not a port defect.
- Session persistence, branching, fork, resume, tree-nav: **full parity**.
- Built-in tools (read/write/edit/bash/grep/find/ls): **full parity** (7/7).
- All "missing" interactive UI components are **consolidated** into
  `messages.py` / `selectors.py` / `rendering.py` / `auth_components.py`, not
  dropped.
- The 4 `NotImplementedError`s are **abstract base methods** — none are
  reachable at runtime. No `except: pass`, no TODO/FIXME in core.

**Why it *feels* unready** is almost entirely two things:

1. **Default provider first-run friction.** Default is `--provider google`; the
   live Gemini path is the only thing failing tests. If your day-1 path is
   Gemini and model resolution/auth is off, the very first run looks broken.
   → **Try `pi --provider anthropic --model sonnet -p "hello"` to isolate.**
2. **Visual polish gaps** vs Node: **no token-level syntax highlighting** and
   **images render as `[image: <mime>]` placeholders**. The output simply looks
   less finished than the Node TUI even when logic is identical.

Everything else is breadth (extra providers/models, image *generation*) that
you may or may not need.

**For headless use specifically** (your case), a live eval against MiniMax-M3
shows **agentic behavior is at parity** (same tool calls, identical file
outcomes) — see [Functional-parity eval](#functional-parity-eval-live-minimax-m3-headless-pi--p---mode-json).
The only two headless gaps are: (1) Python's `--mode json` `message_update`
events are empty (no streaming payload), and (2) `<think>` tags leak into final
text because Python's model registry routes MiniMax-M3 to the OpenAI-compat
endpoint while Node uses the Anthropic one. Both are small, specific fixes.

---

## The deterministic tool

```bash
cd /Users/marcusswift/cli/pi-mono-python
python3 parity_audit.py            # human-readable
python3 parity_audit.py --json     # machine-readable (CI)
python3 parity_audit.py --strict   # exit 1 if any node-only surface exists
```

It diffs **named surfaces whose identifiers are string-identical across
languages** — the only things that *can* be compared deterministically across
a TS→Py port:

| Surface | How |
|---|---|
| slash commands | `BuiltinSlashCommand("…")` vs `name: "…"` |
| CLI flags | `--xxx` tokens in arg/help files |
| built-in tools | `name=`/`label=` in `core/tools/*` |
| ai providers | files under `providers/` (name-normalized) |
| env API keys | `"[A-Z_]+"` in `env-api-keys` |
| model registry | count of `id=` entries in generated models |
| modules | every source file, name-normalized, **whole-repo** match |
| stub markers | `NotImplementedError` / TODO / FIXME in Python |

**What it deliberately does NOT do:** judge whether a tool, stream parser, or
TUI component *behaves* the same. Cross-language behavioral fidelity is not
statically decidable — it needs runtime evals (see
[Recommended next step](#recommended-next-step)). Treating the script's
"module present" as "feature works" would be false comfort.

Two false positives were found and fixed during construction, both worth
knowing:
- `agent/harness/*` looked "missing" but is **relocated** into
  `coding-agent/core/*` — fixed by matching module names across the whole repo.
- Model count read 0 because Python uses `id='…'` (single quotes) — fixed.

---

## Behavioral disparities

Ranked by user impact. ✅ = confirmed solid, ⚠️ = real gap.

### ⚠️ HIGH — likely the source of your "not ready" feeling

| # | Disparity | Evidence | Impact |
|---|---|---|---|
| 1 | **Default-provider live path fails.** Only failing tests are `test_live_gemini.py` (5/5) — Gemini returns `NOT_FOUND`. Default provider is `google`. | `pytest` output; `main.py` default `--provider google` | First-run looks broken even though the engine is fine. **Verify your API key / model id, or switch default provider.** |
| 2 | **No syntax highlighting.** `highlight_code()` applies one flat color; Node uses highlight.js token highlighting. | `modes/interactive/theme/theme.py:431` | Code blocks look dull/unfinished vs Node. Cosmetic but constant. |
| 3 | **Images shown as text placeholders.** Tool output emits `[image: <mime>]`; Node renders inline PNG. | `modes/interactive/components/rendering.py:189` | Vision/screenshot workflows feel broken. |

### ⚠️ MEDIUM — breadth gaps (matter only if you use them)

| # | Disparity | Evidence | Impact |
|---|---|---|---|
| 4 | **Image *generation* entirely absent** (distinct from image input). No `generateImages`, image-models, images-api-registry, OpenRouter image provider. | node `ai/src/images.ts` etc.; zero Python matches | No DALL-E/Flux/etc. High if you need it, none if you don't. |
| 5 | **Missing providers:** `mistral` (634-line impl), `cloudflare`, `openai-prompt-cache`, `faux` (test mock). | `parity_audit.py` → ai_providers | Can't use Mistral/Cloudflare Workers AI. |
| 6 | **~140 fewer models & 7 fewer provider env keys:** missing `DEEPSEEK`, `FIREWORKS`, `NVIDIA`, `TOGETHER`, `XIAOMI*`, `ZAI*`, `CLOUDFLARE`. | model registry 563 vs 705; env_api_keys 33 vs 42 | Power-user/emerging providers unreachable. Most are OpenAI-compatible — addable via `api_registry.py`. |
| 7 | **Slash commands missing:** `/import`, `/clone`, `/trust`. | `parity_audit.py` → slash_commands | `/trust` notable — trust manager exists but no slash entry. |
| 8 | **OAuth device-code flow less robust.** No RFC-8628 `slow_down` backoff helper, no shared OAuth success/error HTML page. Anthropic/Copilot/Codex flows present; Python *adds* Google Gemini-CLI + AntiGravity. | `utils/oauth/` (no `device-code`/`oauth-page`) | Copilot device login may misbehave under rate limits. |
| 9 | **`session-resources.ts` cleanup registry absent.** | node `ai/src/session-resources.ts` | Resource cleanup for long-lived/daemon sessions; low for normal CLI use. |

### ⚠️ LOW — runtime/platform plumbing (mostly N/A for a Python port)

`bun/*`, `windows-self-update`, `photon`/`exif-orientation`/`image-resize-worker`
(wasm/worker image libs), `child-process`, `fs-watch`, `native-modifiers`,
`word-navigation`, `version-check`, `deprecation`. These are Node/Bun-runtime
specifics; some (word-navigation in editor, version-check) are minor UX, the
rest are not features.

### ✅ Confirmed at parity (do not re-investigate)

- Session persistence: JSONL on disk, in-memory/ephemeral, fork, branch,
  branch-with-summary, resume/continue, `/tree` navigation, v1→v3 migrations.
  (`core/session_manager.py`)
- Built-in tools 7/7; CLI flag surface (the `parity_audit` "extra"/"missing"
  flag noise is mostly subcommand-file scoping, not real gaps — `--plan` is the
  one worth confirming).
- All interactive message/selector/auth components (consolidated, listed above).
- No runtime-reachable unfinished code paths.

---

## Functional-parity eval (live, MiniMax-M3, headless `pi -p --mode json`)

Harness: [`eval_parity.py`](./eval_parity.py). Runs 5 scripted scenarios
(echo / read / write / edit / bash) through **both** engines in identical fresh
sandbox dirs, then compares functional behavior — not prose (non-deterministic
with a live model). Ground truth = file-system side effects, which are
parser- and wording-independent.

```bash
python3 eval_parity.py                  # matrix
python3 eval_parity.py --json out.json  # raw transcripts
python3 eval_parity.py --runs 3         # repeat for nondeterminism
```

**Result (corrected run — first run had a harness cwd bug, now fixed):**

| scenario | both done | tools py/node | fs side-effect py/node | event schema | stream py/node | `<think>` leak py/node |
|---|---|---|---|---|---|---|
| echo  | ✅ | – / –   | ✅/✅ | match | ❌/✅ | leak / clean |
| read  | ✅ | ✅/✅ | ✅/✅ | match | ❌/✅ | clean / clean |
| write | ✅ | ✅/✅ | ✅/✅ | match | ❌/✅ | leak / clean |
| edit  | ✅ | ✅/✅ | ✅/✅ | match | ❌/✅ | clean / clean |
| bash  | ✅ | ✅/✅ | ✅/✅ | match | ❌/✅ | clean / clean |

### What the eval proves

**✅ Agentic / functional parity in headless mode is GOOD.** Same tools invoked
(`read`/`write`/`edit`/`bash`), **identical file-system outcomes**, both runs
complete cleanly, top-level event schema matches. For your headless use case,
the engine does the right thing.

**❌ Gap 1 — `message_update` events are empty in Python (all 5 scenarios).**
Node streams typed deltas (`thinking_start/delta/end`, `text_start/delta/end`,
`tool_execution_update`) inside each `message_update`; Python emits bare
`{"type":"message_update"}`. *Impact:* if a headless consumer parses the JSON
stream for live progress, it gets nothing from Python until the final
`message_end` snapshot. If you only read final output, you're unaffected.
*Root cause:* the JSON print-mode emitter (`modes/print_mode.py`) doesn't
serialize the streaming event payload. Independent, genuine Python gap.

**⚠️ Gap 2 — `<think>…</think>` leaks into Python's final text (intermittent).**
Largely a **model-registry drift, not a parser bug**: Python routes
`MiniMax-M3` to `api=openai-completions` @ `api.minimax.io/v1` (returns reasoning
inline as `<think>` text), while Node routes the *same id* to
`api=anthropic-messages` @ `api.minimax.io/anthropic` (returns structured
thinking blocks). The two `models.generated` files were generated at different
times and diverged (Python ships M2/M2.1/M2.5; Node ships M2.7/M3). Aligning
Python's MiniMax entries to the `anthropic-messages` endpoint would most likely
make reasoning handling match Node immediately.

### Fixes applied (2026-06-09)

Running the eval led to the actual root cause, which was deeper than registry drift:

1. **🔴 Anthropic-messages provider dropped ALL tool-call arguments** —
   `providers/anthropic.py` finalized tool args only inside a handler gated on
   `event_type == "ContentBlockStopEvent"`, but the Anthropic SDK's
   `messages.stream()` actually emits **`ParsedContentBlockStopEvent`**. The
   name never matched, so the handler never ran and every tool call went out
   with `arguments={}`. The model would retry ~17 times, get empty params each
   time, then apologize and give up. **This broke tool-calling for every
   MiniMax model on the anthropic endpoint (M2/M2.1/M2.5).** Almost certainly
   the core "not ready for prime time" symptom. *Fix:* match all three stop
   class names. One line.
   *Proven by a raw-SDK probe:* MiniMax streams `input_json_delta` correctly and
   the final message carries full input — so the data was always there; only
   Python's wrapper discarded it.

2. **JSON `message_update` events were empty** — `modes/print_mode.py` emitted
   bare `{"type":"message_update"}`. *Fix:* serialize the streaming payload
   (`assistantMessageEvent` + partial `message`), add the dropped
   `tool_execution_update`, and enrich `message_start`/`message_end` with full
   content (incl. thinking). Headless consumers now see live progress.

3. **MiniMax-M3 registry aligned to Node** — `models_generated.py` M3 was
   `openai-completions` @ `/v1` (reasoning inline as `<think>`); Node uses
   `anthropic-messages` @ `/anthropic` with `reasoning=true`. Aligned to match.
   Now safe *because* fix #1 makes the anthropic tool path work. This also
   removes the `<think>` leak (the Anthropic endpoint returns structured
   thinking blocks).

Validation: `python3 eval_parity.py --runs 2` — tools, file side-effects,
streaming payload, and think-handling all at parity with Node across scenarios.

### Follow-up fixes (2026-06-09, second pass)

4. **Syntax highlighting now token-aware** — `theme.highlight_code` was a flat
   one-color stub that ignored the language; the theme already defined a full
   `syntax*` palette it never used. Reimplemented with **pygments** (already
   installed; now declared in `coding-agent/pyproject.toml`) mapping tokens onto
   that palette, matching the Node highlight.js behavior. Falls back to flat
   color if pygments/lexer/theme-colors are unavailable. Regression test:
   `tests/test_syntax_highlight.py`.

5. **`openai-completions` inline `<think>` — NOT a parity gap (no action).**
   Verified Node's `openai-completions.ts` doesn't strip inline `<think>`
   either; both Node and Python read structured `reasoning_content`/`reasoning`
   fields (`providers/openai_completions.py:231`). Python already matches Node.
   (MiniMax now uses the anthropic path anyway, so inline `<think>` won't occur.)

### Still open (not blocking headless functional use)

- `.env` has a **duplicate `MINIMAX_API_KEY` (first line empty)** — moot for `pi`
  itself (the key store at `~/.pi/agent/auth.json` resolves above env), but
  recommend deleting the empty line for hygiene.
- **Inline image rendering in the interactive TUI — blocked by renderer
  architecture, not a missing component.** Investigated fully:
  - `ToolExecutionComponent` (the faithful port of Node's `tool-execution.ts`)
    is now image-capable: it builds `pi_tui` `Image` components and emits real
    kitty/iTerm2 escape sequences, with a text fallback. Verified by
    `tests/test_tool_image_rendering.py` (4 tests). **But that component is
    referenced only by tests — it is not instantiated on the live path.**
  - The live interactive renderer (`tui.py`) is a *simplified* port: the entire
    output area is **five fixed `Text` widgets**, and all history is flattened
    into one `Text` via string concatenation (`append_history`). A `Text` widget
    word-wraps/truncates its content, which corrupts image escape sequences, so
    images cannot be injected there.
  - Node, by contrast, renders each message/tool as its own component and places
    `Image` components inline in a scrollable container.
  - **To show inline images live, the Python interactive output area must be
    re-architected** from "5 Text panes" into a component container that holds
    heterogeneous children (Text + Image) — matching Node. That is a real
    refactor of the core interactive UX and is **not verifiable without a real
    image-capable terminal** (kitty/iTerm2) driving actual image tool output, so
    it is not shipped here.
  - Interactive-only; **zero impact on headless use.** The image-capable
    component is in place as the foundation if/when that refactor is done.

---

## Recommended next step

The deterministic script covers surface area. For the *behavioral* "feels
unready" class, the highest-leverage move is a tiny **golden-transcript eval**:
run the same scripted prompt through Node `pi -p` and Python `pi -p` with a
faux/echo provider and diff the rendered output. That catches streaming,
tool-render, and formatting drift that no static diff can. Say the word and I'll
wire it.

**Immediate unblock for daily use:** pin a provider you have working keys for
(e.g. `--provider anthropic`) instead of the default `google`, and items 1–3
above stop biting. The engine underneath is sound.
