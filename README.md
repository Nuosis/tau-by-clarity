# Tau by Clarity

> A multi-provider AI coding agent **and** agent-building framework for Python —
> an interactive TUI, a headless CLI (`tau`), an agent loop, file tools, and
> built-in **Tau by Clarity** PII tokenization and active context compression.

Run it (`tau`), embed it (`import pi_coding_agent`), or build your own agents on it
(see [Building agents](#building-agents--tau-as-an-agent-framework)). Tau builds on
the **PI** project — see [Credits & lineage](#credits--lineage).

---

## Installation

Install the `tau` CLI from PyPI ([tau-by-clarity](https://pypi.org/project/tau-by-clarity/)):

```bash
uv tool install tau-by-clarity      # recommended — puts `tau` on your PATH
# or:  pipx install tau-by-clarity
# or:  pip install tau-by-clarity
```

Requires **Python 3.11+**. Verify with `tau --help`.

To embed the library instead of the CLI, `pip install tau-by-clarity` then
`import pi_coding_agent` (or the Tau alias `import tau_coding_agent`).

### Updating Tau

If Tau was installed from PyPI, update it from the CLI:

```bash
tau update          # update Tau itself from PyPI
tau update all      # update installed packages/extensions, then Tau
```

More targeted forms are also supported:

```bash
tau update self
tau update tau
tau update --extensions
tau update --extension <source>
```

<details>
<summary><b>From source (development)</b></summary>

```bash
git clone https://github.com/Nuosis/tau-by-clarity.git
cd tau-by-clarity
uv sync               # workspace + dev deps
uv run tau            # run from the checkout
```

</details>

### Dependencies

Installed automatically by `uv sync` (declared in `pyproject.toml`):

| Required (runtime) | Purpose |
|---|---|
| `pydantic` (≥2) | typed models / contracts |
| `anthropic`, `openai`, `google-genai`, `boto3` | provider SDKs (Anthropic, OpenAI-compatible, Gemini, Bedrock) |
| `httpx` | HTTP transport |

| Optional (feature-gated, lazy-imported — not required) | Enables |
|---|---|
| `presidio-analyzer` | Tau by Clarity's NER detector (regex detection works without it) |
| Ollama + `nomic-embed-text` (local, `http://localhost:11434`) | local semantic recall embeddings for project-local memory |

Dev/test extras (`pytest`, `pytest-asyncio`, …) install with `uv sync --extra dev`.

---

## Quick Start

### 1. Launch the Interactive TUI

```bash
tau
```

This opens the full-featured terminal UI where you can chat with the coding agent.
If no provider is configured yet, Tau will prompt you to log in.

### 2. Log in and choose a model

Use the built-in slash commands from the TUI:

```text
/login
/model
```

`/login` opens the provider/auth flow and stores the credential in Tau's auth
storage. For API-key providers you can also paste the key directly:

```text
/login openai sk-...
/login anthropic sk-ant-...
/login google ...
```

Then use `/model` to select the default model for the current/global settings.
`/models` is an alias. For provider tier mappings, use `/set` interactively or
set one directly:

```text
/set openai strong gpt-5.1
/set google standard gemini-2.5-pro
/set anthropic weak claude-haiku-4-5
```

Valid tiers are `strong`, `standard`, and `weak`. Tau still supports
environment variables such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and
`GEMINI_API_KEY` for CI/headless one-off runs, but shell exports are not the
normal setup path.

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line in input |
| `/` | Slash command completion |
| `@` | File path completion |
| `Ctrl+P` | Cycle to next model |
| `Ctrl+C` / `Esc` | Quit |

### 3. Try a Simple Task

Type in the terminal:

```
Create a Python function to calculate fibonacci numbers
```

The agent will write the code and save it to your current directory.

---

## Common Use Cases

### Single Prompt (Non-Interactive)

For scripting or quick tasks:

```bash
tau --print "Write a quicksort in Python"
```

The agent's response prints to stdout and exits.

### Switch Models

```bash
# Use a specific model
tau --model gemini-2.5-pro-preview

# Use a provider + model name
tau --provider google --model gemini-2.0-flash

# List all available models
tau --list-models
```

### Resume Previous Sessions

```bash
# Continue the most recent session
tau --continue

# Pick from a list of previous sessions
tau --resume
```

### Slash Commands in TUI

Type `/` in the interactive TUI to see available commands:

| Command | Description |
|---------|-------------|
| `/login [provider] [api_key]` | Store provider credentials through Tau auth storage |
| `/model [provider/model]` | Select and persist the current/default model |
| `/models [provider/model]` | Alias for `/model` |
| `/set <provider> <tier> <model>` | Set a provider tier mapping (`strong`, `standard`, `weak`) |
| `/thinking <level>` | Set thinking detail: `minimal` · `low` · `medium` · `high` · `xhigh` |
| `/compact` | Compress conversation context to save tokens |
| `/session` | Show session statistics (tokens used, cost estimate) |
| `/tools` | List all active tools available to the agent |

### Full CLI Help

```bash
tau --help
tau update --help
```

---

## Building agents — tau as an agent framework

Tau isn't only the bundled coding agent — it's a framework for building your own
agents and subagents. The **agent directory is the deployment unit**: prompts,
tools, skills, subagents, and evals all live inside it.

### Agent directory layout

```
my-agent/
├── OBJECTIVES.md             # user stories, success conditions, I/O artifact contracts
└── .tau/
    ├── settings.json         # provider/model, tool allow-list, extensions, name
    ├── SYSTEM.md             # brief system prompt: identity, hard rules, voice
    ├── extensions/           # your tools (extension_factory) — auto-discovered
    ├── skills/               # agent-local procedural knowledge
    └── subagents/<name>/     # each subagent is itself a full agent dir
```

Point the runtime at it with `PI_CODING_AGENT_DIR=/path/to/my-agent/.tau`, then
run headless (`tau --mode json -p "..."`) or in the TUI.

### Tools are extensions

Register a tool from an `extension_factory(pi)` in `.tau/extensions/*.py`, with a
typed parameter schema and structured result:

```python
def extension_factory(pi):
    async def execute(tool_call_id, params, signal, on_update, ctx):
        return {"content": [{"type": "text", "text": "..."}], "details": {...}}

    pi.register_tool(
        name="my_tool", label="My Tool", description="What it does.",
        parameters={"type": "object", "properties": {...}, "required": [...]},
        execute=execute,
    )

activate = extension_factory   # loader alias
```

`settings.json` `tools` / `extensions` lists are structural access control — an
empty `tools` list is a deliberate denial of the default tools, not a hint.

### Subagents

A subagent is a full tau agent under `.tau/subagents/<name>/`. The parent spawns
it in isolation (its own settings, `SYSTEM.md`, extensions), hands it a typed input
artifact, and reads back a typed output artifact.

### The build discipline

Agent-authoring guidance — directory layout, `OBJECTIVES.md` contracts, prompt-last
sequencing, and the **compile → unit-test → live-eval** gates — ships with the
harness as a **bundled, first-class skill** (`agent-build-pattern`). It is baked
in: the loader picks it up on every launch, and the default system prompt
advertises it as the `Agent build discipline` reference. You don't need to copy
it into your `~/.tau/agent/skills/` or project `.tau/skills/` to use it.

Source of truth: [`skills/agent-build-pattern/SKILL.md`](skills/agent-build-pattern/SKILL.md).
At runtime the loader resolves the same file at
`pi_coding_agent/bundled_skills/agent-build-pattern/SKILL.md` (wheel) or
`<repo>/skills/agent-build-pattern/SKILL.md` (dev). Disable per-run with
`PI_NO_BUNDLED_SKILLS=1`.

---

## Privacy — Tau by Clarity (default ON) — `pi_coding_agent.clarity_pii`

Tau by Clarity tokenizes personal data **before it reaches any model provider**, for
**every LLM call regardless of source** (agent sessions, the outer loop, evals, any
direct `pi_ai` use) — installed at the universal `pi_ai` dispatch hook. Real values
never leave the machine: the model sees stable tokens like `[PII:EMAIL:1]`, and the
reply is detokenized transparently.

- **Reversible per-session vault.** Token↔value mappings persist as a **lazy,
  session-referenced** artifact at `pii_vault/<session>.json` — written **only when
  a session actually contains PII**, carrying `{schema, session_id, created_at,
  updated_at}`. No-PII sessions create no artifact.
- **Detection.** Built-in high-confidence regex (email, US SSN, phone, credit card
  with Luhn, IPv4, IBAN, AWS keys) plus optional **Presidio** NER (lazy-imported —
  never a hard dependency).
- **Control.** On by default; disable process-wide with `PI_CLARITY_PII_DISABLED=1`,
  or inspect/toggle in the TUI with `/pii` (`status | on | off | vault | reveal
  <text> | clear`).

PII tokenization runs **after** active compression at the same `pi_ai` chokepoint,
so compressed tool outputs are tokenized before they are sent.

---

## Context management & compression

The harness reduces context with one primary mechanism and one fallback. They are
mutually exclusive — see `design/context-and-memory-management.md` §12.

### Active compression (default ON) — `pi_coding_agent.active_compression`

Content-aware, **reversible** compression of large **tool-output** payloads,
applied universally at the `pi_ai` dispatch layer (every LLM call, any source;
only `toolResult` messages, never the live prompt). JSON arrays are sampled with
**error/anomaly items always kept**; logs keep error lines; big text keeps
head+tail. The original is cached in a local hash-indexed **CCR** store (SQLite)
and is recoverable:

- the model can call the **`ccr_retrieve`** tool with a `[CCR:<handle>]` handle, and
- the harness **auto-rehydrates** a compressed block in place when the model
  references its handle (so recovery doesn't depend on the model calling a tool).

Control with the `active_compression` flag in `settings.json` — **on if the key is
absent**. Disable per-project with `"active_compression": false`, or process-wide
with the env var `PI_ACTIVE_COMPRESSION_DISABLED=1`.

```jsonc
// .tau/settings.json
{ "active_compression": true }   // omit entirely for the same (default-on) effect
```

### Summarization compaction — the fallback

The older threshold-based summarization compaction (§7) now runs **only when
active compression is off**. When active compression is on, it owns context
reduction and proactive compaction stands down. (Emergency *overflow* compaction
remains unconditional as a hard-limit safety net.)

### Position-based working-context compression — removed

The memory module's positional middle-compression (`compress_working_context`)
was **dropped** (§12); active compression replaces it. Project-local **memory**
(`memory_enabled`) now only **records and recalls** atomic facts — it no longer
compresses the working set.

---

## Running Tests

### All tests

```bash
uv run pytest
```

### Per-package

```bash
uv run pytest packages/tui/tests/          # TUI components
uv run pytest packages/ai/tests/           # AI providers
uv run pytest packages/agent/tests/        # Agent core
uv run pytest packages/coding-agent/tests/ # CLI + coding agent
```

### Live API tests (requires `GEMINI_API_KEY`)

```bash
uv run pytest packages/ai/tests/ --live -v

# Or via environment variable
LIVE_TESTS=1 uv run pytest packages/ai/tests/ -v
```

> All tests run against mocks by default — no API key required, no quota consumed.

---

## Test Status

| Package | Tests | Status |
|---------|-------|--------|
| `pi_tui` | 135 | ✅ passed |
| `pi_ai` + `pi_agent` | 156 | ✅ passed (7 skipped = live-only) |
| `pi_coding_agent` | 287 | ✅ passed |
| **Total** | **578** | **✅ all passing** |

---

## Project Structure

```
tau-by-clarity/
├── .env                          ← optional local env overrides for dev/CI (never commit)
├── pyproject.toml                ← uv workspace root
├── conftest.py                   ← global pytest config (.env loader)
└── packages/
    ├── ai/                       ← LLM provider layer
    │   └── src/pi_ai/
    │       ├── providers/        ← google.py, openai.py, anthropic.py, …
    │       ├── stream.py         ← unified stream_simple() / complete_simple()
    │       └── utils/            ← overflow detection, JSON parse, …
    ├── agent/                    ← core agent loop
    │   └── src/pi_agent/
    │       ├── agent.py          ← main run loop
    │       ├── tools/            ← tool registry & execution
    │       └── session.py        ← session state
    ├── coding-agent/             ← CLI entry point & extensions
    │   └── src/pi_coding_agent/
    │       ├── cli.py            ← `pi` command
    │       ├── core/             ← AgentSession, system prompt, tools
    │       └── modes/interactive/← TUI interactive mode
    └── tui/                      ← terminal UI library
        └── src/pi_tui/
            ├── components/       ← Editor, SelectList, Markdown, …
            ├── tui.py            ← differential rendering engine
            └── keys.py           ← Kitty keyboard protocol parser
```

---

## TypeScript → Python Mapping

| TypeScript | Python |
|---|---|
| `interface X {}` | `class X(BaseModel):` or `@dataclass` |
| `type X = A \| B` | `X = Union[A, B]` |
| `async function f()` | `async def f()` |
| `AsyncIterable<T>` | `AsyncGenerator[T, None]` |
| `AbortSignal` | `asyncio.Event` (cancellation token) |
| `EventEmitter` | `dict[str, list[Callable]]` |
| TypeBox schema | `pydantic.BaseModel` |
| `vitest` | `pytest` + `pytest-asyncio` |

---

## FAQ

| Problem | Solution |
|---------|----------|
| `uv: command not found` | Run the install script: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| No model or API key found | Run `tau`, then use `/login` and `/model` |
| `GEMINI_API_KEY not set` in live tests | Add a temporary env var or `.env` value for the live test run |
| `ModuleNotFoundError: pi_tui` | Use `uv run tau` instead of `python` directly |
| TUI shows garbled characters | Ensure your terminal supports UTF-8 (iTerm2, Warp, or any modern terminal) |
| Tests are skipped | Add `--live` to run real API tests |
| `400 thought_signature` error | Upgrade to the latest version — this is fixed in the google provider |

---

## Credits & lineage

Tau stands on the shoulders of the **PI** project. Its architecture, algorithms,
package boundaries, and the `pi_*` import namespaces come from there:

- **PI (`pi-mono`)** — the original TypeScript coding-agent monorepo by
  **Mario Zechner** ([@badlogic](https://github.com/badlogic/pi-mono),
  `@mariozechner/*`). Tau mirrors its design directly.
- **PI for Python** — Tau is forked from the Python port at
  [openxjarvis/pi-mono-python](https://github.com/openxjarvis/pi-mono-python).

Package lineage (and why the import names are `pi_*`):

| PI (TypeScript) | Tau (Python) | Layer |
|---|---|---|
| `@mariozechner/pi-ai` | `pi_ai` | Unified LLM streaming (Google, Anthropic, OpenAI, Bedrock, …) |
| `@mariozechner/pi-agent-core` | `pi_agent` | Agent loop, tool execution, state |
| `@mariozechner/pi-coding-agent` | `pi_coding_agent` | Coding agent + file tools |
| `@mariozechner/pi-tui` | `pi_tui` | Terminal UI rendering engine |

With gratitude to the PI authors and contributors.
