# Model invocation hook and router assignments

Tau's agent loop exposes `before_model_invocation` after context transformation and
message conversion, before API-key resolution and provider dispatch. It runs for
the initial call and each tool continuation. It does not run for every internal
HTTP retry, or auxiliary calls such as compaction outside the agent loop.

## Hook contract

Import `ModelInvocation` and `ModelInvocationSelection` from `pi_agent` (also
exported through the Tau alias package). A standalone agent accepts
`AgentOptions(before_model_invocation=callback)`; an extension registers
`pi.on("before_model_invocation", handler)`.

The standalone callback receives a `ModelInvocation`. Extension handlers receive
`event["invocation"]` plus the normal extension context. The invocation contains:

- `model`: configured model, including its provider and API adapter;
- `reasoning`: configured effort, or `None`;
- `context`: the transformed provider-neutral system prompt, messages and tools;
- `session_id`: session identifier when supplied by the owner.

The snapshot is isolated from the active context. Mutating it does not rewrite
the request. Return `None` to retain the assignment, or return a
`ModelInvocationSelection(model=resolved_model, reasoning=effort)`; both fields
are required and `reasoning=None` explicitly disables reasoning. Sync and async
callbacks are supported. Extension handlers can also return the equivalent dict.
Handlers run in registration order; later handlers see earlier selections.

A selection applies only to this invocation. It does not change `/model`, the
session default, or future calls. A router must return its decision for each call.
Tau resolves credentials for the selected provider and will not carry a static
API key from the configured provider to a different provider. Handler exceptions
and malformed results stop dispatch and surface through the existing session
error state. They do not silently choose another tier.

The hook performs structural validation, not configuration eligibility checks.
Routing policy, metadata extraction, and compatibility validation are owned by
the configuration/router layer. No model classifier is called by this hook.

`tau.model_invocation` instrumentation records configured and selected
provider/model/reasoning plus whether a handler returned an assignment. Existing
provider request/response instrumentation remains downstream of dispatch.

## Configure a routing level

Use `/set` and select **Router tier mapping**, or enter:

```text
/set router ultra-light
/set router light
/set router default
/set router max
```

Each opens provider/model and reasoning inputs. Tab completion offers the router
levels. The direct form is also supported:

```text
/set router ultra-light openai/gpt-5.6-luna low
/set router light openai/gpt-5.6-luna high
/set router default openai/gpt-5.6-sol high
/set router max openai/gpt-6-astra medium
```

Use `off` to store null reasoning. The four mappings above are editor suggestions
from the routing sketch, not activated routing policy. The command configures a
level's assignment; it does not force the session to that tier or enable routing.

Assignments are stored under `router.levels` in the same `get_models_path()` file
used by existing `/set` provider mappings: normally `~/.tau/agent/models.json`,
with the existing agent-directory environment override honored. Other mappings
are preserved, and cancelled/invalid edits do not write. Example:

```json
{
  "router": {
    "levels": {
      "light": {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "reasoning": "high"
      }
    }
  }
}
```

`load_router_selections(path, registry)` resolves the configured mappings once
into `ModelInvocationSelection` values. Missing models and reasoning enabled on
a non-reasoning model are rejected during configuration. Custom effort strings
remain supported as in Tau's existing thinking controls; providers can impose
additional constraints. A router configuration owner must validate its complete
policy, including required tiers and provider-specific capabilities, before use.
Only explicitly saved levels are returned; missing levels are not filled silently.
The owner retains these resolved selections and returns the chosen entry from its
hook. Configuration changes require the owner to reload at a configuration boundary.

## Verification and remaining work

Focused tests exercise the extension through `AgentSession.prompt`, selection
before credential resolution, consecutive provider changes around Tau's disk
read tool, unchanged session defaults on later turns, explicit reasoning-off,
handler failures, static-key scoping, persisted mappings and reloads, and the
`/set` handler/menu/completion. The provider boundary is a test double; no paid
model call or terminal UI automation was used. The saved configuration also drives
a standalone Agent's provider boundary through the hook.

28 selected tests passed across the router configuration, SDK hook, existing
provider hooks, existing `/set` commands, and static-key boundary tests. During a
broader loop check, two steering tests failed; both reproduce using the unchanged
`HEAD` version of `agent_loop.py`:

- `test_agent_loop_injects_steering_after_next_tool_call_completes`
- `test_agent_loop_injects_steering_mid_parallel_batch`

The initial new tests also exposed test-fixture assumptions: the chosen OpenAI
catalog ID was absent, a non-reasoning baseline clamped effort to off, and prompt
failures are recorded in session state rather than raised. Direct catalog output
and failed assertions established these causes; fixtures now use explicit test
model descriptors and assert session errors plus zero provider dispatch. Tau's
ThinkingLevel is an extensible string, so malformed-selection testing omits the
required reasoning field rather than asserting an unsupported enum restriction.

Typed continuation metadata transport and the deterministic routing matrix still
need to be connected to this hook. Routing quality, paid-provider switching, and
savings remain unproven. This source change does not publish or install a release.
