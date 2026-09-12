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

When routing is off, the standalone callback receives a `ModelInvocation`. Extension handlers receive
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
level's assignment. Use `/model router` to enable routing afterward.

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

## Activate routing

Run `/model router`, or select **Router** from `/model`. Activation resolves all
four configured levels and checks provider API support and authentication before
setting the active state. Missing or incompatible configuration leaves the
previous mode and footer unchanged. Routing is opt-in for the current session;
new sessions start in their usual fixed-model mode.

While active, the footer replaces its model and thinking fields with exactly
`router on`. Selecting a concrete model through `/model` (including cycling)
disables routing and restores the model/thinking display. Mode changes wait until
the current response is finished. After editing tier configuration, reselect
`/model router` to load the new assignments.

The first invocation of each new user prompt uses the configured `default` tier.
Every routed generation returns a strict `submit_response` envelope containing
ordinary text, native tool requests, the next-invocation metadata, and the
unresolved-choice probe. Tau validates the envelope, converts its tool requests
back to normal tool calls, and runs them through the existing tool execution
path. Those tool results are visible to the next selected model. The working
model emits its own subsequent classification; there is no separate classifier
call. Responses are buffered until the envelope validates, so carrier JSON never
appears as user-facing response text. A malformed envelope cannot execute tools,
and its reported usage is retained on the error response.

`model_router.routing_level` owns the frozen sketch policy:

- Planning, competing explanations, or mutually constraining unresolved choices:
  `max`.
- Otherwise, method design or inferential interpretation: `default`.
- Otherwise, adaptation, bounded interpretation, or one-way unresolved choices:
  `light`.
- Otherwise: `ultra-light`.

Choice interaction is derived deterministically from the probe, rather than
trusting the redundant generated category. No `unknown` category or semantic
fallback is introduced. Null continuation is allowed only when no tools are
pending. Each invocation still follows the normal loop termination rules.
While selected, the built-in router owns assignment; extension assignment hooks
resume when a concrete model is selected.

This carrier currently supports Responses and Codex Responses adapters. Tools
must have strict-compatible parameter schemas; arbitrary-key object arguments
are rejected with the tool name at activation or a tool configuration change.
Model registry/eligibility work occurs at activation, not each invocation.
Auxiliary calls outside the main agent loop, such as compaction, are unchanged.

See [router activation verification](router-activation-verification.md) for the
source checks, terminal check and live-provider evidence. Broad routing quality
and cost savings remain unproven.
