"""Opt-in invocation routing with a strict continuation-metadata carrier."""
from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
from pi_ai.types import EventDone, EventError, EventStart, TextContent, Tool, ToolCall

from .router_config import ROUTER_LEVELS, load_router_selections

CARRIER_NAME = "submit_response"
SUPPORTED_APIS = {"openai-responses", "openai-codex-responses"}
TEMPLATE = json.loads(Path(__file__).with_name("router_response.schema.json").read_text())


def derived_coupling(probe):
    if probe is None or not (probe["a_unresolved"] and probe["b_unresolved"]):
        return "independent"
    if probe["a_affects_b"] and probe["b_affects_a"]:
        return "tightly_coupled"
    if probe["a_affects_b"] or probe["b_affects_a"]:
        return "interacting"
    return "independent"


def routing_level(metadata, probe):
    """Frozen sketch policy; only this function owns metadata-to-tier mapping."""
    if metadata is None:
        return "default", "initial_invocation"
    coupling = derived_coupling(probe)
    if metadata["work_kind"] == "plan":
        return "max", "planning"
    if metadata["interpretation_required"] == "competing_explanations":
        return "max", "competing_explanations"
    if coupling == "tightly_coupled":
        return "max", "mutual_unresolved_choices"
    if metadata["method_certainty"] == "needs_design" or metadata["interpretation_required"] == "inferential":
        return "default", "design_or_inference"
    if (metadata["method_certainty"] == "adaptation_required"
            or metadata["interpretation_required"] == "bounded" or coupling == "interacting"):
        return "light", "adaptation_interpretation_or_one_way_choices"
    return "ultra-light", "specified_direct_independent"


def _strict_parameters(schema):
    """Represent omitted optional arguments as null in the strict carrier.

    Keep the native tool schema as the execution validator; this only adapts its
    transport representation. Arbitrary-key dictionaries cannot be represented
    by this carrier and are rejected at activation/tool configuration changes.
    """
    schema = copy.deepcopy(schema)
    schema.pop("default", None)
    if schema.get("type") == "object":
        if schema.get("additionalProperties") not in (None, False):
            raise ValueError("Router tools cannot use arbitrary-key object arguments")
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})
        schema["properties"] = {
            key: _strict_parameters(value) if key in required else
            {"anyOf": [_strict_parameters(value), {"type": "null"}]}
            for key, value in properties.items()
        }
        schema["required"] = list(properties)
        schema["additionalProperties"] = False
    elif schema.get("type") == "array" and "items" in schema:
        schema["items"] = _strict_parameters(schema["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            if key != "anyOf":
                raise ValueError(f"Router tool schema does not support {key}")
            schema[key] = [_strict_parameters(value) for value in schema[key]]
    if "$defs" in schema:
        schema["$defs"] = {key: _strict_parameters(value) for key, value in schema["$defs"].items()}
    return schema


def response_schema(tools):
    schema = copy.deepcopy(TEMPLATE)
    branches = []
    for index, tool in enumerate(tools):
        try:
            parameters = _strict_parameters(tool.parameters)
        except ValueError as exc:
            raise ValueError(f"Router cannot encode tool {tool.name}: {exc}") from exc
        # Tool-local references must remain valid when nested in the carrier.
        definitions = parameters.pop("$defs", {})
        refs = {key: f"router_tool_{index}_{key}" for key in definitions}

        def relocate(value):
            if isinstance(value, dict):
                return {key: (f"#/$defs/{refs[item[8:]]}" if key == "$ref" and item.startswith("#/$defs/")
                              else relocate(item)) for key, item in value.items()}
            return [relocate(item) for item in value] if isinstance(value, list) else value

        for key, value in definitions.items():
            schema["$defs"][refs[key]] = relocate(value)
        branches.append({"type": "object", "additionalProperties": False,
                         "required": ["name", "arguments"], "properties": {
                             "name": {"type": "string", "enum": [tool.name], "description": tool.description},
                             "arguments": relocate(parameters),
                         }})
    schema["$defs"]["tool_call"] = ({"anyOf": branches} if branches else
                                    {"type": "object", "properties": {}, "required": [], "additionalProperties": False})
    if not tools:
        for branch in schema["properties"]["response"]["anyOf"]:
            branch["properties"]["tool_calls"]["maxItems"] = 0
    Draft202012Validator.check_schema(schema)
    return schema


def native_arguments(value, schema, root=None):
    """Remove transport-only omitted values recursively before native validation."""
    root = schema if root is None else root
    if "$ref" in schema and schema["$ref"].startswith("#/"):
        target = root
        for part in schema["$ref"][2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        return native_arguments(value, target, root)
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            candidate = native_arguments(value, branch, root)
            if Draft202012Validator({"$defs": root.get("$defs", {}), **branch}).is_valid(candidate):
                return candidate
    if isinstance(value, dict) and schema.get("type") == "object":
        required = set(schema.get("required", []))
        return {key: native_arguments(item, schema.get("properties", {}).get(key, {}), root)
                for key, item in value.items() if item is not None or key in required}
    if isinstance(value, list) and schema.get("type") == "array":
        return [native_arguments(item, schema.get("items", {}), root) for item in value]
    return value


class ModelRouter:
    def __init__(self, path, registry, tools, emit):
        self.selections = load_router_selections(path, registry)
        missing = set(ROUTER_LEVELS) - set(self.selections)
        if missing:
            raise ValueError("Configure router levels with /set router: " + ", ".join(sorted(missing)))
        for level, selection in self.selections.items():
            if selection.model.api not in SUPPORTED_APIS:
                raise ValueError(f"Router {level} requires a Responses API model; got {selection.model.api}")
        self.emit = emit
        self.configure_tools(tools)
        self.reset()

    def configure_tools(self, tools):
        signature = json.dumps([(t.name, t.description, t.parameters) for t in tools], sort_keys=True)
        if signature != getattr(self, "_tools_signature", None):
            schema = response_schema(tools)
            self.schema = schema
            self.validator = Draft202012Validator(schema)
            self._tools_signature = signature

    def reset(self):
        self.next_metadata = None
        self.next_probe = None

    def select(self):
        level, rule = routing_level(self.next_metadata, self.next_probe)
        selection = self.selections[level]
        self.emit("tau.router_selection", metadata={
            "level": level, "rule": rule, "model": selection.model.id,
            "provider": selection.model.provider, "reasoning": selection.reasoning,
            "next_invocation": self.next_metadata, "dependency_probe": self.next_probe,
        })
        return selection

    @staticmethod
    def prepare_payload(payload):
        payload = dict(payload)
        tools = payload.get("tools", [])
        if len(tools) != 1 or tools[0].get("name") != CARRIER_NAME:
            raise ValueError("Router carrier was replaced before provider dispatch")
        payload["tools"] = [{**tools[0], "strict": True}]
        payload["tool_choice"] = {"type": "function", "name": CARRIER_NAME}
        payload["parallel_tool_calls"] = False
        return payload

    async def stream(self, provider, model, context, options):
        self.configure_tools(context.tools or [])
        schema = self.schema
        validator = self.validator
        tool_map = {tool.name: tool for tool in context.tools or []}
        carrier_context = context.model_copy(update={"tools": [Tool(
            name=CARRIER_NAME,
            description="Return the ordinary response, requested actions and classification of the next invocation.",
            parameters=schema,
        )]})
        final = None
        async for event in provider(model, carrier_context, options):
            if event.type == "error":
                yield event
                return
            if event.type == "done":
                final = event.message
        if final is None:
            raise ValueError("Router provider completed without a response")
        try:
            carriers = [item for item in final.content if isinstance(item, ToolCall)]
            if len(carriers) != 1 or carriers[0].name != CARRIER_NAME:
                raise ValueError("Router response must contain one submit_response call")
            wire = carriers[0].arguments
            validator.validate(wire)
            response = wire["response"]
            content = [TextContent(text=wire["text"])] if wire["text"] else []
            for action in response["tool_calls"]:
                tool = tool_map[action["name"]]
                arguments = native_arguments(action["arguments"], tool.parameters)
                Draft202012Validator(tool.parameters).validate(arguments)
                content.append(ToolCall(id="call_" + uuid.uuid4().hex,
                                        name=action["name"], arguments=arguments))
            stop = "toolUse" if response["tool_calls"] else "stop"
            message = final.model_copy(update={"content": content, "stop_reason": stop})
            self.next_metadata = response["next_invocation"]
            self.next_probe = wire["dependency_probe"]
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            error = final.model_copy(update={"content": [], "stop_reason": "error", "error_message": str(exc)})
            yield EventError(reason="error", error=error)
            return
        self.emit("tau.router_metadata", metadata={"model": model.id,
                  "next_invocation": self.next_metadata, "dependency_probe": self.next_probe})
        # Buffer the transport envelope so it cannot appear in the transcript or
        # reach tool execution before structural validation has completed.
        yield EventStart(partial=message)
        yield EventDone(reason=stop, message=message)
