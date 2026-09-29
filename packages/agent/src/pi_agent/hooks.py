"""Lifecycle hook dispatch for tau agents.

Why this exists rather than APPEND_SYSTEM.md
--------------------------------------------
A system prompt is read once at session start and then sits behind everything
that follows. By turn forty it is thousands of tokens back, competing with the
whole conversation. Instructions about how to write the NEXT message need to be
adjacent to that message.

Claude Code and Codex both solve this with a UserPromptSubmit hook that injects
fresh context immediately before generation, every turn. Measured on Claude Code
that mechanism visibly changed output; the system-prompt route was never tested
because the hook was available.

tau's equivalent position is on_payload, which every provider calls via
apply_on_payload() at the last moment before the request leaves — after context
assembly, after compaction, after tool results. This module turns that callback
into a hook system with the same hooks.json contract the other two harnesses
use, so hook scripts port between them unchanged.

Config, first match wins:
    <cwd>/.tau/hooks.json
    ~/.tau/hooks.json

    {"hooks": {
       "PreGeneration": [{"hooks": [{"type": "command",
                                     "command": "/path/to/inject.sh",
                                     "timeout": 10}]}],
       "BeforeFinalOutput": [{"hooks": [{"type": "command",
                                          "command": "/path/to/finalize.py",
                                          "timeout": 15}]}],
       "Stop":          [{"hooks": [{"type": "command",
                                     "command": "/path/to/audit.py",
                                     "timeout": 15}]}]}}

A PreGeneration hook prints JSON on stdout; its
hookSpecificOutput.additionalContext is appended to the final user message of
the outgoing payload — the same semantic as UserPromptSubmit.

A BeforeFinalOutput hook runs in the core agent loop after a tool-free response
has finished generating but before any part of it is emitted or persisted. Its
``replacementText`` replaces the candidate, while ``appendText`` provides a
minimal way to prove the boundary with a sentinel word.

A TurnEnd hook runs after the working response has no tool calls. Its
``additionalContext`` is added to the system prompt for one tool-disabled
finalization call, and ``replacementPrompt`` becomes the internal follow-up
request for that call. This is the hook for output-only controls that must not
steer tool selection or working turns.

A Stop hook receives the message that will actually be delivered. Its output
is discarded; it is for measuring, and it must not be able to block a turn.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

_MAX_OUTPUT = 256 * 1024
_FINAL_OUTPUT_PROBE = "final_output_probe"
_FINAL_OUTPUT_PROBE_TEXT = (
    "PRUEBA: el borrador interno fue reemplazado. TURN_END_HOOK_FIRED"
)


def _config_path(cwd: str | None = None) -> Path | None:
    for p in (Path(cwd or os.getcwd()) / ".tau" / "hooks.json",
              Path.home() / ".tau" / "hooks.json"):
        if p.is_file():
            return p
    return None


def load_hooks(event: str, cwd: str | None = None) -> list[dict]:
    path = _config_path(cwd)
    if not path:
        return []
    try:
        cfg = json.loads(path.read_text())
    except Exception:
        return []
    out = []
    for group in (cfg.get("hooks") or {}).get(event, []) or []:
        for hook in group.get("hooks", []) or []:
            if hook.get("type") == "command" and hook.get("command"):
                out.append(hook)
            elif (
                hook.get("type") == "builtin"
                and hook.get("name") == _FINAL_OUTPUT_PROBE
            ):
                out.append(hook)
    return out


def run_hook(hook: dict, payload: dict) -> dict | None:
    """Run one command hook. Never raises; a broken hook must not break a turn."""
    if hook.get("type") == "builtin":
        if hook.get("name") != _FINAL_OUTPUT_PROBE:
            return None
        if payload.get("hook_event_name") != "BeforeFinalOutput":
            return None
        return {
            "hookSpecificOutput": {
                "replacementText": _FINAL_OUTPUT_PROBE_TEXT,
            }
        }
    try:
        proc = subprocess.run(
            hook["command"], shell=True, input=json.dumps(payload),
            capture_output=True, text=True, timeout=hook.get("timeout", 15),
        )
    except Exception:
        return None
    out = (proc.stdout or "")[:_MAX_OUTPUT].strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def gather_context(session_id: str | None, cwd: str | None = None) -> str:
    """Run PreGeneration hooks and concatenate their additionalContext."""
    parts = []
    for hook in load_hooks("PreGeneration", cwd):
        res = run_hook(hook, {"hook_event_name": "PreGeneration",
                              "session_id": session_id or "",
                              "cwd": cwd or os.getcwd()})
        if not isinstance(res, dict):
            continue
        ctx = (res.get("hookSpecificOutput") or {}).get("additionalContext")
        if isinstance(ctx, str) and ctx.strip():
            parts.append(ctx.strip())
    return "\n\n".join(parts)


def gather_turn_end(
    message: Any,
    session_id: str | None,
    cwd: str | None = None,
) -> dict[str, str] | None:
    """Return one finalization request assembled from configured TurnEnd hooks."""
    text = _extract_text(message).strip()
    if not text:
        return None
    contexts: list[str] = []
    replacement_prompt = ""
    for hook in load_hooks("TurnEnd", cwd):
        res = run_hook(
            hook,
            {
                "hook_event_name": "TurnEnd",
                "session_id": session_id or "",
                "cwd": cwd or os.getcwd(),
                "last_assistant_message": text,
            },
        )
        if not isinstance(res, dict):
            continue
        specific = res.get("hookSpecificOutput") or {}
        if not isinstance(specific, dict):
            continue
        context = specific.get("additionalContext")
        if isinstance(context, str) and context.strip():
            contexts.append(context.strip())
        prompt = specific.get("replacementPrompt")
        if isinstance(prompt, str) and prompt.strip():
            replacement_prompt = prompt.strip()
    if not contexts:
        return None
    if not replacement_prompt:
        replacement_prompt = (
            "Rewrite your previous answer for delivery using the finalization "
            "instructions. Preserve every fact, decision, refusal, evidence "
            "limit, name, date, amount, and reference. Return only the "
            "replacement answer."
        )
    return {
        "additional_context": "\n\n".join(contexts),
        "replacement_prompt": (
            f"{replacement_prompt}\n\n"
            "<draft_to_finalize>\n"
            f"{text}\n"
            "</draft_to_finalize>"
        ),
    }


def apply_before_final_output(
    message: Any,
    session_id: str | None,
    cwd: str | None = None,
) -> Any | None:
    """Apply the native, pre-emission final-output hook to one candidate.

    Returning ``None`` means no configured hook replaced the candidate. Hook
    failures remain fail-open, matching the other command lifecycle hooks.
    """
    text = _extract_text(message).strip()
    if not text:
        return None
    replacement: str | None = None
    for hook in load_hooks("BeforeFinalOutput", cwd):
        result = run_hook(
            hook,
            {
                "hook_event_name": "BeforeFinalOutput",
                "session_id": session_id or "",
                "cwd": cwd or os.getcwd(),
                "last_assistant_message": text,
            },
        )
        if not isinstance(result, dict):
            continue
        specific = result.get("hookSpecificOutput") or {}
        if not isinstance(specific, dict):
            continue
        replacement_text = specific.get("replacementText")
        append_text = specific.get("appendText")
        if isinstance(replacement_text, str) and replacement_text.strip():
            replacement = replacement_text.strip()
        elif isinstance(append_text, str) and append_text.strip():
            replacement = f"{replacement or text} {append_text.strip()}"
    if replacement is None:
        return None
    return _replace_text_content(message, replacement)


def _replace_text_content(message: Any, replacement: str) -> Any:
    """Replace delivered text while preserving non-text response metadata."""
    from pi_ai.types import TextContent

    content = getattr(message, "content", None)
    if not isinstance(content, list) or not hasattr(message, "model_copy"):
        return message
    non_text = [
        block
        for block in content
        if not (
            isinstance(block, TextContent)
            or (isinstance(block, dict) and block.get("type") == "text")
        )
    ]
    return message.model_copy(
        update={"content": [*non_text, TextContent(type="text", text=replacement)]}
    )


def _append_to_last_user_message(params: Any, text: str) -> bool:
    """Attach text to the final user turn, mirroring UserPromptSubmit semantics.

    Returns False when the payload shape is not recognised, so the caller can
    leave the request untouched rather than corrupting it.
    """
    if not isinstance(params, dict):
        return False
    for key in ("messages", "input", "contents"):
        msgs = params.get(key)
        if not isinstance(msgs, list) or not msgs:
            continue
        for msg in reversed(msgs):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") not in ("user", None):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = f"{content}\n\n{text}"
                return True
            if isinstance(content, list):
                item_type = "input_text" if key == "input" else "text"
                content.append({"type": item_type, "text": text})
                return True
            if isinstance(msg.get("parts"), list):      # google shape
                msg["parts"].append({"text": text})
                return True
        return False
    return False


def _is_tool_result(msg: Any) -> bool:
    c = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(c, list):
        return any(isinstance(b, dict) and b.get("type") in
                   ("tool_result", "function_call_output", "tool_call_output")
                   for b in c)
    return False


def _starts_a_turn(payload: Any) -> bool:
    """True only on the FIRST model call of a turn.

    tau calls on_payload for every generation, including the ones that only pick
    the next tool. Claude Code's UserPromptSubmit fires once per user prompt, and
    firing per generation is measurably worse: the model reads "answer now" while
    still mid-search and narrates its progress instead of working.
    """
    if not isinstance(payload, dict):
        return False
    for key in ("messages", "input", "contents"):
        msgs = payload.get(key)
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if not isinstance(last, dict):
                return False
            return last.get("role") in ("user", None) and not _is_tool_result(last)
    return False


def _already_injected(payload: Any, marker: str) -> bool:
    try:
        import json as _json
        return marker in _json.dumps(payload, default=str)
    except Exception:
        return False


def inject_pregeneration(payload: Any, session_id: str | None = None,
                         cwd: str | None = None) -> Any:
    """Append PreGeneration context, once per turn. Never raises.

    Two guards, both learned the hard way. Fire only when the payload starts a
    turn, matching UserPromptSubmit. And refuse to append when the context is
    already present: tau reuses the same message dicts across a turn, so an
    unguarded append stacked one copy per model call -- 28 tool calls meant 28
    copies of the instructions.
    """
    try:
        if not _starts_a_turn(payload):
            return payload
        text = gather_context(session_id, cwd)
        if not text:
            return payload
        marker = text.strip().split("\n", 1)[0][:40]
        if marker and _already_injected(payload, marker):
            return payload
        _append_to_last_user_message(payload, text)
    except Exception:
        pass
    return payload


def report_stop(response: Any, session_id: str | None = None,
                cwd: str | None = None) -> None:
    """Hand the finished message to Stop hooks. Observing only; cannot block."""
    try:
        text = _extract_text(response)
        if not text.strip():
            return
        for hook in load_hooks("Stop", cwd):
            run_hook(hook, {"hook_event_name": "Stop",
                            "session_id": session_id or "",
                            "cwd": cwd or os.getcwd(),
                            "last_assistant_message": text})
    except Exception:
        pass


def _extract_text(response: Any) -> str:
    final = response.get("final_message") if isinstance(response, dict) else None
    if final is None:
        final = response
    content = getattr(final, "content", None)
    if content is None and isinstance(final, dict):
        content = final.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            (b.get("text") or "") if isinstance(b, dict) else (getattr(b, "text", "") or "")
            for b in content)
    return ""


def make_on_payload(session_id: str | None = None, cwd: str | None = None,
                    chain: Callable | None = None) -> Callable:
    """on_payload that injects PreGeneration context, then defers to `chain`.

    Chaining matters: an agent that already supplies on_payload keeps it.
    """
    async def _on_payload(params: Any, model: Any):
        text = gather_context(session_id, cwd)
        if text:
            _append_to_last_user_message(params, text)
        if chain is not None:
            result = chain(params, model)
            if hasattr(result, "__await__"):
                result = await result
            if result is not None:
                return result
        return params
    return _on_payload


def make_on_response(session_id: str | None = None, cwd: str | None = None,
                     chain: Callable | None = None) -> Callable:
    """on_response that reports the finished message to Stop hooks."""
    async def _on_response(response: Any, model: Any):
        try:
            final = None
            if isinstance(response, dict):
                final = response.get("final_message")
            text = ""
            if final is not None:
                content = getattr(final, "content", None)
                if content is None and isinstance(final, dict):
                    content = final.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "".join(
                        (b.get("text") or "") if isinstance(b, dict)
                        else (getattr(b, "text", "") or "")
                        for b in content)
            if text.strip():
                for hook in load_hooks("Stop", cwd):
                    run_hook(hook, {"hook_event_name": "Stop",
                                    "session_id": session_id or "",
                                    "cwd": cwd or os.getcwd(),
                                    "last_assistant_message": text})
        except Exception:
            pass
        if chain is not None:
            result = chain(response, model)
            if hasattr(result, "__await__"):
                await result
    return _on_response
