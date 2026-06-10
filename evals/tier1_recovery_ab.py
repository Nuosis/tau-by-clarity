#!/usr/bin/env python3
"""
Tier-1 recovery A/B — does active-compression + model-driven `expand` produce answers
non-inferior to full context, on the low-recency regime?

This is the experiment Tier-0 structurally couldn't run: it needs live model calls and
a tool loop. For chosen "recall" turns in a real Claire chat session we generate two
answers with a cheap model and have a judge compare them:

  FULL arm        : full uncompressed context up to the turn  -> answer_full
  COMPRESSED arm  : head/tail verbatim + middle as TF-IDF keyword-cue stubs, each with a
                    [ref] the model can pull back via an expand(ref) tool -> answer_comp
  JUDGE           : is answer_comp NON-INFERIOR to answer_full (same substantive facts,
                    no important omission/error)? A/B order randomised per turn.

Logs whether the model actually called expand (recovery firing) and how often. A high
non-inferiority rate => active-compression + model-driven expand is sound WITHOUT
embeddings; failures where expand was not called locate the paraphrase tail (model
didn't realise it needed a compressed block).

    python evals/tier1_recovery_ab.py evals/fixtures/clarify_chat/<id>.jsonl \
        --turns 5 --model qwen3:30b

Live LLM calls. Defaults to a LOCAL Ollama model (qwen3:30b) so customer content
never leaves the box. Point --base-url / --model elsewhere if desired.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recovery_eval import compressed_ids, keyword_cues  # noqa: E402
from tier0_context_replay import Block, lex_set, load_session  # noqa: E402


def extract_json(text: str) -> dict:
    """Robustly pull the first JSON object out of a model reply (qwen3 emits
    <think> tokens around it)."""
    if not text:
        return {}
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j <= i:
        return {}
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return {}


def make_client(base_url: str):
    import openai
    if "localhost" in base_url or "127.0.0.1" in base_url:
        return openai.OpenAI(base_url=base_url, api_key="ollama")  # local: key ignored
    key = ""
    for line in Path(".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY=") and line.split("=", 1)[1].strip():
            key = line.split("=", 1)[1].strip()
    return openai.OpenAI(base_url=base_url, api_key=key)


def to_chat(blocks: list[Block]) -> list[dict]:
    """Original-fidelity chat messages (collapse non-user roles to assistant)."""
    out = []
    for b in blocks:
        role = "user" if b.role == "user" else "assistant"
        out.append({"role": role, "content": b.text})
    return out


def to_compressed_chat(blocks: list[Block], comp: set[str],
                       cues: dict[str, set[str]]) -> list[dict]:
    out = []
    for b in blocks:
        role = "user" if b.role == "user" else "assistant"
        if b.eid in comp:
            kw = " ".join(sorted(cues.get(b.eid, set()))) or "(none)"
            out.append({"role": role,
                        "content": f"[compressed block ref={b.eid}; keywords: {kw}. "
                                   f"Call expand('{b.eid}') for the full text if needed.]"})
        else:
            out.append({"role": role, "content": b.text})
    return out


EXPAND_TOOL = [{
    "type": "function",
    "function": {
        "name": "expand",
        "description": (
            "Retrieve the FULL original text of an earlier conversation turn that has "
            "been compressed to a keyword placeholder. Call this whenever you need the "
            "exact details (names, numbers, wording, decisions) of a block shown as "
            "'[compressed block ref=<id>; keywords: ...]'. Returns the verbatim text."),
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ref": {"type": "string",
                        "description": "the block ref id exactly as shown, e.g. m42"}},
            "required": ["ref"],
        },
    },
}]

SYS_BASE = ("You are the assistant in this ongoing conversation. Respond to the user's "
            "most recent message, using the conversation so far.")
SYS_COMPRESSED = SYS_BASE + (
    " NOTE: to save space, some earlier turns have been replaced by placeholders of the "
    "form '[compressed block ref=<id>; keywords: ...]'. The full text of those turns is "
    "NOT shown. If answering accurately needs the exact details of any compressed block, "
    "you MUST call the expand(ref) tool to retrieve its full text BEFORE answering. Never "
    "guess or fabricate the contents of a compressed block.")


def answer(client, model: str, messages: list[dict], full_text: dict[str, str] | None):
    """Single completion, with an expand tool loop if full_text lookup is provided."""
    msgs = list(messages)
    tools = EXPAND_TOOL if full_text is not None else None
    expand_calls: list[str] = []
    for _ in range(6):
        # no temperature: gpt-5.x reasoning models reject non-default values
        kw = {"tools": tools, "tool_choice": "auto"} if tools else {}
        resp = client.chat.completions.create(model=model, messages=msgs, **kw)
        m = resp.choices[0].message
        if getattr(m, "tool_calls", None):
            msgs.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in m.tool_calls]})
            for tc in m.tool_calls:
                try:
                    ref = json.loads(tc.function.arguments).get("ref", "")
                except json.JSONDecodeError:
                    ref = ""
                expand_calls.append(ref)
                msgs.append({"role": "tool", "tool_call_id": tc.id,
                             "content": full_text.get(ref, f"(no block {ref})")})
            continue
        return (m.content or ""), expand_calls
    return "(tool loop exhausted)", expand_calls


def judge(client, model: str, question: str, a: str, b: str) -> dict:
    sys_p = ("You compare two assistant answers to the same user message. "
             "Decide whether answer B is NON-INFERIOR to answer A: does B convey the "
             "same substantive facts, decisions, and specifics, with no important "
             "omission or error relative to A? Minor wording differences are fine. "
             'Reply ONLY JSON: {"non_inferior": true|false, "reason": "<short>"}.')
    u = f"USER MESSAGE:\n{question}\n\n--- ANSWER A ---\n{a}\n\n--- ANSWER B ---\n{b}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": u}])
    parsed = extract_json(resp.choices[0].message.content or "")
    return parsed or {"non_inferior": None, "reason": "unparseable"}


def pick_turns(blocks: list[Block], comp_full: set[str], cues: dict[str, set[str]],
               n: int) -> list[int]:
    """User turns most likely to need compressed content: highest overlap of the user
    message with the union of compressed-block keyword cues, in the latter half."""
    comp_terms: set[str] = set()
    for eid in comp_full:
        comp_terms |= cues.get(eid, set())
    half = len(blocks) // 2
    scored = []
    for i, b in enumerate(blocks):
        if b.role == "user" and i >= half:
            scored.append((len(b.toks & comp_terms), i))
    scored.sort(reverse=True)
    return sorted(i for _, i in scored[:n])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--model", default="qwen3:30b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--head-tok", type=int, default=1500)
    ap.add_argument("--tail-tok", type=int, default=1500)
    ap.add_argument("--cue-k", type=int, default=40)
    args = ap.parse_args()

    client = make_client(args.base_url)

    blocks = load_session(args.session, per_message_turns=True)
    cues = keyword_cues(blocks, args.cue_k)
    comp_full = compressed_ids(blocks, args.head_tok, args.tail_tok)
    turns = pick_turns(blocks, comp_full, cues, args.turns)
    print(f"session {os.path.basename(args.session)}  blocks={len(blocks)}  "
          f"model={args.model}  test turns={turns}\n")

    ni = expanded = total = 0
    for idx in turns:
        prefix = blocks[:idx]
        question = blocks[idx].text
        comp = compressed_ids(prefix, args.head_tok, args.tail_tok)
        full_text = {b.eid: b.text for b in prefix if b.eid in comp}
        if not comp:
            print(f"turn {idx}: no compressed middle at this depth — skip")
            continue

        full_msgs = ([{"role": "system", "content": SYS_BASE}]
                     + to_chat(prefix) + [{"role": "user", "content": question}])
        comp_msgs = ([{"role": "system", "content": SYS_COMPRESSED}]
                     + to_compressed_chat(prefix, comp, cues)
                     + [{"role": "user", "content": question}])

        a_full, _ = answer(client, args.model, full_msgs, None)
        a_comp, calls = answer(client, args.model, comp_msgs, full_text)

        # fixed order: A = full (reference), B = compressed. The judge question
        # "is B non-inferior to A" then reads directly as "is comp non-inferior to
        # full". (Fixed order => possible position bias; noted in the writeup.)
        v = judge(client, args.model, question, a_full, a_comp)
        comp_non_inferior = v.get("non_inferior")
        total += 1
        ni += 1 if comp_non_inferior else 0
        expanded += 1 if calls else 0
        q1 = question.replace("\n", " ")[:60]
        print(f"turn {idx}: comp_non_inferior={comp_non_inferior}  "
              f"expand_calls={calls or '[]'}  q={q1!r}")
        print(f"         judge: {v.get('reason','')[:140]}")

    print(f"\nSUMMARY  turns={total}  "
          f"non-inferior={ni}/{total} ({100*ni/total if total else 0:.0f}%)  "
          f"used-expand={expanded}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
