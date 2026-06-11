"""JSON convenience helpers."""
from __future__ import annotations


def strip_json_comments(input_text: str) -> str:
    """Strip // comments and trailing commas, preserving string literals."""
    result: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(input_text):
        char = input_text[i]
        next_char = input_text[i + 1] if i + 1 < len(input_text) else ""
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue
        if char == "/" and next_char == "/":
            while i < len(input_text) and input_text[i] != "\n":
                i += 1
            continue
        if char == ",":
            j = i + 1
            while j < len(input_text) and input_text[j].isspace():
                j += 1
            if j < len(input_text) and input_text[j] in "}]":
                i += 1
                continue
        result.append(char)
        i += 1
    return "".join(result)


__all__ = ["strip_json_comments"]
