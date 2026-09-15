"""PII detection — built-in high-confidence regex recognizers plus optional
Presidio NER (lazy-imported, never a hard dependency).

Ported from the Devin prototype; the detection logic is unchanged.
"""

from __future__ import annotations

import re
from typing import Any

TOKEN_PREFIX = "PII"
# e.g. [PII:EMAIL:1] — bracketed + colon-delimited so it round-trips through the
# model verbatim and is trivial to detect/reverse with an exact string match.
#
# Older/dev transcripts may contain [PII:EMAIL=1]. Accept that shape on read so
# tool-call detokenization does not leak a shell-unsafe bracket token into bash.
TOKEN_RE = re.compile(r"\[" + re.escape(TOKEN_PREFIX) + r":[A-Z_]+(?::|=)\d+\]")


def make_token(label: str, n: int) -> str:
    return f"[{TOKEN_PREFIX}:{label}:{n}]"


# Map detector entity-types → short token labels.
_LABELS = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CC",
    "US_SSN": "SSN",
    "IP_ADDRESS": "IP",
    "IBAN_CODE": "IBAN",
    "PERSON": "NAME",
    "LOCATION": "LOCATION",
    "US_BANK_NUMBER": "BANK",
    "US_DRIVER_LICENSE": "DL",
    "US_PASSPORT": "PASSPORT",
    "CRYPTO": "CRYPTO",
    "MEDICAL_LICENSE": "MEDLIC",
    "AWS_KEY": "AWS_KEY",
    # Weak, phone-shaped numerics that lack strong phone signals (no country
    # code, no parenthesized area code, <10 digits). Masked generically so a
    # tracking id / local extension isn't asserted to be a real phone number.
    "AMBIGUOUS_NUMERIC": "REDACTED",
}


def label_for(entity_type: str) -> str:
    return _LABELS.get(entity_type, re.sub(r"[^A-Z0-9_]", "_", entity_type.upper()))


# ---- built-in regex recognizers (high-confidence, structured PII) ---------- #

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA|AIDA|AGPA|ANPA|AROA)[A-Z0-9]{16}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{2,4}[\s.\-]\d{2,4}[\s.\-]?\d{2,4}(?!\w)"
)
_CC_RE = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")

# Dates and timestamps the phone regex would otherwise misfire on. Dates are
# not PII: `2026-07-25` and `2026-07-25-12-04-51` / `2026-07-25T12:04:51` are
# calendar values, not phone numbers. Tokenizing them corrupts file paths and
# downstream reasoning, so any phone-shaped match that contains one of these is
# excluded from PHONE detection entirely.
_DATETIME_RES = (
    # YYYY-MM-DD optionally followed by a [T|-|space]HH:MM[:SS] time part.
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T\-\s]\d{2}:\d{2}(?::\d{2})?)?\b"),
    # YYYY-MM-DD-HH-MM-SS (all dash-separated, e.g. artifact filename stamps).
    re.compile(r"\b\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\b"),
)


def _looks_like_datetime(s: str) -> bool:
    return any(rx.search(s) for rx in _DATETIME_RES)


def _is_strong_phone(val: str) -> bool:
    """A match is a *real* phone only with strong signal: an international
    prefix (``+``), a parenthesized area code, or 10+ digits. Anything weaker
    (7-9 bare digits, no formatting) is treated as an ambiguous numeric."""
    if "+" in val:
        return True
    if "(" in val and ")" in val:
        return True
    return sum(c.isdigit() for c in val) >= 10


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _regex_detect(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for m in _EMAIL_RE.finditer(text):
        found.append((m.group(0), "EMAIL_ADDRESS"))
    for m in _SSN_RE.finditer(text):
        found.append((m.group(0), "US_SSN"))
    for m in _IPV4_RE.finditer(text):
        found.append((m.group(0), "IP_ADDRESS"))
    for m in _IBAN_RE.finditer(text):
        found.append((m.group(0), "IBAN_CODE"))
    for m in _AWS_KEY_RE.finditer(text):
        found.append((m.group(0), "AWS_KEY"))
    for m in _CC_RE.finditer(text):
        val = m.group(0).strip()
        # A dash-separated timestamp (e.g. 2026-07-25-12-04-51) can have 13-19
        # digits and coincidentally pass Luhn — it is a date, not a card number.
        if _looks_like_datetime(val):
            continue
        if _luhn_ok(val):
            found.append((val, "CREDIT_CARD"))
    # Match phones with complete dates masked out first. Otherwise a value at
    # the end of a log line can join the start of the next date (503\n2026-09),
    # yielding a partial match that _looks_like_datetime cannot recognize.
    # Word characters block numeric matches while preserving genuine adjacent
    # phones, including phone numbers wrapped across lines.
    phone_text = text
    for rx in reversed(_DATETIME_RES):  # Mask longer all-dash timestamps first.
        phone_text = rx.sub(lambda match: "x" * len(match.group(0)), phone_text)
    for m in _PHONE_RE.finditer(phone_text):
        val = m.group(0).strip()
        if sum(c.isdigit() for c in val) < 7:
            continue
        # Dates/timestamps are not phone numbers — never tokenize them.
        if _looks_like_datetime(val):
            continue
        if _is_strong_phone(val):
            found.append((val, "PHONE_NUMBER"))
        else:
            found.append((val, "AMBIGUOUS_NUMERIC"))
    return found


# ---- optional Presidio NER detector (lazy) -------------------------------- #

_presidio_analyzer: Any = None
_presidio_tried = False


def get_presidio() -> Any:
    global _presidio_analyzer, _presidio_tried
    if _presidio_tried:
        return _presidio_analyzer
    _presidio_tried = True
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore

        _presidio_analyzer = AnalyzerEngine()
    except Exception:
        _presidio_analyzer = None
    return _presidio_analyzer


def _presidio_detect(text: str) -> list[tuple[str, str]]:
    engine = get_presidio()
    if engine is None:
        return []
    try:
        results = engine.analyze(text=text, language="en")
    except Exception:
        return []
    out: list[tuple[str, str]] = []
    for r in results:
        try:
            if getattr(r, "score", 1.0) < 0.5:
                continue
            out.append((text[r.start : r.end], r.entity_type))
        except Exception:
            continue
    return out


def detect(text: str) -> list[tuple[str, str]]:
    """Detect PII values in text. Returns [(value, entity_type), …].

    Combines Presidio (when available) with built-in regex recognizers, drops
    values that are substrings of a longer detected value, and never matches our
    own tokens.
    """
    if not text:
        return []
    raw = _regex_detect(text) + _presidio_detect(text)
    by_value: dict[str, str] = {}
    for value, etype in raw:
        value = value.strip()
        if not value or TOKEN_RE.fullmatch(value):
            continue
        by_value.setdefault(value, etype)
    values = sorted(by_value, key=len, reverse=True)
    kept: list[str] = []
    for v in values:
        if any(v != k and v in k for k in kept):
            continue
        kept.append(v)
    return [(v, by_value[v]) for v in kept]
