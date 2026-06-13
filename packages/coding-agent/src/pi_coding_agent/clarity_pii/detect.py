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
TOKEN_RE = re.compile(r"\[" + re.escape(TOKEN_PREFIX) + r":[A-Z_]+:\d+\]")


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
        if _luhn_ok(m.group(0)):
            found.append((m.group(0).strip(), "CREDIT_CARD"))
    for m in _PHONE_RE.finditer(text):
        val = m.group(0).strip()
        if sum(c.isdigit() for c in val) >= 7:
            found.append((val, "PHONE_NUMBER"))
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
