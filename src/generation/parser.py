"""Answer parser: extract typed values from raw model output.

Shared by all experiments. Parser failure → MALFORMED marker (scored as 0).
"""

from __future__ import annotations

import json
import re
from typing import Any

MALFORMED = "_malformed_"

# Regex for dates
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Regex for numbers (int or float, possibly negative)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_answer(raw_output: str, answer_type: str) -> tuple[Any, bool]:
    """Parse model output per answer_type.

    Returns (parsed_value, is_malformed).
    Detects [] abstention first — valid for any type, not malformed.
    """
    stripped = raw_output.strip()

    # Detect [] abstention (valid signal, not malformed)
    if stripped == "[]":
        return [], False

    parser = _PARSERS.get(answer_type)
    if parser is None:
        return MALFORMED, True
    return parser(stripped)


def _parse_boolean(raw: str) -> tuple[Any, bool]:
    lowered = raw.lower()
    if "true" in lowered:
        return True, False
    if "false" in lowered:
        return False, False
    return MALFORMED, True


def _parse_number(raw: str) -> tuple[Any, bool]:
    # Remove common formatting: commas, currency symbols
    cleaned = raw.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    match = _NUMBER_RE.search(cleaned)
    if match:
        val = match.group()
        return float(val) if "." in val else int(val), False
    return MALFORMED, True


def _parse_name(raw: str) -> tuple[Any, bool]:
    # Take first non-empty line, strip
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line, False
    return MALFORMED, True


def _parse_names(raw: str) -> tuple[Any, bool]:
    # Try JSON array first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(n).strip() for n in parsed if str(n).strip()], False
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find a JSON array embedded in the text
    bracket_start = raw.find("[")
    bracket_end = raw.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        try:
            parsed = json.loads(raw[bracket_start : bracket_end + 1])
            if isinstance(parsed, list):
                return [str(n).strip() for n in parsed if str(n).strip()], False
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: split by commas or newlines
    parts = re.split(r"[,\n]+", raw)
    names = [p.strip().strip('"').strip("'") for p in parts if p.strip()]
    if names:
        return names, False
    return MALFORMED, True


def _parse_date(raw: str) -> tuple[Any, bool]:
    match = _DATE_RE.search(raw)
    if match:
        return match.group(), False
    return MALFORMED, True


def _parse_free_text(raw: str) -> tuple[Any, bool]:
    # Take full output, truncate at 280 chars
    text = raw.strip()
    if not text:
        return MALFORMED, True
    return text[:280], False


_PARSERS = {
    "boolean": _parse_boolean,
    "number": _parse_number,
    "name": _parse_name,
    "names": _parse_names,
    "date": _parse_date,
    "free_text": _parse_free_text,
}
