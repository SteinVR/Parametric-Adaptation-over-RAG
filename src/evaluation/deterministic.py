"""Deterministic per-type scorers for S_det computation."""

from __future__ import annotations

from typing import Any

from src.generation.parser import MALFORMED


def score_deterministic(
    predicted: Any,
    gold: Any,
    answer_type: str,
    is_unanswerable: bool,
) -> float:
    """Dispatch to type-specific scorer. Returns float in [0, 1].

    Malformed predictions always score 0.
    Unanswerable (gold is None) takes priority over answer_type.
    """
    if predicted == MALFORMED:
        return 0.0

    if is_unanswerable:
        return _score_unanswerable(predicted)

    scorer = _SCORERS.get(answer_type)
    if scorer is None:
        return 0.0
    return scorer(predicted, gold)


def _score_unanswerable(predicted: Any) -> float:
    """Gold is null → expected response is []. Both [] → 1.0, else 0.0."""
    if isinstance(predicted, list) and len(predicted) == 0:
        return 1.0
    return 0.0


def _score_boolean(predicted: Any, gold: Any) -> float:
    """Exact match on boolean value."""
    if isinstance(predicted, bool) and isinstance(gold, bool):
        return 1.0 if predicted == gold else 0.0
    # Try coercing strings
    pred_str = str(predicted).lower().strip()
    gold_str = str(gold).lower().strip()
    return 1.0 if pred_str == gold_str else 0.0


def _score_number(predicted: Any, gold: Any, tolerance: float = 0.01) -> float:
    """Numeric match with ±tolerance (default 1%)."""
    try:
        pred_num = float(predicted)
        gold_num = float(gold)
    except (TypeError, ValueError):
        return 0.0

    if gold_num == 0:
        return 1.0 if pred_num == 0 else 0.0
    return 1.0 if abs(pred_num - gold_num) / abs(gold_num) <= tolerance else 0.0


def _score_name(predicted: Any, gold: Any) -> float:
    """Normalized exact string match (lowercased, stripped)."""
    pred_norm = str(predicted).lower().strip()
    gold_norm = str(gold).lower().strip()
    return 1.0 if pred_norm == gold_norm else 0.0


def _score_names(predicted: Any, gold: Any) -> float:
    """Jaccard similarity over normalized string sets."""
    pred_set = _normalize_name_set(predicted)
    gold_set = _normalize_name_set(gold)

    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0

    intersection = pred_set & gold_set
    union = pred_set | gold_set
    return len(intersection) / len(union)


def _score_date(predicted: Any, gold: Any) -> float:
    """Exact ISO 8601 match (YYYY-MM-DD)."""
    return 1.0 if str(predicted).strip() == str(gold).strip() else 0.0


def _normalize_name_set(value: Any) -> set[str]:
    """Convert value to a normalized set of strings."""
    if isinstance(value, list):
        return {str(v).lower().strip() for v in value if str(v).strip()}
    if isinstance(value, str):
        return {value.lower().strip()} if value.strip() else set()
    return set()


_SCORERS = {
    "boolean": _score_boolean,
    "number": _score_number,
    "name": _score_name,
    "names": _score_names,
    "date": _score_date,
}
