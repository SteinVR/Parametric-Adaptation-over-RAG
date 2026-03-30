"""Per-document deterministic sanity checks for EXP-004."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.deterministic import score_deterministic
from src.evaluation.schemas import Prediction


@dataclass(frozen=True, slots=True)
class DocSanityResult:
    """Diagnostic score for one document adapter."""

    question_count: int
    deterministic_count: int
    s_det: float
    malformed_count: int


def select_doc_train_refs(
    *,
    train_refs: list[dict[str, Any]],
    doc_id: str,
) -> list[dict[str, Any]]:
    """Select single-document S2-train references for one document."""
    selected: list[dict[str, Any]] = []
    for ref in train_refs:
        gold_docs = {entry["doc_id"] for entry in ref.get("gold_retrieval", [])}
        if gold_docs == {doc_id} and ref.get("answer_type") != "free_text":
            selected.append(ref)
    return selected


def score_deterministic_subset(
    *,
    predictions: list[Prediction],
    refs: list[dict[str, Any]],
) -> DocSanityResult:
    """Compute deterministic sanity metrics without judge calls."""
    preds_by_id = {pred.question_id: pred for pred in predictions}
    det_scores: list[float] = []
    malformed_count = 0
    for ref in refs:
        qid = str(ref["question_id"])
        pred = preds_by_id.get(qid)
        if pred is None:
            continue
        if pred.is_malformed:
            malformed_count += 1
            det_scores.append(0.0)
            continue
        is_unanswerable = ref["answer"] is None
        score = score_deterministic(
            pred.parsed_answer,
            ref["answer"],
            str(ref["answer_type"]),
            is_unanswerable,
        )
        det_scores.append(score)
    deterministic_count = len(det_scores)
    s_det = sum(det_scores) / deterministic_count if deterministic_count else 0.0
    return DocSanityResult(
        question_count=len(refs),
        deterministic_count=deterministic_count,
        s_det=s_det,
        malformed_count=malformed_count,
    )
