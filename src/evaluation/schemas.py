"""Pydantic schemas for predictions and evaluation results.

These types define the contract between experiment systems and the eval framework.
Every experiment (S1-S6) produces list[Prediction]; the eval runner scores them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PageRef(BaseModel):
    """A (doc_id, page_number) pair for grounding evaluation."""

    doc_id: str
    page_number: int


class Prediction(BaseModel):
    """Standardized output from any experiment system.

    All experiments produce a list of these. The eval module scores them
    without knowing which system generated them.
    """

    question_id: str
    raw_output: str
    parsed_answer: Any  # bool | float | str | list[str] | None
    answer_type: str
    is_malformed: bool = False
    predicted_pages: list[PageRef] = Field(default_factory=list)
    ttft_ms: float | None = None
    latency_ms: float | None = None


class QuestionScore(BaseModel):
    """Per-question evaluation result."""

    question_id: str
    answer_type: str
    difficulty: str
    is_unanswerable: bool

    # Deterministic (None for free_text)
    s_det: float | None = None

    # LLM judge (None for non-free_text)
    s_asst: float | None = None
    judge_criteria: dict[str, int] | None = None

    # Grounding (None when not applicable)
    grounding_precision: float | None = None
    grounding_recall: float | None = None
    grounding_f_beta: float | None = None

    is_malformed: bool = False


class EvalReport(BaseModel):
    """Complete evaluation output for one experiment run."""

    system_id: str
    experiment_id: str

    # Per-question
    question_scores: list[QuestionScore]

    # Aggregate
    q_main: float
    s_det: float
    s_asst: float
    grounding_f_beta: float | None = None

    # Breakdown: {answer_type: {metric_name: value}}
    breakdown_by_type: dict[str, dict[str, float]]

    # Systems metrics aggregate
    ttft_median_ms: float | None = None
    ttft_p95_ms: float | None = None
    latency_median_ms: float | None = None
    latency_p95_ms: float | None = None
    malformed_rate: float = 0.0
