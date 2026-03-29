"""Grounding scorer: F_β on (doc_id, page_number) pairs."""

from __future__ import annotations

from src.evaluation.schemas import PageRef


def compute_grounding(
    predicted_pages: list[PageRef],
    gold_pages: list[PageRef],
    beta: float = 2.5,
) -> tuple[float, float, float]:
    """Compute grounding precision, recall, and F_β.

    Returns (precision, recall, f_beta).
    Both empty → (1.0, 1.0, 1.0). One empty → (0.0, 0.0, 0.0).
    """
    pred_set = {(p.doc_id, p.page_number) for p in predicted_pages}
    gold_set = {(p.doc_id, p.page_number) for p in gold_pages}

    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0

    intersection = pred_set & gold_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gold_set)

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

    return precision, recall, f_beta


def expand_gold_retrieval(gold_retrieval: list[dict]) -> list[PageRef]:
    """Expand goldset gold_retrieval field into flat list of PageRef.

    Input: [{"doc_id": "hash", "page_numbers": [1, 2]}, ...]
    Output: [PageRef("hash", 1), PageRef("hash", 2), ...]
    """
    pages: list[PageRef] = []
    for entry in gold_retrieval:
        doc_id = entry.get("doc_id", "")
        for page_num in entry.get("page_numbers", []):
            pages.append(PageRef(doc_id=doc_id, page_number=page_num))
    return pages
