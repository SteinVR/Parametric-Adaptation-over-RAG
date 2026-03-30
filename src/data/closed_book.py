"""Closed-book dataset generation for EXP-003b."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data.io import load_goldset, load_json
from src.data.raft import RaftExample, serialize_training_answer
from src.generation.prompt import format_closed_book_prompt

logger = logging.getLogger(__name__)


def build_closed_book_examples(
    *,
    goldset_path: Path,
    split_path: Path,
) -> list[RaftExample]:
    """Generate closed-book training examples from the S2 train split.

    Each example: closed-book prompt (question + answer_type) → answer.
    No context, no gold chunks, no distractors.
    Reuses RaftExample type for training code compatibility.
    """
    refs = load_goldset(goldset_path)
    refs_by_id = {ref["question_id"]: ref for ref in refs}
    split = load_json(split_path)
    train_ids = [str(qid) for qid in split["s2_train"]]

    examples: list[RaftExample] = []
    for question_id in train_ids:
        ref = refs_by_id[question_id]
        prompt = format_closed_book_prompt(
            question=str(ref["question"]),
            answer_type=str(ref["answer_type"]),
        )
        examples.append(
            RaftExample(
                question_id=question_id,
                answer_type=str(ref["answer_type"]),
                prompt=prompt,
                answer=serialize_training_answer(ref["answer"]),
                gold_pages=[],
                distractor_pages=[],
            )
        )

    if len(examples) != len(train_ids):
        raise ValueError(f"Expected {len(train_ids)} examples, built {len(examples)}")
    logger.info("Built %d closed-book examples", len(examples))
    return examples


def save_closed_book_jsonl(examples: list[RaftExample], output_path: Path) -> None:
    """Write closed-book examples as newline-delimited JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")
    logger.info("Saved closed-book dataset to %s", output_path)
