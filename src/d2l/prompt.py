"""Frozen no-retrieval prompt content for EXP-004.

The shared generation pipeline already applies the tokenizer chat template,
so this module provides only the user-message body from the frozen spec.
"""

from __future__ import annotations

from src.generation.prompt import ANSWER_TYPE_INSTRUCTIONS

D2L_NO_RETRIEVAL_PROMPT_TEMPLATE = """\
Answer the question based on your knowledge. If you are not confident in the answer, respond with [].

Question: {question}
Expected answer format: {answer_type_instruction}"""


def format_d2l_no_retrieval_prompt(question: str, answer_type: str) -> str:
    """Build the frozen EXP-004 closed-book prompt."""
    instruction = ANSWER_TYPE_INSTRUCTIONS.get(answer_type, "")
    return D2L_NO_RETRIEVAL_PROMPT_TEMPLATE.format(
        question=question,
        answer_type_instruction=instruction,
    )
