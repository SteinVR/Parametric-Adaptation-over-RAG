"""LLM judge client for free_text scoring via OpenAI API."""

from __future__ import annotations

import json
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating a legal QA system's response.\n"
    "Score each criterion as 1 (met) or 0 (not met). Return ONLY a JSON object."
)

JUDGE_USER_TEMPLATE = """\
Question: {question}
Reference answer: {reference_answer}
System response: {system_response}

Criteria:
1. correctness: Does the response contain the key information from the reference and no factual errors?
2. completeness: Does the response address all aspects of the question?
3. grounding: Is every claim supported by plausible legal reasoning (no hallucinated specifics)?
4. calibration: Does the response appropriately express uncertainty when information is missing?
5. clarity: Is the answer clear, concise, and directly addresses the question?

Return JSON: {{"correctness": 0|1, "completeness": 0|1, "grounding": 0|1, "calibration": 0|1, "clarity": 0|1}}"""

CRITERIA_KEYS = ["correctness", "completeness", "grounding", "calibration", "clarity"]
ALL_ZERO = {k: 0 for k in CRITERIA_KEYS}


class LLMJudge:
    """Judge client using OpenAI API (gpt-5.4-mini)."""

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        reasoning: str = "medium",
        max_retries: int = 1,
    ) -> None:
        self.model = model
        self.reasoning = reasoning
        self.max_retries = max_retries
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — judge will return all-zero scores")
            self.client = None
        else:
            self.client = OpenAI()

    def score(
        self,
        question: str,
        reference_answer: str,
        system_response: str,
    ) -> dict[str, int]:
        """Score a single free_text response.

        Returns dict with 5 binary criteria.
        On parse failure after retries, returns all-zero.
        """
        user_msg = JUDGE_USER_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer,
            system_response=system_response,
        )

        if self.client is None:
            logger.warning("No OpenAI client — returning all-zero for judge")
            return dict(ALL_ZERO)

        for attempt in range(1 + self.max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    reasoning={"effort": self.reasoning},
                    input=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                )
                text = response.output_text.strip()
                return self._parse_criteria(text)
            except Exception:
                logger.warning(
                    "Judge call failed (attempt %d/%d)",
                    attempt + 1,
                    1 + self.max_retries,
                    exc_info=True,
                )

        logger.error("Judge failed after all retries, returning all-zero")
        return dict(ALL_ZERO)

    @staticmethod
    def _parse_criteria(text: str) -> dict[str, int]:
        """Parse JSON criteria from judge response."""
        # Try to extract JSON from the response
        # The judge might include markdown code blocks
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

        data = json.loads(text)
        result = {}
        for key in CRITERIA_KEYS:
            val = data.get(key, 0)
            result[key] = 1 if val else 0
        return result
