"""Generation pipeline: prompt → constrained generate → parse.

Uses outlines for constrained decoding on structured answer types.
Falls back to unconstrained generation + parsing for free_text.
"""

from __future__ import annotations

import logging
import time
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.evaluation.schemas import Prediction
from src.generation.parser import parse_answer
from src.generation.prompt import format_prompt

logger = logging.getLogger(__name__)

# Outlines regex patterns per answer_type (constrained decoding).
# Only for types where the model struggles with format — NOT for date/number/name
# where unconstrained generation is more accurate (outlines distorts logits).
ANSWER_TYPE_REGEX: dict[str, str] = {
    "boolean": r"(true|false|True|False)",
    "names": r'\["[^"]*"(,\s*"[^"]*")*\]|\[\]',
}
# date, number, name, free_text: unconstrained generation + rule-based parser


class GenerationPipeline:
    """Reusable generation pipeline with outlines constrained decoding."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        max_retries: int = 1,
        use_outlines: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_retries = max_retries
        self.use_outlines = use_outlines

        # Lazy-init outlines generators
        self._outlines_model = None
        self._generators: dict[str, object] = {}

    def _get_outlines_model(self):
        if self._outlines_model is None:
            import outlines
            self._outlines_model = outlines.from_transformers(self.model, self.tokenizer)
        return self._outlines_model

    def _get_generator(self, answer_type: str):
        if answer_type not in self._generators:
            import outlines
            regex_pattern = ANSWER_TYPE_REGEX.get(answer_type)
            if regex_pattern is None:
                return None
            om = self._get_outlines_model()
            output_type = outlines.regex(regex_pattern)
            self._generators[answer_type] = outlines.Generator(om, output_type)
        return self._generators[answer_type]

    def generate_answer(
        self,
        question: str,
        answer_type: str,
        context: str = "",
        question_id: str = "",
    ) -> Prediction:
        """Generate and parse an answer, using constrained decoding when available."""
        prompt_text = format_prompt(question, answer_type, context)

        # Try constrained generation for structured types
        if self.use_outlines and answer_type in ANSWER_TYPE_REGEX:
            return self._generate_constrained(prompt_text, answer_type, question_id)

        # Unconstrained generation + parse for free_text (and fallback)
        for attempt in range(1 + self.max_retries):
            raw_output, ttft_ms, latency_ms = self._generate_raw(prompt_text)
            parsed_answer, is_malformed = parse_answer(raw_output, answer_type)

            if not is_malformed or attempt >= self.max_retries:
                return Prediction(
                    question_id=question_id,
                    raw_output=raw_output,
                    parsed_answer=parsed_answer,
                    answer_type=answer_type,
                    is_malformed=is_malformed,
                    ttft_ms=ttft_ms,
                    latency_ms=latency_ms,
                )
            logger.warning("Malformed output for %s (attempt %d), retrying", question_id, attempt + 1)

        return Prediction(
            question_id=question_id,
            raw_output="",
            parsed_answer="_malformed_",
            answer_type=answer_type,
            is_malformed=True,
        )

    def _generate_constrained(
        self, prompt_text: str, answer_type: str, question_id: str
    ) -> Prediction:
        """Generate with outlines regex constraint. Guaranteed valid format."""
        generator = self._get_generator(answer_type)
        if generator is None:
            # Fallback to unconstrained
            raw_output, ttft_ms, latency_ms = self._generate_raw(prompt_text)
            parsed_answer, is_malformed = parse_answer(raw_output, answer_type)
            return Prediction(
                question_id=question_id, raw_output=raw_output,
                parsed_answer=parsed_answer, answer_type=answer_type,
                is_malformed=is_malformed, ttft_ms=ttft_ms, latency_ms=latency_ms,
            )

        messages = [{"role": "user", "content": prompt_text}]
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        t_start = time.perf_counter()
        raw_output = generator(prompt_tokens, max_new_tokens=self.max_new_tokens)
        total_ms = (time.perf_counter() - t_start) * 1000

        raw_output = raw_output.strip()
        parsed_answer, is_malformed = parse_answer(raw_output, answer_type)

        return Prediction(
            question_id=question_id,
            raw_output=raw_output,
            parsed_answer=parsed_answer,
            answer_type=answer_type,
            is_malformed=is_malformed,
            ttft_ms=total_ms,  # outlines doesn't expose TTFT separately
            latency_ms=total_ms,
        )

    def _generate_raw(self, prompt_text: str) -> tuple[str, float, float]:
        """Unconstrained generation with TTFT measurement via streamer."""
        messages = [{"role": "user", "content": prompt_text}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "streamer": streamer,
        }
        if self.do_sample and self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        t_start = time.perf_counter()
        ttft_recorded = None

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        output_parts: list[str] = []
        for token_text in streamer:
            if ttft_recorded is None:
                ttft_recorded = (time.perf_counter() - t_start) * 1000
            output_parts.append(token_text)

        thread.join()
        total_ms = (time.perf_counter() - t_start) * 1000
        raw_output = "".join(output_parts).strip()

        return raw_output, ttft_recorded or total_ms, total_ms
