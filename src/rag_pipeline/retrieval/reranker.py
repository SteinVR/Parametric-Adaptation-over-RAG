"""Qwen3-based cross-encoder reranker for retrieval candidates."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_QWEN3_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
DEFAULT_QWEN3_RERANK_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
DEFAULT_QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "for", "how", "in",
        "is", "of", "or", "the", "to", "what", "which", "who",
    }
)


# ---------------------------------------------------------------------------
# Backend: Qwen3 cross-encoder scoring
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TransformersQwenRerankerBackend:
    """Lazy Transformers backend for ``Qwen/Qwen3-Reranker-0.6B``.

    Rerank scores come from the next-token probability of answering "yes"
    versus "no" for a formatted query/document prompt.
    """

    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    device: str | None = None
    batch_size: int = 8
    max_length: int = 8192
    instruction: str | None = DEFAULT_QWEN3_RERANK_INSTRUCTION
    system_prompt: str = DEFAULT_QWEN3_RERANK_SYSTEM_PROMPT
    trust_remote_code: bool = True
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _runtime_device: str | None = field(default=None, init=False, repr=False)
    _token_false_id: int | None = field(default=None, init=False, repr=False)
    _token_true_id: int | None = field(default=None, init=False, repr=False)
    _prefix_token_ids: list[int] | None = field(default=None, init=False, repr=False)
    _suffix_token_ids: list[int] | None = field(default=None, init=False, repr=False)

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one rerank score per candidate document."""
        if not documents:
            return []

        self._ensure_loaded()

        import torch

        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch_documents = documents[start : start + self.batch_size]
            model_inputs = self._tokenize_pairs(
                [self._format_pair(query, document) for document in batch_documents]
            )

            with torch.no_grad():
                outputs = self._model(**model_inputs)
                batch_logits = outputs.logits[:, -1, :]
                true_vector = batch_logits[:, self._token_true_id]
                false_vector = batch_logits[:, self._token_false_id]
                rerank_logits = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(rerank_logits, dim=1)[:, 1].exp()

            scores.extend(float(s) for s in batch_scores.detach().float().cpu().tolist())

        return scores

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        runtime_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = snapshot_download(repo_id=self.model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )
        model.to(runtime_device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._runtime_device = runtime_device
        self._token_false_id = _single_token_id(tokenizer, "no")
        self._token_true_id = _single_token_id(tokenizer, "yes")
        self._prefix_token_ids = list(
            tokenizer.encode(
                f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n",
                add_special_tokens=False,
            )
        )
        self._suffix_token_ids = list(
            tokenizer.encode(DEFAULT_QWEN3_RERANK_SUFFIX, add_special_tokens=False)
        )

        logger.info("Loaded reranker model %s on %s", self.model_name, runtime_device)

    def _format_pair(self, query: str, document: str) -> str:
        instruction = self.instruction or DEFAULT_QWEN3_RERANK_INSTRUCTION
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"

    def _tokenize_pairs(self, pairs: list[str]) -> dict[str, Any]:
        assert self._tokenizer is not None
        assert self._runtime_device is not None
        assert self._prefix_token_ids is not None
        assert self._suffix_token_ids is not None

        content_max_length = max(
            self.max_length - len(self._prefix_token_ids) - len(self._suffix_token_ids), 1
        )
        tokenized = self._tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=content_max_length,
        )
        for index, input_ids in enumerate(tokenized["input_ids"]):
            tokenized["input_ids"][index] = (
                self._prefix_token_ids + list(input_ids) + self._suffix_token_ids
            )

        padded_inputs = self._tokenizer.pad(
            tokenized,
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(self._runtime_device) for key, value in padded_inputs.items()}


# ---------------------------------------------------------------------------
# Reranker: wraps a backend and operates on SearchResult lists
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Reranker:
    """Cross-encoder reranker that scores query/document pairs via a Qwen3 backend.

    Falls back to :class:`LexicalFallbackReranker` if the model backend is
    unavailable or raises at runtime.
    """

    backend: TransformersQwenRerankerBackend = field(
        default_factory=TransformersQwenRerankerBackend,
    )
    use_lexical_fallback: bool = True

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Score and re-sort *candidates* by cross-encoder relevance.

        Each returned dict gains a ``rerank_score`` key.
        """
        result_limit = max(int(top_k), 0)
        if result_limit == 0 or not candidates:
            return []

        try:
            scores = self.backend.score(
                query,
                [str(c.get("text") or "") for c in candidates],
            )
            if len(scores) != len(candidates):
                raise ValueError("Reranker returned mismatched score count.")
            reranked = [
                {**dict(c), "rerank_score": float(s)}
                for c, s in zip(candidates, scores, strict=True)
            ]
        except Exception:
            if not self.use_lexical_fallback:
                raise
            logger.warning(
                "Cross-encoder reranker failed; falling back to lexical reranker.",
                exc_info=True,
            )
            return LexicalFallbackReranker().rerank(query, candidates, result_limit)

        reranked.sort(
            key=lambda c: (
                -c["rerank_score"],
                -_safe_float(c.get("retrieval_score")),
                str(c.get("chunk_id") or ""),
            ),
        )
        return reranked[:result_limit]


# ---------------------------------------------------------------------------
# Lexical fallback (no model required)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LexicalFallbackReranker:
    """Lightweight token-overlap reranker that needs no model."""

    lexical_weight: float = 0.7
    retrieval_weight: float = 0.3
    stopwords: frozenset[str] = field(default_factory=lambda: DEFAULT_STOPWORDS)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        result_limit = max(int(top_k), 0)
        query_tokens = self._tokenize(query)
        reranked: list[dict[str, Any]] = []

        for candidate in candidates:
            normalized = dict(candidate)
            retrieval_score = _safe_float(candidate.get("retrieval_score"))
            overlap = self._overlap_score(
                query_tokens,
                self._tokenize(str(candidate.get("text") or "")),
            )
            normalized["rerank_score"] = (
                overlap * self.lexical_weight + retrieval_score * self.retrieval_weight
            )
            reranked.append(normalized)

        reranked.sort(
            key=lambda c: (
                -c["rerank_score"],
                -_safe_float(c.get("retrieval_score")),
                str(c.get("chunk_id") or ""),
            ),
        )
        return reranked[:result_limit]

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in TOKEN_PATTERN.findall(text.lower())
            if token and token not in self.stopwords
        }

    def _overlap_score(self, query_tokens: set[str], candidate_tokens: set[str]) -> float:
        if not query_tokens or not candidate_tokens:
            return 0.0
        return len(query_tokens & candidate_tokens) / len(query_tokens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_token_id(tokenizer: Any, text: str) -> int:
    input_ids = getattr(tokenizer(text, add_special_tokens=False), "input_ids", None)
    if not input_ids or len(input_ids) != 1:
        raise RuntimeError(
            f"Expected '{text}' to map to a single token for Qwen reranker scoring."
        )
    return int(input_ids[0])


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
