"""Dense and sparse embedding components for the indexing pipeline."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Mapping

from qdrant_client import models

LOGGER = logging.getLogger(__name__)

DEFAULT_QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QWEN3_PROMPT_NAME = "document"
DEFAULT_EMBED_BATCH_SIZE = 16
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Dense embedder
# ---------------------------------------------------------------------------


class DenseEmbedder:
    """Dense embedding protocol for the indexing pipeline."""

    model_name: str
    prompt_name: str | None
    normalize_embeddings: bool

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class Qwen3DenseEmbedder(DenseEmbedder):
    """Local Qwen3 embedding path used by the indexing baseline."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QWEN3_EMBEDDING_MODEL,
        prompt_name: str | None = DEFAULT_QWEN3_PROMPT_NAME,
        normalize_embeddings: bool = True,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._model: Any | None = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._load_model()
        encode_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize_embeddings,
            "show_progress_bar": False,
        }
        if self.prompt_name:
            encode_kwargs["prompt_name"] = self.prompt_name

        try:
            vectors = model.encode(texts, **encode_kwargs)
        except TypeError:
            encode_kwargs.pop("prompt_name", None)
            vectors = model.encode(texts, **encode_kwargs)

        if hasattr(vectors, "tolist"):
            return [[float(value) for value in row] for row in vectors.tolist()]
        return [[float(value) for value in row] for row in vectors]

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required to build dense indices with Qwen3-Embedding-0.6B."
            ) from exc

        try:
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Qwen3 embedding model '{self.model_name}' is not available locally. "
                "Pre-download it before running build-indices without a test stub."
            ) from exc
        return self._model


def build_dense_embedder() -> DenseEmbedder:
    """Build the dense embedder used for index construction (prompt_name='document')."""

    return Qwen3DenseEmbedder()


def build_query_embedder() -> DenseEmbedder:
    """Build the dense embedder used for query-time retrieval (prompt_name='query')."""

    return Qwen3DenseEmbedder(prompt_name="query")


# ---------------------------------------------------------------------------
# BM25 sparse encoder
# ---------------------------------------------------------------------------


class BM25SparseEncoder:
    """BM25 Okapi sparse encoder with retrieval-facing metadata."""

    def __init__(
        self,
        texts: list[str],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.k1 = k1
        self.b = b
        tokenized_texts = [_tokenize(text) for text in texts]
        self.document_count = len(tokenized_texts)
        self.document_lengths = [max(len(tokens), 1) for tokens in tokenized_texts]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths) if self.document_lengths else 1.0
        )

        document_frequency: Counter[str] = Counter()
        for tokens in tokenized_texts:
            document_frequency.update(set(tokens))
        if not document_frequency:
            document_frequency["_empty"] = 1

        self.term_index = {term: index for index, term in enumerate(sorted(document_frequency))}
        self.inverse_document_frequency = {
            term: math.log(1.0 + ((self.document_count - frequency + 0.5) / (frequency + 0.5)))
            for term, frequency in document_frequency.items()
        }

    def encode(self, text: str) -> models.SparseVector:
        return self.encode_tokens(_tokenize(text))

    def encode_tokens(self, tokens: list[str]) -> models.SparseVector:
        frequencies = Counter(tokens or ["_empty"])
        document_length = max(len(tokens), 1)
        weighted_terms: list[tuple[int, float]] = []

        for term, frequency in frequencies.items():
            term_id = self.term_index.get(term)
            if term_id is None:
                continue
            idf = self.inverse_document_frequency.get(term, 0.0)
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * (document_length / self.average_document_length)
            )
            weight = idf * ((frequency * (self.k1 + 1.0)) / denominator)
            weighted_terms.append((term_id, float(weight)))

        weighted_terms.sort(key=lambda item: item[0])
        if not weighted_terms:
            weighted_terms = [(0, 0.0)]
        return models.SparseVector(
            indices=[index for index, _ in weighted_terms],
            values=[value for _, value in weighted_terms],
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "scheme": "bm25_okapi",
            "tokenizer": "regex-lowercase",
            "k1": self.k1,
            "b": self.b,
            "average_document_length": self.average_document_length,
            "document_count": self.document_count,
            "vocabulary_size": len(self.term_index),
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "metadata": self.metadata(),
            "term_index": dict(self.term_index),
            "inverse_document_frequency": dict(self.inverse_document_frequency),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "BM25SparseEncoder":
        metadata = state.get("metadata")
        term_index_raw = state.get("term_index")
        idf_raw = state.get("inverse_document_frequency")
        if not isinstance(metadata, Mapping):
            raise ValueError("sparse encoder state must include metadata.")
        if not isinstance(term_index_raw, Mapping) or not term_index_raw:
            raise ValueError("sparse encoder state must include term_index mapping.")
        if not isinstance(idf_raw, Mapping) or not idf_raw:
            raise ValueError("sparse encoder state must include inverse_document_frequency mapping.")

        encoder = cls.__new__(cls)
        encoder.k1 = float(metadata.get("k1", BM25_K1))
        encoder.b = float(metadata.get("b", BM25_B))
        encoder.document_count = int(metadata.get("document_count", 1))
        encoder.average_document_length = float(metadata.get("average_document_length", 1.0))
        encoder.document_lengths = [1] * max(encoder.document_count, 1)
        encoder.term_index = {str(term): int(index) for term, index in term_index_raw.items()}
        encoder.inverse_document_frequency = {
            str(term): float(weight) for term, weight in idf_raw.items() if str(term) in encoder.term_index
        }
        if not encoder.inverse_document_frequency:
            raise ValueError("sparse encoder inverse_document_frequency cannot be empty.")
        return encoder


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
