"""RAFT-style dataset generation for EXP-003."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import random
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from external.pdf_rag_pipeline import build_corpus, parse_pdf, serialize_document_tables
from src.data.io import load_goldset, load_json
from src.generation.prompt import format_context_from_chunks, format_prompt
from src.retrieval.indexer import build_doc_id_map

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PageChunk:
    """Full-page chunk used for RAFT gold pages and distractors."""

    chunk_id: str
    doc_id: str
    page_span: list[int]
    text: str

    @property
    def page_number(self) -> int:
        return self.page_span[0]


@dataclass(frozen=True, slots=True)
class RaftExample:
    """Serialized training example for supervised QLoRA."""

    question_id: str
    answer_type: str
    prompt: str
    answer: str
    gold_pages: list[dict[str, int]]
    distractor_pages: list[dict[str, int]]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def load_page_chunks(index_output_dir: Path) -> dict[tuple[str, int], PageChunk]:
    """Load all page-family chunks from the frozen EXP-002 Qdrant index."""

    manifest_path = index_output_dir / "index" / "index_manifest.json"
    qdrant_dir = index_output_dir / "index" / "qdrant"
    collection_name = "document_index"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        collection_name = str(manifest.get("collection_name") or collection_name)

    client = QdrantClient(path=str(qdrant_dir))
    key_to_chunk: dict[tuple[str, int], PageChunk] = {}
    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="chunk_type",
                match=models.MatchValue(value="page"),
            )
        ]
    )
    offset: Any = None

    try:
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                payload = dict(point.payload or {})
                doc_id = str(payload.get("doc_id") or "").strip()
                page_span = [int(p) for p in payload.get("page_span") or [] if int(p) > 0]
                text = str(payload.get("text") or "")
                if not doc_id or len(page_span) != 1 or not text:
                    continue
                page_number = page_span[0]
                key_to_chunk[(doc_id, page_number)] = PageChunk(
                    chunk_id=str(payload.get("chunk_id") or f"{doc_id}-p{page_number}-page"),
                    doc_id=doc_id,
                    page_span=[page_number],
                    text=text,
                )

            if offset is None:
                break
    finally:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            close_fn()

    logger.info("Loaded %d page chunks from %s", len(key_to_chunk), qdrant_dir)
    return key_to_chunk


def build_fallback_page_chunks(
    corpus_dir: Path,
    goldset_path: Path,
) -> dict[tuple[str, int], PageChunk]:
    """Reparse corpus into full-page records for missing gold-page fallback."""

    stem_to_sha = build_doc_id_map(goldset_path)
    fallback: dict[tuple[str, int], PageChunk] = {}
    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        document = parse_pdf(pdf_path)
        table_blocks = serialize_document_tables(document)
        pages = build_corpus(document, table_blocks)
        remapped_doc_id = stem_to_sha.get(document.doc_id, document.doc_id)
        for page in pages:
            fallback[(remapped_doc_id, page.page_number)] = PageChunk(
                chunk_id=f"{remapped_doc_id}-p{page.page_number}-page-fallback",
                doc_id=remapped_doc_id,
                page_span=[page.page_number],
                text=page.text,
            )

    logger.info("Built %d fallback page chunks from source PDFs", len(fallback))
    return fallback


def build_raft_examples(
    *,
    goldset_path: Path,
    split_path: Path,
    index_output_dir: Path,
    corpus_dir: Path,
    distractor_seed: int = 42,
) -> list[RaftExample]:
    """Generate frozen RAFT training examples from the S2 train split."""

    refs = load_goldset(goldset_path)
    refs_by_id = {ref["question_id"]: ref for ref in refs}
    split = load_json(split_path)
    train_ids = [str(qid) for qid in split["s2_train"]]

    page_chunks = load_page_chunks(index_output_dir)
    fallback_chunks: dict[tuple[str, int], PageChunk] | None = None
    rng = random.Random(distractor_seed)
    all_page_chunks = sorted(
        page_chunks.values(),
        key=lambda chunk: (chunk.doc_id, chunk.page_number, chunk.chunk_id),
    )

    examples: list[RaftExample] = []
    for question_id in train_ids:
        ref = refs_by_id[question_id]
        gold_docs = {entry["doc_id"] for entry in ref.get("gold_retrieval", [])}
        gold_chunks = []
        for gold_entry in sorted(ref.get("gold_retrieval", []), key=lambda item: item["doc_id"]):
            doc_id = str(gold_entry["doc_id"])
            for page_number in sorted(int(page) for page in gold_entry["page_numbers"]):
                chunk = page_chunks.get((doc_id, page_number))
                if chunk is None:
                    if fallback_chunks is None:
                        fallback_chunks = build_fallback_page_chunks(corpus_dir, goldset_path)
                    chunk = fallback_chunks[(doc_id, page_number)]
                gold_chunks.append(chunk)

        distractor_pool = [chunk for chunk in all_page_chunks if chunk.doc_id not in gold_docs]
        if len(distractor_pool) < 2:
            raise ValueError(f"Not enough distractor page chunks for question {question_id}")
        distractors = rng.sample(distractor_pool, 2)

        context = format_context_from_chunks([*gold_chunks, *distractors])
        prompt = format_prompt(
            question=str(ref["question"]),
            answer_type=str(ref["answer_type"]),
            context=context,
        )
        examples.append(
            RaftExample(
                question_id=question_id,
                answer_type=str(ref["answer_type"]),
                prompt=prompt,
                answer=serialize_training_answer(ref["answer"]),
                gold_pages=[
                    {"doc_id": chunk.doc_id, "page_number": chunk.page_number}
                    for chunk in gold_chunks
                ],
                distractor_pages=[
                    {"doc_id": chunk.doc_id, "page_number": chunk.page_number}
                    for chunk in distractors
                ],
            )
        )

    validate_raft_examples(examples, expected_count=len(train_ids))
    logger.info("Built %d RAFT examples", len(examples))
    return examples


def serialize_training_answer(answer: Any) -> str:
    """Serialize gold answers into the exact target text used for training."""

    if answer is None:
        return "[]"
    if isinstance(answer, bool):
        return str(answer).lower()
    if isinstance(answer, list):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def save_raft_jsonl(examples: list[RaftExample], output_path: Path) -> None:
    """Write RAFT examples as newline-delimited JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")

    logger.info("Saved RAFT dataset to %s", output_path)


def load_raft_jsonl(path: Path) -> list[RaftExample]:
    """Load RAFT examples from the persisted JSONL dataset."""

    examples: list[RaftExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            examples.append(RaftExample(**json.loads(line)))
    return examples


def validate_raft_examples(
    examples: list[RaftExample],
    *,
    expected_count: int | None = None,
) -> None:
    """Validate frozen RAFT dataset invariants required by the spec."""

    if expected_count is not None and len(examples) != expected_count:
        raise ValueError(
            f"Expected {expected_count} RAFT examples, found {len(examples)}"
        )

    seen_ids: set[str] = set()
    for example in examples:
        if example.question_id in seen_ids:
            raise ValueError(f"Duplicate RAFT example for question {example.question_id}")
        seen_ids.add(example.question_id)

        if len(example.distractor_pages) != 2:
            raise ValueError(
                f"Question {example.question_id} has {len(example.distractor_pages)} distractors, expected 2"
            )
        gold_docs = {item["doc_id"] for item in example.gold_pages}
        distractor_docs = [item["doc_id"] for item in example.distractor_pages]
        if any(doc_id in gold_docs for doc_id in distractor_docs):
            raise ValueError(
                f"Question {example.question_id} has distractors from a gold document"
            )
