"""Frozen prompt template and answer type instructions.

Shared by all experiments (S1-S6). Do not modify without updating ARCHITECTURE.md.
"""

from __future__ import annotations

ANSWER_TYPE_INSTRUCTIONS: dict[str, str] = {
    "boolean": "Answer true or false.",
    "number": "Answer with a number only.",
    "name": "Answer with the exact name.",
    "names": 'Answer with a JSON array of names, e.g. ["Name1", "Name2"].',
    "date": "Answer with a date in YYYY-MM-DD format.",
    "free_text": "Answer in 1-3 sentences (max 280 characters).",
}

PROMPT_TEMPLATE = """\
Answer the question using ONLY the provided context. \
If the information is not in the context, respond with [] for factual questions \
or state that the information is not available for free-text questions.

Context:
{context}

Question: {question}
Expected answer format: {answer_type_instruction}"""


def format_prompt(question: str, answer_type: str, context: str = "") -> str:
    """Build the complete prompt string using the frozen template."""
    instruction = ANSWER_TYPE_INSTRUCTIONS.get(answer_type, "")
    return PROMPT_TEMPLATE.format(
        context=context or "(no context provided)",
        question=question,
        answer_type_instruction=instruction,
    )


def format_context_from_chunks(chunks: list) -> str:
    """Format retrieved chunks into a context string for the prompt.

    Each chunk is rendered as:
        [Doc: {doc_id}, Pages: {page_span}]
        {chunk_text}

    Chunks separated by blank line + separator.
    Accepts RetrievedChunk objects (or anything with doc_id, page_span, text attrs).
    """
    if not chunks:
        return ""
    parts = []
    for chunk in chunks:
        doc_id = getattr(chunk, "doc_id", "unknown")
        page_span = getattr(chunk, "page_span", [])
        text = getattr(chunk, "text", "")
        pages_str = ", ".join(str(p) for p in page_span) if page_span else "?"
        parts.append(f"[Doc: {doc_id}, Pages: {pages_str}]\n{text}")
    return "\n\n---\n\n".join(parts)
