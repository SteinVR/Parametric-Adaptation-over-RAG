# SPEC: EXP-002 — S1 Classical RAG Baseline

**System:** S1 | **Wave:** 1 | **Depends on:** EXP-001 | **Blocks:** EXP-003, EXP-006, EXP-007

## Goal

Build working RAG pipeline on Gemma-2-2b-it over 8-doc corpus. Establish nonparametric baseline. Freeze shared infrastructure (prompt template, retriever, answer parser) used by all downstream experiments.

## Pipeline

1. **Text extraction:** PyMuPDF, all 8 docs, page-level text with page metadata
2. **Chunking:** hierarchical page → chunk → subchunk system (user-provided implementation from external project). At integration time: pin source repo commit hash, document chunking config parameters, freeze both in REPORT.md.
3. **Embedding:** `Qwen3-Embedding-0.6B` — shared across S1 retrieval and S4 routing
4. **Index:** FAISS IndexFlatIP (inner product on normalized vectors = cosine) over chunk embeddings. Each indexed chunk MUST carry metadata: `(doc_id, page_number)` for grounding computation. Assert this at index build time.
5. **Retrieval:** top-5 chunks by cosine similarity, fixed k=5
6. **Generation:** Gemma-2-2b-it (4-bit NF4 via bitsandbytes), prompt template below
7. **Answer parsing:** rule-based parser per answer_type (see below)
8. **Scoring:** Q_main, G, systems metrics on 50 eval questions

## Prompt Template (frozen for all systems)

```
<start_of_turn>user
Answer the question using ONLY the provided context. If the information is not in the context, respond with [] for factual questions or state that the information is not available for free-text questions.

Context:
{retrieved_chunks}

Question: {question}
Expected answer format: {answer_type_instruction}
<end_of_turn>
<start_of_turn>model
```

`answer_type_instruction` per type:
- `boolean` → "Answer true or false."
- `number` → "Answer with a number only."
- `name` → "Answer with the exact name."
- `names` → "Answer with a JSON array of names, e.g. [\"Name1\", \"Name2\"]."
- `date` → "Answer with a date in YYYY-MM-DD format."
- `free_text` → "Answer in 1-3 sentences (max 280 characters)."

## Answer Parser

Per answer_type:
- `boolean`: search for "true"/"false" in output, case-insensitive
- `number`: extract first numeric value (int or float)
- `name`: strip whitespace, take first line
- `names`: try JSON parse; fallback: split by commas/newlines
- `date`: regex `\d{4}-\d{2}-\d{2}`, take first match
- `free_text`: take full output, truncate at 280 chars
- If parsing fails → `_malformed_` (distinct from intentional `[]` abstention; scored as 0)

## Frozen Decisions

| Decision | Value |
|----------|-------|
| Embedding model | Qwen3-Embedding-0.6B |
| Index type | FAISS IndexFlatIP |
| Top-k | 5 |
| Generation model | Gemma-2-2b-it 4-bit NF4 |
| Prompt template | As above |
| Answer parser | Rule-based per type |

## Metrics

- Q_main = 0.7 × S_det + 0.3 × S_asst
- Grounding G (F_β=2.5) on page-level (doc_id, page_number)
- S_det breakdown by answer_type
- S_asst via gpt-5.4-mini judge
- TTFT (median, p95), end-to-end latency, peak VRAM
- Malformed output rate

## Output

- `src/retrieval/` — chunking, embedding, index, retriever modules
- `src/generation/` — Gemma loader, prompt builder, answer parser
- `src/evaluation/` — Q_main scorer, grounding scorer, judge client
- Eval baseline metrics CSV in `results/EXP-002/`
- `experiments/EXP-002/REPORT.md`
