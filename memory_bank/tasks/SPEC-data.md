# SPEC: Data Handling

> Detail spec for corpus, goldset, splits, leakage rules. Parent: `memory_bank/ARCHITECTURE.md`

---

## Corpus

- **Source:** `data/150/documents/pdfs/` — 65 PDF files
- **Domain:** DIFC (Dubai International Financial Centre) legal documents
- **Source types:** statutes (61 QA refs), case law (48 refs), cross-case (41 refs)
- **Manifest:** `data/manifests/corpus_manifest.csv` — one row per document with:
  - doc_id (hash filename)
  - page_count
  - approximate token count
  - source_type if identifiable
- **Freeze rule:** corpus manifest must be frozen before any main experiment

---

## Goldset

- **Source:** `data/150/dev-gold-150-v1.benchmark.json`
- **Composition:** 100 warmup + 50 full phase = 150 QA pairs
- **Schema per reference:**
  - `question_id`, `question`, `answer`, `answer_type`
  - `gold_retrieval`: list of `{doc_id, page_numbers}`
  - `source_type`, `difficulty`, `tags`
- **Answer type distribution:**
  - free_text: 44, boolean: 42, number: 25, name: 21, names: 11, date: 7
- **Difficulty:** easy: 101, medium: 43, hard: 6
- **Tags:** single_document: 103, multi_document: 47, comparative: 45, negative: 33
- **No synthetic expansion** in core scope

---

## Split Protocol

- **Locked test:** 30 questions (20%)
- **Development:** 120 questions (80%)
- **Stratification:** by answer_type + difficulty + single/multi-doc
- **Sparse strata handling:** date (7) and hard (6) are very small — ensure at least 1-2 per stratum in test
- **Split is created once and frozen** in `data/splits/split_v1.json`
- **No cross-validation.** S2 uses 3 random seeds for variance estimation.

---

## S2 Training Data Format (RAFT-style)

From the 120 dev questions (train portion per seed):

```
System prompt: [domain context]
User: [question]
Context: [gold evidence chunk(s)] [optional distractor chunk(s)]
Assistant: [answer]
```

- Gold chunks extracted from `gold_retrieval` page references
- Distractor policy: 1-2 random non-gold chunks from same document or corpus (TBD at EXP-003)
- Answer formatted according to answer_type rules

---

## Doc-to-LoRA Corpus Segmentation (S3/S4)

- Documents chunked into segments fitting Doc-to-LoRA context window (~32K tokens)
- Segment boundaries respect page breaks where possible
- Per-segment metadata: doc_id, page_range, token_count
- Segmentation is deterministic and reproducible from manifest + config

For S4: clustering applied at **document level** before segmentation.
Each cluster's documents segmented independently.

---

## Leakage Rules

1. No locked test question may influence: prompt tuning, hyperparameter selection, adapter choice for S5, routing calibration
2. Clustering (S4) uses document embeddings only — no QA-derived features
3. Near-duplicate / paraphrase questions must be grouped before splitting (check during EXP-001)
4. S2 trains only on the dev split's train portion for each seed
5. Doc-to-LoRA (S3/S4) ingests documents, not QA pairs — no QA leakage by design
