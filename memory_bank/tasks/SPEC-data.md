# SPEC: Data Handling

> Detail spec for corpus, goldset, splits, leakage rules. Parent: `memory_bank/ARCHITECTURE.md`

---

## Corpus

- **Source:** `data/corpus/` — 8 PDF files, human-selected
- **Domain:** DIFC (Dubai International Financial Centre) legal documents
- **Total:** 176 pages, ~141K tokens
- **Each document fits Doc-to-LoRA single pass** (all ≤30K tokens)

| # | File | Doc ID | Pages | ~Tokens | Type |
|---|------|--------|-------|---------|------|
| 1 | doc1_general_partnership_law.pdf | `302a0bd8d677...` | 23 | 18,400 | Statute |
| 2 | doc2_crs_regulations.pdf | `04be93255ec4...` | 26 | 20,800 | Regulation |
| 3 | doc3_techteryx_v_aria.pdf | `3f8a5ea0e051...` | 25 | 20,000 | Case (first instance) |
| 4 | doc4_bond_v_tr88house.pdf | `ad76dc709385...` | 23 | 18,400 | Case (first instance) |
| 5 | doc5_personal_property_law.pdf | `536bbce854b9...` | 21 | 16,800 | Statute |
| 6 | doc6_securities_regulations.pdf | `3fa59589a91b...` | 24 | 19,200 | Regulation |
| 7 | doc7_ozias_v_obadiah.pdf | `5d3df6d69fac...` | 19 | 15,200 | Case (appeal) |
| 8 | doc8_lxt_v_sir_realestate.pdf | `437568a80111...` | 15 | 12,000 | Case (appeal) |

---

## Goldset

- **Source:** `data/goldset/goldset.benchmark.json` (merged from 2 batches)
- **Composition:** 200 QA pairs (100 per batch of 4 docs)
- **Schema per reference:**
  - `question_id`, `question`, `answer`, `answer_type`
  - `gold_retrieval`: list of `{doc_id, page_numbers}`
  - `source_type`, `difficulty`, `tags`
- **Answer type distribution:**
  - free_text: 53, boolean: 48, number: 36, name: 30, names: 17, date: 16
- **Difficulty:** easy: 98, medium: 71, hard: 31
- **Multi-document:** 26 questions (13%), all within same-batch doc pairs
- **Negative/unanswerable:** 17 (9 deterministic null + 8 free_text negative)
- **Near-duplicates:** 1 pair (grouped at split)
- **No cross-batch multi-doc questions** (limitation noted)

---

## Split Protocol

- **Locked test:** 40 questions (20%)
- **Development:** 160 questions (80%)
- **Stratification:** by answer_type + difficulty + single/multi-doc
- **Near-duplicate pair grouped** in same split
- **Split is created once and frozen** in `data/splits/split_v1.json`
- **No cross-validation.** S2 uses 3 random seeds for variance estimation.
- **Split needed because:** S2 trains on QA pairs → must not evaluate on training data. All systems evaluated on same test set for fairness.

---

## S2 Training Data Format (RAFT-style)

From the 160 dev questions (train portion per seed):

```
System prompt: [domain context]
User: [question]
Context: [gold evidence chunk(s)] [optional distractor chunk(s)]
Assistant: [answer]
```

- Gold chunks extracted from `gold_retrieval` page references
- Distractor policy: 1-2 random non-gold chunks from corpus (TBD at EXP-003)
- Answer formatted according to answer_type rules

---

## Doc-to-LoRA Packaging (S3/S4)

- **Each document processed individually** via hypernetwork (no chunking needed — all fit)
- S3: 8 per-doc adapters → merge into 1 monolithic adapter
- S4: per-cluster adapters (e.g. 4 clusters of 2 docs each) → route at inference
- No sub-document segmentation required (key advantage of 8-doc corpus)

---

## Leakage Rules

1. No locked test question may influence: prompt tuning, hyperparameter selection, adapter choice for S5, routing calibration
2. Clustering (S4) uses document embeddings only — no QA-derived features
3. Near-duplicate questions grouped before splitting
4. S2 trains only on the dev split's train portion for each seed
5. Doc-to-LoRA (S3/S4) ingests documents, not QA pairs — no QA leakage by design
