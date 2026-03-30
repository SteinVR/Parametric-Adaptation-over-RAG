# SPEC: Data Handling

> Detail spec for corpus, goldset, splits, leakage rules. Parent: `memory_bank/ARCHITECTURE.md`

---

## Corpus

- **Source:** `data/corpus/` — 8 PDF files, human-selected
- **Domain:** DIFC (Dubai International Financial Centre) legal documents
- **Total:** 176 pages, ~115K tokens
- **Each document ≤30K tokens**

| # | File | Doc ID | Pages | Tokens | Type |
|---|------|--------|-------|--------|------|
| 1 | doc1_general_partnership_law.pdf | `302a0bd8d677...` | 23 | 13,268 | Statute |
| 2 | doc2_crs_regulations.pdf | `04be93255ec4...` | 26 | 22,359 | Regulation |
| 3 | doc3_techteryx_v_aria.pdf | `3f8a5ea0e051...` | 25 | 17,445 | Case (first instance) |
| 4 | doc4_bond_v_tr88house.pdf | `ad76dc709385...` | 23 | 14,950 | Case (first instance) |
| 5 | doc5_personal_property_law.pdf | `536bbce854b9...` | 21 | 13,003 | Statute |
| 6 | doc6_securities_regulations.pdf | `3fa59589a91b...` | 24 | 11,319 | Regulation |
| 7 | doc7_ozias_v_obadiah.pdf | `5d3df6d69fac...` | 19 | 13,032 | Case (appeal) |
| 8 | doc8_lxt_v_sir_realestate.pdf | `437568a80111...` | 15 | 9,658 | Case (appeal) |

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
- **Unanswerable:** 17 total. 9 deterministic (goldset `answer=null`, expected system response `[]`) + 8 free_text negative (expected: text stating info absent). Derived from `answer is None` at runtime (no explicit `is_unanswerable` field in goldset JSON), not a separate answer_type.
- **Near-duplicates:** 1 pair (grouped at split)
- **No cross-batch multi-doc questions** (limitation noted)

---

## Split Protocol

- **Train-150 (`S2-train`):** 150 questions — used for both S2 closed-book and S2+R RAFT training
- **Eval:** 50 questions — ALL systems evaluated on these same 50
- **Stratification:** by answer_type + difficulty + single/multi-doc
- **Near-duplicate pair grouped** in same split
- **Split is created once and frozen** in `data/splits/split_v1.json`
- **No cross-validation.** S2/S3 use 3 random seeds for variance estimation.
- **Split needed because:** supervised QLoRA systems (S2 and S2+R) train on QA pairs → must not evaluate on training data. Single eval set for all systems eliminates selection bias.

---

## S2+R Training Data Format (RAFT-style)

From 150 Train-150 (`S2-train`) questions:

```
Input:  RAG prompt template(question, [gold_chunks..., distractor_1, distractor_2])
Output: answer
```

- **Gold chunks:** page-family IndexChunks from EXP-002 index matching `gold_retrieval` pages. One chunk per gold page, sorted by (doc_id, page_number). Multi-doc questions include all gold pages from all gold documents.
- **Distractors:** exactly 2 random page-family chunks from documents OTHER than any gold document
- Distractors sampled once, frozen in training set
- Context assembly: gold chunks first, then distractors, separated by `\n\n---\n\n`
- All examples oracle (no distractor-only)
- Answer formatted per answer_type: true/false, number, date ISO, `[]` for unanswerable

---

## CLM Pretraining (S3/S3+R)

- **All 8 corpus documents concatenated** into a single training corpus (~115K tokens)
- **Tokenized as plain text** — no chat template, no prompt/answer structure, no label masking
- **Training objective:** standard causal LM (next-token prediction on all tokens)
- **No QA pairs used** — adapter learns document content only
- **No sub-document segmentation** — documents concatenated with separator tokens

---

## Leakage Rules

1. No eval question (50) may influence: prompt tuning, hyperparameter selection.
2. Near-duplicate questions grouped before splitting.
3. S2 and S2+R train only on Train-150 (`S2-train`), never on 50 eval.
4. CLM (S3/S3+R) trains on document text, not QA pairs — no QA leakage by design.
