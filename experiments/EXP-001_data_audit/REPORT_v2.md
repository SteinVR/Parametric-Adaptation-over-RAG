# EXP-001 v2: Data Audit Report

**Date:** 2026-03-28
**Status:** Complete
**Corpus:** 8 documents (hand-selected from 65)
**Goldset:** 200 QA (merged from 2 batches)

## Corpus

| Document | Pages | Tokens (est) | Fits D2L |
|----------|-------|-------------|----------|
| doc1_general_partnership_law | 23 | 13,268 | Yes |
| doc2_crs_regulations | 26 | 22,359 | Yes |
| doc3_techteryx_v_aria | 25 | 17,445 | Yes |
| doc4_bond_v_tr88house | 23 | 14,950 | Yes |
| doc5_personal_property_law | 21 | 13,003 | Yes |
| doc6_securities_regulations | 24 | 11,319 | Yes |
| doc7_ozias_v_obadiah | 19 | 13,032 | Yes |
| doc8_lxt_v_sir_realestate | 15 | 9,658 | Yes |

**Total:** 176 pages, 115,034 tokens
**All fit D2L single pass:** True

## Goldset

- QA pairs: 200
- All gold doc_ids in corpus: YES
- Schema validation: PASS
- Format validation: PASS
- Near-duplicate groups: 1

### Answer type distribution

| Type | Count | % |
|------|-------|---|
| free_text | 53 | 26% |
| boolean | 48 | 24% |
| number | 36 | 18% |
| name | 30 | 15% |
| names | 17 | 8% |
| date | 16 | 8% |

### Difficulty distribution

| Difficulty | Count | % |
|------------|-------|---|
| easy | 98 | 49% |
| medium | 71 | 36% |
| hard | 31 | 16% |

### Key stats
- Multi-document questions: 26 (13%)
- Unanswerable: 17 (8.5%) — 9 null + 8 free_text negative
- Evidence pages/question: median=1, max=4

## Split

| Split | Size | Types | Difficulty |
|-------|------|-------|------------|
| dev | 160 | {'number': 30, 'date': 12, 'names': 14, 'free_text': 42, 'boolean': 39, 'name': 23} | {'hard': 24, 'easy': 78, 'medium': 58} |
| test | 40 | {'free_text': 11, 'number': 6, 'date': 4, 'name': 7, 'names': 3, 'boolean': 9} | {'hard': 7, 'medium': 13, 'easy': 20} |

- Near-duplicates grouped: OK
- Split saved: `data/splits/split_v1.json`
- **FROZEN**

## Capacity Audit

- All 8 docs fit D2L individually: **YES**
- No sub-document chunking needed
- S3: merge 8 per-doc adapters
- S4-doc: 8 adapters, zero merge, per-doc routing
- S4-cluster: 4 adapters (merge 2 per cluster)

## Conclusion

Data artifacts validated and consistent. Proceed to EXP-002.
