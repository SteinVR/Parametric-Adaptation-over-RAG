# Experiment Report: EXP-004 - S3 Doc-to-LoRA Monolithic

**Date:** 2026-03-30
**Status:** Blocked

## 1. Goal

- Generate 8 per-document Doc-to-LoRA adapters, validate each on its own train-document deterministic subset, merge them by simple average, and evaluate the monolithic adapter without retrieval.

## 2. Validation

- Corpus documents: 8
- Goldset references: 200
- S2-train questions: 150
- Eval questions: 50
- Checkpoint status: missing: /home/xeliaray/Projects/Term-Paper/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin

## 3. Blocker

- missing: /home/xeliaray/Projects/Term-Paper/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin


## 8. Artifacts

- Generation records: `/home/xeliaray/Projects/Term-Paper/results/EXP-004/document_generation.json`
- Sanity results: `/home/xeliaray/Projects/Term-Paper/results/EXP-004/doc_sanity.json`
- Merge summary: `/home/xeliaray/Projects/Term-Paper/results/EXP-004/merge_summary.json`
- Validation summary: `/home/xeliaray/Projects/Term-Paper/results/EXP-004/validation.json`
- Predictions: `/home/xeliaray/Projects/Term-Paper/results/EXP-004/predictions.json`
- Eval outputs: `/home/xeliaray/Projects/Term-Paper/results/EXP-004`
- Adapters: `/home/xeliaray/Projects/Term-Paper/models/d2l`
