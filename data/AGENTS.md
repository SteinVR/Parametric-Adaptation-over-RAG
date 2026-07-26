## Structure
- **corpus/** — base directory for corpus versions and variants.
- **corpus4-100/** — primary corpus (v4, 100 documents) with gold standards, benchmarks, question pools, shard manifests, and candidate/draft/review documents.
- **corpus4_2-100/** — enhanced corpus variant (v4.2, 100 documents) with improved annotations, smoke test reports, and comprehensive text views.
- **goldset/** — gold standard reference dataset containing validated question-answer pairs, benchmark results, and dataset merging utilities.
- **manifests/** — metadata manifests tracking corpus versions, shard distributions, question inventory, and structural versioning.
- **processed/** — cleaned and preprocessed data ready for model training and evaluation.
- **splits/** — data splits for train/dev/test phases used in experiments.
- **old_corpus/** — archive of legacy corpus versions for historical reference.

## Active Dataset for All Experiments

**Primary source:** `goldset/`

All experiments (EXP-002 through EXP-010) use the **goldset** as the canonical evaluation dataset:
- **goldset.benchmark.json** — reference Q&A pairs with benchmark metadata for evaluation
- **goldset.questions.json** — complete question inventory
- **Dataset scale:** 8 documents, 200 question-answer pairs
- **Train/eval split:** 150 training examples (S2), 50 evaluation examples
- **Status:** Frozen before experiments; all experiment configurations reference `GOLDSET_PATH` from `config.py`

The source corpora (corpus4-100 and corpus4_2-100) are processed and merged into this goldset prior to experimental runs.