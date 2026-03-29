# SPEC: EXP-005b — S4-cluster Cluster-Routed Doc-to-LoRA

**System:** S4-cluster | **Wave:** 3 | **Depends on:** EXP-004 (per-doc adapters, frozen merge strategy) | **Blocks:** EXP-006

## Goal

Cluster 8 documents into 4 groups. Merge 2 per-doc adapters within each cluster. Route questions to cluster adapter.

## Pipeline (inference)

1. **Route:** embed question via Qwen3-Embedding-0.6B → cosine similarity to 4 cluster centroids → hard top-1
2. **Generate:** load selected cluster adapter → no-retrieval prompt → generate → parse
3. **Score** on 50 eval questions

## Packaging

### Clustering (offline, one-time)
- Embed all 8 documents via Qwen3-Embedding-0.6B (same doc embeddings as EXP-005a)
- k-means, k=4, on document embeddings
- Log cluster assignments, silhouette score, cluster sizes
- **Imbalanced clusters:** if a cluster has 1 doc → use its per-doc adapter as-is (note: identical to S4-doc for that doc — flag in analysis). If 3+ docs → merge all (same strategy as S3). Document actual composition.

### Per-cluster adapter merge
- For each cluster: merge its doc adapters using frozen simple average from EXP-004
- Result: 4 cluster adapters

### Routing details
- Cluster centroids = mean of doc embeddings per cluster
- Log all 4 similarity scores per question

## Analysis

- Cluster composition: which docs landed together? Does k-means reproduce the natural type grouping (statutes/regs/first-instance/appeals)?
- Routing accuracy per cluster
- S4-cluster vs S4-doc vs S3: gradient analysis (8 adapters → 4 adapters → 1 adapter)
- Multi-doc questions: does cluster-level routing handle them better than per-doc routing?

## Metrics

- Q_main, S_det, S_asst
- Routing accuracy (overall + per-cluster)
- Cluster balance, silhouette score (diagnostic)
- Comparison delta vs S3 and S4-doc
- Breakdown by answer_type

## Output

- Cluster assignments: `results/EXP-005b/clusters.json`
- 4 cluster adapters: `models/d2l/cluster_{0-3}/`
- Routing logs: `results/EXP-005b/routing_log.csv`
- `experiments/EXP-005b/REPORT.md`

## Definition of Done

- [ ] k-means clustering done — `clusters.json` with 4 cluster assignments and silhouette score
- [ ] 4 cluster adapters merged and saved — `models/d2l/cluster_{0-3}/`
- [ ] Full 50-question eval — `predictions.json` has 50 entries
- [ ] Judge scored all free_text questions via OpenAI API
- [ ] Routing log saved — `routing_log.csv` with 50 rows
- [ ] Routing accuracy reported: overall + per-cluster
- [ ] Q_main, S_det, S_asst reported
- [ ] Delta vs S3 and S4-doc computed
- [ ] Breakdown by answer_type
- [ ] All results committed to git
- [ ] `experiments/EXP-005b/REPORT.md` written (including cluster composition analysis)
