# SPEC: EXP-005b — S4-cluster Cluster-Routed Doc-to-LoRA

**System:** S4-cluster | **Wave:** 3 | **Depends on:** EXP-004 (per-doc adapters, frozen merge strategy) | **Blocks:** EXP-006

## Goal

Cluster 8 documents into 4 groups. Merge 2 per-doc adapters within each cluster. Route questions to cluster adapter.

## Pipeline

### Step 1: Clustering
- Embed all 8 documents via Qwen3-Embedding-0.6B (same doc embeddings as EXP-005a)
- k-means, k=4, on document embeddings
- Log cluster assignments, silhouette score, cluster sizes
- **Imbalanced clusters:** if a cluster has 1 doc → use its per-doc adapter as-is (no merge; note this is identical to S4-doc's adapter for that doc — flag in analysis). If a cluster has 3+ docs → merge all adapters in that cluster (same strategy as S3). Document actual composition.

### Step 2: Per-cluster adapter merge
- For each cluster: merge its 2 doc adapters using frozen merge strategy from EXP-004 (simple average)
- Result: 4 cluster adapters

### Step 3: Router
- Compute cluster centroids (mean of doc embeddings per cluster)
- Embed question → cosine similarity to 4 centroids → hard top-1
- Log all 4 similarity scores per question

### Step 4: Generation
- Load selected cluster adapter
- No retrieved context — adapter parameters only
- Same no-retrieval prompt template as EXP-004
- Same answer parser

### Step 5: Evaluation on 50 eval questions

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
