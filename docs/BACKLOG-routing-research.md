# Backlog: Routing & Merge Research Space

> Potential experiments beyond core scope.
> Pick from here if time permits or for appendix/future work.
> Core scope: S3 (full merge), S4-doc (per-doc top-1), S4-cluster (k=4 top-1).

---

## Axis 1: Granularity (number of adapters)

Core covers 3 points: 1, 4, 8 adapters. Extensions:

- [ ] **Type-based split (k=2):** statutes/regs → 1 adapter, cases → 1 adapter. Simplest meaningful clustering. Tests whether legal-domain structure matters.
- [ ] **k=3, k=5, k=6 clustering:** optimal k search via silhouette + downstream Q_main. Is k=4 actually optimal or arbitrary?
- [ ] **Hierarchical routing:** first route by type (statute vs case), then within type by doc similarity. Two-stage routing.

## Axis 2: Routing mechanism

Core uses hard top-1. Extensions:

- [ ] **Soft weighted routing:** cosine similarity → softmax → weighted combination of adapter outputs. May solve multi-doc question problem.
- [ ] **Top-2 hard routing:** activate 2 closest adapters, average their delta-W or outputs. Simple middle ground between top-1 and soft.
- [ ] **Retrieval-based routing:** use same retriever as S1 — whichever doc retriever finds first determines the adapter. Ties routing directly to retrieval quality.
- [ ] **Learned router:** small MLP on question embedding → cluster assignment. Risk: overfitting on 150 train questions. Only viable as appendix.

## Axis 3: What gets clustered

Core uses document-level k-means on embeddings. Extensions:

- [ ] **Metadata-based clustering:** deterministic groups by doc type (statute/regulation/first-instance/appeal). No embeddings needed, fully interpretable.
- [ ] **Chunk-level clustering:** embed individual chunks, cluster those. One document may span multiple clusters. More granular but router becomes chunk-level, not doc-level.
- [ ] **Hybrid:** cluster by metadata first, refine by embeddings within groups.

## Axis 4: Merge strategies

Core uses simple average for S3/S4-cluster. Extensions:

- [ ] **Weighted average by doc token count:** larger docs get more weight in merge.
- [ ] **TIES-Merging:** trim, elect signs, then merge. May preserve more information.
- [ ] **DARE:** random drop + rescale before merge. Complementary to TIES.
- [ ] **Task Arithmetic:** treat each doc adapter as a "task vector", compose via addition with scaling.
- [ ] **Dynamic merge at inference (soft routing):** don't pre-merge; compute weighted sum of adapter delta-Ws on the fly based on query similarity. Essentially S4-soft from a different angle.

## Axis 5: Multi-document question handling

- [ ] **Analyse S4-doc failures on 26 multi-doc questions:** is routing the bottleneck?
- [ ] **Oracle routing experiment:** give S4-doc the correct document(s) — what's the ceiling?
- [ ] **Multi-adapter activation:** for detected multi-doc questions, activate 2+ adapters. Detection could be: retriever returns docs from multiple clusters.

## Axis 6: Adapter analysis (interpretability)

- [ ] **Pairwise cosine similarity matrix** between 8 doc adapters (flattened delta-W). Are statute adapters more similar to each other than to case adapters?
- [ ] **Per-adapter performance breakdown:** which adapter is best on which question types?
- [ ] **Merge degradation curve:** plot Q_main as function of number of adapters merged (1, 2, 4, 8). At what point does merge quality collapse?

---

## Prioritization for appendix

If time permits, highest-value extensions in order:
1. **Merge degradation curve** (Axis 6) — directly supports RQ2, easy to compute
2. **Soft weighted routing** (Axis 2) — tests dynamic merge, addresses multi-doc limitation
3. **Metadata-based clustering** (Axis 3) — zero-cost comparison to k-means, fully interpretable
4. **Adapter similarity matrix** (Axis 6) — one heatmap, strong interpretability contribution
