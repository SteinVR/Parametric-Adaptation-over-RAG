# Deep Analysis

## 1. Ranking and Margin Structure

- Global quality leader: **S7** with `Q_main=0.7045`.
- Merge effect (S7 vs S2+R): `ΔQ_main=+0.0356`, `ΔS_det=+0.0310`, `ΔS_asst=+0.0462`.
- Trade-off persists in base hybrids: S2+R dominates deterministic score (`S_det=0.6479`), S3+R dominates assistant score (`S_asst=0.8256`).

## 2. Error Topology and Complementarity

- Mean pairwise failure-overlap (headline systems) Jaccard: `0.714`.
- Lower overlap means systems fail on different subsets, creating room for fusion/selection strategies.
- Pairwise win-rate matrix quantifies question-level dominance and reveals where apparent aggregate ties hide local regime shifts.

## 3. Stability and Variance

- Highest run-to-run variance (Q_main std) is visible in: `S7`.
- Mean per-question score std highlights systems with unstable behavior across seeds, not just unstable global means.

## 4. Difficulty and Judge-Dimension Behavior

- Best performer on `hard` questions: **S7** (`0.552`).
- Free-text grounding criterion leader: **S3+R**.
- Criterion-level profiles show where quality gains come from (correctness/completeness) versus stylistic clarity.

## 5. Cost-Quality Frontier

- Pareto-optimal systems on (offline cost, Q_main): `S7`.
- Systems outside Pareto front are strictly dominated and can be deprioritized in practical deployment decisions.

## 6. Practical Recommendations

- Keep S2+R and S3+R both in discussion as base hybrids; they encode different strengths and should not be collapsed into a single narrative.
- Promote S7 as best observed post-hoc merged configuration, but preserve caveat: it is a merge-based conclusion, not independent retraining evidence.
- Use question-level win-rate and overlap artifacts to motivate targeted ensemble/routing hypotheses in future work.
