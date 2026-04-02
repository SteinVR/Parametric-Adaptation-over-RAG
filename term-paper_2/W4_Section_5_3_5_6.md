## 5.3 By Answer Type

Aggregate scores hide several material behavioral differences. On free-text questions, S3+R reaches the strongest assistant-style score at 0.8256 ± 0.0622, clearly above S1 at 0.7385 and S2+R at 0.7179 ± 0.0178. This supports the interpretation that CLM continued pretraining helps local contextualization and discourse quality when the answer requires synthesized prose. By contrast, S2+R improves several deterministic categories relative to the baseline, including boolean questions at 0.8889 and date questions at 0.4000, while S3+R remains closer to the baseline on those answer types.

The per-type breakdown also shows that none of the systems is uniformly strong. The names category remains difficult for the adapted systems, with both S2+R and S3+R underperforming S1 there, and multi-name extraction remains unstable across the board. S7 performs best on number and name questions, but that improvement is reported as secondary because the system is post-hoc. The main interpretive point is that near-equal aggregate scores conceal distinct answer behaviors that align with the different adaptation signals.

*Figure 3 about here: Per-Type Score Heatmap.*

## 5.4 Retrieval Contribution and the Limits of Pure Parametric Memory

Retrieval remains indispensable on this benchmark. The supervised control S2 reaches only 0.2630 ± 0.0046 in Q_main, whereas S2+R reaches 0.6689 ± 0.0137. The CLM control S3 performs even worse at 0.1854 ± 0.0027, while S3+R reaches 0.6671 ± 0.0229. The deltas are large: retrieval adds 0.4059 Q_main points to the supervised system and 0.4817 Q_main points to the CLM system. These gaps are too large to treat retrieval as a minor convenience or as a redundant supplement to parametric adaptation.

The legacy D2L branch supports the same conclusion from a separate engineering path. S3-legacy reaches a Q_main of 0.2100, with S_det at 0.1351 and S_asst at 0.3846. Although the D2L implementation is not directly comparable to the active CLM setup, it remains useful as a negative control. The result indicates that document-internalized adaptation without retrieval did not become competitive in this implementation regime, and the main thesis of the paper does not depend on it doing so.

## 5.5 Multi-Document Difficulty

The split between single-document and multi-document questions yields one of the clearest analytical results in the paper. Multi-document questions are substantially harder for all headline systems. S1 drops from 0.6958 on single-document items to 0.3100 on multi-document items. S3+R shows an even larger contrast, moving from 0.7222 to 0.3100. S2+R remains lower on single-document questions than S3+R, at 0.6938, but it retains more quality on multi-document questions, reaching 0.4367. This difference is important because it reveals a sharper behavioral distinction than the aggregate table alone.

The contrast supports a bounded complementarity interpretation. CLM adaptation appears strongest when the task is local and context-sensitive within a single document, while RAFT-style supervision appears more robust when the answer depends on aggregation or comparison across documents. S7 reaches 0.7184 on single-document questions and 0.5233 on multi-document questions, which is consistent with partial combination of both effects. Even so, the headline analytical result belongs to the contrast between S2+R and S3+R, not to the merged system.

*Figure 4 about here: Single-doc vs Multi-doc Comparison.*

## 5.6 Exploratory Adapter Fusion

The merged adapter provides evidence that the two adaptation signals are not redundant. Relative to S2+R, S7 improves Q_main by 0.0356, S_det by 0.0310, and S_asst by 0.0462. Relative to S3+R, it improves Q_main by 0.0374 and S_det by 0.0799, while reducing S_asst by 0.0615. This pattern is consistent with partial complementarity: the merged system appears to preserve some of the CLM advantage in assistant-style quality while recovering part of the deterministic advantage associated with RAFT-style supervision.

The result remains exploratory for two reasons. First, S7 is a post-hoc merge rather than a separately trained system. Second, its practical cost is not directly comparable to the headline systems because it inherits prior adaptation cost from both source adapters. For that reason, the merged system supports interpretation rather than practical winner selection. The main claim of the paper remains intact without S7, which is the correct standard for treating it as a secondary finding.
