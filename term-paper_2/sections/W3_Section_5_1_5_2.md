# 5. Results

## 5.1 Main Comparison

The main comparison begins with a strong baseline. S1 reaches a Q_main of 0.6425, with S_det at 0.6014 and S_asst at 0.7385, which establishes a difficult starting point for any adapted retrieval-aware system. Against that baseline, both headline adapted systems provide moderate but meaningful improvement. S2+R reaches 0.6689 ± 0.0137, while S3+R reaches 0.6671 ± 0.0229. The size of the gain is not large, but that is precisely what makes it informative in this setup: the baseline is already strong, and improvements are measured against a fixed retrieval stack rather than against a weak generator-only system.

The two headline adapted systems remain close in aggregate quality. S2+R holds a marginal aggregate edge over S3+R, but the difference is too small to support a claim of practical dominance. The observed pattern is better interpreted as a trade-off among answer-quality dimensions.

S7 reaches the highest observed aggregate score at 0.7045 ± 0.0345, with S_det at 0.6790 ± 0.0481 and S_asst at 0.7641 ± 0.0178. That result is reported after the headline comparison because S7 is a post-hoc adapter merge rather than a separately trained system. It strengthens the case for partial complementarity between the two adaptation signals, but it does not redefine the central comparison on which the paper's main claim depends.

*Table 3 about here: Main Results.*

## 5.2 Trade-off Between RAFT-style and CLM Adaptation

The central scientific result of the paper is the difference in quality profile between RAFT-style adaptation and CLM continued pretraining. S2+R reaches a higher deterministic score than S3+R, with S_det values of 0.6479 ± 0.0150 and 0.5991 ± 0.0156 respectively. S3+R, however, reaches a substantially higher assistant-style quality score, with S_asst at 0.8256 ± 0.0622 compared with 0.7179 ± 0.0178 for S2+R. The systems are therefore close in Q_main while behaving differently at the component level.

The delta-to-S1 view makes this contrast clearer. Relative to the baseline, S2+R improves Q_main by 0.0265 and S_det by 0.0466, while slightly reducing S_asst by 0.0205. S3+R improves Q_main by 0.0246 and S_asst by 0.0872, while leaving S_det slightly below the baseline by 0.0023. This pattern supports the interpretation that training signal matters more than the mere presence of an adapter. RAFT-style supervision appears to strengthen deterministic extraction, whereas CLM adaptation appears to strengthen free-text response quality.

The practical interpretation remains mixed. The comparison records a tie on Q_main and grounding between S2+R and S3+R, with S2+R favored on deterministic extraction and S3+R favored on assistant-style quality and offline cost. The offline cost difference is substantial, with S2+R at 1205.5 seconds and S3+R at 581.4 seconds. Under consumer-hardware constraints, that asymmetry matters. The result therefore supports a bounded conclusion: both adaptation strategies add value over strong RAG, but they do so through different quality profiles and different practical costs.

*Figure 2 about here: Delta-to-S1 Bar Chart.*
