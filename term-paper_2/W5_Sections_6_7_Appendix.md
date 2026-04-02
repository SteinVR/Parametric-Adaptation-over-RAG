# Title Page Placeholder

University, department, module, semester, author details, supervisor details, and submission date should be inserted here according to the institutional template.

# Declaration of Academic Integrity Placeholder

The declaration should be inserted here in the exact wording required by the institution.

# Table of Contents Placeholder

The final table of contents should be generated after the manuscript structure is fixed.

# 6. Discussion and Limitations

## 6.1 Answer to RQ1

The results indicate that parametric adaptation does add value beyond a strong RAG baseline within this setup. Both S2+R and S3+R improve over S1 in Q_main, but the gain is moderate rather than transformative. That magnitude is important. Because S1 is already a strong baseline, modest gains are more informative than they would be in a weak-baseline setting. They indicate that adaptation can still matter after retrieval is strong, but they do not support the claim that retrieval-aware adaptation fundamentally changes the problem.

The value of adaptation is best understood as a change in quality profile. RAFT-style supervision raises deterministic extraction, while CLM continued pretraining raises assistant-style answer quality. The paper therefore answers RQ1 positively, but in a qualified form: parametric adaptation is useful on top of strong RAG, and the specific training signal matters more than the mere fact that an adapter is present.

## 6.2 Answer to RQ2

RQ2 asks whether pure parametric systems can substitute for retrieval on this benchmark. The answer is negative within the present setup. Both retrieval-free controls perform far below the retrieval-aware systems, and the deltas between S2 and S2+R as well as between S3 and S3+R are large enough to make the conclusion unambiguous. Retrieval remains the dominant memory mechanism for this document-grounded legal QA task.

This conclusion should be stated narrowly. It applies to the evaluated corpus, split, backbone, and hardware regime. It does not imply that parametric memory is irrelevant in general. It indicates that, on this benchmark, retrieval is indispensable as the main carrier of document knowledge, while parametric adaptation is better interpreted as a complementary method for improving how retrieved evidence is used.

## 6.3 Error Analysis

Error overlap clarifies both the shared difficulty of the benchmark and the limits of any single system improvement. Fifteen evaluation questions are answered incorrectly by all headline systems, which indicates that a substantial portion of the remaining difficulty is benchmark-level rather than model-specific. At the same time, the overlap is not total. Two questions are answered correctly only by S1, two only by S3+R, and none only by S2+R or only by S7. This distribution suggests that local non-overlapping strengths exist, but they are sparse and do not overturn the aggregate-level interpretation.

The qualitative failures support the same conclusion. Recurrent misses include multi-document synthesis, date recovery, and exact name or form-list extraction. Several of these errors persist even when retrieval is available, which implies that access to evidence is necessary but not sufficient. Some failures reflect remaining difficulty in mapping retrieved context to precise answer behavior, while others likely reflect the compactness of the benchmark and the limited number of examples for harder question types.

## 6.4 Limitations

The findings are bounded in several straightforward ways. The benchmark is compact, the evaluation split is fixed, and the study uses one backbone family under one hardware regime. Free-text scoring depends on a frozen judge rubric rather than on human adjudication for every answer. The retrieval stack is fixed across retrieval-aware systems, which strengthens interpretability but limits how far the conclusions can speak to alternative retrieval designs. The merged adapter result is also bounded because S7 is post-hoc and inherits prior adaptation cost.

The D2L branch should be read even more narrowly. It supports a negative finding for the present implementation and engineering regime, not a broad claim about document-conditioned adapter generation in general. More broadly, the paper studies a compact legal corpus rather than a large or heterogeneous legal benchmark, so the conclusions should be understood as benchmark-specific and hardware-specific rather than universal.

# 7. Conclusion

## 7.1 Main Findings

This study examined whether parametric adaptation adds value beyond a strong fixed RAG baseline for document-grounded legal QA under consumer-hardware constraints. The results support three main conclusions. First, the RAG baseline is already strong and difficult to surpass. Second, parametric adaptation does provide additional value, but that value depends on the adaptation signal rather than on adapter use alone. RAFT-style supervision is associated with stronger deterministic extraction, whereas CLM continued pretraining is associated with stronger assistant-style answer quality. Third, retrieval remains indispensable, because retrieval-free controls perform far below the retrieval-aware systems.

The exploratory merged-adapter result suggests that the two adaptation signals capture partially complementary strengths. Even so, the scientific contribution of the paper does not depend on the merged system. The practical takeaway is therefore compact: when retrieval is already strong, adaptation can still improve quality, but the relevant decision is which quality profile is needed and whether the additional offline cost is justified under the available hardware budget.

# Bibliography

The bibliography will be completed in a dedicated reference pass. The final version should include only the works that are directly used in the argument, with method references focused on RAG, LoRA or QLoRA, RAFT-style training, and continued pretraining where actually discussed.

# Appendix

## Appendix A - Hyperparameters and Prompts

This appendix should report the training hyperparameters for S2+R, S3+R, and the retrieval-free controls, together with the frozen judge-prompt summary and any implementation details needed for reproducibility. The main text should refer here rather than reproduce configuration blocks inline.

## Appendix B - Extra Tables and Figures

Appendix Table B1 should summarize the practical trade-off among S1, S2+R, S3+R, S2, and S3. S7 should not appear in the cost-comparable body of that table because it inherits prior training cost from both source adapters. Appendix Figure B2 should present the error-overlap UpSet plot. Appendix Figure B3 should present the judge criteria profile. Additional auxiliary figures should be added only if they directly support a discussion point in the main text.

## Appendix C - D2L Engineering Note

This appendix should explain that D2L was originally intended to test document-conditioned adapter generation, but the implemented system required chunk-level workaround packaging in practice. The note should state plainly that the branch is retained as a legacy engineering diagnostic and negative control rather than as a headline result.

## Appendix D - Use of Generative AI

This appendix should name the tools used, describe the scope of assistance, and state that responsibility for the final manuscript remains with the author. If the institutional template requires explicit marking of substantially AI-assisted passages in the manuscript itself, that marking should be applied during the final formatting pass.
