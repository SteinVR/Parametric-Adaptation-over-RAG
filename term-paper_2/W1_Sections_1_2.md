# 1. Introduction

## 1.1 Problem and Motivation

Document-grounded legal question answering places unusual pressure on both factual precision and answer discipline. The target answer is often tied to a specific provision, date, list element, or procedural distinction, which makes unsupported generation especially costly. In that setting, improvements should not be credited merely because a model appears more fluent. They should be credited only when they improve document-bound answering under controlled conditions.

This constraint becomes more consequential on consumer hardware. When training and inference must fit within an RTX 4060 with 8 GB of VRAM and 32 GB of RAM, the design space shifts away from large-scale retraining and toward retrieval engineering, compact backbones, and parameter-efficient adaptation. The practical research question is therefore whether adaptation remains useful once the baseline is already a strong retrieval-augmented generation pipeline.

The present study addresses that question on a compact legal benchmark built from eight DIFC documents and a frozen evaluation protocol. It asks whether parametric adaptation adds measurable value beyond a strong fixed RAG baseline and whether different adaptation signals produce different quality profiles under the same infrastructure constraints.

## 1.2 Research Questions and Scope

The study is intentionally narrow. It uses one backbone family, one shared retrieval stack, one frozen benchmark split, and one hardware regime in order to isolate the effect of adaptation signal. The benchmark contains eight DIFC legal documents and 200 human-authored question-answer pairs. The final split is fixed at 150 train / 50 eval, and all compared systems are evaluated on the same 50-item evaluation set.

Within that setup, the main comparison is restricted to three retrieval-aware systems: the strong RAG baseline S1, the RAFT-style adapted system S2+R, and the CLM-adapted system S3+R. This comparison answers the main research question: whether parametric adaptation improves a strong RAG baseline, and how supervised retrieval-conditioned adaptation differs from supervision-free continued pretraining when both are used inside the same retrieval pipeline. A second research question examines the limits of pure parametric memory by comparing retrieval-aware systems with no-retrieval controls.

The scope is deliberately bounded. The paper does not claim to settle the general value of parametric memory for legal QA, and it does not compare many backbones, retrieval stacks, or training recipes. Its contribution depends on holding those factors constant so that the contrast between adaptation signals remains interpretable.

## 1.3 Contributions

The contribution of the paper is a controlled empirical comparison rather than a new architecture. First, it compares RAFT-style supervised adaptation and CLM continued pretraining on top of the same fixed RAG pipeline, which makes the difference in training signal analytically visible. Second, it quantifies how far pure parametric systems can go without retrieval on the same benchmark and under the same hardware constraints, thereby clarifying whether retrieval remains indispensable. Third, it reports a post-hoc adapter-merge result, S7, as exploratory evidence that supervised and corpus-level adaptation may encode partially complementary strengths, while keeping that result secondary to the headline comparison.

## 1.4 Structure of the Paper

The remainder of the paper follows a compact experimental structure. Section 2 introduces the background needed to position retrieval and parameter-efficient adaptation in document-grounded legal QA. Section 3 describes the benchmark, hardware setting, and fixed retrieval backbone. Section 4 presents the compared systems and the evaluation protocol. Section 5 reports the main empirical results, including the headline comparison, per-type breakdowns, retrieval controls, and the single-document versus multi-document analysis. Section 6 discusses the answers to the research questions, the error topology, and the main limitations. Section 7 concludes. The appendix collects hyperparameters, extra tables and figures, the D2L engineering note, and the disclosure of generative AI use.

# 2. Background and Related Work

## 2.1 RAG as Nonparametric Memory

Retrieval-augmented generation can be treated as a nonparametric memory mechanism. Instead of requiring the model to encode all relevant information in its parameters, the system retrieves external evidence at inference time and conditions generation on that evidence. In legal QA, this distinction is particularly useful because answer quality depends less on open-ended completion ability than on the accurate use of document-bound facts. A system that can recover the relevant pages or clauses at inference time has a direct path to grounded answering that does not rely on internalizing the entire corpus.

That perspective is methodologically important for the present study. The baseline is a retrieval-aware pipeline with hybrid search, reranking, and evidence compression. This means that any gain from downstream adaptation must be interpreted relative to an already strong external memory mechanism. The paper therefore studies parametric adaptation as an addition to nonparametric memory while treating retrieval as the foundation of the comparison.

## 2.2 Parameter-Efficient Adaptation on Consumer Hardware

Parameter-efficient fine-tuning is central to the project because the hardware budget is not incidental. QLoRA makes it possible to adapt a compact instruction-tuned model within the memory limits of consumer-grade hardware by freezing the base model and training a small number of low-rank parameters under 4-bit quantization. Under the present resource constraints, that design is not merely convenient. It is what makes adaptation experimentally feasible without changing the backbone family or requiring large-scale compute.

This constraint also improves interpretability. Because the compared adapted systems rely on the same PEFT basis, differences between them can be attributed to their training signals rather than to different adaptation mechanisms. The paper therefore treats PEFT as part of the experimental design, not as an optimization detail.

## 2.3 RAFT-style Adaptation vs. CLM Continued Pretraining

The central comparison in this paper is between two adaptation signals. RAFT-style adaptation uses retrieval-conditioned supervision: the model is trained on question-answer examples paired with supporting evidence, so the adaptation objective directly reflects the downstream QA task. CLM continued pretraining uses a different signal. It exposes the model to corpus text through next-token prediction without explicit QA supervision, allowing the adapter to absorb corpus-level distributional regularities without training on labeled answers.

Because both approaches are applied to the same backbone and later evaluated inside the same retrieval stack, the contrast isolates a substantive methodological difference. RAFT-style adaptation may favor deterministic extraction because it is trained directly on answer production under evidence conditioning. CLM adaptation may favor assistant-style answer quality because it continues to shape the model's local contextualization behavior without task-specific labels. The empirical question is which of these tendencies becomes visible once both are tested against the same strong RAG baseline.

## 2.4 Research Gap and Positioning

The study is positioned against broad comparisons among abstract memory families. That framing often obscures the more practical question faced in a constrained experimental setting: if a retrieval pipeline is already strong and fixed, does adapter-based adaptation still add value, and what kind of adaptation signal matters most? This paper answers that narrower question under one benchmark, one backbone family, and one hardware regime.

That scope makes the result more controlled, even if it makes the claims less general. The paper does not argue that any single method family is universally best for legal QA. It argues that, on this benchmark and under these constraints, a strong retrieval baseline is difficult to beat, adaptation can still provide moderate gains, and those gains depend on the training signal rather than on the mere presence of an adapter.
