# {Title Page Placeholder}

University, department, module, semester, author details, supervisor details, and submission date should be inserted here according to the institutional template.

---
# Declaration of Academic Integrity Placeholder

The declaration should be inserted here in the exact wording required by the institution.

---
# Table of Contents

1. Introduction
2. Background and Related Work
3. Benchmark and Experimental Setup
4. Compared Systems and Evaluation Protocol
5. Results
6. Discussion and Limitations
7. Conclusion
- References
- Appendix A: Hyperparameters and Prompts
- Appendix B: Supplementary Tables and Figures
- Appendix C: D2L Engineering Note
- Appendix D: Use of Generative AI

---

## 1. Introduction

### 1.1 Problem and Motivation

Legal document-grounded question answering requires systems that can extract precise factual details from authoritative texts while producing well-reasoned explanations when a direct lookup is insufficient. In practical deployments on consumer hardware, where GPU memory is limited to a single 8 GB card, scaling to larger language models is not an option. Instead, practitioners face a concrete engineering choice: invest in a strong retrieval-augmented generation (RAG) pipeline, apply parameter-efficient fine-tuning to the generator, or combine both.

This study investigates whether parametric adaptation retains practical value when the baseline is not a bare generator but a carefully engineered document-grounded RAG system. The question is not whether RAG helps --- it demonstrably does --- but whether *additionally* adapting the generator's parameters yields measurable gains once the retrieval backbone is already strong. The experimental setting is deliberately constrained to consumer hardware, where retrieval engineering and parameter-efficient methods represent the most feasible adaptation strategies.

### 1.2 Research Questions and Scope

To isolate the effect of parametric adaptation from confounding factors, the study adopts a deliberately narrow scope: one compact legal benchmark, one frozen language model backbone, one hardware configuration, and one fixed retrieval stack. Within this controlled setting, two research questions guide the investigation:

**RQ1.** Does parametric adaptation yield measurable gains over a strong RAG baseline on a compact legal benchmark under consumer-hardware constraints, and how do RAFT-style supervised adaptation and supervision-free CLM continued pretraining differ as retrieval-conditioned generators?

**RQ2.** How far can pure parametric systems reach without retrieval on this benchmark, and does retrieval remain indispensable?

The benchmark comprises 8 legal documents from the Dubai International Financial Centre (DIFC) regulatory corpus and 200 human-authored QA pairs spanning six answer types. All systems share the same frozen 50-question evaluation set, while supervised training uses the remaining 150 questions. This narrow scope is methodologically intentional: it permits a controlled comparison of training signals under identical infrastructure.

### 1.3 Contributions

The paper makes three contributions:

1. A controlled comparison between RAFT-style supervised adaptation and CLM continued pretraining on top of an identical RAG backbone, isolating training signal as the sole variable.
2. A quantification of the limits of pure parametric memory by contrasting retrieval-aware systems against no-retrieval controls.
3. A post-hoc adapter-merge result suggesting partial complementarity between the two training signals, reported as an exploratory finding.

### 1.4 Structure of the Paper

Section 2 provides background on RAG, parameter-efficient adaptation, and the two training paradigms under comparison. Section 3 describes the benchmark corpus, hardware constraints, and the fixed retrieval backbone. Section 4 defines the compared systems and the evaluation protocol. Section 5 presents the experimental results, including aggregate comparisons, per-type analyses, and a single-document versus multi-document breakdown. Section 6 discusses the findings in light of the research questions, analyzes common error patterns, and acknowledges the study's limitations. Section 7 concludes with practical takeaways and directions for future work.

---

## 2. Background and Related Work

### 2.1 RAG as Nonparametric Memory

Retrieval-augmented generation (RAG) delegates document knowledge to an external retrieval pipeline rather than requiring the generator to memorize corpus content in its parameters (Lewis et al., 2020). At inference time, a query is used to retrieve relevant passages from an indexed corpus, and the retrieved evidence is prepended to the generator's input context. The generator then conditions its output on both the query and the retrieved text.

For legal QA, this externalization is particularly attractive because answers must be traceable to specific regulatory provisions. A RAG system can, in principle, cite the pages from which an answer was derived, providing a form of evidence grounding that purely parametric systems lack. On consumer hardware, where model size is constrained, retrieval also compensates for the limited world knowledge that a compact model can encode in its parameters.

### 2.2 Parameter-Efficient Adaptation on Consumer Hardware

Full fine-tuning of even a 2-billion-parameter model requires storing optimizer states and gradients for all parameters, which exceeds the memory budget of an 8 GB consumer GPU. Low-Rank Adaptation (LoRA) addresses this by freezing the pretrained weights and injecting small trainable rank-decomposition matrices into selected attention layers (Hu et al., 2022). QLoRA extends this approach by quantizing the frozen backbone to 4-bit NormalFloat (NF4) precision, reducing memory consumption further while preserving adaptation quality (Dettmers et al., 2023). A comprehensive survey of PEFT methods, including LoRA variants and their trade-offs, is provided by Han et al. (2024). In this study, QLoRA enables controlled adaptation of the Gemma-2-2b-it backbone within a consumer-hardware budget. Parameter-efficient adaptation constitutes part of the experimental constraint rather than an incidental implementation detail: the same QLoRA configuration is shared across both adapted systems to prevent PEFT architecture from confounding the training-signal comparison.

### 2.3 RAFT-style Adaptation vs. CLM Continued Pretraining

The central experimental axis of this study contrasts two training signals applied to the same PEFT architecture.

**RAFT-style supervised adaptation.** Inspired by Retrieval-Augmented Fine-Tuning (Zhang et al., 2024), the adapter is trained on question--answer pairs where the input includes retrieved evidence chunks (both gold and distractor passages). This directly optimizes answer generation from evidence-rich contexts, exposing the adapter to the QA task distribution. The training signal is supervised: labels are the reference answers.

**CLM continued pretraining.** The adapter is trained on the raw corpus text using a standard causal language modeling (CLM) objective --- next-token prediction on all tokens. No QA labels or task-specific formatting are used. The adapter is exposed to the corpus distribution without any task-specific supervision, relying solely on the language modeling objective to absorb domain patterns.

These two paradigms represent fundamentally different assumptions about how parametric adaptation should interact with retrieval. RAFT-style training teaches the generator *how to use* retrieved evidence; CLM pretraining teaches it *what the corpus contains*.

### 2.4 Research Gap and Positioning

High-level discussions of parametric versus nonparametric knowledge injection are abundant in the literature. In the legal domain specifically, benchmarks such as LegalBench (Guha et al., 2023), LegalBench-RAG (Pipitone & Alami, 2024), and evaluation frameworks like LRAGE (Park et al., 2025) have evaluated LLM capabilities for legal reasoning and retrieval-augmented legal QA. However, controlled studies that isolate the training signal while holding the retrieval infrastructure, PEFT architecture, backbone, and evaluation protocol fixed remain scarce. Most comparisons involve different model families, different retrieval stacks, or different evaluation setups, making it difficult to attribute performance differences to the training signal alone. This study fills that gap for a specific, practically motivated setting: compact legal QA on consumer hardware.

---

## 3. Benchmark and Experimental Setup

### 3.1 Corpus and Benchmark

The benchmark is built on 8 PDF documents from the DIFC legal corpus, comprising statutes, regulations, and court judgments. Together, the documents span approximately 176 pages and 115,000 tokens. A pool of 200 question--answer pairs was authored by domain experts, covering six answer types: free-text explanations (53 questions), boolean lookups (48), numeric extractions (36), named entity lookups (30), multi-name lists (17), and date extractions (16). The distribution includes 26 multi-document comparative questions (13%) and 17 unanswerable questions (8.5%), ensuring that evaluation is not limited to simple single-document lookups.

The benchmark combines heterogeneous answer types --- from boolean lookups to free-text legal explanations --- rather than reducing evaluation to a single regime. This heterogeneity exposes distinct failure modes and prevents aggregate scores from masking type-specific weaknesses.

The 200 questions are split into 150 training questions and 50 evaluation questions, stratified by answer type, difficulty, and single-/multi-document status. All systems are evaluated on the identical 50-question evaluation set. Supervised training (RAFT-style adaptation) uses only the 150 training questions; CLM continued pretraining uses the raw document text and is therefore independent of the QA split.

### 3.2 Hardware, Shared Backbone, and Variance Policy

All experiments run on a single NVIDIA RTX 4060 with 8 GB VRAM and 32 GB system RAM. The language model backbone is Gemma-2-2b-it, an instruction-tuned model with approximately 2 billion parameters. The backbone is held constant across all systems to prevent architectural variance from confounding the training-signal comparison.

For systems that involve training (S2+R, S3+R, and their no-retrieval controls), three random seeds (42, 123, 777) are used, and results are reported as mean +/- standard deviation. No cross-validation is performed: the single frozen split is shared across all evaluations, and seed-level variance captures only the stochasticity introduced by the training process.

### 3.3 Fixed Retrieval Backbone

The retrieval stack is held constant across all retrieval-aware systems (S1, S2+R, S3+R, S7). It comprises a five-stage pipeline:

1. **Ingestion and hierarchical chunking.** Documents are parsed and split into five chunk families: page-level, section-level, clause-level, microchunks (300 tokens, 50-token overlap), and table blocks. Metadata --- including entities, dates, heading paths, and BM25 terms --- is extracted for each chunk.

2. **Hybrid retrieval.** Each query is embedded using Qwen3-Embedding-0.6B (384 dimensions) for dense retrieval and tokenized for BM25 sparse retrieval (k1 = 1.5, b = 0.75). Both channels prefetch 30 candidates.

3. **Reciprocal Rank Fusion (RRF).** Dense and sparse candidate lists are fused with equal weights and k = 60, producing a ranked list of 10 candidates.

4. **Cross-encoder reranking.** The top 10 candidates are reranked using Qwen3-Reranker-0.6B, and the top 5 are retained.

5. **Evidence compression.** A page-diverse compressor selects up to 3 chunks (at most one per physical page), and the corresponding (doc\_id, page\_number) pairs are lifted for grounding evaluation.

Because the retrieval backbone is identical for all retrieval-aware systems, performance differences should be attributed to how the generator utilizes retrieved evidence, not to differences in evidence selection. Exact configuration parameters are listed in Appendix A.

---

## 4. Compared Systems and Evaluation Protocol

### 4.1 System Inventory

The study evaluates seven systems that serve distinct roles. Three form the headline comparison, one provides an exploratory post-hoc result, and three serve as negative controls. Table 1 summarizes their key characteristics.

**Table 1. Compared systems and their roles.**

| System | Retrieval | Training signal | Supervision | Role |
|--------|-----------|-----------------|-------------|------|
| S1 | Yes | None | --- | Headline baseline |
| S2+R | Yes | RAFT-style QA | Supervised | Headline |
| S3+R | Yes | CLM on corpus | Unsupervised | Headline |
| S7 | Yes | Merged S2+R + S3+R | Post-hoc | Exploratory |
| S2 | No | RAFT-style QA | Supervised | Control |
| S3 | No | CLM on corpus | Unsupervised | Control |
| S3-legacy | No | D2L hypernetwork | Supervised | Legacy control |

*[Figure 1. System overview schematic --- to be inserted. Shows the pipeline for each system class: S1 routes queries through the shared retrieval stack to the base generator; S2+R and S3+R route through retrieval to an adapted generator; S7 uses a merged adapter; controls bypass retrieval entirely.]*

**S1 (Classical RAG)** serves as the nonparametric baseline: the frozen Gemma-2-2b-it generator receives retrieved evidence and produces answers without any adapter.

**S2+R (RAFT-style QLoRA + retrieval)** and **S3+R (CLM + retrieval)** are the two headline adapted systems. Both use the same retrieval stack as S1 and the same QLoRA architecture (rank 32, alpha 32, dropout 0.05, targeting q\_proj and v\_proj). They differ only in training signal: S2+R is trained on question--answer pairs with retrieved context (RAFT-style), while S3+R is pretrained on raw corpus text via causal language modeling.

**S7 (Post-hoc adapter merge)** linearly interpolates the S2+R and S3+R adapters with equal weights (alpha = 0.5) without any additional training. It is reported separately as an exploratory result because it represents a post-hoc interpolation rather than a retrained system.

**S2 (closed-book RAFT)** and **S3 (CLM without retrieval)** are parametric controls that use the same trained adapters as S2+R and S3+R but bypass retrieval at inference time, receiving only the question. **S3-legacy (D2L)** is a historical negative control using a Doc-to-LoRA hypernetwork approach (Charakorn et al., 2026) that required a chunk-level workaround due to memory constraints; engineering details are in Appendix C.

### 4.2 Training Setups

Both S2+R and S3+R employ identical QLoRA configurations applied to the same frozen backbone; the sole difference is the training objective. The comparison thus isolates training signal rather than model family or PEFT architecture.

**S2+R training.** The adapter is fine-tuned for 3 epochs on the 150 training questions in RAFT format. Each training example consists of the question, gold evidence chunks (matched to gold retrieval pages), and 2 distractor chunks from unrelated documents. The target is the reference answer. Learning rate is 2 x 10^-4 with cosine decay and 3% warmup. Maximum sequence length is 4096 tokens. Training takes approximately 20 minutes per seed on the RTX 4060.

**S3+R training.** The adapter is pretrained for 5 epochs on the concatenated corpus text (~115K tokens) using a CLM objective. Learning rate is 5 x 10^-5 with cosine decay and 10% warmup. Maximum sequence length is limited to 512 tokens because computing CLM cross-entropy over the full Gemma vocabulary (~256K tokens) at longer sequences exceeds 8 GB VRAM. Training takes approximately 10 minutes per seed.

**S7.** No training is performed. The S2+R and S3+R adapter weight matrices are linearly interpolated: W\_merged = 0.5 * W\_S2+R + 0.5 * W\_S3+R.

### 4.3 Evaluation Protocol

The evaluation protocol combines deterministic scoring for structured answer types with judge-based assessment for free-text responses.

**Composite metric.** The primary metric is Q\_main = 0.7 * S\_det + 0.3 * S\_asst, weighting deterministic extraction at 0.7 and judged free-text quality at 0.3.

**Deterministic score (S\_det).** For boolean, number, name, and date questions, scoring is binary exact match (with minor normalization). For multi-name lists, the score is the Jaccard similarity between predicted and gold name sets. Unanswerable questions expect an empty response; both empty yields 1.0, any mismatch yields 0.0.

**Free-text score (S\_asst).** Free-text responses are evaluated by GPT-5.4-mini (OpenAI, reasoning effort = medium) against 5 binary criteria: correctness, completeness, grounding, calibration, and clarity (following the LLM-as-judge paradigm; see Pradhan et al., 2025 for a discussion of this approach in legal RAG evaluation). The per-question score is the mean of the 5 criteria; S\_asst is the mean across all free-text questions.

**Grounding (G).** For retrieval-aware systems, grounding is computed as F\_beta (beta = 2.5) on page-level (doc\_id, page\_number) pairs, comparing the final evidence set against gold retrieval references. The elevated beta emphasizes recall, penalizing missing gold pages more than including extra pages. Because the retrieval stack is fixed, grounding primarily reflects the shared evidence pipeline rather than adapter-specific behavior. The constant G = 0.567 across all retrieval-aware systems is a direct consequence of this design choice: the adapters change only how the generator uses evidence, not which evidence is retrieved.

**Operational metrics.** Latency (time-to-first-token and end-to-end), peak inference VRAM, offline training cost, and malformed output rate are reported for all systems.

---

## 5. Results

### 5.1 Main Comparison

Table 2 presents the aggregate results across all systems. The headline systems are grouped at the top, followed by the exploratory post-hoc merge, and then the negative controls.

**Table 2. Main results on the 50-question evaluation set.** Trained systems report mean +/- std across 3 seeds. Offline cost is per-seed wall-clock training time. S7 inherits the combined offline cost of S2+R and S3+R.

| | Q\_main | S\_det | S\_asst | G | Latency (ms) | VRAM (MB) | Offline (s) |
|---|---------|--------|---------|------|--------------|-----------|------------|
| **Headline** | | | | | | | |
| S1 (RAG) | 0.643 | 0.601 | 0.739 | 0.567 | 479 | 5201 | --- |
| S2+R (RAFT) | 0.669 +/- 0.014 | 0.648 +/- 0.015 | 0.718 +/- 0.018 | 0.567 | 492 | 3069 | 1206 |
| S3+R (CLM) | 0.667 +/- 0.023 | 0.599 +/- 0.016 | 0.826 +/- 0.062 | 0.567 | 525 | 3069 | 581 |
| **Post-hoc** | | | | | | | |
| S7 (merge) | 0.705 +/- 0.035 | 0.679 +/- 0.048 | 0.764 +/- 0.018 | 0.567 | 527 | 3069 | 1787* |
| **Controls** | | | | | | | |
| S2 (closed) | 0.263 +/- 0.005 | 0.270 | 0.246 | --- | 257 | 3067 | 88 |
| S3 (no retr.) | 0.185 +/- 0.003 | 0.135 | 0.303 | --- | 195 | 3077 | 581 |
| S3-leg. (D2L) | 0.210 | 0.135 | 0.385 | --- | 179 | 3072 | 3932 |

\* S7 offline cost = S2+R (1206 s) + S3+R (581 s); the merge itself is instantaneous.

The nonparametric baseline S1 already reaches Q\_main = 0.643, establishing a high bar. Both retrieval-aware adapters improve over S1: S2+R attains 0.669 +/- 0.014 and S3+R attains 0.667 +/- 0.023. The improvements are moderate --- +0.026 for S2+R and +0.025 for S3+R --- but consistent across seeds.

S7 reaches the highest observed score (0.705 +/- 0.035) through post-hoc adapter interpolation. However, its higher seed variance (std = 0.035 vs. 0.014 and 0.023 for the headline adapters) and post-hoc nature warrant cautious interpretation. It should not supersede the controlled headline comparison.

*[Figure 2. Improvement over S1 baseline (Delta bars) --- to be inserted. Grouped bar chart showing Delta-Q\_main, Delta-S\_det, Delta-S\_asst for S2+R, S3+R, and S7 relative to S1. Illustrates that RAFT raises S\_det, CLM raises S\_asst, and the merge raises both.]*

### 5.2 Trade-off Between S2+R and S3+R

The headline comparison does not yield a single dominant system. Instead, the two adapters improve different quality dimensions.

S2+R achieves higher deterministic extraction scores (S\_det = 0.648 vs. 0.599), reflecting its supervised exposure to question--answer pairs with evidence context. S3+R achieves substantially higher free-text answer quality (S\_asst = 0.826 vs. 0.718), suggesting that CLM pretraining improves the generator's ability to produce well-structured legal explanations. On the aggregate Q\_main, the two systems are near-tied (delta = 0.002), with S2+R marginally ahead.

S3+R also incurs lower offline cost (581 s vs. 1206 s per seed), as it requires no task-specific label generation --- a consideration relevant under consumer-hardware constraints where training time competes with other workloads.

*[Figure 3. Judge criteria profile --- to be inserted. Radar or grouped bar chart comparing S1, S2+R, S3+R, and S7 on the 5 judge criteria (correctness, completeness, grounding, calibration, clarity). Shows that S3+R's advantage is concentrated in free-text quality dimensions.]*

### 5.3 By Answer Type

Type-level analysis reveals that performance differences between systems are concentrated in specific answer categories rather than distributed uniformly. Table 3 presents Q\_main scores broken down by the six answer types.

**Table 3. Scores by answer type on the 50-question evaluation set.** Headline and exploratory systems only; control systems are in Appendix B.

| | Boolean (n=12) | Number (n=7) | Name (n=8) | Names (n=5) | Date (n=5) | Free-text (n=13) |
|---|----------------|-------------|------------|-------------|------------|-------------------|
| S1 | 0.833 | 0.714 | 0.500 | 0.450 | 0.200 | 0.739 |
| S2+R | 0.889 | 0.714 | 0.625 | 0.261 | 0.400 | 0.718 |
| S3+R | 0.833 | 0.714 | 0.583 | 0.300 | 0.200 | 0.826 |
| S7 | 0.889 | 0.810 | 0.708 | 0.224 | 0.400 | 0.764 |

The strongest divergences appear between deterministic extraction and free-text explanation. S2+R outperforms S1 on boolean (+0.056), name (+0.125), and date (+0.200) types, consistent with its supervised training on structured answer extraction. S3+R shows its advantage primarily on free-text (+0.087 vs. S1), where judged quality benefits from the CLM adapter's exposure to corpus-level language patterns.

Date extraction and multi-name normalization remain weak spots across all systems: even the best system (S7) achieves only 0.400 on dates (n=5) and 0.224 on multi-name lists (n=5). These results point to persistent formatting and evidence-utilization limitations that neither training signal fully addresses.

S7 achieves the highest score in 4 of 6 types, including number (0.810) and name (0.708), providing further evidence that the two training signals are partially complementary.

*[Figure 4. Per-type score heatmap --- to be inserted. Heatmap showing S1, S2+R, S3+R, and S7 across the 6 answer types, with sample sizes in labels.]*

### 5.4 Retrieval Contribution and the Limits of Pure Parametric Memory

Removing retrieval causes severe quality collapse for both adaptation paradigms. Q\_main drops from 0.669 to 0.263 for RAFT (S2+R to S2, a gap of 0.406) and from 0.667 to 0.185 for CLM (S3+R to S3, a gap of 0.482). This pattern holds across both S\_det and S\_asst: for the CLM system, S\_det drops from 0.599 to 0.135 and S\_asst from 0.826 to 0.303.

The D2L legacy control (S3-legacy) reaches Q\_main = 0.210, slightly above the pure CLM control but far below any retrieval-aware system. Its S\_asst = 0.385 suggests that the hypernetwork-generated adapter retains some corpus-level language patterns, but without evidence retrieval this is insufficient for factual legal QA.

These results indicate that retrieval remains the dominant memory mechanism in this setting. Parametric adaptation without evidence access is insufficient, regardless of whether the adapter was trained with supervised QA labels (S2) or corpus-level language modeling (S3). This is one of the study's strongest conclusions, not a secondary observation.

### 5.5 Single-Document vs. Multi-Document Difficulty and Exploratory Adapter Fusion

Multi-document questions remain substantially harder than single-document questions across all systems. Table 4 presents this breakdown.

**Table 4. Q\_main by document scope (headline and exploratory systems).** Based on 42 single-document and 8 multi-document evaluation questions.

| | Single-doc | Multi-doc | Delta |
|---|-----------|-----------|-------|
| S1 | 0.694 | 0.338 | -0.356 |
| S2+R | 0.692 | 0.529 | -0.163 |
| S3+R | 0.719 | 0.338 | -0.381 |
| S7 | 0.716 | 0.621 | -0.095 |

S1 and S3+R both drop to Q\_main = 0.338 on multi-document items, while S2+R reaches 0.529 and S7 reaches 0.621. This pattern is the most granular evidence for signal complementarity observed in the study.

CLM continued pretraining appears to benefit single-document contextualization: S3+R achieves the highest single-doc score at 0.719, suggesting that corpus-level exposure helps the generator make better use of evidence from a single source. However, S3+R offers no improvement over S1 on multi-doc questions (both at 0.338), indicating that the CLM signal does not help with cross-document aggregation.

RAFT-style supervision confers greater robustness to multi-document composition: S2+R's multi-doc score (0.529) represents a 56% relative improvement over S1's 0.338. The RAFT training format, which includes distractors alongside gold chunks, may teach the generator to discriminate between relevant and irrelevant evidence --- a skill particularly valuable when evidence spans multiple documents.

The merged adapter S7 partially combines both advantages, achieving the highest scores in both regimes (0.716 single-doc, 0.621 multi-doc) and showing the smallest single-to-multi-doc gap (delta = -0.095 vs. -0.356 for S1). This provides the most direct evidence that the two training signals are partially complementary, though the result remains exploratory given S7's post-hoc nature.

*[Figure 5. Single-doc vs. multi-doc comparison --- to be inserted. Paired bar chart with systems on the x-axis, Q\_main on the y-axis, and two series (single-doc, multi-doc). Annotated with per-system delta.]*

---

## 6. Discussion and Limitations

### 6.1 Answer to RQ1

RQ1 asked whether parametric adaptation yields measurable gains over a strong RAG baseline and how RAFT-style and CLM adaptation differ. The answer is a qualified affirmative.

Both S2+R and S3+R improve over the nonparametric baseline S1, but the gains are moderate rather than transformative (+0.026 and +0.025 Q\_main respectively). The choice of training signal proves more consequential than the presence of an adapter per se. RAFT-style supervision improves deterministic extraction (S\_det: +0.047 over S1) at the cost of a slight decrease in free-text quality (S\_asst: -0.021). CLM continued pretraining improves free-text answer quality (S\_asst: +0.087) while leaving deterministic extraction essentially unchanged (S\_det: -0.002). These complementary profiles mean that the optimal system depends on the deployment priority: factual precision favors S2+R, while explanation quality favors S3+R.

The post-hoc merge S7 achieves the highest aggregate score (0.705), suggesting that the two signals are partially complementary. Recent work on LoRA adapter composition (Prabhakar et al., 2024) has shown that carefully designed merge schemes can approach multi-task training quality without retraining; more structured alternatives such as rank-wise clustering (Zhao et al., 2024) suggest further room for improvement. S7's result is consistent with these findings. However, because S7 was not retrained and was identified post-hoc, this finding should be interpreted as a direction for future work rather than a validated deployment recommendation.

### 6.2 Answer to RQ2

RQ2 asked whether pure parametric systems can substitute for retrieval. The answer is unambiguous: retrieval remains indispensable in this setting.

Neither supervised closed-book adaptation (S2, Q\_main = 0.263) nor corpus-level CLM pretraining without retrieval (S3, Q\_main = 0.185) provides a viable substitute for external evidence retrieval. The D2L legacy control (Q\_main = 0.210) corroborates this from a third direction. On a compact legal benchmark where the corpus fits within the token budgets of larger models, a 2-billion-parameter model simply cannot internalize sufficient factual detail to answer legal questions without external evidence.

### 6.3 Error Analysis

Error overlap analysis reveals that 15 of the 50 evaluation questions are missed by all headline systems (S1, S2+R, S3+R, and S7), indicating hard cases likely rooted in retrieval coverage gaps or inherent question ambiguity. The Jaccard overlap coefficient across headline systems is 0.714, confirming that the systems share most of their failure modes.

Persistent failure patterns include date extraction (scores at or below 0.400 for all systems), multi-name list normalization (at or below 0.450), and cross-document composition. Among the 15 universally missed questions, recurring themes include unanswerable questions where the gold answer is null, questions requiring information from document regions not well covered by the 3-chunk evidence budget, and questions demanding multi-step cross-document reasoning.

Local wins by individual systems are sparse: 2 questions are answered correctly only by S1, 2 only by S3+R, and 0 only by S2+R. This limited local complementarity suggests that while the systems have different strengths in aggregate, their per-question advantages rarely translate into exclusive wins --- consistent with the modest aggregate deltas observed in Section 5.1.

### 6.4 Limitations

These findings are bounded in several important respects:

- **Compact corpus.** The benchmark comprises only 8 documents (~115K tokens). Results may not generalize to larger, more heterogeneous corpora.
- **Small evaluation set.** With 50 evaluation questions, per-type sample sizes are small (as few as n=5 for dates and multi-name lists), limiting statistical power for type-level conclusions.
- **Single backbone.** All experiments use Gemma-2-2b-it. Different model families or scales might alter the relative benefit of parametric adaptation.
- **Fixed retrieval stack.** Because retrieval is frozen, the study measures differences in evidence-conditioned generation but cannot assess how adapters interact with retrieval quality.
- **Judge-based free-text scoring.** S\_asst depends on GPT-5.4-mini evaluations rather than human judgments, introducing potential systematic biases.
- **Post-hoc S7.** The adapter merge was identified and evaluated after the main experiments; it was not included in the original experimental plan and should not be treated as a pre-registered result.
- **D2L as legacy control.** The Doc-to-LoRA approach required an engineering workaround (chunk-level adapter merging) that may not reflect its theoretical potential; it is included only as a historical data point.

---

## 7. Conclusion

This study investigated whether parametric adaptation adds measurable value on top of a strong RAG baseline for document-grounded legal QA on consumer hardware. The main findings are:

1. **A strong RAG baseline is already highly effective.** The nonparametric S1 system achieves Q\_main = 0.643 on the DIFC legal benchmark, setting a high bar for any further adaptation.

2. **Parametric adaptation helps, but training signal matters more than the mere presence of an adapter.** Both RAFT-style supervised adaptation (S2+R) and CLM continued pretraining (S3+R) improve over S1, but they do so along different quality axes: RAFT improves deterministic extraction, CLM improves free-text answer quality. The aggregate gains are moderate (+0.026 and +0.025 Q\_main).

3. **Retrieval remains non-substitutable.** Pure parametric controls without evidence access collapse to Q\_main below 0.27, regardless of training signal. On this compact legal benchmark with a 2-billion-parameter model, retrieval is the dominant memory mechanism.

4. **Post-hoc adapter fusion suggests complementarity.** The merged S7 system achieves the highest observed Q\_main (0.705) and shows the strongest multi-document performance, indicating that supervised and unsupervised training signals encode partially complementary information.

The practical takeaway is that, under consumer-hardware constraints, investing in retrieval engineering yields the largest gains; parametric adaptation provides additional --- but modest --- improvement, and its value depends on the type of quality sought. Future work should target multi-document reasoning (where even the best system achieves only 0.621 Q\_main), unanswerable-question calibration, and retrieval-aware adaptation strategies that explicitly model cross-document evidence composition.

---

## References

- Charakorn, R., Cetin, E., Uesaka, S., & Lange, R. T. (2026). Doc-to-LoRA: Learning to instantly internalize contexts. *arXiv preprint arXiv:2602.15902*. https://arxiv.org/abs/2602.15902

- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems, 36*. https://arxiv.org/abs/2305.14314

- Guha, N., Nyarko, J., Ho, D. E., Re, C., Chilton, A., Narayana, A., & others. (2023). LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. *arXiv preprint arXiv:2308.11462*. https://arxiv.org/abs/2308.11462

- Han, Z., Gao, C., Liu, J., Zhang, J., & Zhang, S. Q. (2024). Parameter-efficient fine-tuning for large models: A comprehensive survey. *arXiv preprint arXiv:2403.14608*. https://arxiv.org/abs/2403.14608

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *Proceedings of ICLR 2022*. https://arxiv.org/abs/2106.09685

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459--9474. https://arxiv.org/abs/2005.11401

- Park, M., Oh, H., Choi, E., & Hwang, W. (2025). LRAGE: Legal retrieval augmented generation evaluation tool. *arXiv preprint arXiv:2504.01840*. https://arxiv.org/abs/2504.01840

- Pipitone, N., & Alami, G. H. (2024). LegalBench-RAG: A benchmark for retrieval-augmented generation in the legal domain. *arXiv preprint arXiv:2408.10343*. https://arxiv.org/abs/2408.10343

- Prabhakar, A., Li, Y., Narasimhan, K., Kakade, S., Malach, E., & Jelassi, S. (2024). LoRA Soups: Merging LoRAs for practical skill composition tasks. *arXiv preprint arXiv:2410.13025*. https://arxiv.org/abs/2410.13025

- Pradhan, A., Ortan, A., Verma, A., & Seshadri, M. (2025). LLM-as-a-Judge: Rapid evaluation of legal document recommendation for retrieval-augmented generation. *arXiv preprint arXiv:2509.12382*. https://arxiv.org/abs/2509.12382

- Zhang, T., Patil, S. G., Jain, N., Shen, S., Zaharia, M., Stoica, I., & Gonzalez, J. E. (2024). RAFT: Adapting language model to domain specific RAG. *arXiv preprint arXiv:2403.10131*. https://arxiv.org/abs/2403.10131

- Zhao, Z., Shen, T., Zhu, D., Li, Z., Su, J., Wang, X., Kuang, K., & Wu, F. (2024). Merging LoRAs like playing LEGO: Pushing the modularity of LoRA to extremes through rank-wise clustering. *arXiv preprint arXiv:2409.16167*. https://arxiv.org/abs/2409.16167

---

## Appendix A. Hyperparameters and Prompts

### A.1 QLoRA Configuration (Shared)

| Parameter      | Value                          |
| -------------- | ------------------------------ |
| PEFT method    | QLoRA                          |
| Rank           | 32                             |
| Alpha          | 32                             |
| Dropout        | 0.05                           |
| Target modules | q\_proj, v\_proj               |
| Quantization   | 4-bit NF4, double quantization |
| Optimizer      | Paged AdamW 8-bit              |
| Scheduler      | Cosine                         |
| Weight decay   | 0.01                           |

### A.2 Training-Signal-Specific Parameters

| Parameter | S2+R (RAFT) | S3+R (CLM) |
|-----------|-------------|------------|
| Learning rate | 2 x 10^-4 | 5 x 10^-5 |
| Epochs | 3 | 5 |
| Warmup ratio | 0.03 | 0.10 |
| Max seq. length | 4096 | 512 |
| Effective batch size | 4 | 4 |
| Training data | 150 QA pairs (RAFT format) | ~115K tokens (raw corpus) |
| Supervision | Supervised (question -> answer) | Unsupervised (next-token) |

### A.3 Retrieval Pipeline Parameters

| Parameter | Value |
|-----------|-------|
| Embedding model | Qwen3-Embedding-0.6B (384-dim) |
| Sparse encoder | BM25 Okapi (k1=1.5, b=0.75) |
| Chunk size (microchunk) | 300 tokens |
| Chunk overlap | 50 tokens |
| Chunk families | page, section, clause, microchunk, table |
| Candidate prefetch | 30 (per channel) |
| RRF k | 60 |
| RRF weights | dense=1.0, sparse=1.0 |
| Post-fusion candidates | 10 |
| Reranker | Qwen3-Reranker-0.6B |
| Rerank budget | 5 |
| Evidence budget | 3 |
| Max chunks per page | 1 (page-diverse) |

### A.4 Generation Parameters

| Parameter | Value |
|-----------|-------|
| Model | Gemma-2-2b-it |
| Temperature | 0.0 (greedy) |
| Max new tokens | 256 |
| Constrained decoding | Boolean and names types (via Outlines) |

### A.5 Judge Prompt (Frozen)

**System:** "You are an impartial judge evaluating a legal QA system's response. Score each criterion as 1 (met) or 0 (not met). Return ONLY a JSON object."

**User template:**

```
Question: {question}
Reference answer: {reference_answer}
System response: {system_response}

Criteria:
1. correctness: Does the response contain the key information from the reference
   and no factual errors?
2. completeness: Does the response address all aspects of the question?
3. grounding: Is every claim supported by plausible legal reasoning
   (no hallucinated specifics)?
4. calibration: Does the response appropriately express uncertainty when
   information is missing?
5. clarity: Is the answer clear, concise, and directly addresses the question?

Return JSON: {"correctness": 0|1, "completeness": 0|1, "grounding": 0|1,
              "calibration": 0|1, "clarity": 0|1}
```

**Judge model:** GPT-5.4-mini (OpenAI), reasoning effort = medium.

---

## Appendix B. Supplementary Tables and Figures

### B.1 Control System Per-Type Breakdown

**Table B1. Scores by answer type for control systems.**

| | Boolean (n=12) | Number (n=7) | Name (n=8) | Names (n=5) | Date (n=5) | Free-text (n=13) |
|---|----------------|-------------|------------|-------------|------------|-------------------|
| S2 | 0.750 | 0.143 | 0.000 | 0.000 | 0.000 | 0.246 |
| S3 | 0.333 | 0.000 | 0.125 | 0.000 | 0.000 | 0.303 |
| S3-leg. | 0.333 | 0.000 | 0.125 | 0.000 | 0.000 | 0.385 |

### B.2 Seed-Level Variance

**Table B2. Per-seed Q\_main for trained systems.**

| Seed | S2+R | S3+R | S7 | S2 | S3 |
|------|------|------|------|------|------|
| 42 | 0.673 | 0.674 | 0.678 | 0.263 | 0.182 |
| 123 | 0.654 | 0.664 | 0.692 | 0.263 | 0.187 |
| 777 | 0.680 | 0.664 | 0.743 | 0.263 | 0.187 |
| Std | 0.014 | 0.023 | 0.035 | 0.005 | 0.003 |

*[Figure B1. Error overlap --- UpSet plot or heatmap showing set intersections of correct/incorrect answers across headline systems. To be generated.]*

*[Figure B2. Seed stability --- bar chart of per-seed Q\_main for trained systems. To be generated.]*

*[Figure B3. Pairwise win rates --- heatmap of head-to-head comparisons. To be generated.]*

---

## Appendix C. D2L Engineering Note

The Doc-to-LoRA (D2L) approach generates document-specific LoRA adapters via a hypernetwork, conditioning the adapter weights on document content. In principle, this would allow the generator to specialize to each document without supervised QA labels.

In this study's implementation, however, D2L required a chunk-level workaround. Although a preliminary token-based audit suggested that all 8 documents would fit a single-pass D2L encoding, the released D2L implementation imposed stricter effective memory limits, necessitating chunked packaging: each document was split into chunks, a separate adapter was generated for each chunk, and the resulting adapters were merged via linear interpolation. This multi-stage process introduced engineering complexity and a substantial offline cost (3932 seconds, compared to 1206 seconds for S2+R RAFT training).

The resulting system (S3-legacy) achieved Q\_main = 0.210 without retrieval, placing it between the two pure parametric controls (S2 = 0.263, S3 = 0.185) but far below any retrieval-aware system. Given these results and the engineering constraints, D2L was archived after the pilot evaluation and retained only as a legacy negative control. It should not be interpreted as a competitive alternative to the RAFT or CLM approaches evaluated in this study.

---

## Appendix D. Use of Generative AI

The following generative AI tools were used during the preparation of this work:

- **Claude (Anthropic):** Experiment orchestration, code generation for the evaluation and training pipelines, data analysis, and writing assistance (structure development, draft editing).
- **GPT-5.4-mini (OpenAI):** Used as the judge model for free-text answer evaluation (S\_asst scoring). The judge prompt is reproduced in Appendix A.5.
