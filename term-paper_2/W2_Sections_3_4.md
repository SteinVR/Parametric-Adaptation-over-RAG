# 3. Benchmark and Experimental Setup

## 3.1 Corpus and Benchmark

The benchmark is built from eight DIFC legal documents comprising 176 pages and approximately 115 thousand tokens. The corpus includes statutes, regulations, and case decisions, which gives the evaluation set a mix of local extraction tasks and broader interpretive questions while remaining small enough for controlled experimentation on consumer hardware.

The gold set contains 200 human-authored question-answer pairs. Answer types are distributed across free-text, boolean, number, name, names, and date questions, and the benchmark also includes unanswerable items. Difficulty labels span easy, medium, and hard cases. A total of 26 questions, or 13 percent of the benchmark, require evidence from more than one document. This is a useful property because it creates a natural stress test for systems that may differ in local contextualization versus cross-document aggregation.

The final split is fixed at 150 train / 50 eval. The split is stratified by answer type, difficulty, and the single-document versus multi-document distinction. This frozen split is used throughout the paper. The compact size of the benchmark limits the breadth of claims that can be made, but it also makes controlled cross-system comparison feasible under one hardware regime.

Table 1 should be placed here.

## 3.2 Hardware, Shared Backbone, and Variance Policy

All systems are evaluated under the same hardware constraint: an RTX 4060 with 8 GB of VRAM and 32 GB of RAM. The shared backbone across the active systems is Gemma-2-2b-it. This common infrastructure is important because it removes a large source of confounding variation. The headline systems differ in adaptation signal, not in backbone family or deployment environment.

For trained systems, variance is handled through three random seeds, namely 42, 123, and 777, and mean plus standard deviation is reported where relevant. This variance policy is modest, but it provides a clearer view of stability than a single run would. The evaluation remains anchored to one frozen evaluation split rather than cross-validation, which keeps all compared systems on the same test set.

Detailed hyperparameters are not repeated in the main text because they are not the center of the argument. They are deferred to the appendix, where they support reproducibility without interrupting the logic of the comparison.

## 3.3 Fixed Retrieval Backbone

The retrieval backbone is held constant across all retrieval-aware systems. It combines hybrid dense and sparse retrieval, reciprocal-rank fusion, reranking, and evidence compression. In operational terms, that means S1, S2+R, S3+R, and S7 receive their evidence through the same retrieval pipeline rather than through system-specific retrieval variants.

This frozen retrieval design is essential for interpretation. Because the evidence path is constant, differences among retrieval-aware systems should be read primarily as differences in how the generator uses the same retrieved context. The grounding metric therefore functions as a control on the shared pipeline. The nearly identical grounding values across retrieval-aware systems do not suggest that adaptation is irrelevant. They indicate that the main source of variation lies in generation conditioned on fixed evidence rather than in evidence selection itself.

The retrieval description is kept in prose because the visual budget of the paper is better spent on cross-system comparisons. The main methodological point is that retrieval is strong, shared, and frozen before adaptation results are interpreted.

# 4. Compared Systems and Evaluation Protocol

## 4.1 System Inventory

The compared systems occupy different methodological roles. The headline comparison consists of S1, S2+R, and S3+R. S1 is the strong nonparametric baseline, S2+R is the supervised retrieval-aware adapter trained with RAFT-style open-book supervision, and S3+R is the retrieval-aware CLM adapter obtained through continued pretraining on the corpus text. These three systems define the main thesis comparison.

S7 is reported separately as an exploratory post-hoc result. It is obtained by linearly merging the CLM and RAFT adapters without retraining, then evaluating the merged adapter inside the same S1 retrieval stack. Because S7 inherits prior training effort and is not a separately trained system, it is reported outside the headline branch.

The pure parametric systems S2 and S3 serve as controls. They clarify the limits of parametric memory without retrieval rather than competing for the main claim. The legacy D2L branch, labeled S3-legacy in comparison tables, is retained as an engineering negative control. It documents a non-competitive document-conditioned packaging route under the present implementation constraints. Figure 1 and Table 2 should be placed here to summarize this system inventory.

## 4.2 Training Setups

The adapted systems differ in signal rather than in surrounding scaffolding. S2+R is trained with retrieval-conditioned supervision built from the 150 train portion of the benchmark. Each training instance pairs a question with gold evidence chunks and distractors, and the model is trained to produce the answer from that evidence-conditioned prompt. This makes the training objective closely aligned with downstream document-grounded QA.

S3 is trained differently. Its adapter is learned through causal language modeling on the concatenated text of all eight corpus documents, without question-answer labels. The corresponding retrieval-aware system S3+R then uses that adapter inside the fixed S1 retrieval pipeline. This yields a symmetric comparison with S2+R: same backbone, same PEFT basis, same retrieval pipeline, but a different adaptation signal.

The no-retrieval controls remove retrieval at inference time and thereby measure the limits of internalized or partially internalized knowledge. The D2L branch is described only briefly in the main text because its full engineering details do not pay for their page budget here. The relevant point is that a token-level audit suggested single-pass feasibility, but the released implementation imposed stricter effective limits in practice, which required chunked packaging and left D2L as a legacy negative control. Full training details and hyperparameters are reported in Appendix A.

## 4.3 Evaluation Protocol

The evaluation combines aggregate answer quality, component-level answer quality, grounding, and practical systems metrics. The primary score is Q_main, defined as 0.7 times S_det plus 0.3 times S_asst. This weighting prioritizes deterministic extraction while still crediting assistant-style quality on free-text answers. S_det captures exact or near-exact correctness for deterministic answer types such as boolean, number, name, names, and date. S_asst captures judged free-text answer quality.

Grounding is reported only for retrieval-aware systems and is measured as page-level F-beta with beta equal to 2.5. In this paper, grounding should be read as a control on the fixed retrieval backbone rather than as a primary differentiator among retrieval-aware systems. Because S1, S2+R, S3+R, and S7 share the same retrieval stack, identical grounding values indicate common evidence access rather than identical generation behavior.

The protocol also records systems metrics such as time to first token, end-to-end latency, peak inference VRAM, and offline cost. These metrics matter for practical interpretation, but the paper avoids collapsing them into a misleading latency-only cost narrative. Instead, quality and resource expenditure are interpreted together, with direct offline-cost comparison restricted to systems that are genuinely comparable in training or packaging effort.
