# Term Paper Blueprint

## Purpose

This blueprint consolidates the two advisor reviews into a single writing plan for the final paper.

Base structure: Advisor 2's seven-section paper architecture plus appendix.

Integrated additions from Advisor 1:
- a compact system overview schematic in the methods section;
- a delta-to-S1 comparison figure to make the training-signal trade-off visually explicit;
- stronger treatment of single-document vs. multi-document performance as a central analytical result;
- an UpSet plot or compact overlap table as the preferred appendix treatment for shared failures;
- a practical quality-cost presentation that avoids the misleading latency-only "cost-quality" framing.

The paper should be written as a compact experimental study, not as a broad survey of memory paradigms. The main comparison is `S1` vs `S2+R` vs `S3+R`. `S7` remains exploratory. Pure parametric systems remain controls. `D2L` remains a legacy negative control.

## Working Title Options

Primary title:

**Parametric Adaptation over a Strong RAG Baseline for Document-Grounded Legal QA on Consumer Hardware: RAFT-style QLoRA vs. CLM Continued Pretraining**

Shorter alternative:

**Does Parametric Adaptation Add Value Beyond Strong RAG? A Compact Legal QA Study on Consumer Hardware**

Signal-focused alternative:

**Training Signal Matters: RAFT-style and CLM Adaptation over Fixed RAG in Legal QA**

## Research Questions

**RQ1.** Does parametric adaptation add value on top of a strong RAG baseline on a compact legal benchmark under consumer-hardware constraints, and how do RAFT-style supervised adaptation and CLM continued pretraining differ as retrieval-conditioned generators?

**RQ2.** How far can pure parametric systems go without retrieval on this benchmark, and does retrieval remain indispensable?

## Core Narrative

Use this as the paper's main line of argument:

> On a compact legal benchmark, a strong fixed RAG pipeline already provides a difficult-to-beat baseline. Parametric adaptation on top of that baseline yields moderate but meaningful quality gains, yet the type of training signal matters more than the mere presence of an adapter: RAFT-style supervision improves deterministic extraction, while corpus-level CLM improves assistant-style answer quality. Retrieval remains indispensable, because removing it leads to a large performance drop. A post-hoc merge of the two adapted systems suggests partial complementarity, but that result is exploratory rather than central.

Supplementary analytical emphasis:
- Multi-document questions are the strongest interpretive result after the headline comparison.
- `CLM+R` is stronger as a local contextualizer on single-document questions.
- `RAFT+R` is more helpful on multi-document aggregation/comparison.
- `S7` partially combines both effects.

## Writing Rules

Apply these style rules throughout the paper:
- Prefer positive declarative claims over contrast templates such as "not X but Y".
- Use cautious academic wording: `observed improvement`, `suggests`, `is consistent with`, `within this setup`.
- Avoid informal evaluators such as `very good`, `really strong`, `clean story`.
- Avoid overclaiming generality beyond this benchmark, backbone, and hardware setup.
- Default to prose paragraphs in the main text. Use bullet lists or numbered lists only when they communicate structure more clearly than prose and cannot be replaced cleanly by a paragraph or table.
- Prefer compact tables over bullet inventories for benchmark facts, system comparisons, and trade-off summaries.
- Treat `S7` as exploratory/post-hoc in every section where it appears.
- Treat `D2L` as a legacy engineering diagnostic / negative control, not as a co-equal branch of the thesis.
- State explicitly that grounding is effectively constant across retrieval-aware systems because the retrieval stack is fixed; improvements come from better use of the same retrieved context.
- Use only the final split description: `150 train / 50 eval`.
- Use one fixed visual plan unless page overflow forces appendix relocation; do not redesign the figure package while drafting.
- If needed, resolve the D2L audit inconsistency with one sentence: token-based audit suggested single-pass feasibility, but the released implementation imposed stricter effective limits or memory behavior, so chunked packaging was required in practice.
- In any practical cost comparison, do not plot `S7` as a cheap standalone system. Either exclude it from cost-comparable rows or label it explicitly as inheriting prior `S2+R` and `S3+R` training cost.
- If the institutional template requires explicit marking of GenAI-generated passages, mark every such passage in the manuscript itself and list the tools separately in the appendix.

## Science Work Style

Use this as the explicit writing standard for the final paper.

The paper should read like a compact academic study, not like a project memo, pitch, or advisor-facing outline. The default unit of writing is a short analytical paragraph. Claims should be stated directly, scoped carefully, and supported by results rather than rhetoric.

Preferred style:
- use positive declarative statements;
- use hedged academic phrasing where needed, such as `observed improvement`, `is consistent with`, `suggests`, `within this setup`, `under the present benchmark and hardware constraints`;
- make scope boundaries explicit instead of implying generality;
- describe results in terms of patterns, trade-offs, and bounded conclusions;
- prefer precise nouns such as `baseline`, `adaptation signal`, `retrieval-aware system`, `control`, `exploratory result`, and `evaluation setup`.

Avoid these red-flag patterns:
- contrast slogans such as `is not X but Y`;
- informal framing such as `clean story`, `good story`, `nice story`, `strong story`;
- vague praise such as `very good`, `really strong`, `super interesting`, `clearly better` without qualification;
- inflated claims such as `proved`, `definitively showed`, `solved`, `replaced retrieval`, `internalized the corpus`;
- thesis-defense filler such as `it is worth mentioning that`, `it should be noted that`, `in today’s world`, `as we all know`.

Use these replacements instead:
- replace `is not X but Y` with a direct positive statement of what the result is;
- replace `clean story` or `strong story` with `coherent empirical pattern` or `consistent interpretation`;
- replace `very good` or `really strong` with the specific metric-backed observation;
- replace `proved` with `showed`, `indicated`, or `suggested`, depending on evidential strength;
- replace broad claims with scoped forms such as `within this setup` or `on the evaluated benchmark`.

Non-negotiable drafting rule:
- if a sentence would sound natural in a product pitch, casual review, or lab chat, rewrite it into academic prose before keeping it.

## Front Matter

### Title Page

Include:
- university / department / module / semester;
- full paper title;
- author information;
- supervisor / lecturer information;
- submission date.

### Declaration of Academic Integrity

Place immediately after the title page.

### Table of Contents

Include all numbered sections and subsections. Start regular page numbering from the introduction.

## Section-by-Section Blueprint

## 1. Introduction (1.5-2 pages)

### 1.1 Problem and Motivation

**Section thesis**

Document-grounded legal QA on consumer hardware requires a stronger justification than simply adding more model capacity, which makes the value of parametric adaptation over a strong RAG baseline a practical research question.

**Content guidance**
- Motivate legal QA as a setting that requires factual grounding, precise extraction, and stable handling of document-bound answers.
- Explain why consumer-hardware constraints matter methodologically, not just practically.
- Introduce the central question as whether adaptation still matters once retrieval is already strong.
- Keep the opening tightly problem-driven; do not start with a generic history of memory-augmented NLP.

**Ready thesis statements**
- "This paper studies whether parametric adaptation remains useful when the baseline is not a weak generator-only system, but a strong document-grounded RAG pipeline."
- "The practical setting is deliberately constrained to consumer hardware, where retrieval engineering and parameter-efficient adaptation are more realistic than full model retraining."

**Figure/table placement**
- No figures here.

### 1.2 Research Questions and Scope

**Section thesis**

The study is intentionally narrow in order to isolate the effect of training signal under fixed infrastructure.

**Content guidance**
- State the benchmark boundaries: 8 DIFC documents, 200 QA pairs, 150/50 split.
- State the shared backbone and fixed retrieval pipeline.
- Introduce `RQ1` and `RQ2` directly.
- Clarify that the main comparison is `S1` vs `S2+R` vs `S3+R`; controls and `S7` serve different roles.

**Ready thesis statements**
- "The study is bounded to a compact DIFC benchmark with one frozen evaluation split, one backbone, and one shared retrieval stack."
- "This narrow design is a strength of the paper because it isolates the effect of adaptation signal under otherwise identical conditions."

**Figure/table placement**
- No figures here.

### 1.3 Contributions

**Section thesis**

The contribution is a controlled empirical comparison rather than a new architecture.

**Content guidance**
- Present the contributions as one compact prose paragraph.
- Only convert them into a short bullet list if the institutional template or supervisor explicitly expects list formatting in the introduction.
- Include the controlled comparison of RAFT-style vs CLM adaptation over the same RAG backbone.
- Include the control finding that retrieval remains indispensable.
- Include the exploratory complementarity signaled by `S7`, but mark it clearly as secondary.

**Ready thesis statements**
- "The paper contributes a controlled comparison between RAFT-style supervised adaptation and CLM continued pretraining on top of the same RAG baseline."
- "It also quantifies the limits of pure parametric memory by evaluating no-retrieval controls under the same benchmark and hardware constraints."
- "Finally, it reports a post-hoc adapter-merge result that suggests partial complementarity between the two adaptation signals."

**Figure/table placement**
- No figures here.

### 1.4 Structure of the Paper

**Section thesis**

The remainder of the paper follows a compact experimental-paper structure.

**Content guidance**
- One short paragraph summarizing Sections 2-7 and the appendix.

**Ready thesis statement**
- "The remainder of the paper introduces the necessary background, describes the benchmark and compared systems, presents the evaluation results, and then discusses their implications and limitations."

**Figure/table placement**
- No figures here.

## 2. Background and Related Work (1.5-2 pages)

### 2.1 RAG as Nonparametric Memory

**Section thesis**

RAG serves here as a nonparametric memory mechanism that externalizes corpus knowledge into retrieval rather than forcing the model to memorize it.

**Content guidance**
- Define RAG in 1-2 compact paragraphs.
- Explain why this matters for legal QA.
- Keep retrieval discussion conceptual; avoid implementation detail here.

**Ready thesis statements**
- "RAG can be viewed as a nonparametric memory mechanism in which external retrieval supplies task-relevant evidence at inference time."
- "This externalization is particularly relevant in legal QA, where answer quality depends on document-grounded factual precision rather than unconstrained generation."

**Figure/table placement**
- No figures here.

### 2.2 Parameter-Efficient Adaptation on Consumer Hardware

**Section thesis**

Parameter-efficient adaptation is methodologically central because the project is constrained to consumer hardware.

**Content guidance**
- Define LoRA and QLoRA briefly.
- State why QLoRA is appropriate for an 8GB setup.
- Emphasize that both adapted headline systems share the same PEFT basis.

**Ready thesis statements**
- "QLoRA enables controlled adaptation of a compact instruction-tuned model within the memory limits of consumer-grade hardware."
- "In this study, PEFT is not a secondary optimization choice but part of the experimental constraint itself."

**Figure/table placement**
- No figures here.

### 2.3 RAFT-style Adaptation vs. CLM Continued Pretraining

**Section thesis**

The paper compares two distinct adaptation signals rather than two arbitrary model variants.

**Content guidance**
- Explain RAFT-style adaptation as retrieval-conditioned supervised training.
- Explain CLM continued pretraining as corpus-level next-token training without QA labels.
- Emphasize that the key distinction is supervision type and training signal.

**Ready thesis statements**
- "RAFT-style adaptation uses retrieval-conditioned supervision to train answer generation from question-evidence pairs."
- "CLM continued pretraining instead adapts the model through corpus-level language modeling without explicit QA labels."
- "The comparison therefore isolates differences in adaptation signal rather than differences in backbone or retrieval infrastructure."

**Figure/table placement**
- No figures here.

### 2.4 Research Gap and Positioning

**Section thesis**

The paper addresses a narrower question than generic "memory family" comparisons by asking whether adaptation remains useful after strong retrieval has already been established.

**Content guidance**
- Position the study against overly broad memory-taxonomy framings.
- State that the paper is not trying to prove universal superiority of any method family.
- Set up the main comparison for the methods sections.

**Ready thesis statements**
- "The relevant question in this setup is not which memory paradigm is universally best, but whether adaptation still adds value once retrieval is already strong."
- "This framing yields a more controlled and practically meaningful comparison than a broad taxonomy-level survey of memory mechanisms."

**Figure/table placement**
- No figures here.

## 3. Benchmark and Experimental Setup (2 pages)

### 3.1 Corpus and Benchmark

**Section thesis**

The benchmark is compact but sufficiently structured to support a controlled analysis of grounded legal QA behavior.

**Content guidance**
- Describe the 8 DIFC documents, page count/token scale, and 200 QA pairs.
- Summarize answer-type distribution, multi-doc presence, and unanswerable cases.
- Use only the final `150 train / 50 eval` split.
- If space allows, give one sentence on why the compact size is useful for controlled experimentation.

**Ready thesis statements**
- "The benchmark consists of eight DIFC legal documents paired with 200 human-authored question-answer examples spanning multiple answer types."
- "Its compact size makes it feasible to compare retrieval and adaptation strategies under a controlled consumer-hardware setting."

**Figure/table placement**
- `Table 1. Benchmark Summary` here.
- Keep it compact: corpus size, QA count, split, answer types, multi-doc share, unanswerable share.

### 3.2 Hardware, Shared Backbone, and Variance Policy

**Section thesis**

All systems are compared under one fixed hardware and model regime to keep the experimental contrast interpretable.

**Content guidance**
- State backbone and hardware.
- State that headline systems use the same model family and PEFT basis.
- Explain seed reporting / variance treatment briefly.
- Avoid dumping full hyperparameters here.

**Ready thesis statements**
- "All systems were evaluated under the same hardware constraint and shared backbone, which keeps the comparison focused on training signal rather than infrastructure differences."
- "Variance is reported where relevant in order to distinguish stable trends from isolated runs."

**Figure/table placement**
- No figures here.

### 3.3 Fixed Retrieval Backbone

**Section thesis**

The retrieval stack is intentionally fixed so that changes in outcome can be attributed to how the generator uses the same evidence.

**Content guidance**
- Describe ingestion, hierarchical chunking, hybrid dense+sparse retrieval, fusion/reranking, and evidence compression at a compact level.
- State explicitly that this stack is shared by all retrieval-aware systems.
- Introduce the grounding interpretation here: `G` tracks a fixed retrieval pipeline, not adapter-specific evidence selection.

**Ready thesis statements**
- "The retrieval backbone was held constant across all retrieval-aware systems in order to isolate generator-side differences."
- "Because the evidence pipeline is fixed, differences among retrieval-aware systems should be interpreted primarily as differences in how they use the same retrieved context."

**Figure/table placement**
- Do not add a separate retrieval pipeline figure in the main text.
- Keep retrieval description in prose so the visual budget stays focused on the cross-system comparison.

## 4. Compared Systems and Evaluation Protocol (2-2.5 pages)

### 4.1 System Inventory

**Section thesis**

The compared systems occupy distinct methodological roles: headline systems, exploratory fusion, and controls.

**Content guidance**
- Introduce `S1`, `S2+R`, `S3+R`, `S7`, `S2`, `S3`, and `S3-legacy` in a role-aware order.
- Explicitly label headline systems, exploratory result, and controls.
- Keep `S6` out of the main narrative.

**Ready thesis statements**
- "The headline comparison is restricted to `S1`, `S2+R`, and `S3+R`, which differ in adaptation while sharing the same retrieval backbone and backbone model."
- "`S7` is evaluated as an exploratory post-hoc fusion rather than as a co-equal trained system."
- "Pure parametric systems and the legacy D2L branch are treated as controls that clarify the role of retrieval rather than as expected winners."

**Figure/table placement**
- `Figure 1. System Overview Schematic` here.
- The figure should show: `S1: retrieval -> generator`, `S2+R: retrieval -> RAFT-adapted generator`, `S3+R: retrieval -> CLM-adapted generator`, `S7: retrieval -> merged-adapter generator`, and `controls: no retrieval`.
- Size target: simple half-page schematic, not a dense engineering diagram.
- `Table 2. System Overview` here or immediately after Figure 1.
- Recommended columns: `System`, `Retrieval`, `Training Signal`, `Supervision`, `Role in Paper`.

### 4.2 Training Setups

**Section thesis**

The adapted systems differ in signal, not in the surrounding experimental scaffolding.

**Content guidance**
- Briefly describe the RAFT-style training data format.
- Briefly describe CLM corpus training.
- Briefly describe no-retrieval controls and legacy D2L.
- Move detailed hyperparameters to Appendix A.

**Ready thesis statements**
- "The RAFT-style system is trained on retrieval-conditioned supervision built from question-answer pairs and supporting evidence."
- "The CLM system is trained on the corpus text alone via continued next-token prediction, without explicit QA labels."
- "Detailed hyperparameters are reported in the appendix because they support reproducibility but are not central to the paper's argument."

**Figure/table placement**
- No new figures here.

### 4.3 Evaluation Protocol

**Section thesis**

The evaluation combines aggregate quality, component-wise answer quality, and practical constraints.

**Content guidance**
- Define `Q_main`, `S_det`, `S_asst`, grounding, latency, and cost framing.
- State how judge-based scoring is used.
- Explain the meaning of identical grounding values across retrieval-aware systems.
- Clarify that practical interpretation uses both quality and resource cost.

**Ready thesis statements**
- "The main evaluation combines aggregate answer quality with sub-metrics that distinguish deterministic extraction from assistant-style answer quality."
- "Grounding is reported as a control on the shared retrieval pipeline, while the main differences among retrieval-aware systems emerge in answer generation."

**Figure/table placement**
- No main figures here.
- Hyperparameter tables and judge prompt details go to Appendix A.

## 5. Results (4-4.5 pages)

This is the center of the paper. It should receive the highest visual density.

### 5.1 Main Comparison

**Section thesis**

A strong fixed RAG baseline is already competitive, and both adapted retrieval-aware systems produce modest but real gains over it.

**Content guidance**
- Lead with `S1`, `S2+R`, `S3+R`.
- Mention `S7` only after the headline comparison is established.
- Do not let controls interrupt the first narrative pass.
- Emphasize that there is no single practical winner between `S2+R` and `S3+R`.

**Ready thesis statements**
- "The strong RAG baseline establishes a high starting point, which makes subsequent gains harder to obtain and therefore more informative."
- "Both retrieval-aware adapted systems improve over the baseline, but they do so without producing a single dominant practical winner across all dimensions."
- "The post-hoc merged system reaches the highest observed aggregate score, but this result is interpreted separately because it is exploratory and inherits prior training effort."

**Figure/table placement**
- `Table 3. Main Results` here as the central table.
- Organize rows into `Headline`, `Exploratory`, and `Controls`.
- Mark `S7` as `post-hoc, no retraining`.

### 5.2 Trade-off Between RAFT-style and CLM Adaptation

**Section thesis**

The main scientific result is not a universal winner, but a difference in quality profile driven by training signal.

**Content guidance**
- Show that RAFT-style improves `S_det`.
- Show that CLM improves `S_asst`.
- Explain why this supports the training-signal interpretation.
- This subsection should make the paper's central claim explicit.

**Ready thesis statements**
- "The comparison indicates that training signal matters more than the mere presence of an adapter."
- "RAFT-style supervision is associated with stronger deterministic extraction, whereas CLM adaptation is associated with stronger assistant-style answer quality."
- "These differences are consistent with the distinct objectives used during adaptation."

**Figure/table placement**
- `Figure 2. Delta-to-S1 Bar Chart` here.
- Show deltas for at least `Q_main`, `S_det`, and `S_asst`.
- Include `S2+R`, `S3+R`, and `S7`; include `S2` and `S3` if the figure remains readable.
- This figure is a required Advisor 1 integration.
- If space is very tight, make it compact and keep only the most important systems.

### 5.3 By Answer Type

**Section thesis**

Answer-type analysis shows that the systems are not interchangeable even when their aggregate scores are close.

**Content guidance**
- Use this subsection to explain where each headline system is strong or weak.
- Keep the narrative selective; do not comment on every small cell.
- Mention small-sample categories cautiously.

**Ready thesis statements**
- "Per-type results show that near-equal aggregate scores can hide materially different strengths across answer categories."
- "This reinforces the interpretation that the systems differ in response behavior rather than only in overall score magnitude."

**Figure/table placement**
- `Figure 3. Per-Type Score Heatmap` here.
- Prefer showing `S1`, `S2+R`, `S3+R`, and `S7` only.
- Add sample counts `n` in type labels.
- Use raw per-type values rather than delta-to-S1, because the delta view is already covered by `Figure 2`.

### 5.4 Retrieval Contribution and the Limits of Pure Parametric Memory

**Section thesis**

Removing retrieval causes a large quality drop, which shows that retrieval remains indispensable in this benchmark.

**Content guidance**
- Compare no-retrieval controls with retrieval-aware systems.
- Present D2L briefly as a legacy negative control.
- Keep this subsection compact and decisive.

**Ready thesis statements**
- "The no-retrieval controls perform far below the retrieval-aware systems, indicating that retrieval remains indispensable in this setup."
- "The legacy D2L branch is retained as an engineering negative control rather than as part of the headline comparison."

**Figure/table placement**
- Reuse `Table 3`; no extra figure required.

### 5.5 Multi-Document Difficulty

**Section thesis**

Single-document versus multi-document behavior is one of the strongest analytical results in the paper and should stand on its own rather than serving as a lead-in to `S7`.

**Content guidance**
- Elevate this analysis beyond a minor side observation.
- Show that multi-doc questions remain substantially harder.
- Interpret `CLM+R` as stronger on single-doc local contextualization and `RAFT+R` as more helpful on multi-doc aggregation/comparison.
- Keep the main narrative focused on `S1`, `S2+R`, and `S3+R`.

**Ready thesis statements**
- "The single-document versus multi-document split reveals the clearest behavioral contrast among the headline systems."
- "CLM adaptation appears strongest on single-document contextualization, whereas RAFT-style supervision appears more helpful on multi-document aggregation and comparison."

**Figure/table placement**
- `Figure 4. Single-doc vs Multi-doc Comparison` here.
- Use grouped bars.
- This figure is a required high-priority visual.
- Do not add a support table for this subsection in the main text.

### 5.6 Exploratory Adapter Fusion

**Section thesis**

`S7` is best used as a short exploratory result that supports complementarity without reshaping the core claim of the paper.

**Content guidance**
- Introduce `S7` only after the multi-doc result is already established.
- Interpret `S7` as evidence that the two adaptation signals may be partially complementary.
- Keep the subsection short and explicitly non-headline.
- Discuss practical trade-off in prose, with reference to Appendix B rather than a main-text cost figure.

**Ready thesis statements**
- "The merged system partially combines the strengths associated with the two adapted headline systems, which is consistent with a complementarity interpretation."
- "This result remains exploratory because `S7` is a post-hoc merge rather than a separately trained and directly cost-comparable system."

**Figure/table placement**
- No dedicated main-text figure here.
- Refer to `Appendix Table B1. Practical Trade-off Summary` for offline cost and latency context.

## 6. Discussion and Limitations (2-2.5 pages)

### 6.1 Answer to RQ1

**Section thesis**

Parametric adaptation adds value beyond strong RAG, but the gain is moderate and depends on the type of adaptation signal.

**Content guidance**
- Answer `RQ1` explicitly in prose.
- Emphasize that the gain is real but not revolutionary.
- Tie the answer back to the trade-off between `S_det` and `S_asst`.

**Ready thesis statements**
- "Within this setup, parametric adaptation provides measurable improvement beyond a strong RAG baseline."
- "The observed gain is best understood as a change in quality profile rather than as a uniform increase along every dimension."

**Figure/table placement**
- No new figures.

### 6.2 Answer to RQ2

**Section thesis**

Retrieval remains the dominant memory mechanism on this benchmark, and pure parametric controls do not substitute for it.

**Content guidance**
- Answer `RQ2` explicitly.
- Summarize the no-retrieval drop.
- State the conclusion narrowly and cautiously.

**Ready thesis statements**
- "Within the evaluated benchmark, pure parametric systems do not approach the quality of retrieval-aware systems."
- "This indicates that retrieval remains indispensable as the primary memory mechanism under the present data, model, and hardware constraints."

**Figure/table placement**
- No new figures.

### 6.3 Error Analysis

**Section thesis**

Error overlap clarifies both the shared limits of the benchmark and the partial complementarity of the headline systems.

**Content guidance**
- Discuss questions missed by all headline systems.
- Mention unique wins if they support the complementarity story.
- Keep this analysis compact and interpretive.

**Ready thesis statements**
- "The error topology indicates that some failures are shared across all headline systems, which points to benchmark-level difficulty rather than model-specific weakness."
- "At the same time, a small number of non-overlapping successes supports the interpretation that the systems are partially complementary."

**Figure/table placement**
- Keep the overlap visualization in the appendix, not the main text.
- Preferred appendix asset: `Appendix Figure B2. Error Overlap UpSet Plot`.
- This is the required Advisor 1 integration that replaces a less informative heatmap.

### 6.4 Limitations

**Section thesis**

The claims are informative but bounded by benchmark size, fixed infrastructure, and evaluation design.

**Content guidance**
- List compact corpus, one backbone, one frozen split, judge-based evaluation, and the exploratory nature of `S7`.
- Note D2L as a limited engineering pilot only.
- Do not defend away the limitations; present them plainly in compact prose rather than as a bullet list.

**Ready thesis statements**
- "The findings are bounded to a compact benchmark, one backbone family, one hardware regime, and a fixed retrieval pipeline."
- "The `S7` result should be interpreted cautiously because it is post-hoc and inherits prior adaptation cost."
- "The D2L branch does not support a broad claim about document-conditioned adapter generation in general; it only supports a negative finding for the present implementation and setup."

**Figure/table placement**
- No figures.

## 7. Conclusion (0.5-1 page)

### 7.1 Main Findings

**Section thesis**

The conclusion should restate the scientific answer in a compact, non-inflated form.

**Content guidance**
- Reaffirm the three-part result: strong RAG baseline, useful but signal-dependent adaptation, indispensable retrieval.
- Mention complementarity as promising but exploratory.
- End with one practical takeaway.

**Ready thesis statements**
- "A strong document-grounded RAG baseline already performs well on the compact legal benchmark studied here."
- "Parametric adaptation on top of that baseline provides additional value, but the effect depends on the adaptation signal rather than on adapter use alone."
- "Retrieval remains indispensable, while post-hoc fusion suggests that supervised and corpus-level adaptation may capture partially complementary strengths."

**Figure/table placement**
- No figures.

## Bibliography

Use standard academic references. Keep method citations targeted:
- RAG;
- LoRA / QLoRA;
- RAFT-style tuning or the closest cited formulation actually used;
- continued pretraining / CLM references as needed;
- any benchmark-specific or legal QA references actually cited in the text.

Do not inflate the bibliography with loosely related "memory systems" papers that are not used in the argument.

## Appendix Structure

### Appendix A - Hyperparameters and Prompts

Include:
- training hyperparameters for `S2+R`, `S3+R`, and controls where relevant;
- judge prompt/rubric summary;
- implementation details that support reproducibility but would overload the main text.

### Appendix B - Extra Tables and Figures

Priority order:
1. `Appendix Table B1. Practical Trade-off Summary`;
2. `Appendix Figure B3. Judge Criteria Profile`;
3. `Appendix Figure B2. Error Overlap UpSet Plot`;
4. additional per-type or seed-stability figures only if they clearly support a discussion point.

`Appendix Table B1. Practical Trade-off Summary` should use rows `S1`, `S2+R`, `S3+R`, `S2`, and `S3`.

Do not include `S7` in the cost-comparable body of the table. Add a note below the table stating that `S7` inherits prior adaptation cost from `S2+R` and `S3+R` and is therefore not directly comparable in offline-cost terms.

Keep out of the main text unless absolutely needed:
- latency-grounding scatter;
- pairwise win heatmap;
- pareto frontier;
- multiple seed-stability plots.

### Appendix C - D2L Engineering Note

Include 1 short subsection:
- what D2L was intended to test;
- why the practical implementation required workaround packaging;
- why the branch is treated as a legacy negative control rather than a headline result.

### Appendix D - Use of Generative AI

Include:
- tools used by name;
- what they were used for;
- scope limits;
- statement that responsibility for the final text remains with the author.
- If required by the institutional template, list which passages were GenAI-generated or substantially AI-assisted and mark them explicitly in the manuscript.

## Visual and Table Plan

### Main Text Visual Priority

Use this priority order when space is limited:
1. `Table 3. Main Results`
2. `Figure 1. System Overview Schematic`
3. `Figure 4. Single-doc vs Multi-doc Comparison`
4. `Figure 2. Delta-to-S1 Bar Chart`
5. `Figure 3. Per-Type Score Heatmap`
6. `Table 2. System Overview`
7. `Table 1. Benchmark Summary`

### Appendix Visual Priority

1. `Appendix Table B1. Practical Trade-off Summary`
2. `Appendix Figure B2. Error Overlap UpSet Plot`
3. `Appendix Figure B3. Judge Criteria Profile`
4. supporting stability/auxiliary plots only if referenced

## Figure and Table Numbering Map

Use this numbering unless layout changes force small shifts:
- `Table 1. Benchmark Summary`
- `Figure 1. System Overview Schematic`
- `Table 2. System Overview`
- `Table 3. Main Results`
- `Figure 2. Delta-to-S1 Bar Chart`
- `Figure 3. Per-Type Score Heatmap`
- `Figure 4. Single-doc vs Multi-doc Comparison`

Appendix numbering defaults:
- `Appendix Table B1. Practical Trade-off Summary`
- `Appendix Figure B2. Error Overlap UpSet Plot`
- `Appendix Figure B3. Judge Criteria Profile`

## Term Placement Guide

Define briefly on first mention:
- `RAG` in Section 2.1;
- `LoRA` and `QLoRA` in Section 2.2;
- `RAFT-style training` and `CLM continued pretraining` in Section 2.3;
- `Q_main`, `S_det`, `S_asst`, and grounding in Section 4.3.

Keep short or defer to appendix:
- mathematical details of LoRA;
- Gemma architecture internals;
- BM25/RRF/Qdrant mechanics;
- D2L internals.

## Drafting Checklist

Before writing the paper body, make sure the draft follows this checklist:
- All 7 main sections plus bibliography and appendix are present.
- The main comparison is always `S1` vs `S2+R` vs `S3+R`.
- `S7` is always marked exploratory/post-hoc.
- Retrieval-free systems are always treated as controls.
- The body text is prose-first; bullet and numbered lists are used only where they are clearly justified.
- Multi-doc analysis is elevated as a central analytical result.
- The system overview schematic is planned in Section 4.1.
- The delta-to-S1 chart is planned in Section 5.2.
- The multi-doc result is presented before `S7` and stands as an independent analytical finding.
- The error-overlap appendix uses an UpSet plot rather than a main-text heatmap.
- The practical trade-off comparison appears in `Appendix Table B1`, excludes `S7` from cost-comparable rows, and avoids a misleading latency-only cost framing.
- All thesis statements are written in formal academic English.
- The split is consistently described as `150 train / 50 eval`.
- Grounding is interpreted correctly as a control on the fixed retrieval backbone.
- No claim implies that D2L "failed in general"; the wording stays limited to this implementation and setup.

## One-Paragraph Executive Summary

If you need a single paragraph to guide the full draft, use this:

This paper should be written as a controlled study of whether parametric adaptation adds value beyond a strong fixed RAG baseline in compact legal QA under consumer-hardware constraints. The headline comparison is `S1` versus `S2+R` versus `S3+R`, with `S7` presented only as an exploratory post-hoc fusion and pure parametric systems retained as controls. The central interpretation is that training signal matters more than adapter presence alone: RAFT-style supervision improves deterministic extraction, CLM continued pretraining improves assistant-style answer quality, retrieval remains indispensable, and single-document versus multi-document behavior provides the clearest evidence of complementary strengths.
