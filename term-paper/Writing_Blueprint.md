# Writing Blueprint — Unified Plan

> **Base:** Advisor 2 structure + thesis statements
> **Integrated:** Advisor 1 additions (system schematic, delta bars, multi-doc emphasis, UpSet plot, cost-quality fix)
> **Polish:** Thesis statements revised for academic register

---

## Working Title

**Primary (precise):**
Parametric Adaptation over a Strong RAG Baseline for Document-Grounded Legal QA on Consumer Hardware: RAFT-style QLoRA vs. CLM Continued Pretraining

**Short (question-form):**
Does Parametric Adaptation Add Value Beyond Strong RAG? A Compact Legal QA Study on Consumer Hardware

## Research Questions

**RQ1 (main):**
Does parametric adaptation yield measurable gains over a strong RAG baseline on a compact legal benchmark under consumer-hardware constraints, and how do RAFT-style supervised adaptation and supervision-free CLM continued pretraining differ as retrieval-conditioned generators?

**RQ2 (secondary):**
How far can pure parametric systems reach without retrieval on this benchmark, and does retrieval remain indispensable?

## Core Narrative (keep in mind while writing)

A strong fixed RAG baseline already achieves high quality on this benchmark. Parametric adaptation on top of it yields moderate gains, yet the choice of training signal proves more consequential than the mere presence of an adapter: RAFT-style supervision improves deterministic extraction, while corpus-level CLM improves free-text answer quality at lower offline cost. Removal of retrieval leads to severe quality collapse across both paradigms, confirming that retrieval remains the dominant memory mechanism under consumer-hardware constraints. A post-hoc adapter merge further suggests partial complementarity between the two signals, though this finding remains exploratory.

---

## Front Matter

### Title Page
- University / department / subject
- Module / seminar / semester / submission date
- Thesis title
- Lecturer / supervisor
- Author details

### Declaration of Academic Integrity
- Separate page after the title page
- Include GenAI statement: tools used, purpose, scope
- Detailed tool listing deferred to Appendix D

### Table of Contents
- Page numbering starts with Introduction as page 1
- Title page and ToC not counted in normal numbering

---

## 1. Introduction (1.5–2 pages)

### 1.1 Problem and Motivation

**Section thesis:**
Legal document-grounded QA demands factual precision and verifiable grounding; on consumer hardware, the practical question is whether adapting the generator yields additional value once a strong retrieval backbone is already in place.

**Content:**
- Legal QA requires factual grounding and robustness to lookup-style questions
- Consumer hardware precludes scaling to larger models; retrieval engineering and PEFT are the realistic levers
- Central tension: if retrieval already supplies relevant evidence, does parametric adaptation still contribute measurable improvement?

**Draft thesis statements:**
- "This study investigates whether parametric adaptation retains practical value when the baseline comprises a carefully engineered document-grounded RAG system rather than a bare generator."
- "The experimental setting is deliberately constrained to consumer hardware, where retrieval engineering and parameter-efficient fine-tuning represent the most feasible adaptation strategies."

**Insertions:** None. The introduction should remain prose-only.

---

### 1.2 Research Questions and Scope

**Section thesis:**
The study adopts a deliberately narrow scope — one compact benchmark, one frozen backbone, one hardware configuration, and one fixed retrieval stack — to isolate the effect of training signal on generator behavior.

**Content:**
- 8 DIFC legal documents, 200 QA pairs
- Split: 150 train / 50 eval
- Backbone: Gemma-2-2b-it
- Hardware: RTX 4060 8 GB VRAM
- Comparison conducted on a fixed retrieval backbone

**Draft thesis statements:**
- "The benchmark comprises 8 DIFC legal documents and 200 human-authored QA pairs spanning six answer types."
- "All systems share the same frozen 50-question evaluation set; supervised training uses the remaining 150 questions."
- "This narrow scope is methodologically intentional: it permits a controlled comparison of training signals under identical infrastructure."

**Insertions:** None.

---

### 1.3 Contributions

**Section thesis:**
The contribution is an empirical comparison under controlled conditions, yielding a nuanced picture of when and how parametric adaptation helps.

**Content:**
- Strong S1 baseline establishes a high bar
- S2+R vs S3+R isolates training signal at fixed PEFT architecture
- Controls quantify the limits of pure parametric memory
- S7 provides exploratory evidence of signal complementarity

**Draft thesis statements:**
- "The paper contributes a controlled comparison between RAFT-style supervised adaptation and CLM continued pretraining on top of an identical RAG backbone."
- "It further quantifies the limits of pure parametric memory by contrasting retrieval-aware systems against no-retrieval controls."
- "A post-hoc adapter-merge result suggests partial complementarity between the two training signals, reported as an exploratory finding."

**Insertions:** None.

---

### 1.4 Structure of the Paper

One short paragraph: Section 2 provides background; Sections 3–4 describe the benchmark, systems, and evaluation; Section 5 presents results; Section 6 discusses findings and limitations; Section 7 concludes.

Required by formal guidelines: include justification of the chosen structure.

---

## 2. Background and Related Work (1.5–2 pages)

### 2.1 RAG as Nonparametric Memory

**Section thesis:**
Retrieval-augmented generation externalizes document knowledge into an evidence pipeline, avoiding the need for the model to memorize corpus content in its parameters.

**Content:**
- Concise explanation of RAG (1–2 sentences at first mention)
- Relevance to legal QA: grounded answers from authoritative sources
- Why retrieval is especially important with compact models

**Do not:** derive BM25/RRF formulas or provide a multi-paragraph history of RAG.

**Draft thesis statements:**
- "RAG delegates document knowledge to a retrieval pipeline rather than requiring the generator to internalize it parametrically."
- "For legal QA, this externalization is particularly attractive because answers must be traceable to specific regulatory provisions."

**Insertions:** None.

---

### 2.2 Parameter-Efficient Adaptation on Consumer Hardware

**Section thesis:**
Parameter-efficient adaptation is employed here because the hardware constraint rules out full fine-tuning; it is an experimental boundary condition rather than an optimization preference.

**Content:**
- LoRA in 1–2 sentences
- QLoRA (4-bit NF4 + adapters) in 2–3 sentences
- Why this fits 8 GB VRAM

**Draft thesis statements:**
- "QLoRA enables controlled adaptation of a small instruction-tuned model within a consumer-hardware budget."
- "In this study, parameter-efficient adaptation constitutes part of the experimental constraint rather than an incidental implementation detail."

**Insertions:** None.

---

### 2.3 RAFT-style Adaptation vs. CLM Continued Pretraining

**Section thesis:**
The central experimental axis contrasts two distinct training signals applied to the same PEFT architecture: supervised retrieval-conditioned QA generation versus unsupervised corpus-level language modeling.

**Content:**
- RAFT-style: question + gold chunks + distractors → answer (retrieval-conditioned supervision)
- CLM continued pretraining: raw corpus text, next-token prediction, no QA labels
- Why these represent fundamentally different training signals

**Draft thesis statements:**
- "RAFT-style adaptation directly optimizes answer generation from evidence-rich contexts, exposing the adapter to the QA task distribution."
- "CLM continued pretraining exposes the adapter to the corpus distribution without any task-specific supervision, relying solely on the next-token objective."

**Insertions:** None.

---

### 2.4 Research Gap and Positioning

**Section thesis:**
High-level discussions of parametric versus nonparametric knowledge are abundant, but controlled studies that isolate training signal while holding retrieval infrastructure and PEFT architecture fixed remain scarce.

**Content:**
- One paragraph on positioning
- Why this work qualifies as a controlled experimental comparison rather than a benchmark report

**Insertions:** None.

---

## 3. Benchmark and Experimental Setup (2 pages)

### 3.1 Corpus and Benchmark

**Section thesis:**
A compact benchmark suffices for a meaningful study provided the evaluation design is carefully controlled and the answer-type distribution is heterogeneous.

**Content:**
- 8 PDF legal documents, ~115 K tokens, 176 pages
- 200 QA pairs authored by domain experts
- Answer types: free_text, boolean, number, name, names, date
- Multi-document and unanswerable cases present
- Split: 150 train / 50 eval

**Draft thesis statements:**
- "The benchmark combines heterogeneous answer types — from boolean lookups to free-text legal explanations — rather than reducing evaluation to a single regime."
- "This heterogeneity exposes distinct failure modes and prevents aggregate scores from masking type-specific weaknesses."

**Insertions:** None. A dedicated dataset table is optional; preserving page budget for results is preferable.

---

### 3.2 Hardware, Shared Backbone, and Variance Policy

**Section thesis:**
The study is intentionally bounded to a single small backbone and a single realistic consumer-hardware configuration to keep the comparison tractable.

**Content:**
- Gemma-2-2b-it (frozen across all systems)
- RTX 4060 8 GB VRAM, 32 GB RAM
- 3 random seeds for trained systems; mean ± std reported
- No cross-validation (fixed split by design)

**Draft thesis statements:**
- "The backbone is held constant across all systems to prevent architectural variance from confounding the training-signal comparison."
- "Three random seeds are reported for each trained system to capture training variance without excessive experimental cost."

**Insertions:** None.

---

### 3.3 Fixed Retrieval Backbone

**Section thesis:**
The retrieval stack is strong and frozen across all retrieval-aware systems; this design choice converts the S2+R vs. S3+R comparison into a test of generator adaptation rather than retrieval engineering.

**Content:**
- Ingestion pipeline
- Hierarchical chunking
- Dense + sparse hybrid retrieval
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranker
- Evidence compression

**Depth:** One compact subsection. Not "how RAG works in general" but "what is held fixed in this study and why." Exact parameters (chunk_size, overlap, budgets, weights) go to appendix or a small config table.

**Draft thesis statements:**
- "The retrieval backbone is intentionally held constant across all retrieval-aware systems."
- "Consequently, performance differences between S1, S2+R, S3+R, and S7 should be attributed to how the generator utilizes retrieved evidence, not to differences in evidence selection."

**Insertions:** None. No retrieval pipeline figure in this section — the system overview schematic belongs in 4.1.

---

## 4. Compared Systems and Evaluation Protocol (2–2.5 pages)

### 4.1 System Inventory

**Section thesis:**
The study distinguishes explicitly between headline systems (S1, S2+R, S3+R), a post-hoc exploratory result (S7), and negative controls (S2, S3, S3-legacy).

**Content:**
- Headline: S1, S2+R, S3+R
- Post-hoc: S7 (linear adapter merge, no retraining)
- Controls: S2 (closed-book RAFT), S3 (CLM without retrieval), S3-legacy (D2L)
- S6 excluded from the narrative (archived ablation)

**Draft thesis statements:**
- "The seven evaluated systems serve distinct roles: three form the headline comparison, one provides an exploratory post-hoc result, and three serve as negative controls."

**INSERT: Table 1 — Compared Systems and Roles** (at the start of 4.1)

| System | Retrieval | Training Signal | Supervision | Role |
|--------|-----------|-----------------|-------------|------|
| S1 | yes | none | — | headline baseline |
| S2+R | yes | RAFT-style QA | supervised | headline |
| S3+R | yes | CLM on corpus | unsupervised | headline |
| S7 | yes | merged S2+R × S3+R | post-hoc | exploratory |
| S2 | no | RAFT-style QA | supervised | control |
| S3 | no | CLM on corpus | unsupervised | control |
| S3-leg. | no | D2L hypernetwork | supervised | legacy control |

> **⊕ FROM ADVISOR 1 — System Overview Schematic**
>
> **INSERT: Figure 1 — System Overview Schematic** (immediately after Table 1)
>
> A half-page pipeline diagram showing:
> - S1: `[retrieval] → [base generator]`
> - S2+R: `[retrieval] → [RAFT-adapted generator]`
> - S3+R: `[retrieval] → [CLM-adapted generator]`
> - S7: `[retrieval] → [merged-adapter generator]`
> - Controls: `[no retrieval] → [adapted generator]`
>
> This is the single most impactful addition for reader orientation.

---

### 4.2 Training Setups

**Section thesis:**
The headline comparison isolates training signal because S2+R and S3+R share the same backbone, PEFT architecture, and retrieval stack but differ solely in training objective.

**Content:**
- S2+R: RAFT-style open-book training on 150 train questions
- S3+R: CLM continued pretraining on concatenated corpus text
- Same PEFT config for both (QLoRA rank-32, LR 5e-5, 5 epochs, warmup 0.1, seq_len 512)
- S7: linear interpolation of S2+R and S3+R adapters (0.5 / 0.5), no additional training
- D2L: legacy implementation requiring a chunk-level workaround due to memory constraints

**Draft thesis statements:**
- "Both S2+R and S3+R employ identical QLoRA configurations applied to the same frozen backbone; the sole difference is the training objective."
- "The comparison thus isolates training signal rather than model family or PEFT architecture."
- "S7 is reported as a separate exploratory result because it represents a post-hoc interpolation rather than a retrained system."
- "D2L is retained only as a historical negative control; the engineering constraints that necessitated a chunk-level workaround are documented in Appendix C."

**Insertions:** None.

---

### 4.3 Evaluation Protocol

**Section thesis:**
The evaluation protocol combines deterministic scoring for structured answer types with judge-based assessment for free-text responses, reflecting the benchmark's heterogeneous nature.

**Content:**
- `Q_main = 0.7 × S_det + 0.3 × S_asst`
- Deterministic scoring for number / boolean / name / names / date
- Judge-based scoring (GPT-4.1-mini) for free_text
- Grounding `G` (F-beta) for retrieval-aware systems only
- Operational metrics: latency, VRAM, offline cost

**Critical interpretation (state explicitly in text):**
"Because the retrieval stack is fixed, grounding G primarily reflects the shared evidence pipeline rather than adapter-specific behavior. The constant G = 0.567 across retrieval-aware systems is a direct consequence of this design choice."

**Draft thesis statements:**
- "The composite metric Q_main weights deterministic extraction at 0.7 and judged free-text quality at 0.3, balancing factual precision with explanation quality."
- "Grounding scores are reported for retrieval-aware systems but should be interpreted as a property of the shared retrieval backbone rather than of individual adapters."

**Insertions:** None. Q_main formula given inline.

---

## 5. Results (4–4.5 pages)

### 5.1 Main Comparison

**Section thesis:**
The strong RAG baseline already achieves high quality, so improvements from parametric adaptation are incremental rather than self-evident.

**Content:**
- S1 reaches Q_main = 0.643
- S2+R and S3+R each improve over S1, but modestly
- S7 attains the highest aggregate score but remains secondary in interpretation due to its post-hoc nature

**Draft thesis statements:**
- "The nonparametric baseline S1 already reaches Q_main = 0.643, establishing a high bar that makes further gains difficult to achieve."
- "Both retrieval-aware adapters improve over S1: S2+R attains 0.669 ± 0.014 and S3+R attains 0.667 ± 0.023."
- "S7 reaches the highest observed score (0.705 ± 0.035) through post-hoc adapter interpolation; however, it should not supersede the controlled headline comparison."

**INSERT: Table 2 — Main Results** (at the start of 5.1, before text)

Group systems into blocks:

| Block | System | Q_main | S_det | S_asst | G | Latency | Offline cost |
|-------|--------|--------|-------|--------|---|---------|-------------|
| Headline | S1 | 0.643 | 0.601 | 0.739 | 0.567 | ... | 0 |
| Headline | S2+R | 0.669±0.014 | 0.648 | 0.718 | 0.567 | ... | ... |
| Headline | S3+R | 0.667±0.023 | 0.599 | 0.826 | 0.567 | ... | ... |
| Post-hoc | S7 | 0.705±0.035 | 0.679 | 0.764 | 0.567 | ... | note¹ |
| Control | S2 | 0.263±0.005 | 0.270 | 0.246 | — | ... | ... |
| Control | S3 | 0.185±0.003 | 0.135 | 0.303 | — | ... | ... |
| Control | S3-leg. | 0.210 | ... | ... | — | ... | ... |

¹ S7 inherits offline cost of both S2+R and S3+R; mark explicitly.

> **⊕ FROM ADVISOR 1 — Delta-to-S1 Bar Chart**
>
> **INSERT: Figure 2 — Improvement over S1 Baseline** (after Table 2, still in 5.1)
>
> Grouped bar chart showing ΔQ_main, ΔS_det, ΔS_asst for S2+R, S3+R, S7 (and optionally S2, S3 as negative deltas).
> Instantly communicates the central finding: RAFT raises S_det, CLM raises S_asst, merge raises both.
>
> *Replaces the former "Cost-Quality Scatter" in this position.*

> **⊕ FROM ADVISOR 1 — Cost-Quality Fix**
>
> Advisor 2's "Figure 1 — Latency–Quality Scatter" is **replaced**: latency and offline-cost columns are embedded directly into Table 2.
> This eliminates a separate figure that added little insight (latency is similar across retrieval-aware systems) and frees page budget for the new Δ-bars and single-doc/multi-doc figures.
> If a standalone cost figure is still desired, use `x = offline cost, y = Q_main` — but the table route is preferred for page economy.

---

### 5.2 Trade-off Between S2+R and S3+R

**Section thesis:**
The headline comparison yields a trade-off rather than a universal winner: the two adapters improve different quality dimensions.

**Content:**
- S2+R stronger on S_det (0.648 vs 0.599)
- S3+R stronger on S_asst (0.826 vs 0.718)
- S3+R cheaper offline (no QA label generation required)
- Aggregate Q_main is near-tied (Δ = 0.002)

**Draft thesis statements:**
- "The headline comparison does not yield a single dominant system."
- "S2+R achieves higher deterministic extraction scores (S_det = 0.648 vs. 0.599), whereas S3+R achieves higher free-text answer quality (S_asst = 0.826 vs. 0.718)."
- "S3+R also incurs lower offline cost, as it requires no task-specific label generation — a consideration relevant under consumer-hardware constraints."

**INSERT: Figure 3 — Judge Criteria Profile** (inside 5.2, after the first paragraph)

Show only S1, S2+R, S3+R, S7. Omit controls to avoid clutter.
Purpose: demonstrate that S3+R's advantage is concentrated in free-text quality dimensions.

---

### 5.3 By Answer Type

**Section thesis:**
Type-level evaluation reveals that the systems diverge in non-uniform ways, with the strongest contrasts between deterministic extraction and free-text explanation.

**Content:**
- Breakdown by boolean / number / name / date / names / free_text
- S2+R vs S3+R diverge differently by type
- Date and names remain weak across all systems
- Discuss patterns, not every individual cell

**Draft thesis statements:**
- "Type-level analysis reveals that performance differences between systems are concentrated in specific answer categories rather than distributed uniformly."
- "The strongest divergences appear between deterministic extraction types and free-text explanation."
- "Date extraction and multi-name normalization remain weak spots across all systems, pointing to persistent formatting and evidence-utilization limitations."

**INSERT: Figure 4 — Per-Type Score Heatmap** (in 5.3)

Show S1, S2+R, S3+R, S7. Controls to appendix if included at all.
Add sample sizes to type labels: `date (n=5)`, `free_text (n=13)`, etc.

---

### 5.4 Retrieval Contribution and the Limits of Pure Parametric Memory

**Section thesis:**
Removing retrieval causes severe quality collapse for both adaptation paradigms, establishing that retrieval remains the dominant memory mechanism in this setup.

**Content:**
- S2 → S2+R: +0.406 Q_main
- S3 → S3+R: +0.482 Q_main
- S3-legacy (D2L): Q_main = 0.210, legacy control only
- This is one of the paper's strongest conclusions, not a sidebar

**Draft thesis statements:**
- "Removing retrieval causes a large quality collapse for both training paradigms: Q_main drops from 0.669 to 0.263 for RAFT and from 0.667 to 0.185 for CLM."
- "This indicates that retrieval remains the dominant memory mechanism in this setting, and that parametric adaptation without evidence access is insufficient."
- "The D2L legacy control (Q_main = 0.210) corroborates this finding from a third, independent direction."

**Insertions:** None. Table 2 already shows all relevant numbers.

---

### 5.5 Single-Document vs. Multi-Document Difficulty and Exploratory Adapter Fusion

> **⊕ FROM ADVISOR 1 — Elevated Multi-doc Analysis**
>
> Advisor 1 identifies this as *"the one smart analytical result to elevate to the center of the paper."*
> This subsection receives fuller treatment than in the Advisor 2 baseline.

**Section thesis:**
Multi-document questions remain substantially harder than single-document questions across all systems; the divergence in system behavior on multi-doc items provides the most granular evidence for signal complementarity.

**Content:**
- Single-doc vs multi-doc Q_main:
  - S1: 0.696 vs 0.310
  - S2+R: 0.694 vs 0.437
  - S3+R: 0.722 vs 0.310
  - S7: 0.718 vs 0.523
- CLM+R performs best on single-doc (local contextualization)
- RAFT+R degrades less on multi-doc (cross-document aggregation)
- S7 partially combines both effects
- This pattern provides the strongest interpretive support for signal complementarity

**Draft thesis statements:**
- "Multi-document questions remain substantially harder than single-document questions for all systems studied."
- "S1 and S3+R both drop to Q_main = 0.310 on multi-document items, while S2+R reaches 0.437 and S7 reaches 0.523."
- "CLM continued pretraining appears to benefit single-document contextualization (S3+R achieves the highest single-doc score at 0.722), whereas RAFT-style supervision confers greater robustness to multi-document composition."
- "The merged adapter S7 partially combines both advantages, providing the most direct evidence of signal complementarity — though this remains an exploratory finding."

**INSERT: Figure 5 — Single-doc vs. Multi-doc Comparison** (end of 5.5)

Paired bar chart or dumbbell plot:
- x-axis: systems (S1, S2+R, S3+R, S7)
- y-axis: Q_main
- Two series: single-doc and multi-doc
- Optional annotation: Δ(multi − single) per system

This is the highest-priority new figure.

---

## 6. Discussion and Limitations (2–2.5 pages)

### 6.1 Answer to RQ1

**Section thesis:**
Parametric adaptation yields measurable gains over a strong RAG baseline, but the improvement is moderate and profile-dependent rather than uniform.

**Content:**
- Strong baseline already high — gains are incremental
- Both S2+R and S3+R improve over S1
- Training signal determines the quality profile: RAFT → S_det, CLM → S_asst

**Draft thesis statements:**
- "RQ1 receives a qualified affirmative: parametric adaptation adds value on top of strong RAG, but the gain is moderate rather than transformative."
- "The choice of training signal proves more consequential than the presence of an adapter per se: RAFT-style supervision improves deterministic extraction, while CLM continued pretraining improves free-text answer quality."

---

### 6.2 Answer to RQ2

**Section thesis:**
Pure parametric systems fail to approach the retrieval-augmented baseline, confirming that retrieval remains indispensable under the studied constraints.

**Content:**
- S2 and S3 severely underperform their retrieval-aware counterparts
- D2L also weak
- Retrieval is a non-optional component

**Draft thesis statements:**
- "RQ2 receives an unambiguous answer: retrieval remains indispensable in this setting."
- "Neither supervised closed-book adaptation (S2) nor corpus-level CLM pretraining (S3) provides a viable substitute for external evidence retrieval."

---

### 6.3 Error Analysis

**Section thesis:**
The residual errors are structured rather than random, which makes them analytically informative and suggests directions for future work.

**Content:**
- 15 questions missed by all headline systems (common hard cases)
- Recurring failure modes: date extraction, long-list normalization, cross-document composition, unanswerable calibration
- Limited local complementarity: 0 questions solved only by S2+R, 2 only by S3+R, 2 only by S1
- Multi-doc remains the hardest regime

**Draft thesis statements:**
- "Error overlap analysis reveals 15 questions that all headline systems fail to answer correctly, indicating hard cases likely rooted in retrieval coverage or question ambiguity."
- "Persistent failure modes include date extraction, long-list normalization, and cross-document composition."
- "Local wins exist (2 questions answered only by S1, 2 only by S3+R, 0 only by S2+R) but remain too sparse to overturn the aggregate-level trade-off."

> **⊕ FROM ADVISOR 1 — UpSet Plot**
>
> Consider replacing or supplementing the error-overlap heatmap with an **UpSet plot** or a compact set-intersection table:
> - All headline systems wrong: 15
> - Only S1 correct: 2
> - Only S2+R correct: 0
> - Only S3+R correct: 2
>
> More informative than a pairwise heatmap for showing complementarity patterns.
> UpSet plot preferred; error-overlap heatmap is acceptable if UpSet is not feasible.
> Place in **Appendix B** and reference from this subsection.

**Insertions in main text:** None. Reference appendix figure.

---

### 6.4 Limitations

**Section thesis:**
The conclusions are internally valid but deliberately bounded in scope.

**Content:**
- Compact corpus (8 documents)
- Small evaluation set (50 questions)
- Single 2B backbone
- One fixed retrieval stack
- Judge-based free-text scoring (not human evaluation)
- S7 is post-hoc and not retrained
- D2L used an engineering workaround; serves only as legacy control

**Draft thesis statements:**
- "These findings are bounded to one compact legal corpus, one evaluation split, one backbone, and one consumer-hardware configuration."
- "Because the retrieval backbone is fixed, the study primarily measures differences in evidence-conditioned generation rather than differences in retrieval quality."
- "The D2L branch is included solely as a legacy negative control and should not be over-interpreted as a competitive alternative in this setting."

---

## 7. Conclusion (0.5–1 page)

**Section thesis:**
The paper's core contribution is a controlled answer to a practical question: whether small-model parametric adaptation retains value once RAG is already strong.

**Content:**
- 3–4 concluding statements answering RQ1 and RQ2
- One practical takeaway
- Future work directions

**Draft thesis statements:**
- "A strong compact RAG baseline already delivers high quality on the studied benchmark."
- "Parametric adaptation on top of it is beneficial, but the gain is moderate and strongly dependent on the choice of training signal."
- "Retrieval remains non-substitutable under the studied constraints."
- "Future work should target multi-document reasoning, unanswerable calibration, and retrieval-aware adaptation strategies that explicitly model cross-document composition."

**Insertions:** None.

---

## Bibliography

- Only actually cited literature
- Alphabetical order
- No padding sources

---

## Appendix

### Appendix A — Hyperparameters and Prompts
- QLoRA configuration details
- CLM training settings
- Judge prompt (full text)
- Scoring rubric details

### Appendix B — Supplementary Tables and Figures

| ID | Content | Source |
|----|---------|--------|
| Table A1 | Full per-type numeric breakdown | Existing |
| Figure A1 | Error overlap — UpSet plot or heatmap ⊕ | New / reworked |
| Figure A2 | Seed stability across 3 seeds | Existing |
| Figure A3 | Pairwise win heatmap | Existing |

### Appendix C — D2L Engineering Note
- Short explanation: D2L required chunk-level workaround despite token-audit suggesting single-pass feasibility
- Why it is a legacy control, not an equal methodological branch
- One sentence resolving the documentation inconsistency:
  "Token-based audit indicated single-pass feasibility, but the released D2L implementation imposed stricter effective memory limits, necessitating chunked packaging."

### Appendix D — Use of Generative AI
- List tool names and versions
- Purpose of use (structure discussion, draft editing, brainstorming, experiment orchestration)
- Scope and extent
- Statement that the author bears full responsibility

---

## Visualization Map

### Main Text — Figures and Tables (numbered in order of appearance)

| # | Type | Title | Section | Status |
|---|------|-------|---------|--------|
| Table 1 | Table | Compared Systems and Roles | 4.1 | Existing data, new table |
| Figure 1 | Figure | System Overview Schematic | 4.1 | **New** ⊕ Advisor 1 |
| Table 2 | Table | Main Results | 5.1 | Existing data, reformat |
| Figure 2 | Figure | Improvement over S1 Baseline (Δ bars) | 5.1 | **New** ⊕ Advisor 1 |
| Figure 3 | Figure | Judge Criteria Profile | 5.2 | Existing, filter to headline + S7 |
| Figure 4 | Figure | Per-Type Score Heatmap | 5.3 | Existing, add sample sizes |
| Figure 5 | Figure | Single-doc vs. Multi-doc Comparison | 5.5 | **New** ⊕ Advisor 1 |

### Appendix Figures

| # | Type | Title | Status |
|---|------|-------|--------|
| Table A1 | Table | Full per-type numeric breakdown | Existing |
| Figure A1 | Figure | Error overlap (UpSet plot preferred; heatmap as fallback) | **Reworked** ⊕ Advisor 1 |
| Figure A2 | Figure | Seed stability | Existing |
| Figure A3 | Figure | Pairwise win heatmap | Existing |

### Removed from Main Text
- ~~Latency-grounding scatter~~ (G is constant; uninformative)
- ~~Cost-quality scatter~~ (replaced by Δ bars + trade-off discussion in text)
- ~~Pareto frontier~~ (S7 cost accounting unclear)
- ~~Seed stability~~ (moved to appendix)

### Priority Ranking (if page budget tight)

**Must keep:**
1. Table 1 (systems)
2. Table 2 (results)
3. Figure 4 (per-type heatmap)
4. Figure 5 (single-doc vs multi-doc) ⊕

**Should keep:**
5. Figure 1 (system schematic) ⊕
6. Figure 2 (Δ bars) ⊕
7. Figure 3 (judge criteria)

**Cut first:**
- Error overlap from main text
- Seed stability
- Pairwise wins

---

## Documentation Fixes (pre-writing checklist)

1. **Split inconsistency:** Unify all references to 150/50. If the early 160/40 appears in any cited audit, add one sentence: "An earlier data audit recorded a preliminary 160/40 split; the final frozen split used throughout all experiments is 150 train / 50 eval."

2. **D2L audit inconsistency:** Resolve in Appendix C with the sentence provided above.

3. **S7 offline cost:** Explicitly note in Table 2 that S7 inherits the combined offline cost of S2+R and S3+R; do not report it as 0.

---

## Language Guidelines

### Prefer
- "observed improvement" / "measured gain"
- "trade-off" / "profile-dependent"
- "bounded to this benchmark / backbone / hardware"
- "retrieval remains indispensable / non-substitutable in this setup"
- Hedged language where appropriate: "suggests," "indicates," "appears to"

### Avoid
- "proved" / "significant" (without statistical test)
- "internalized the corpus"
- "replaced retrieval"
- "clean" (as a quality adjective for experiments)
- "is not X but Y" constructions (rephrase as positive statements)
- Informal intensifiers: "very good," "really strong," "very important"
- "story" (when referring to the paper's argument or narrative informally)

---

## Term Placement Guide

*(Ported from Advisor 2's "Где объяснять термины")*

### Define in Section 2 (Background), at first mention
- **RAG** — 1–2 sentences
- **LoRA / QLoRA** — 2–4 sentences, functional description, no math
- **CLM / continued pretraining** — use full form "causal language modeling (CLM) continued pretraining" at first mention, then "CLM" throughout
- **RAFT-style training** — explain in context of this study's data format (question + gold chunks + distractors → answer)

### Define in Section 4 (Evaluation Protocol)
- **Grounding F-beta** and **Q_main** formula

### Define in Section 3 (Setup), briefly
- Ingestion / indexing / retrieval pipeline — one compact paragraph, not a deep technical excursion

### Keep short or defer to appendix
- **D2L** — 3–4 sentences max in Section 4 as legacy control; engineering detail in Appendix C
- **S6** — do not include in the narrative; archived ablation, excluded from active thesis set
- **Dualhead LoRA** — omit entirely from the body; at most one footnote in the introduction if needed to acknowledge the project's evolution
