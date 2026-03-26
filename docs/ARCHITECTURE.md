# Project Architecture: Parameter-Efficient Knowledge Injection for Document-Grounded QA on Consumer Hardware

**Status:** SSOT (single source of truth)  
**Version:** 1.0  
**Last updated:** 2026-03-24  
**Project shorthand:** `lora-rag-knowledge-injection`  
**Primary practical setting:** domain-specific / legal-style document-grounded QA on a fixed corpus  

> This document is the authoritative source for scope, experiment design, evaluation logic, and repository structure. If a concrete experiment diverges from this document, the deviation must be explicitly documented in the corresponding `experiments/EXP-XXX_*/REPORT.md` and, if the deviation is strategic rather than incidental, this file must be updated first.

---

## 1. Problem Statement & Success Criteria

> Context: Define the problem clearly, what success looks like.

### Problem Definition

The project studies **knowledge injection into a frozen LLM under consumer-hardware constraints**.
The practical task is **document-grounded QA** over a **fixed corpus of 30 documents** and a **human-authored goldset of 150 question-answer pairs**. The domain is legal / legal-style QA, but the research focus is **not legal QA as such**; the corpus is a controlled case study for comparing knowledge-injection strategies.

The central comparison is between three families of approaches:

1. **Nonparametric knowledge injection** — Classical RAG over a fixed retrieval/indexing stack.
2. **Parametric knowledge injection** — LoRA-based adaptation of a fixed backbone model.
3. **Hybrid knowledge injection** — retrieval + the best parametric adapter, with HyDE used only through an adapter already adapted to the corpus.

The core scientific question is:

**Under a fixed trainable-parameter budget and on consumer hardware, which strategy yields the best trade-off between answer quality, grounding, and latency for document-grounded QA: Classical RAG, single-head LoRA, 4-branch MH-LoRA, cluster-routed single-head LoRA, or a hybrid RAG + best-adapter + HyDE system?**

### Research Scope (Frozen)

#### In scope

- Controlled comparison of **five systems** on the same corpus, goldset, backbone, prompt family, and evaluation protocol.
- Comparison of **single-head LoRA** vs **4-branch MH-LoRA** vs **cluster-routed single-head LoRA** under a matched trainable-parameter budget.
- Analysis of whether **implicit specialization** inside a decomposed adapter differs from **explicit specialization** via routed adapters.
- Final hybrid system where **HyDE is generated only by the best corpus-adapted adapter** and is used as the retrieval query representation.
- Measurement of **quality**, **grounding**, **TTFT**, **end-to-end latency**, **VRAM**, and **specialization behavior**.

#### Out of scope

- Benchmarking multiple retriever architectures, rerankers, chunking strategies, or vector databases as primary research axes.
- Large-scale synthetic QA generation as a core data source.
- Full MoE training, hypernetwork LoRA, or token-level routing as main systems.
- Comparing many backbone models. Backbone choice is a feasibility gate; after selection, it is frozen.
- Claiming universal conclusions beyond the setting of a **small fixed corpus**, **small human goldset**, and **consumer hardware**.

### System Inventory (Frozen)

| ID | System | Family | Definition | Parameter Budget Logic | Role |
|----|--------|--------|------------|------------------------|------|
| S1 | Classical RAG | Nonparametric | Existing vector-base RAG pipeline over the fixed corpus | No trainable adapter parameters | Main nonparametric baseline |
| S2 | Single-head LoRA | Parametric | One LoRA adapter, rank 32 | Reference parametric budget | Main parametric baseline |
| S3 | 4-branch MH-LoRA | Parametric | Four independent branches with their own `(A_i, B_i)`, each rank 8; contributions are summed | `4 x rank-8 = rank-32` total trainable budget | Tests decomposed rank under equal budget |
| S4 | Cluster-routed single-head LoRA | Parametric | Four separate single-head LoRA adapters, rank 8 each, one adapter selected by router | Total stored trainable budget comparable to S2/S3 | Tests explicit specialization via routing |
| S5 | Hybrid RAG + best adapter + HyDE | Hybrid | Best adapter from S2-S4 generates HyDE text for retrieval and answers with retrieved context | Uses selected best adapter only | Practical top-line |

### Architectural Clarifications (Mandatory)

1. **MH-LoRA definition in this project:**
   - Multi-head means **four independent LoRA branches with their own `A_i` and `B_i`**.
   - This is **not** HydraLoRA-style shared-`A`, multi-`B`.
   - All four branches are active simultaneously in S3.

2. **Implication for interpretation:**
   - S3 does **not** primarily test higher expressive power than rank-32 LoRA.
   - Since `sum_i B_i A_i` is still a low-rank update, S3 is treated as a study of **parameter decomposition, optimization dynamics, and potential specialization**, not as a claim of a larger effective parameter budget.

3. **Cluster-routed design in this project:**
   - S4 means **4 clusters -> 4 separate single-head LoRA adapters**.
   - It does **not** mean “4-head LoRA inside each cluster”.
   - The core comparison is:
     - **implicit specialization inside one decomposed adapter** (S3)
     - vs **explicit specialization by routing between four separate adapters** (S4)

4. **HyDE usage is deliberately restricted:**
   - HyDE is **not** treated as a standalone retrieval toggle for vanilla RAG.
   - Rationale: a non-adapted model is expected to generate low-value / noisy hypothetical texts on this corpus, which can degrade retrieval.
   - Therefore HyDE is only used in **S5**, where the hypothetical document is produced by the **best corpus-adapted adapter**.

### Working Hypotheses

**H1.** The best overall practical system will be **hybrid** (S5), because retrieval supplies explicit evidence while the selected adapter improves domain alignment of both HyDE generation and final answer generation.

**H2.** Plain 4-branch MH-LoRA (S3) may not strongly outperform single-head LoRA (S2) under equal parameter budget unless branch diversity is encouraged; if improvement appears, it should be interpreted as an optimization / decomposition effect rather than a simple capacity increase.

**H3.** Cluster-routed single-head LoRA (S4) can outperform S2 and S3 if the four clusters capture stable substructures of the task space (e.g., question types, reasoning modes, or document subdomains).

**H4.** If routing quality is weak or clusters are imbalanced, S4 may underperform S2 despite stronger specialization intuition.

**H5.** In this setting, the final research value is not only “which system wins”, but also **how specialization emerges or collapses** under equal budget.

### Anchor Literature (Design Rationale)

| Work | Role in this architecture |
|------|---------------------------|
| LoRA — *Low-Rank Adaptation of Large Language Models* | Base PEFT method and parameter-budget reference |
| QLoRA — *Efficient Finetuning of Quantized LLMs* | Training recipe that makes the project feasible on consumer hardware |
| HyDE — *Precise Zero-Shot Dense Retrieval without Relevance Labels* | Retrieval-time hypothetical document generation, adapted here into a domain-aware hybrid variant |
| RAFT — *Adapting Language Model to Domain Specific RAG* | Background for document-grounded adaptation and evidence-aware evaluation; not a core training recipe here |
| HydraLoRA — *An Asymmetric LoRA Architecture for Efficient Fine-Tuning* | Important contrast case showing what this project explicitly does **not** mean by MH-LoRA |
| MELoRA — *Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning* | Parallel-adapter intuition and decomposition motivation |
| Multi-Head Adapter Routing for Cross-Task Generalization | Adapter-routing rationale for modular specialization |
| R-LoRA — *Randomized Multi-Head LoRA for Efficient Multi-Task Learning* | Strong motivation for branch diversification and anti-collapse analysis |
| Poly-PRAG — *Parametric Retrieval-Augmented Generation using Latent Routing of LoRA Adapters* | Literature bridge between routed adapters and parametric retrieval ideas |

### Success Criteria

#### Primary selection logic

The architecture uses a **two-level evaluation policy**:

1. **Universal cross-system comparison** (applies to S1-S5):
   - Primary metric: `Q_main = 0.7 * S_det + 0.3 * S_asst`
2. **Retrieval-aware comparison** (applies to S1 and S5 only):
   - Secondary metric: grounding `G = F_beta(beta=2.5)`
   - Optional appendix metric: `Q_grounded = Q_main * G`

This avoids unfairly penalizing pure parametric systems for not emitting retrieval telemetry while preserving grounding as a critical axis for retrieval-based systems.

#### Success Criteria Table

| Metric | Baseline | Target | Rationale |
|--------|----------|--------|-----------|
| `Q_main` on locked test | S1 Classical RAG | Best final system improves over S1 by at least a practically meaningful margin (`>= 3` percentage points preferred) or shows clearly better trade-off at equal quality | Primary thesis result |
| `Q_main` among S2-S4 | S2 Single-head LoRA | Either S3 or S4 should beat S2 on development means, or the study must clearly explain why decomposition / routing failed | Parametric comparison is the scientific core |
| Grounding `G` (S1, S5) | S1 Classical RAG | S5 should not degrade grounding materially; target is `>= S1` or within `2` points if compensated by a clear `Q_main` gain | Hybrid must stay evidence-aware |
| TTFT | S1 Classical RAG and S2 Single-head | Parametric systems should remain faster than retrieval-heavy systems; S5 should stay within a practical latency envelope (`<= 1.5x` S1 median TTFT preferred) | Consumer-hardware realism |
| Peak VRAM | Hardware limit | All training runs must fit RTX 4060 8GB + 32GB RAM using quantization / grad accumulation | Hard operational constraint |
| Stability | Single run can be noisy | Mean/std over seeds must be reported for development experiments; preferred `std <= 3` pp on `Q_main` | Small-data robustness |
| Interpretability | No baseline | At least one of the two specialization analyses must show non-trivial structure rather than uniform collapse | Scientific value beyond benchmarking |

### Constraints (if applicable)

- **Latency:** TTFT and end-to-end latency must be measured on local consumer hardware; latency is a first-class metric, not an afterthought.
- **Interpretability:** The work must include at least two compact but defensible analyses of head / adapter specialization.
- **Resources:** Training and inference must fit **RTX 4060 8GB + 32GB RAM**. Quantization and QLoRA-style training are expected by default.
- **Data:** The core dataset is limited to **30 fixed documents** and **150 human-authored QA pairs**.
- **Fairness:** Retriever, prompt family, backbone, target modules, and parameter budget are fixed across comparative runs unless the deviation is explicitly logged.
- **HyDE restriction:** HyDE is only evaluated through the selected best adapter; no standalone vanilla-HyDE baseline is part of the core scope.

---

## 2. Experiment Pipeline

> Context: The iterative workflow for running experiments. Unlike product development, this is a cycle, not a linear flow.

```
[Corpus + Goldset Audit] -> [Backbone Feasibility Freeze] -> [Classical RAG Baseline] -> [Parametric Experiments S2-S4] -> [Select Best Adapter] -> [Hybrid S5]
            ^                                                                                                                                    |
            |____________________________________________________________________________________________________________________________________|
                                                  [Error Analysis, Specialization Analysis, Split Revision only if leakage/bias is found]
```

### Pipeline Stages

#### 1. Data Preparation

- **Load from `data/raw/`**
  - `documents/`: 30 fixed source documents
  - `goldset/`: 150 human-authored question-answer pairs
- **Normalize and preprocess**
  - preserve document/page mappings
  - unify document identifiers
  - normalize answer formats for deterministic types
  - attach evidence page metadata where available
- **Save to `data/processed/`**
  - normalized documents
  - normalized goldset
  - split manifests
  - clustering features / embeddings cache
- **Goldset expansion policy**
  - the goldset is expanded from 100 to **150** QA pairs by adding **50 additional human-authored QA pairs**
  - the added examples must follow the same answer-type schema and evidence annotation style
  - no synthetic QA generation is part of the core SSOT scope
- **Split protocol**
  - Create a **locked final test set of 30 questions** (20%)
  - Keep **120 questions** for development / model selection
  - Stratify by:
    - answer type (`number`, `boolean`, `name`, `names`, `date`, `free_text`, `null/unanswerable`)
    - approximate difficulty
    - document coverage
    - single-document vs multi-document reasoning when identifiable
- **Development protocol**
  - On the 120-question development pool, run **5-fold cross-validation**
  - Each fold recomputes all train-time artifacts using train split only:
    - cluster centroids
    - routed-adapter assignments
    - tuned hyperparameters if nested search is used
- **Leakage rules**
  - No cluster fitting, centroid estimation, or routing calibration may see validation/test examples
  - If paraphrase duplicates are found, they must be grouped before splitting

#### 2. Exploratory Data Analysis (EDA)

- Distribution analysis:
  - answer-type balance
  - question length
  - answer length
  - evidence-page count
  - document coverage frequency
- Retrieval sanity checks:
  - current Classical RAG hit-rate / recall at `k`
  - failure modes on raw-question retrieval
  - unanswerable handling
- Leakage and quality checks:
  - duplicate / paraphrase detection
  - ambiguous questions
  - inconsistent evidence mapping
  - malformed deterministic answers
- Clustering sanity for S4:
  - question-embedding visualization
  - rough balance of 4 clusters
  - silhouette / separability only as diagnostics, not as optimization target
- Document EDA artifacts in `eda/reports/EDA-Report.md`

#### 3. Deep Feature Engineering

> For this project, “feature engineering” means **task / representation engineering**, not tabular features.

- Analyze candidate representations after EDA findings are stable
- Prioritize high-signal candidates by expected quality impact and runtime cost:
  - question embeddings for clustering / routing
  - prompt templates for parametric QA
  - prompt template for HyDE generation
  - evidence formatting for hybrid answering
- Frozen choices after feasibility stage:
  - one backbone model family
  - one tokenizer
  - one set of LoRA target modules
  - one retriever pipeline
- Recommended representation strategy:
  - Use the **same dense embedding model as the retriever** for cluster construction in S4 unless a strong reason exists to separate them
- Document feature / representation strategy in `eda/reports/EDA-Insights.md`

#### 4. Baseline Establishment

- **System S1: Classical RAG**
  - fixed ingestion/indexing/retrieval pipeline
  - no adapter training
  - serves as main nonparametric baseline
- **Internal sanity baseline (optional, not headline)**
  - prompt-only no-RAG answer generation on the frozen backbone
  - used only to quantify the value of retrieval itself
- **System S2: Single-head LoRA**
  - first parametric baseline
  - rank 32 under the frozen target-module set
- Save benchmark artifacts, cached retrieval results, and logs for comparison

#### 5. Experimentation Cycle

- Formulate hypothesis/task in `memory_bank/TASKS.md`
- Keep working notes in `memory_bank/tasks/{TASK_ID}.md`
- Implement each experiment in `experiments/EXP-XXX_*/`
- Train and evaluate using `main_exp.py`
- Log metrics, latency, GPU memory, judge outputs, and specialization artifacts in the experiment report

##### 5.1 Experiment Phases (Frozen Order)

| EXP ID | Goal | Core Output |
|--------|------|-------------|
| EXP-000_data_audit | Validate goldset v150, evidence pages, answer-type consistency | Clean processed dataset + split manifest |
| EXP-010_backbone_feasibility | Choose and freeze one backbone that fits the hardware and provides acceptable baseline quality | Frozen backbone choice |
| EXP-020_classical_rag | Measure S1 and retrieval diagnostics | Nonparametric baseline report |
| EXP-030_single_head_lora | Train and evaluate S2 | Main parametric baseline |
| EXP-040_mh_lora_4branch | Train and evaluate S3 | Decomposed-rank comparison |
| EXP-050_cluster_routed_lora | Train and evaluate S4 | Explicit-routing comparison |
| EXP-060_select_best_adapter | Choose best parametric adapter based on development results | Frozen adapter choice for S5 |
| EXP-070_hybrid_best_adapter_hyde | Build and evaluate S5 | Hybrid top-line |
| EXP-080_specialization_analysis | Generate the two analysis artifacts | Interpretability section figures |
| EXP-090_locked_test | Final retrain on dev pool and evaluate once on locked test | Final thesis tables |

##### 5.2 Core System Definitions

###### S2 — Single-head LoRA

- One adapter
- Rank = 32
- Same target modules across all parametric systems
- Same optimizer family, same scheduler family, same prompt format as S3/S4

###### S3 — 4-branch MH-LoRA

- Four independent branches
- Each branch rank = 8
- Final update is the sum of branch updates
- Default anti-collapse policy:
  - independent random seeds per branch
  - branch dropout during training is allowed and recommended
- Interpretation rule:
  - any gain over S2 is interpreted as **decomposition / optimization / specialization benefit**, not naive capacity gain

###### S4 — Cluster-routed single-head LoRA

- Four separate single-head adapters
- Each adapter rank = 8
- Training data is partitioned by cluster assignment
- Only one adapter is active per query at inference
- Router design (default, frozen unless explicitly revised):
  - encode questions into dense embeddings
  - fit `k-means` with `k=4` on **train split only**
  - use nearest centroid by cosine similarity at inference
- Router simplicity principle:
  - no learned router in the core scope
  - learned routing is appendix-only if time remains
- Cluster health rule:
  - if any cluster gets too few samples to train meaningfully, rerun clustering with a different seed or rebalance before training; do not silently accept pathological clusters

###### S5 — Hybrid RAG + Best Adapter + HyDE

- Select the best adapter from S2-S4 using development results
- Use this adapter to generate a **hypothetical document / answer-like text** for the user question
- Feed the resulting HyDE text into the retriever instead of the raw question
- Retrieve real corpus evidence
- Generate the final answer with the same selected best adapter conditioned on retrieved evidence
- This is the **only** core use of HyDE in the project

##### 5.3 Parameter Budget Policy

To keep comparisons as fair as possible:

- S2 uses rank 32
- S3 uses `4 x rank 8`
- S4 uses `4 adapters x rank 8`

This equalizes the **stored trainable-parameter budget at the system level** as closely as possible under the chosen architecture definitions.

##### 5.4 Backbone & Training Policy

- One backbone only after feasibility freeze
- Preferred model class: **instruction-tuned open model in the 1B-3B range**
- Default decision rule:
  - choose the strongest model that reliably fits training/inference on RTX 4060 8GB with 4-bit quantization
- Quantization / PEFT policy:
  - QLoRA-style 4-bit training is the default
- Recommended default training settings (subject to feasibility confirmation):
  - optimizer: AdamW / paged AdamW
  - quantization: NF4 4-bit
  - precision: bf16 if available, otherwise fp16
  - max sequence length: 1024-2048 depending corpus chunk format and hardware
  - effective batch size: achieved via gradient accumulation
- Hyperparameter sweep (minimum required):
  - learning rate: `5e-5`, `1e-4`, `2e-4`, `4e-4`
  - LoRA alpha: `16`, `32`, `64`
  - adapter dropout: `0.0`, `0.05`, `0.1`
  - epochs / max steps: tune with early stopping on development metric
- Fixed target modules after feasibility:
  - default recommendation: `q_proj`, `v_proj`
  - `o_proj` may be included only if done consistently across S2-S4 and parameter accounting is updated in this file

##### 5.5 Evaluation Protocol

###### Universal metrics (all systems)

**Primary cross-system metric**

`Q_main = 0.7 * S_det + 0.3 * S_asst`

Where:
- `S_det` = deterministic score across deterministic answer types
- `S_asst` = LLM-judge score for free-text answers

###### Deterministic scoring rules

- `number` -> exact numeric comparison with tolerance where appropriate
- `boolean` -> exact boolean match
- `name` -> normalized exact string match
- `names` -> set-based overlap / Jaccard-style scoring
- `date` -> exact ISO date match
- `null` -> valid answer when information is absent and should not be inferred

###### Free-text scoring (`S_asst`)

A fixed judge rubric with five binary criteria:
- correctness
- completeness
- grounding / support
- confidence calibration
- clarity & relevance

Judge policy:
- same judge prompt for all runs
- version-pinned judge model
- no self-judging by the evaluated model
- manual audit on a small subset is required before final conclusions

###### Retrieval-aware metrics (S1 and S5 only)

**Grounding metric**

`G = F_beta(beta=2.5)` on retrieved page references.

Grounding is not used as a universal cross-system multiplier because pure parametric systems do not expose retrieval telemetry in the same way.

**Optional retrieval appendix metrics**
- Recall@k
- Hit@k
- MRR / nDCG if already available in the pipeline

###### Systems metrics

- TTFT (median, p95)
- end-to-end latency (median, p95)
- tokens/sec during generation
- peak VRAM for training and inference
- wall-clock training time per epoch / run
- malformed output rate

###### Challenge-style metric adaptation

The original challenge-style formula is **not** the main thesis selector.

Use it as follows:
- keep `S_det` and `S_asst` as the universal quality core
- keep grounding `G` as a retrieval-only metric
- track telemetry failures as a **malformed output rate**, not as a headline multiplicative factor
- track TTFT buckets separately; if needed, reproduce bucketized TTFT as an appendix metric only

##### 5.6 Aggregation & Statistical Reporting

- Development results are reported as **mean +/- std** over seeds and folds where applicable
- Locked test is run once per final chosen configuration after retraining on the full 120-question development pool
- Report both:
  - aggregate metrics
  - breakdown by answer type
- Mandatory breakdowns:
  - `number`
  - `boolean`
  - `name` / `names`
  - `date`
  - `free_text`
  - `null / unanswerable`

##### 5.7 Adapter Selection Rule for S5

The best adapter from S2-S4 is selected using the following priority order:

1. Highest development `Q_main`
2. Better latency / TTFT if `Q_main` is within a small margin
3. Lower variance across seeds
4. Simpler system preferred if quality is effectively tied

This selected adapter is then frozen and used inside S5.

##### 5.8 Specialization / Scientific Analysis (Exactly Two Core Analyses)

**A1. Heatmap: head or adapter usage vs question type**

Purpose:
- detect whether certain branches / routed adapters are disproportionately associated with certain answer or reasoning types

Implementation:
- for S3: quantify per-branch contribution norm or activation contribution by question type
- for S4: visualize routing frequency by cluster/adapter vs question type

**A2. Pairwise similarity matrix between heads / adapters**

Purpose:
- check whether branches / adapters genuinely diverged or effectively collapsed

Implementation options:
- cosine similarity between flattened adapter updates
- similarity of head contributions on a fixed probe set

Only these two analyses are mandatory in the core thesis scope.

#### 6. Final Evaluation

- Final locked test evaluation is performed only after:
  - backbone is frozen
  - hyperparameters are selected
  - best adapter is selected
  - all core systems S1-S5 are implemented and validated
- Deliverables:
  - final comparison table
  - per-answer-type table
  - latency / VRAM table
  - two specialization figures
  - explicit discussion of failure modes and threats to validity
- Model documentation in `models/model_<metric>_<value>/MODEL_REPORT.md`
- Model architecture and validation strategy are documented in model reports, not here

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Small dataset causes noisy conclusions | High | Locked test + dev CV + multiple seeds + error analysis |
| Cluster imbalance in S4 | High | Stratified clustering diagnostics, rerun if pathological |
| S3 heads collapse to similar updates | Medium | Independent branch seeds, dropout, similarity analysis |
| HyDE harms retrieval | High | HyDE only through selected adapted adapter; compare against S1 on grounding and latency |
| Hardware OOM | High | QLoRA, 4-bit quantization, conservative target modules, gradient accumulation |
| Judge instability | Medium | Version-pin judge, manual audit subset, keep deterministic metrics separate |

---

## 3. Technology Stack

> Context: Tools, libraries, and infrastructure for the project.

- **Language:** Python 3.11+
- **Core Libraries:**
  - **Data:** `pandas`, `numpy`, `pyarrow`
  - **ML / Eval:** `scikit-learn`, `scipy`, `rapidfuzz`, `pydantic`
  - **DL / LLM:** `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`
  - **Retrieval / Embeddings:** existing project retriever stack, plus `sentence-transformers` / vector DB client as required by the reused pipeline
  - **Visualization:** `matplotlib`, `seaborn`, `plotly` (optional interactive diagnostics)
- **Environment:** `uv` + local virtual environment; optional Docker only if it does not slow iteration
- **Compute:** local workstation, **RTX 4060 8GB**, **32GB RAM**
- **Tracking / Logging:** local filesystem first; `wandb` or `mlflow` optional but not required by SSOT
- **Model Storage:** local `models/` directory; adapters stored separately from merged checkpoints whenever possible
- **Judge Execution:** fixed external or local stronger model, version-pinned and logged

### Backbone Policy

- The project uses **one frozen backbone after EXP-010**.
- Candidate backbones should come from the **1B-3B instruct** range.
- Selection criteria:
  1. fits RTX 4060 8GB with QLoRA-style training
  2. adequate baseline reasoning quality on the corpus
  3. stable generation format for deterministic answers

### Setup Commands

```bash
uv sync                    # Install dependencies
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Reproducibility Requirements

- All experiment configs are file-based and versioned
- Random seeds must be fixed and logged
- Every run must log:
  - git commit hash or equivalent code version
  - dataset split version
  - backbone name
  - quantization mode
  - adapter config
  - router config (if any)
  - judge version

---

## 4. Code Organization & Conventions

### Project Structure

```text
data/
├── raw/
│   ├── documents/                 # 30 source documents
│   └── goldset/                   # 150 human-authored QA pairs
├── processed/
│   ├── corpus/                    # normalized docs, page mapping, chunk ids
│   ├── goldset/                   # normalized QA and answer types
│   ├── splits/                    # locked test + dev CV manifests
│   ├── retrieval/                 # cached retrieval outputs / diagnostics
│   └── clustering/                # cached embeddings, centroids, assignments
└── interim/                       # optional temporary artifacts, never SSOT

src/
├── __init__.py
├── data/
│   ├── documents.py               # corpus loading / normalization
│   ├── goldset.py                 # QA loading / validation / answer normalization
│   ├── splits.py                  # split generation and CV utilities
│   └── validation.py              # schema and leakage checks
├── rag/
│   ├── ingestion.py               # existing ingestion wrappers
│   ├── indexing.py                # index build / reuse wrappers
│   ├── retrieval.py               # retriever calls and telemetry
│   ├── hyde.py                    # HyDE generation + retrieval glue for S5
│   └── prompts.py                 # RAG / HyDE prompts
├── adapters/
│   ├── single_lora.py             # S2 definition
│   ├── mh_lora.py                 # S3 definition
│   ├── routed_lora.py             # S4 orchestration
│   ├── routing.py                 # clustering, centroid selection, routing logic
│   └── config.py                  # adapter config schemas
├── training/
│   ├── dataset.py                 # parametric training datasets
│   ├── collators.py               # batching logic
│   ├── trainer.py                 # shared train/eval loops
│   ├── checkpoints.py             # adapter save/load helpers
│   └── prompts.py                 # training prompt templates
├── evaluation/
│   ├── deterministic.py           # S_det metrics
│   ├── judge.py                   # S_asst rubric and scoring wrapper
│   ├── grounding.py               # page-level F_beta grounding
│   ├── latency.py                 # TTFT and E2E latency
│   ├── telemetry.py               # malformed output and format checks
│   └── reports.py                 # summary table generation
├── analysis/
│   ├── specialization.py          # heatmap and usage analysis
│   ├── similarity.py              # pairwise head/adapter similarity
│   └── visualization.py           # plot helpers
└── utils/
    ├── io.py
    ├── logging.py
    ├── seeding.py
    └── types.py

main.py                            # Main pipeline (cell-like blocks with # %% separators)
config.py                          # Global configuration and hyperparameters

eda/
├── src/
│   ├── eda.py                     # EDA analysis and visualizations
│   └── deep_eda.py                # Representation / routing diagnostics
├── results/
│   ├── figures/
│   └── tables/
└── reports/
    ├── EDA-Report.md
    └── EDA-Insights.md

experiments/
├── EXP-000_data_audit/
├── EXP-010_backbone_feasibility/
├── EXP-020_classical_rag/
├── EXP-030_single_head_lora/
├── EXP-040_mh_lora_4branch/
├── EXP-050_cluster_routed_lora/
├── EXP-060_select_best_adapter/
├── EXP-070_hybrid_best_adapter_hyde/
├── EXP-080_specialization_analysis/
└── EXP-090_locked_test/
    ├── main_exp.py                # Experiment pipeline (cell-like blocks)
    ├── config.py                  # Experiment-specific config
    └── REPORT.md                  # Experiment report

models/
├── base/
├── adapters/
│   ├── single_head/
│   ├── mh4/
│   ├── routed_cluster/
│   └── hybrid_selected/
└── merged/                        # only if merged checkpoints are needed

logs/
└── [experiment_id]/

memory_bank/
├── ARCHITECTURE.md                # canonical project architecture (this file)
├── STATE.md                       # current project state
└── tasks/
    ├── TASKS.md
    └── {TASK_ID}.md
```

### Code Style

- **Cell-like execution:** Use `# %% [Block Name]` separators in `main.py` and `main_exp.py` for block-by-block execution
- **Typed functions:** All functions should have type hints
- **Reusability:** Functions in `src/` should be reusable across experiments (DRY)
- **Docstrings:** All public functions must have docstrings
- **Config-first design:** Hard-coded experiment values inside scripts are discouraged; configs must live in `config.py` or experiment configs
- **Evaluation isolation:** Metric code must not depend on training code side effects
- **No silent defaults:** Any fallback behavior that changes model, data split, router, or metric logic must be logged explicitly

### Naming Conventions

- **Scripts:** `main.py`, `main_exp.py`, `config.py`
- **Modules:** lowercase with underscores (`single_lora.py`, `routed_lora.py`, `grounding.py`)
- **Experiments:** `EXP-{number}_{description}/`
- **Adapters:** `adapter_{method}_r{rank}_seed{seed}`
- **Routers / centroids:** `router_k{K}_seed{seed}`
- **Models:** `model_{experiment_id}_{metric}_{value}.pkl` or adapter-native format
- **Reports:** `REPORT.md`, `MODEL_REPORT.md`, `ERROR_ANALYSIS.md`
- **Splits:** `split_v{version}_{name}.json`

### Logging

- Training logs in `logs/`, format `[YYYY-MM-DD HH:MM:SS] [LEVEL] - Message`
- Additionally store machine-readable metrics in JSON / JSONL
- Random seeds: always set and document for reproducibility
- Every experiment report must include:
  - goal / hypothesis
  - exact config
  - dataset split version
  - backbone and quantization mode
  - metrics table
  - latency table
  - failure cases
  - decision: continue / revise / stop

### SSOT Rules

- `memory_bank/ARCHITECTURE.md` is the authoritative definition of:
  - core systems
  - split strategy
  - primary metrics
  - parameter-budget logic
  - final evaluation protocol
- `STATE.md` tracks current progress but does not override architecture
- Experiment reports may propose deviations, but they are not canonical until this file is updated
- Any change to the following requires updating this file first:
  - backbone family
  - number of systems in the core comparison
  - rank / parameter-budget policy
  - primary metric
  - split protocol
  - HyDE usage policy

### Definition of Done

The project is considered architecturally complete when all of the following are true:

1. S1-S5 are implemented and evaluated under the frozen protocol
2. Best adapter from S2-S4 is selected transparently
3. Locked test results are produced
4. `Q_main`, grounding, latency, and VRAM tables are available
5. Two specialization analyses are generated
6. Threats to validity and failure modes are explicitly documented

