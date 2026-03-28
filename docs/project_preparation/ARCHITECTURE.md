# Project Architecture: Nonparametric, Supervised Parametric, Downstream Supervision-Free Parametric, and Hybrid Knowledge Injection for Document-Grounded QA on Consumer Hardware

**Status:** SSOT (single source of truth)  
**Version:** 2.0  
**Last updated:** 2026-03-26  
**Project shorthand:** `knowledge-injection-consumer-hardware`  
**Primary practical setting:** domain-specific / legal-style document-grounded QA on a fixed corpus  

> This document is the authoritative source for project scope, experiment design, evaluation logic, repository structure, and terminology. If any concrete experiment diverges from this document, the deviation must be documented in the corresponding `experiments/EXP-XXX_*/REPORT.md`. If the divergence is strategic rather than incidental, this file must be updated first.

---

## 1. Problem Statement & Success Criteria

> Context: Define the problem clearly, what success looks like.

### Problem Definition

The project studies **knowledge injection into a frozen LLM under consumer-hardware constraints**.
The practical task is **document-grounded QA** over a **fixed domain corpus** and a **human-authored goldset of 150 question-answer pairs**.
The domain is legal / legal-style QA, but the research focus is **not legal QA itself**. The domain corpus is only a controlled case study for comparing different ways of providing external knowledge to an LLM.

The project intentionally compares methods that operate under **different natural informational constraints**:

- **Nonparametric retrieval** has access to the full corpus through an index, but depends on retrieval quality.
- **Supervised parametric adaptation** sees human-authored QA supervision, but only where such supervision exists.
- **Downstream supervision-free hypernetwork-based parametric adaptation** sees raw documents through a pre-trained hypernetwork, but is constrained by adapter quality, context packing, and adapter capacity.
- **Hybrid systems** combine external retrieval with an adapted generator, but incur added latency and engineering complexity.

This informational asymmetry is **not a flaw in the design**. It is the main research subject. The project asks what each family can achieve **under its own natural regime**, rather than forcing artificial symmetry by crippling methods until they all use the exact same channel.

#### Main Research Question (RQ1)

**How do four knowledge-injection paradigms compare on consumer hardware for document-grounded QA over a fixed corpus: nonparametric retrieval, supervised parametric adaptation, downstream supervision-free hypernetwork-based parametric adaptation, and hybrid retrieval + adaptation, when each paradigm operates under its natural informational and computational constraints?**

Short form for writing:

**Nonparametric, Supervised Parametric, Downstream Supervision-Free Parametric, and Hybrid Knowledge Injection for Document-Grounded QA on Consumer Hardware**

#### Inner Study / Secondary Research Question (RQ2)

Inside the downstream supervision-free parametric branch, the project also runs a narrower controlled study:

**Does cluster-routed Doc-to-LoRA outperform monolithic Doc-to-LoRA on the same corpus under consumer-hardware constraints, and if yes, is the gain better explained by memory sharding / capacity relief than by simple adapter count increase?**

RQ2 is **not** the top-level project framing. It is an inner study inside RQ1.

#### Terminology Rules (Mandatory)

1. **“Unsupervised parametric”** is acceptable as shorthand only in tables and diagrams.
   In precise prose, use one of:
   - **downstream supervision-free hypernetwork-based parametric adaptation**
   - **amortized parametric knowledge injection**
   - **Doc-to-LoRA-based parametric injection**

2. The project must **not** claim that Doc-to-LoRA is fully supervision-free in an absolute sense.
   The correct interpretation is that **downstream adaptation to the project corpus does not require task-specific QA supervision**, because the hypernetwork is already pre-trained upstream.

3. The supervised LoRA baseline must **not** be described as “the model learned the whole corpus”.
   The correct interpretation is that it learns to answer in a domain-specific setting from **goldset-style supervision**, ideally in a context-aware RAFT-style format.

4. The project must **not** claim full corpus internalization.
   All conclusions are limited to the benchmark slice represented by the corpus, the goldset, the prompt protocol, the chosen model backbone, and the consumer-hardware setting.

#### System Inventory (Frozen Project-Level Matrix)

| ID | System | Family | Informational input | What is adapted / built | Primary role |
|----|--------|--------|---------------------|-------------------------|--------------|
| S1 | Classical RAG | Nonparametric | Full corpus through retrieval index | Index only, no model adaptation | Main nonparametric baseline |
| S2 | QLoRA fine-tuned (RAFT-style) | Supervised parametric | 150 human-authored QA pairs, formatted as question + support chunks + answer | One LoRA adapter on fixed backbone | Main supervised parametric baseline |
| S3 | Doc-to-LoRA (monolithic) | Downstream supervision-free parametric | Full corpus through Doc-to-LoRA hypernetwork | One merged LoRA adapter built from corpus chunks | Main hypernetwork-based parametric baseline |
| S4 | Cluster-routed Doc-to-LoRA | Downstream supervision-free parametric + routing | Full corpus split into semantic clusters | One merged adapter per cluster + router | Main inner-study system |
| S5 | Hybrid: RAG + best adapter | Hybrid | Retrieval + best adapter from S2-S4 | Best adapter + retrieval pipeline | Practical top-line |

#### Family-Specific Clarifications

- **S1 Classical RAG** uses the existing ingestion, indexing, and retrieval stack as the reference nonparametric pipeline.
- **S2 QLoRA fine-tuned** is trained in a **RAFT-style open-book format** whenever feasible: question + gold evidence chunks + answer, optionally with distractors. This keeps S2 aligned with document-grounded QA rather than pure closed-book memorization.
- **S3 Doc-to-LoRA (monolithic)** is the baseline for corpus-to-adapter internalization. Because the corpus is larger than one direct Doc-to-LoRA pass can safely absorb, S3 uses a fixed chunking and adapter-merging procedure to produce one global adapter.
- **S4 Cluster-routed Doc-to-LoRA** is the project’s own main technical extension. Documents are clustered, one Doc-to-LoRA-derived adapter is built per cluster, and a router selects which adapter to activate for a query.
- **S5 Hybrid** combines retrieval with the best adapter selected from S2-S4.
  - **HyDE is evaluated only inside S5**, not as a standalone toggle for vanilla RAG.
  - Within S5, HyDE is treated as a **retrieval ablation / enhancement** generated only by the chosen best adapter.

#### Primary Scientific Value

The main project contribution is **not** “inventing a new LoRA architecture”.
The main contribution is a controlled comparison of **four families of knowledge injection** on one task and one hardware regime, with an embedded study of **monolithic vs routed hypernetwork-based parametric memory**.

In one sentence:

**The project compares external retrieval, supervised PEFT, downstream supervision-free corpus-to-adapter internalization, and hybrid composition on the same document-grounded QA benchmark, then asks whether routing improves hypernetwork-based parametric memory under capacity constraints.**

#### Working Hypotheses (Project-Level)

**H1.** The best overall practical trade-off will come from **S5 Hybrid**, because retrieval supplies explicit evidence while the selected adapter improves domain alignment of query reformulation and answer generation.

**H2.** **S2 Supervised QLoRA** will likely show stronger output formatting discipline and better behavior on highly represented supervised patterns, but its factual coverage will remain bounded by the supervision slice.

**H3.** **S3/S4 Doc-to-LoRA-based systems** may be more competitive than S2 on facts not densely covered by supervised training, because they ingest raw corpus content rather than only annotated QA examples.

**H4.** **S4 Cluster-routed Doc-to-LoRA** should outperform **S3 Monolithic Doc-to-LoRA** if cluster-level sharding preserves information better than a single global merge and if the router is sufficiently aligned with corpus structure.

**H5.** **S1 Classical RAG** will remain strongest on deterministic lookup-heavy questions that depend on exact extraction, attribution, and grounding.

**H6.** Even if S3/S4 do not beat S1, the project still yields a meaningful result by quantifying the practical limits of parametric injection on a small consumer-hardware stack.

#### Inner Study Focus (RQ2 Operationalization)

The inner study concerns only **S3 vs S4**.
Its purpose is to disentangle two explanations:

1. **Capacity / packaging bottleneck** — a single adapter built from many chunk-level internalizations loses information when merged.
2. **Routing benefit** — multiple smaller semantically localized adapters preserve useful specialization and can be selected at inference time.

The inner study is successful if it can answer at least one of the following cleanly:

- Does cluster routing beat a monolithic merged adapter?
- If yes, is the gain consistent across answer types or concentrated in certain regions of the benchmark?
- If not, is the failure due to weak routing, weak clustering, or adapter merge instability?

### Success Criteria

#### Primary selection logic

The project uses a **three-layer evaluation policy**:

1. **Universal cross-system comparison** for S1-S5:
   - `Q_main = 0.7 * S_det + 0.3 * S_asst`
2. **Retrieval-aware evaluation** for systems with retrieval (S1 and S5):
   - `G = F_beta(beta = 2.5)` for grounding / evidence coverage
   - retrieval recall-style metrics (`Recall@k`, evidence hit-rate)
3. **Systems-cost evaluation** for all systems:
   - TTFT
   - end-to-end latency
   - peak VRAM
   - offline packaging cost (index build, training time, or adapter-generation time)
   - storage footprint of artifacts

This structure preserves fairness:

- Pure parametric systems are **not penalized** for lacking retrieval telemetry.
- Retrieval systems are still judged on grounding, which is central to document-grounded QA.
- Consumer-hardware practicality is treated as a first-class result, not an appendix-only consideration.

#### Success Criteria Table

| Metric | Baseline | Target / decision rule | Rationale |
|--------|----------|------------------------|-----------|
| `Q_main` on locked test | S1 Classical RAG | Preferred winning system beats S1 by a practically meaningful margin, or matches it with clearly better cost / simplicity | Main thesis result |
| `Q_main` among S2-S4 | S2 Supervised QLoRA | At least one of S3 or S4 should either exceed S2 or clearly justify why supervised adaptation remains stronger | Core parametric comparison |
| `Q_main` S4 vs S3 | S3 Monolithic D2L | S4 should improve mean quality or reduce failure concentration under same hardware envelope | Main inner-study result |
| Grounding `G` | S1 Classical RAG | S5 should preserve strong grounding even when the adapter changes generator behavior | Hybrid must not break citation value |
| TTFT | S1 Classical RAG | Any hybrid or parametric benefit must be interpreted against added latency | Practicality on consumer hardware |
| Peak VRAM | Feasible local run | All reported main systems must fit the local hardware regime or be explicitly marked infeasible | Hard feasibility gate |
| Offline packaging cost | N/A | Must be measured and reported for every family | Fair comparison across paradigms |
| Malformed answer rate | Existing prompt baseline | Lower is better; used as robustness metric, not headline ranking metric | Format reliability |

#### Reporting Rules

- Every main result must be reported **both as an aggregate and by answer type**:
  - `number`
  - `boolean`
  - `name`
  - `names`
  - `date`
  - `free_text`
  - `null / unanswerable`
- Every retrieval-based result must include both answer quality and grounding.
- Every parametric result must include at least one cost metric beyond quality.
- Claims about “better knowledge injection” require discussion of **quality + cost + grounding trade-off**, not quality alone.

### Constraints (if applicable)

- **Hardware:** RTX 4060 8GB VRAM, 32GB RAM, local consumer machine.
- **Training recipe constraint:** QLoRA-style quantized PEFT is the default path for supervised adaptation.
- **Backbone constraint:** one primary backbone is selected during feasibility and then frozen. A second backbone may be used only for a very small confirmatory run if time permits.
- **Corpus constraint:** the corpus is fixed before the final experiment campaign. Any corpus-count discrepancy must be resolved through a frozen `data/manifests/corpus_manifest.csv` before final runs.
- **Goldset constraint:** the final goldset size is **150 human-authored QA pairs**.
- **Annotation constraint:** no large synthetic QA generation is part of the core project scope.
- **Doc-to-LoRA constraint:** the hypernetwork is **not retrained** in this project. Only downstream use of the available method is in scope.
- **Research-scope constraint:** the project does not attempt to invent a new hypernetwork architecture. Its original contribution is the comparison framework and the routed Doc-to-LoRA study.
- **Evaluation constraint:** no method may see the locked final test questions during training, routing calibration, prompt tuning, or system selection.
- **Claim constraint:** conclusions must remain bounded to this corpus, this benchmark, this backbone, and this hardware regime.

---

## 2. Experiment Pipeline

> Context: The iterative workflow for running experiments. Unlike product development, this is a cycle, not a linear flow.

```
[Data Preparation] -> [EDA] -> [Deep Feature Engineering] -> [Baseline] -> [Hypothesis] -> [Experiment] -> [Evaluate]
                                                                       ^                                  |
                                                                       |__________________________________|
                                                                              (iterate until target met)
```

### Pipeline Stages

#### 1. Data Preparation

- **Corpus freeze**
  - Build `data/manifests/corpus_manifest.csv` with one row per source document.
  - Freeze corpus membership, source identifiers, page mapping, and document-level metadata before main runs.
  - All subsequent preprocessing must be reproducible from manifest + scripts.

- **Goldset freeze**
  - Expand the human-authored goldset to **150 QA pairs**.
  - Standardize schema:
    - `question_id`
    - `question`
    - `answer`
    - `answer_type`
    - `evidence_doc_ids`
    - `evidence_page_ids` or equivalent span/page mapping
    - `is_unanswerable`
  - Keep answer formatting rules explicit for deterministic scoring.

- **Split protocol**
  - Create a **locked final test set of 30 questions**.
  - Keep the remaining **120 questions** for development.
  - Stratify by:
    - answer type
    - rough difficulty
    - document coverage
    - single-document vs multi-document reasoning when identifiable
    - unanswerable presence

- **Development protocol**
  - On the 120-question development pool, run **5-fold cross-validation**.
  - Each fold recomputes all train-time artifacts from the train split only for systems that require them.

- **S2 data formatting**
  - Convert supervised examples into **RAFT-style open-book training format** whenever feasible:
    - question
    - supporting chunk(s)
    - optional distractors
    - answer
  - This format is the default for S2.
  - A pure QA-only S2 variant is allowed only as an appendix ablation if needed.

- **Leakage rules**
  - No test example may influence:
    - prompt tuning
    - hyperparameter selection
    - adapter choice for S5
    - routing calibration using labeled outcomes
  - Near-duplicate or paraphrase pairs must be grouped before splitting.

#### 2. Exploratory Data Analysis (EDA)

- **Goldset EDA**
  - answer-type balance
  - question length distribution
  - answer length distribution
  - evidence-page count
  - document coverage frequency
  - unanswerable share

- **Retrieval sanity checks**
  - baseline hit-rate / recall at `k`
  - failure cases for raw-question retrieval
  - document coverage skew
  - deterministic lookup vs reasoning-heavy failure split

- **Corpus capacity audit**
  - document lengths
  - page counts
  - approximate token counts
  - chunk-length distribution after preprocessing
  - whether the corpus or its subsets fit direct Doc-to-LoRA input assumptions

- **Clustering diagnostics for S4**
  - document-embedding visualization
  - cluster balance at `k = 4`
  - rough semantic coherence of clusters
  - note: silhouette-like statistics are diagnostics only, not optimization targets

- **EDA deliverables**
  - `eda/reports/EDA-Report.md`
  - `eda/reports/EDA-Insights.md`

#### 3. Deep Feature Engineering

> For this project, “feature engineering” means representation and packaging engineering, not tabular feature crafting.

- **Retrieval-side representation decisions**
  - final retriever embedding model
  - chunking strategy
  - top-k retrieval depth
  - reranking usage if already present in the existing stack

- **Supervised parametric representation decisions (S2)**
  - prompt template for RAFT-style training
  - chunk ordering inside open-book context
  - distractor policy
  - answer format template

- **Doc-to-LoRA packaging decisions (S3 / S4)**
  - corpus segmentation window for Doc-to-LoRA ingestion
  - chunk-to-adapter conversion workflow
  - adapter merge rule in delta-weight space or equivalent implementation space
  - cluster granularity for routed variant
  - routing input representation

- **Frozen decisions after feasibility**
  - one primary backbone model
  - one tokenizer
  - one set of LoRA target modules for S2
  - one retrieval pipeline for S1 / S5
  - one primary Doc-to-LoRA packaging strategy for S3 / S4

#### 4. Baseline Establishment

##### S1 — Classical RAG baseline

- Use the existing ingestion, indexing, and retrieval pipeline.
- Freeze retriever / index configuration before the main comparison campaign.
- Evaluate with the same answer-formatting prompt protocol used elsewhere when possible.

##### S2 — Supervised QLoRA baseline

- Train one LoRA adapter on the frozen backbone.
- Default format: **RAFT-style open-book supervision**.
- QLoRA is the default training recipe due to hardware limits.
- S2 is interpreted as **context-use adaptation under human supervision**, not whole-corpus internalization.

##### S3 — Monolithic Doc-to-LoRA baseline

- Build one corpus-wide adapter through a fixed pipeline:
  1. segment corpus into Doc-to-LoRA-compatible windows
  2. generate one intermediate adapter per segment or segment batch
  3. merge intermediates into one final adapter using a fixed merge rule
- S3 is explicitly acknowledged as an **approximate packaging baseline**, not a perfect one-shot internalization of the whole corpus.
- The primary purpose of S3 is to establish whether corpus-to-adapter memory is viable at all in this setting.

##### S4 — Cluster-routed Doc-to-LoRA baseline

- Primary design:
  1. embed documents using the frozen corpus representation
  2. cluster the corpus into **4 semantic clusters**
  3. build one merged Doc-to-LoRA adapter per cluster
  4. route each query to one cluster adapter at inference time
- Default clustering granularity is **document-level** for interpretability and simplicity.
- Chunk-level clustering may be used only as an appendix ablation if document-level clustering proves clearly inadequate.
- Default router:
  - embed the user question
  - compute cosine similarity to cluster centroids
  - activate top-1 cluster adapter
- Learned routers are out of the core scope unless they are trivial to add after the main campaign.

##### S5 — Hybrid baseline

- S5 combines retrieval with the best adapter selected from S2-S4.
- The best adapter is selected on development performance, not on the locked test set.
- S5 must be evaluated in at least two retrieval variants if feasible:
  - **S5a:** raw-question retrieval + best adapter
  - **S5b:** best-adapter-generated HyDE + retrieval + best adapter
- In headline reporting, S5 refers to the best-performing hybrid configuration.
- HyDE is therefore a **sub-ablation inside the hybrid family**, not a standalone global toggle.

#### 5. Experimentation Cycle

- Formulate each concrete experiment in `memory_bank/tasks/TASKS.md`.
- Keep one working note per experiment in `memory_bank/tasks/{TASK_ID}.md`.
- Implement isolated experiment runs in `experiments/EXP-XXX_{description}/`.
- Use config-driven runs with frozen seeds and explicit artifact paths.
- Report every run in `REPORT.md` with:
  - hypothesis
  - exact config
  - data split / fold
  - metrics
  - failure modes
  - interpretation

Recommended experiment ordering:

1. **EXP-001** — Lock S1 Classical RAG baseline.
2. **EXP-002** — Feasibility run for S2 supervised QLoRA on smallest practical setup.
3. **EXP-003** — Feasibility run for S3 monolithic Doc-to-LoRA packaging.
4. **EXP-004** — Corpus clustering study for S4.
5. **EXP-005** — Main comparison S1 vs S2 vs S3 vs S4 on development folds.
6. **EXP-006** — Hybrid study S5a vs S5b.
7. **EXP-007** — Locked test evaluation.
8. **EXP-008** — Error analysis and interpretability.

#### 6. Final Evaluation

- **Locked test execution**
  - Run only after the main system designs are frozen.
  - Use the locked 30-question test set once for headline results.

- **Universal metrics for S1-S5**
  - `S_det` for deterministic answer types
  - `S_asst` for free-text judged answers
  - `Q_main = 0.7 * S_det + 0.3 * S_asst`
  - malformed answer rate
  - TTFT
  - end-to-end latency
  - peak VRAM
  - offline preparation cost:
    - index build time for S1
    - QLoRA training time for S2
    - Doc-to-LoRA adapter generation / merge time for S3-S4

- **Retrieval-aware metrics for S1 and S5**
  - grounding `G = F_beta(beta = 2.5)`
  - retrieval `Recall@k`
  - evidence hit-rate / page overlap where available

- **Interpretation rules**
  - A system is not “better” based on `Q_main` alone if its cost or grounding profile is materially worse.
  - Hybrid wins must be discussed in terms of both accuracy and retrieval dependence.
  - Strong performance by S2 must be interpreted as success of supervised adaptation, not evidence that the whole corpus was internalized.
  - Strong performance by S3/S4 must be interpreted as success of corpus-to-adapter packaging within this benchmark, not universal replacement of retrieval.

- **Inner-study reporting for RQ2**
  - Compare S3 vs S4 overall and by answer type.
  - Report routing distribution across clusters.
  - Report cluster size balance and cluster-level failure concentration.
  - Diagnose whether gains, if any, come from better specialization or simply from avoiding destructive global merge.

---

## 3. Technology Stack

> Context: Tools, libraries, and infrastructure for the project.

- **Language:** Python 3.11+
- **Core Libraries:**
  - Data: `pandas`, `numpy`, `pyarrow`
  - ML / utilities: `scikit-learn`, `scipy`
  - DL: `torch`, `transformers`, `accelerate`
  - PEFT: `peft`, `bitsandbytes`
  - Retrieval / embeddings: `sentence-transformers`, `faiss` or existing vector DB stack
  - Evaluation: custom project metrics, optional `datasets` utilities, optional LLM-judge client wrappers
  - Visualization: `matplotlib`, optional `plotly`
- **External method integration:**
  - Doc-to-LoRA repository / package integration is allowed as a dependency or submodule.
  - The project does **not** retrain the Doc-to-LoRA hypernetwork.
- **Environment:** `uv` preferred; separate environment allowed for Doc-to-LoRA integration if dependency conflicts occur.
- **Compute:** local machine with RTX 4060 8GB + 32GB RAM.
- **Storage note:** sufficient disk must be reserved for adapters, embeddings, logs, and experiment artifacts.
- **Setup Commands:**
  ```bash
  uv sync                    # Install dependencies
  source .venv/bin/activate  # Linux/macOS
  .venv\Scripts\activate     # Windows
  ```

Recommended practical stack defaults:

- **Backbone default:** one small instruct model feasible under QLoRA on 8GB VRAM.
- **Precision / quantization default:** 4-bit where needed for supervised PEFT.
- **Embedding default:** use the same dense embedding family for retrieval and document clustering unless there is a compelling reason to decouple them.
- **Plotting default:** use `matplotlib` for all final charts.

---

## 4. Code Organization & Conventions

### Project Structure

```text
src/                                      # Reusable typed functions (DRY principle)
├── __init__.py
├── config.py                             # Global configuration / dataclasses
├── data/
│   ├── io.py                             # Load corpus, goldset, manifests
│   ├── preprocessing.py                  # Chunking, normalization, schema validation
│   └── splits.py                         # Split and CV utilities
├── rag/
│   ├── indexing.py                       # Index build / load
│   ├── retrieval.py                      # Retrieval and reranking wrappers
│   └── prompting.py                      # RAG answer prompts
├── qlora/
│   ├── dataset.py                        # RAFT-style supervised dataset builders
│   ├── training.py                       # QLoRA training loop / trainer wrappers
│   ├── inference.py                      # Adapter loading and generation
│   └── prompts.py                        # Supervised training / inference templates
├── doc2lora/
│   ├── packaging.py                      # Corpus segmentation for Doc-to-LoRA
│   ├── internalize.py                    # Adapter generation wrappers
│   ├── merge.py                          # Adapter merge logic
│   └── inference.py                      # Adapter loading and generation
├── routing/
│   ├── clustering.py                     # Document clustering
│   ├── centroids.py                      # Cluster summary artifacts
│   └── router.py                         # Query-to-cluster routing
├── evaluation/
│   ├── deterministic.py                  # Deterministic answer scoring
│   ├── judge.py                          # Free-text scoring / LLM judge wrapper
│   ├── grounding.py                      # Retrieval grounding metrics
│   ├── systems.py                        # TTFT, latency, VRAM, storage metrics
│   └── reports.py                        # Aggregation and result tables
├── visualization.py                      # Plots, heatmaps, summary figures
└── utils.py                              # Shared helpers

main.py                                   # Main orchestration script (cell-like blocks)
config.py                                 # Top-level config entrypoint if needed

eda/
├── src/
│   ├── eda.py                            # EDA analysis
│   └── deep_eda.py                       # Representation / clustering analysis
├── results/
│   ├── figures/
│   └── tables/
└── reports/
    ├── EDA-Report.md
    └── EDA-Insights.md

experiments/
└── EXP-XXX_{description}/
    ├── main_exp.py                       # Experiment pipeline (cell-like blocks)
    ├── config.py                         # Experiment-specific config
    ├── artifacts/                        # Saved adapters, caches, tables
    └── REPORT.md                         # Experiment report

memory_bank/
├── ARCHITECTURE.md                       # This SSOT
├── STATE.md                              # Current state / decisions
└── tasks/
    ├── TASKS.md
    └── {TASK_ID}.md

data/
├── raw/
├── processed/
├── manifests/
│   └── corpus_manifest.csv
└── splits/

logs/
models/
results/
```

### Code Style

- **Cell-like execution:** use `# %% [Block Name]` separators in `main.py` and `main_exp.py` for block-by-block execution.
- **Typed functions:** all reusable functions must have type hints.
- **Config-first design:** experiment behavior must be driven by configs, not hidden constants.
- **Reusability:** code in `src/` must be reusable across experiments and not tightly coupled to a single run.
- **Docstrings:** all public functions must have docstrings.
- **No silent fallback behavior:** any fallback logic for missing adapters, empty retrieval, or routing failure must be logged explicitly.
- **Prompt versioning:** prompts used in S1-S5 must be stored as versioned strings or files, not only inline in notebooks.

### Naming Conventions

- **Scripts:** `main.py`, `main_exp.py`, `config.py`
- **Modules:** lowercase with underscores (`routing.py`, `grounding.py`)
- **Experiments:** `EXP-001_baseline_rag/`, `EXP-005_main_family_comparison/`
- **Adapters:**
  - `adapter_s2_qlora_fold{n}`
  - `adapter_s3_doc2lora_global`
  - `adapter_s4_cluster{c}`
- **Reports:** `REPORT.md`, `MODEL_REPORT.md`, `EVAL_SUMMARY.md`
- **Figures:** `fig_{topic}_{split}.png`
- **Tables:** `tbl_{topic}_{split}.csv`

### Logging

- Training / generation logs must follow the format:
  `[YYYY-MM-DD HH:MM:SS] [LEVEL] - Message`
- Every experiment log must record:
  - random seed(s)
  - git commit hash if available
  - hardware summary
  - backbone identifier
  - adapter identifier
  - retrieval configuration
  - prompt version hash
  - split / fold identifier
- Per-query inference logs should capture, where relevant:
  - question ID
  - selected system ID
  - selected adapter / cluster
  - retrieved doc IDs / pages for retrieval systems
  - latency breakdown
  - raw and normalized answers
- Routing logs for S4 must include:
  - question embedding routing target
  - similarity scores to cluster centroids
  - fallback decision if any
- Final reporting must preserve enough metadata to reproduce any headline number.

---

## Change Control

Any change to the following requires a SSOT update before new headline experiments are run:

- main research question
- system inventory S1-S5
- backbone model
- goldset size / split policy
- Doc-to-LoRA packaging strategy
- routing protocol
- primary evaluation metric definition

Minor implementation changes that do not alter scientific interpretation may remain documented only in experiment reports.
