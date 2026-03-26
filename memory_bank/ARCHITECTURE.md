# Project Architecture: Term-Paper

## 1. Problem Statement & Success Criteria

> Context: Define the problem clearly, what success looks like.

### Problem Definition


### Success Criteria

| Metric | Baseline | Target | Rationale |
|--------|----------|--------|-----------|
| [Primary metric, e.g., F1-Score] | [Current/naive baseline] | [Target value] | [Why this target?] |
| [Secondary metric, e.g., AUC-ROC] | [Baseline] | [Target] | [Rationale] |

### Constraints (if applicable)

- **Latency:** [e.g., Inference must be < 100ms]
- **Interpretability:** [e.g., Model must be explainable for regulatory reasons]
- **Resources:** [e.g., Training must fit in 16GB GPU memory]

---

## 2. Experiment Pipeline

> Context: The iterative workflow for running experiments. Unlike product development, this is a cycle, not a linear flow.

```
[Data Collection] -> [EDA] -> [Deep Feature Engineering] -> [Baseline] -> [Hypothesis] -> [Experiment] -> [Evaluate]
                                                                       ^                                  |
                                                                       |__________________________________|
                                                                              (iterate until target met)
```

### Pipeline Stages

1. **Data Preparation**
   - Load from `data/raw/`
   - Clean and preprocess -> save to `data/processed/`
   - Split: train/validation/test (with stratification if needed)

2. **Exploratory Data Analysis (EDA)**
   - Distribution analysis
   - Correlation analysis
   - Leakage and quality checks
   - Document data architecture in `eda/reports/EDA-Report.md`

3. **Deep Feature Engineering**
   - Analyze candidate features after EDA findings are stable
   - Prioritize high-signal candidates by expected metric impact and runtime cost
   - Document feature strategy in `eda/reports/EDA-Insights.md`

4. **Baseline Establishment**
   - Simple model (e.g., LogisticRegression, RandomForest with defaults)
   - Naive baseline (e.g., predict majority class)
   - Save benchmark artifacts and logs for comparison

5. **Experimentation Cycle**
   - Formulate hypothesis/task in `memory_bank/TASKS.md`
   - Keep working notes in `memory_bank/tasks/{TASK_ID}.md`
   - Implement experiment in `experiments/EXP-XXX/`
   - Train and evaluate using `main_exp.py`
   - Log results in experiment report

6. **Final Evaluation**
   - Test set evaluation (only when confident)
   - Error analysis
   - Model documentation in `models/model_<metric>_<value>/MODEL_REPORT.md`
   - Model architecture and validation strategy are documented in model reports, not here

---

## 3. Technology Stack

> Context: Tools, libraries, and infrastructure for the project.

- **Language:** [e.g., Python 3.10+]
- **Core Libraries:** 
  - Data: [e.g., pandas, numpy, polars]
  - ML: [e.g., scikit-learn, XGBoost, LightGBM]
  - DL: [e.g., PyTorch, TensorFlow] (if applicable)
  - Visualization: [e.g., matplotlib, seaborn, plotly]
- **Environment:** [e.g., uv, Docker]
- **Compute:** [e.g., Local, Cloud GPU, Colab]
- **Setup Commands:**
  ```bash
  uv sync                    # Install dependencies
  source .venv/bin/activate  # Linux/macOS
  .venv\Scripts\activate     # Windows
  ```
---

## 4. Code Organization & Conventions

### Project Structure

```
src/                          # Reusable typed functions (DRY principle)
├── __init__.py
├── data.py                   # Data loading, cleaning, transformations
├── features.py               # Feature engineering functions
├── eda.py                    # EDA analysis and visualizations
├── models.py                 # Model initialization, training, inference
├── evaluation.py             # Metrics, validation, error analysis
└── visualization.py          # Plotting functions

main.py                       # Main pipeline (cell-like blocks with # %% separators)
config.py                     # Global configuration and hyperparameters

eda/
├── src/
│   ├── eda.py                # EDA analysis and visualizations
│   └── deep_eda.py           # Deep feature engineering analysis
├── results/
│   ├── figures/
│   └── tables/
└── reports/
    ├── EDA-Report.md
    └── EDA-Insights.md

experiments/                  # Isolated experiments
└── EXP-XXX_{description}/
    ├── main_exp.py           # Experiment pipeline (cell-like blocks)
    ├── config.py             # Experiment-specific config
    └── REPORT.md             # Experiment report

memory_bank/
├── ARCHITECTURE.md
├── STATE.md
└── tasks/
    ├── TASKS.md
    └── {TASK_ID}.md
```

### Code Style

- **Cell-like execution**: Use `# %% [Block Name]` separators in `main.py` and `main_exp.py` for block-by-block execution
- **Typed functions**: All functions should have type hints
- **Reusability**: Functions in `src/` should be reusable across experiments (DRY)
- **Docstrings**: All public functions must have docstrings

### Naming Conventions

- **Scripts**: `main.py`, `main_exp.py`, `config.py`
- **Modules**: lowercase with underscores (`data.py`, `feature_engineering.py`)
- **Models**: `model_{experiment_id}_{metric}_{value}.pkl`
- **Experiments**: `EXP-{number}_{description}/`

### Logging

- Training logs in `logs/`, format `[YYYY-MM-DD HH:MM:SS] [LEVEL] - Message`
- Random seeds: Always set and document for reproducibility
