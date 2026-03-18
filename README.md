# Titanic Agent Tooling Benchmark

**MSIN0097 Predictive Analytics — Group Coursework 2025–26**
University College London

## Overview

This repository contains the code, data, and evidence for a benchmarking study comparing three AI agent tools on realistic data science tasks using the Titanic survival dataset.

**Tools evaluated:** GitHub Copilot (Chat panel, VS Code/Jupyter) · Claude Code (terminal-based agent) · OpenAI Codex (cloud-based agentic framework)

**Tasks (5):**

| # | Task | Type |
|---|------|------|
| T1 | Dataset ingestion and cleaning | Data preparation |
| T2 | Baseline model training + evaluation | ML pipeline |
| T3 | Detecting and fixing data leakage | Bug detection |
| T4 | Debugging a broken pipeline | Bug detection |
| T5 | EDA and insight generation with plots | Exploratory analysis |

**Scoring:** 6 dimensions (Correctness, Statistical validity, Reproducibility, Code quality, Efficiency, Safety) × 0–2 scale = max 12 per task, 60 overall.

## Results

| Tool | T1 | T2 | T3 | T4 | T5 | Total |
|------|-----|-----|-----|-----|-----|-------|
| Copilot | 4 | 6 | 5 | 9 | 5 | **29/60** |
| Claude Code | 12 | 12 | 6 | 12 | 12 | **54/60** |
| Codex | 12 | 12 | 12 | 11 | 12 | **59/60** |

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── Titanic-Dataset.csv              # Raw Titanic dataset (Kaggle)
│   ├── titanic_pipeline_v1.py           # Broken script: data leakage bug (Task 3)
│   └── titanic_pipeline_v2.py           # Broken script: two planted bugs (Task 4)
├── runs/
│   ├── Claude Code/
│   │   ├── preprocess.py                # T1: ingestion and cleaning
│   │   ├── logistic_regression.py       # T2: baseline model (Pipeline)
│   │   ├── titanic_pipeline_V1.py       # T3: annotated leaky script
│   │   ├── titanic_fixed_debug.py       # T4: corrected debug script
│   │   ├── eda_titanic.py               # T5: EDA script
│   │   ├── eda_*.png                    # T5: saved plots
│   │   ├── titanic_clean.csv            # T1 output
│   │   └── Documentation .docx          # Run log and observations
│   ├── Codex/
│   │   ├── titanic_clean.py             # T1: ingestion (type hints, main())
│   │   ├── titanic_logreg.py            # T2: baseline model (dynamic features)
│   │   ├── titanic_fixed_leakage.py     # T3: full Pipeline restructure
│   │   ├── titanic_fixed_debug.py       # T4: corrected debug script
│   │   ├── titanic_eda.py               # T5: EDA script (modular functions)
│   │   ├── eda_*.png                    # T5: saved plots
│   │   ├── codex.ipynb                  # Codex session notebook
│   │   ├── titanic_clean.csv            # T1 output
│   │   └── codex screen/               # Screenshots of Codex IDE
│   └── Co-pilot/
│       ├── copilot.ipynb                # Full Copilot session notebook
│       ├── titanic_fixed_leakage.py     # T3: leakage fix
│       ├── titanic_fixed_debug.py       # T4: debug fix
│       ├── eda_*.png                    # T5: saved plots (3/4, missing heatmap)
│       ├── titanic_clean.csv            # T1 output (contains silent corruption)
│       └── task 1–5/                    # Screenshots per task
└── titanic_benchmark_combined.ipynb     # Combined notebook with all runs
```

## Reproducing Results

```bash
# Clone the repository
git clone https://github.com/<your-username>/titanic-agent-benchmark.git
cd titanic-agent-benchmark

# Install dependencies
pip install -r requirements.txt

# Run any individual script (example: Claude Code Task 1)
cd runs/Claude\ Code
python preprocess.py

# Or open the combined notebook
jupyter notebook titanic_benchmark_combined.ipynb
```

All scripts use `RANDOM_STATE = 42`. Expected baseline metrics with LogisticRegression on an 80/20 stratified split: Accuracy ~0.80, F1 ~0.80 (weighted).

## Key Findings

1. **Execution capability is the defining divide.** Copilot cannot run code or self-verify, scoring 29/60 vs 54 and 59 for execution-capable tools.
2. **Cascading failures** from Copilot's T1 silent `inplace` bug propagated into T2 and T5.
3. **Bug identification is reliable across all tools**; producing verified, saved fixes requires execution capability.
4. **Code quality scales with tool sophistication** — Codex produced the most structured code (type hints, modular functions, dynamic feature detection).

## Dataset

Titanic survival dataset from [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset). 891 passengers, 12 original features, binary classification target (Survived).

## Broken Scripts

Two deliberately broken scripts are provided in `data/` for Tasks 3 and 4:

- `titanic_pipeline_v1.py` — StandardScaler fitted on full dataset before train/test split (data leakage)
- `titanic_pipeline_v2.py` — `shuffle=False` in split + `model.predict(X_train)` compared against `y_test`

These were given to all three tools identically with no hints beyond the standardised prompt.
