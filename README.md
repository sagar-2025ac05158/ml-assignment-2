# ML Assignment 2 — Electrical Grid Stability Classification

## a. Problem Statement

Modern power grids are moving from centralized generation toward
decentralized, demand-responsive control (Decentral Smart Grid Control,
DSGC), where consumers dynamically adjust consumption in response to price
signals instead of relying on constant grid frequency. This introduces a new
risk: unpredictable reaction times and price elasticity across nodes can push
the grid into an unstable operating state. This project builds and compares
multiple supervised classification models that predict whether a simulated
4-node electrical grid is **stable** or **unstable** given each node's
reaction time, power, and price-elasticity characteristics, and exposes the
trained models through an interactive Streamlit web application.

This is a **binary classification** problem (stable vs. unstable).

## b. Dataset Description

- **Dataset:** Electrical Grid Stability Simulated Data
- **Source:** UCI Machine Learning Repository (Arzamasov, V., 2018,
  https://doi.org/10.24432/C5PG66), donated by Karlsruhe Institute of
  Technology. Simulates the local stability of a 4-node star electrical grid
  (1 central producer, 3 consumers) implementing Decentral Smart Grid
  Control, based on the methodology in Schäfer et al. (2016), *"Taming
  instabilities in power grid networks by decentralized control"*.
- **Instances:** 10,000 — well above the minimum of 500.
- **Features:** 12 numeric predictive features — above the minimum of 12.
  - `tau1`–`tau4`: reaction time of each grid participant (seconds)
  - `p1`–`p4`: nominal power produced (`p1`, always positive — the central
    producer) or consumed (`p2`–`p4`, negative — the 3 consumers)
  - `g1`–`g4`: coefficient proportional to price elasticity for each
    participant
- **Target:** `stabf` — binary label, `stable` or `unstable`
  (label-encoded as `stable = 0`, `unstable = 1` in the app/models)
- **Dropped column:** `stab` (the raw real part of the characteristic
  equation root) was excluded from the features — it deterministically
  determines `stabf` and would leak the answer.
- **Class balance:** 3,620 stable, 6,380 unstable (no missing values).
- **Train/Test split:** 80% train (8,000 rows) / 20% test (2,000 rows),
  stratified on the target. The held-out test split is exported as
  `test_data.csv` and is the file used for evaluation in the Streamlit app.

## c. GitHub Repository Link

`<PASTE YOUR GITHUB REPO LINK HERE AFTER PUSHING>`

## d. Models Used

All 5 models below were trained on the **same** dataset and the **same**
train/test split for a fair comparison. Logistic Regression, kNN, and Naive
Bayes were trained on standardized (scaled) features; Decision Tree and
Random Forest were trained on raw features (tree-based splits are
scale-invariant).

### Comparison Table (evaluation metrics on the 20% held-out test split)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8200 | 0.8920 | 0.8408 | 0.8856 | 0.8626 | 0.6039 |
| Decision Tree | 0.8575 | 0.8912 | 0.8874 | 0.8895 | 0.8885 | 0.6912 |
| kNN | 0.8735 | 0.9455 | 0.8569 | 0.9624 | 0.9066 | 0.7243 |
| Naive Bayes | 0.8445 | 0.9210 | 0.8395 | 0.9350 | 0.8847 | 0.6570 |
| Random Forest (Ensemble) | 0.9190 | 0.9810 | 0.9114 | 0.9671 | 0.9384 | 0.8235 |

*(Exact figures are also written programmatically to
`model/metrics_summary.csv` by `model/train_models.py`, and are re-derived
live in the Streamlit app whenever `test_data.csv` is uploaded.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Weakest of the five (Accuracy 0.8200, MCC 0.6039). The relationship between the 12 grid parameters (reaction times, power, elasticity) and stability is governed by a nonlinear characteristic-equation condition, so a purely linear decision boundary struggles to separate the classes cleanly. |
| Decision Tree | A modest improvement over Logistic Regression (Accuracy 0.8575) by capturing some nonlinear feature interactions (e.g. thresholds on `tau` combined with `g`), but a single tree still trails the ensemble and distance-based methods, and its AUC (0.8912) barely beats Logistic Regression's. |
| kNN | Strong performer (Accuracy 0.8735, AUC 0.9455, Recall 0.9624). Because instability is driven by how a node's reaction time and elasticity compare to its neighbors', local-neighborhood-based reasoning fits this problem well once features are standardized. |
| Naive Bayes | Middling accuracy (0.8445) but a respectable AUC (0.9210) — its independence assumption across the 12 correlated grid features is unrealistic, but it still ranks the classes reasonably well even if hard classification suffers. |
| Random Forest (Ensemble) | Clear best performer across every metric (Accuracy 0.9190, AUC 0.9810, MCC 0.8235). Averaging many trees captures the nonlinear, interaction-heavy relationship between node parameters and grid stability far better than any single model, while controlling the overfitting a single Decision Tree is prone to. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — it dominates on all 6 metrics, confirming that grid stability in this DSGC simulation is governed by nonlinear interactions between nodes that ensemble tree methods capture much more effectively than linear, single-tree, or distance/probability-based models. |

## Repository Structure

```
project-folder/
│-- app.py                     # Streamlit app (entry point)
│-- requirements.txt
│-- README.md
│-- test_data.csv              # held-out test split used for evaluation
│-- model/
│   │-- data_prep.py           # shared data load / train-test split (used by all 5 scripts below)
│   │-- logistic_regression.py # trains + saves Logistic Regression
│   │-- decision_tree.py       # trains + saves Decision Tree
│   │-- knn.py                 # trains + saves kNN
│   │-- naive_bayes.py         # trains + saves Naive Bayes
│   │-- random_forest.py       # trains + saves Random Forest (Ensemble)
│   │-- run_all.py             # optional: runs all 5 scripts above in one go
│   │-- raw_source_data.csv    # original UCI dataset (full 10,000 rows)
│   │-- scaler.pkl
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- metrics_summary.csv
│   │-- meta.json
```

## How to Run Locally

```bash
pip install -r requirements.txt

# Regenerate all models + test_data.csv (optional, already included in repo):
python model/run_all.py
# ...or train/re-run a single model, e.g.:
python model/random_forest.py

streamlit run app.py
```

Each of the 5 model scripts (`logistic_regression.py`, `decision_tree.py`,
`knn.py`, `naive_bayes.py`, `random_forest.py`) is self-contained: it loads
the raw dataset via the shared `data_prep.py` helper (which guarantees the
exact same 80/20 stratified split, `random_state=42`, across all models for
a fair comparison), trains its one model, evaluates it with all 6 metrics,
and saves the fitted model to `model/<name>.pkl`. Metrics from each run are
written into the shared `model/metrics_summary.csv`.

## Live App

`<PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE AFTER DEPLOYMENT>`

## Streamlit App Features

- **Dataset upload:** Upload `test_data.csv` (or any CSV with the same 12
  feature columns + a `target` column) via the file uploader.
- **Model selection dropdown:** Choose between Logistic Regression, Decision
  Tree, kNN, Naive Bayes, and Random Forest (Ensemble).
- **Evaluation metrics display:** Accuracy, AUC, Precision, Recall, F1, and
  MCC are computed live on the uploaded data and shown as metric cards.
- **Confusion matrix & classification report:** A heatmap confusion matrix
  and a full per-class classification report are rendered for the selected
  model's predictions on the uploaded data.
