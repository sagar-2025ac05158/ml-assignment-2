"""
data_prep.py
------------
Shared data loading / train-test split used by every per-model training
script (logistic_regression.py, decision_tree.py, knn.py, naive_bayes.py,
random_forest.py) so that all 5 models are trained and evaluated on the
exact same split of the Electrical Grid Stability dataset.

This file does not train anything itself. Each model script imports
`load_split()` from here.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

RANDOM_STATE = 42

FEATURE_COLS = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4",
]


def load_split():
    """
    Loads raw_source_data.csv, encodes the target, performs an 80/20
    stratified train-test split (random_state=42, identical across all
    model scripts), and returns everything a model script needs:

        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled,
        target_names, feature_names

    Also writes model/scaler.pkl (used by the Streamlit app) so it is
    available regardless of which model script is run.
    """
    raw = pd.read_csv(MODEL_DIR / "raw_source_data.csv")

    X = raw[FEATURE_COLS].copy()

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(raw["stabf"]), name="target")
    target_names = list(le.classes_)  # ['stable', 'unstable'] (alphabetical)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "target_names": target_names,
        "feature_names": FEATURE_COLS,
    }


def save_metrics_row(row: dict):
    """
    Appends/updates this model's row in model/metrics_summary.csv,
    keeping the fixed 6-model order used throughout the README.
    """
    order = [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest (Ensemble)",
    ]
    path = MODEL_DIR / "metrics_summary.csv"

    if path.exists():
        df = pd.read_csv(path)
        df = df[df["ML Model Name"] != row["ML Model Name"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df["_order"] = df["ML Model Name"].apply(
        lambda n: order.index(n) if n in order else len(order)
    )
    df = df.sort_values("_order").drop(columns="_order")
    df.to_csv(path, index=False)


def save_test_data_and_meta(X_test, y_test, feature_names, target_names):
    """Writes test_data.csv and model/meta.json (idempotent, same each run)."""
    import json

    test_export = X_test.copy()
    test_export["target"] = y_test.values
    test_export.to_csv(ROOT / "test_data.csv", index=False)

    meta = {"feature_names": feature_names, "target_names": target_names}
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
