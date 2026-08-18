"""
Streamlit demo app — Electrical Grid Stability classification.

Features:
  a. CSV upload of test data
  b. Model selection dropdown
  c. Evaluation metrics display
  d. Confusion matrix + classification report
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Electrical Grid Stability Classifier", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": ("logistic_regression.pkl", True),
    "Decision Tree": ("decision_tree.pkl", False),
    "kNN": ("knn.pkl", True),
    "Naive Bayes": ("naive_bayes.pkl", True),
    "Random Forest (Ensemble)": ("random_forest_ensemble.pkl", False),
}


@st.cache_resource
def load_meta():
    with open(MODEL_DIR / "meta.json") as f:
        return json.load(f)


@st.cache_resource
def load_scaler():
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model(filename):
    with open(MODEL_DIR / filename, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metrics_summary():
    return pd.read_csv(MODEL_DIR / "metrics_summary.csv")


meta = load_meta()
scaler = load_scaler()
feature_names = meta["feature_names"]
target_names = meta["target_names"]  # ['stable', 'unstable'] -> encoded 0, 1

st.title("⚡ Electrical Grid Stability — Model Demo")
st.write(
    "Predicts whether a simulated 4-node smart electrical grid (Decentral "
    "Smart Grid Control) is **stable** or **unstable** from reaction-time, "
    "power, and price-elasticity features of each node. Upload the test "
    "dataset, pick a model, and view its evaluation metrics, confusion "
    "matrix, and classification report."
)

# --- Sidebar: model selection ---------------------------------------
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))
model_file, needs_scaling = MODEL_FILES[model_name]
model = load_model(model_file)

st.sidebar.markdown("---")
st.sidebar.subheader("Training-time metrics (held-out split)")
summary_df = load_metrics_summary()
row = summary_df[summary_df["ML Model Name"] == model_name]
st.sidebar.dataframe(row.set_index("ML Model Name").T, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Label encoding: `{target_names[0]}` = 0, `{target_names[1]}` = 1")

# --- Main: file upload ------------------------------------------------
uploaded_file = st.file_uploader("Upload test CSV (must include a 'target' column)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column with the true labels (0/1).")
        st.stop()

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        st.stop()

    X = df[feature_names]
    y_true = df["target"]

    X_input = scaler.transform(X) if needs_scaling else X

    preds = model.predict(X_input)
    probs = model.predict_proba(X_input)[:, 1]

    st.subheader(f"Results — {model_name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_true, preds):.4f}")
    col1.metric("AUC", f"{roc_auc_score(y_true, probs):.4f}")
    col2.metric("Precision", f"{precision_score(y_true, preds):.4f}")
    col2.metric("Recall", f"{recall_score(y_true, preds):.4f}")
    col3.metric("F1 Score", f"{f1_score(y_true, preds):.4f}")
    col3.metric("MCC", f"{matthews_corrcoef(y_true, preds):.4f}")

    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("### Classification Report")
    report = classification_report(
        y_true, preds, target_names=target_names, output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    with st.expander("Preview uploaded data"):
        st.dataframe(df.head(20), use_container_width=True)
else:
    st.info("Upload `test_data.csv` (provided in the repo) to see predictions and metrics.")

st.markdown("---")
st.subheader("All-Model Comparison (from training-time evaluation)")
st.dataframe(summary_df.set_index("ML Model Name"), use_container_width=True)
