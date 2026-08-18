"""
decision_tree.py
-----------------
Trains a Decision Tree classifier on the Electrical Grid Stability
dataset, evaluates it with 6 metrics, and saves the fitted model to
model/decision_tree.pkl.

Trained on raw (unscaled) features since tree splits are scale-invariant.

Run:  python model/decision_tree.py
"""

import pickle

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from data_prep import MODEL_DIR, RANDOM_STATE, load_split, save_metrics_row, save_test_data_and_meta

MODEL_NAME = "Decision Tree"
OUTPUT_FILE = "decision_tree.pkl"

if __name__ == "__main__":
    data = load_split()

    clf = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
    clf.fit(data["X_train"], data["y_train"])

    preds = clf.predict(data["X_test"])
    probs = clf.predict_proba(data["X_test"])[:, 1]

    metrics = {
        "ML Model Name": MODEL_NAME,
        "Accuracy": round(accuracy_score(data["y_test"], preds), 4),
        "AUC": round(roc_auc_score(data["y_test"], probs), 4),
        "Precision": round(precision_score(data["y_test"], preds), 4),
        "Recall": round(recall_score(data["y_test"], preds), 4),
        "F1": round(f1_score(data["y_test"], preds), 4),
        "MCC": round(matthews_corrcoef(data["y_test"], preds), 4),
    }
    print(metrics)

    with open(MODEL_DIR / OUTPUT_FILE, "wb") as f:
        pickle.dump(clf, f)

    save_metrics_row(metrics)
    save_test_data_and_meta(
        data["X_test"], data["y_test"], data["feature_names"], data["target_names"]
    )
    print(f"Saved model to model/{OUTPUT_FILE}")
