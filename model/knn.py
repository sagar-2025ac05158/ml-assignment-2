"""
knn.py
------
Trains a K-Nearest Neighbors classifier on the Electrical Grid Stability
dataset, evaluates it with 6 metrics, and saves the fitted model to
model/knn.pkl.

Trained on standardized (scaled) features since kNN is distance-based.

Run:  python model/knn.py
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
from sklearn.neighbors import KNeighborsClassifier

from data_prep import MODEL_DIR, load_split, save_metrics_row, save_test_data_and_meta

MODEL_NAME = "kNN"
OUTPUT_FILE = "knn.pkl"

if __name__ == "__main__":
    data = load_split()

    clf = KNeighborsClassifier(n_neighbors=9)
    clf.fit(data["X_train_scaled"], data["y_train"])

    preds = clf.predict(data["X_test_scaled"])
    probs = clf.predict_proba(data["X_test_scaled"])[:, 1]

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
