"""
run_all.py
----------
Convenience script that runs all 5 individual model-training scripts back
to back (not required by the assignment — each model script also runs
fine on its own: `python model/logistic_regression.py`, etc.).

Run:  python model/run_all.py
"""

import runpy
from pathlib import Path

SCRIPTS = [
    "logistic_regression.py",
    "decision_tree.py",
    "knn.py",
    "naive_bayes.py",
    "random_forest.py",
]

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    for script in SCRIPTS:
        print(f"\n{'=' * 60}\nRunning {script}\n{'=' * 60}")
        runpy.run_path(str(HERE / script), run_name="__main__")
    print("\nAll 5 models trained. See model/metrics_summary.csv for the combined results.")
