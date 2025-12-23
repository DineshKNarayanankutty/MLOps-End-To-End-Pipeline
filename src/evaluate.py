import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import yaml


# -----------------------------
# Paths (DVC-friendly)
# -----------------------------
PROCESSED_DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("metrics/metrics.json")
PARAMS_PATH = Path("params.yaml")


# -----------------------------
# Load params
# -----------------------------
def load_params():
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)
    return params["evaluation"]


# -----------------------------
# Load data
# -----------------------------
def load_data():
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()
    return X_test, y_test


# -----------------------------
# Load model
# -----------------------------
def load_model():
    return joblib.load(MODEL_PATH)


# -----------------------------
# Evaluate model
# -----------------------------
def evaluate(model, X_test, y_test, threshold):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


# -----------------------------
# Save metrics
# -----------------------------
def save_metrics(metrics):
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)


# -----------------------------
# Main
# -----------------------------
def main():
    params = load_params()
    threshold = params.get("threshold", 0.5)

    X_test, y_test = load_data()
    model = load_model()

    metrics = evaluate(model, X_test, y_test, threshold)
    save_metrics(metrics)

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
