import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import yaml

# -----------------------------
# Paths
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
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()
    return X_test, y_test


# -----------------------------
# Load model
# -----------------------------
def load_model():
    return joblib.load(MODEL_PATH)


# -----------------------------
# Evaluate model (MULTICLASS)
# -----------------------------
def evaluate(model, X_test, y_test, average):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average),
        "recall": recall_score(y_test, y_pred, average=average),
        "f1_score": f1_score(y_test, y_pred, average=average),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
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
    average = params.get("average", "macro")

    X_test, y_test = load_data()
    model = load_model()

    metrics = evaluate(model, X_test, y_test, average)
    save_metrics(metrics)

    print("Evaluation metrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()
