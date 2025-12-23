"""
Data Preprocessing Module
Splitting, feature scaling, label encoding, and statistics tracking (DVC-compatible)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import logging
from typing import Dict
import yaml
import json

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths (DVC-friendly)
# ------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
PARAMS_PATH = Path("params.yaml")


# ------------------------------------------------------------------
# Preprocessor Class
# ------------------------------------------------------------------
class DataPreprocessor:
    """Handle feature scaling and preprocessing"""

    def __init__(self, scaling_method: str = "standard"):
        if scaling_method != "standard":
            raise ValueError("Only 'standard' scaling is supported for now")

        self.scaler = StandardScaler()
        self.feature_names = None
        MODELS_DIR.mkdir(exist_ok=True)
        logger.info(f"DataPreprocessor initialized (scaling={scaling_method})")

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        logger.info(f"Fitting scaler on {X_train.shape[0]} samples")

        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)

        scaler_path = MODELS_DIR / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✓ Scaler saved to {scaler_path}")

        return X_scaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        logger.info(f"Transforming {X.shape[0]} samples")
        return self.scaler.transform(X)

    def get_feature_stats(self, X: pd.DataFrame) -> Dict:
        return {
            "mean": X.mean().to_dict(),
            "std": X.std().to_dict(),
            "min": X.min().to_dict(),
            "max": X.max().to_dict(),
            "median": X.median().to_dict(),
            "shape": X.shape,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_params():
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Main (DVC Stage Entry Point)
# ------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    params = load_params()

    # Params
    raw_path = params["data"]["raw_path"]
    target_col = params["data"]["target_column"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]
    scaling_method = params["preprocessing"]["scaling"]

    # 1️⃣ Read RAW data
    df = pd.read_csv(raw_path)
    logger.info(f"✓ Raw data loaded: {df.shape}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2️⃣ Encode labels (MANDATORY for Iris)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    logger.info("✓ Label encoder saved")

    # 3️⃣ Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    # 4️⃣ Scaling
    preprocessor = DataPreprocessor(scaling_method=scaling_method)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # 5️⃣ Save outputs (DVC owns this directory)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    pd.Series(y_train).to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    pd.Series(y_test).to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
        PROCESSED_DIR / "X_train_scaled.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
        PROCESSED_DIR / "X_test_scaled.csv", index=False
    )

    # 6️⃣ Save stats (optional, drift-ready)
    stats = preprocessor.get_feature_stats(X_train)
    with open(PROCESSED_DIR / "feature_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    logger.info("✓ Preprocessing completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
