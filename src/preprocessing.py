"""
Data Preprocessing Module
Feature scaling, transformation, and statistics tracking
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging
from typing import Dict, Tuple
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataPreprocessor:
    """Handle feature scaling and preprocessing"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scaler = StandardScaler()
        self.feature_names = None
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        logger.info("DataPreprocessor initialized")
        
    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler on training data and transform
        
        Args:
            X_train (pd.DataFrame): Training features
            
        Returns:
            np.ndarray: Scaled training features
        """
        logger.info(f"Fitting scaler on {X_train.shape[0]} samples...")
        
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        
        logger.info(f"✓ Scaler fitted")
        logger.info(f"  Mean: {self.scaler.mean_}")
        logger.info(f"  Scale: {self.scaler.scale_}")
        
        # Save scaler
        scaler_path = self.models_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✓ Scaler saved to {scaler_path}")
        
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted scaler
        
        Args:
            X (pd.DataFrame): Features to transform
            
        Returns:
            np.ndarray: Scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform() first.")
        
        logger.info(f"Transforming {X.shape[0]} samples...")
        return self.scaler.transform(X)
    
    def load_scaler(self, scaler_path: str = "models/scaler.pkl") -> StandardScaler:
        """Load pre-fitted scaler"""
        logger.info(f"Loading scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)
        return self.scaler
    
    def get_feature_stats(self, X: pd.DataFrame) -> Dict:
        """
        Calculate feature statistics for drift detection
        
        Returns:
            dict: Statistics for each feature
        """
        logger.info(f"Calculating statistics for {X.shape[0]} samples...")
        
        stats = {
            'mean': X.mean().to_dict(),
            'std': X.std().to_dict(),
            'min': X.min().to_dict(),
            'max': X.max().to_dict(),
            'median': X.median().to_dict(),
            'shape': X.shape
        }
        
        logger.info(f"✓ Statistics calculated for {len(stats['mean'])} features")
        
        return stats
    
    def get_scaled_stats(self, X_scaled: np.ndarray) -> Dict:
        """Get statistics of scaled data"""
        logger.info("Calculating scaled data statistics...")
        
        stats = {
            'mean': X_scaled.mean(axis=0).tolist(),
            'std': X_scaled.std(axis=0).tolist(),
            'min': X_scaled.min(axis=0).tolist(),
            'max': X_scaled.max(axis=0).tolist()
        }
        
        logger.info(f"✓ Scaled data statistics:")
        logger.info(f"  Mean (should be ~0): {[round(m, 4) for m in stats['mean']]}")
        logger.info(f"  Std (should be ~1): {[round(s, 4) for s in stats['std']]}")
        
        return stats


def main():
    logger.info("=" * 60)
    logger.info("DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    logger.info(f"✓ Loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    
    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    logger.info(f"✓ X_test_scaled shape: {X_test_scaled.shape}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    
    return X_train_scaled, X_test_scaled, preprocessor


if __name__ == "__main__":
    main()