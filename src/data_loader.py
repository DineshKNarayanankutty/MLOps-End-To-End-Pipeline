"""
Data Loading Module
Handles dataset loading, train/test splitting, and data validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import logging
from typing import Tuple, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load, validate, and split data for ML pipeline"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataLoader
        
        Args:
            test_size (float): Proportion of test data
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        logger.info(f"DataLoader initialized | test_size={test_size}")
        
    def load_iris_dataset(self) -> pd.DataFrame:
        """
        Load Iris dataset and save locally
        
        Returns:
            pd.DataFrame: Dataset with features + target + species
        """
        logger.info("Loading Iris dataset...")
        
        iris = load_iris()
        df = pd.DataFrame(
            data=iris.data,
            columns=iris.feature_names
        )
        
        df['target'] = iris.target
        df['species'] = df['target'].map({
            0: 'setosa',
            1: 'versicolor',
            2: 'virginica'
        })
        
        # Save raw data
        raw_path = self.data_dir / "raw" / "iris_raw.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        
        logger.info(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]-2} features")
        logger.info(f"✓ Saved to {raw_path}")
        
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple:
        """
        Split data into train and test sets
        
        Args:
            df (pd.DataFrame): Input dataset
            target_col (str): Name of target column
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting dataset with test_size={self.test_size}...")
        
        feature_cols = [col for col in df.columns if col not in ['target', 'species']]
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        X_train.to_csv(processed_dir / "X_train.csv", index=False)
        X_test.to_csv(processed_dir / "X_test.csv", index=False)
        pd.Series(y_train).to_csv(processed_dir / "y_train.csv", index=False)
        pd.Series(y_test).to_csv(processed_dir / "y_test.csv", index=False)
        
        logger.info(f"✓ Data splits saved to {processed_dir}")
        
        return X_train, X_test, y_train, y_test
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Validate data quality
        
        Returns:
            dict: Quality report
        """
        logger.info("Validating data quality...")
        
        missing = df.isnull().sum()
        duplicates = df.duplicated().sum()
        
        if missing.any():
            logger.warning(f"Missing values: {missing.sum()}")
        else:
            logger.info("✓ No missing values")
        
        logger.info(f"✓ Duplicate rows: {duplicates}")
        
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': missing.to_dict(),
            'duplicate_rows': int(duplicates),
            'shape': df.shape,
            'columns': df.columns.tolist()
        }
        
        return report


def main():
    logger.info("=" * 60)
    logger.info("DATA LOADING PIPELINE")
    logger.info("=" * 60)
    
    loader = DataLoader(test_size=0.2, random_state=42)
    df = loader.load_iris_dataset()
    quality_report = loader.validate_data_quality(df)
    X_train, X_test, y_train, y_test = loader.split_data(df)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ DATA LOADING COMPLETE")
    logger.info("=" * 60)
    
    return df, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()