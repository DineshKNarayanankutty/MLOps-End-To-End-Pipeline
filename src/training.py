"""
Model Training Module with MLflow Integration
Trains RandomForest model with full experiment tracking
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import logging
import json
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MLOpsTrainer:
    """Train ML models with MLflow tracking"""

    def __init__(self, experiment_name: str = "iris-classification"):
        """
        Initialize trainer with MLflow

        Args:
            experiment_name (str): Name for MLflow experiment
        """
        # Ensure tracking URI is set before creating experiments/runs
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

        self.experiment_name = experiment_name
        self.model = None
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_experiment(self.experiment_name)
        logger.info(f"✓ MLflow experiment: {self.experiment_name}")
        logger.info(f"✓ Tracking URI: {mlflow.get_tracking_uri()}")

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              params: Dict = None) -> RandomForestClassifier:
        """
        Train model with MLflow tracking

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            params (dict): Model hyperparameters

        Returns:
            RandomForestClassifier: Trained model
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }

        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 60)
        logger.info(f"Hyperparameters: {params}")
        logger.info(f"Training data shape: {X_train.shape}")

        # Start a run and log everything within the context
        with mlflow.start_run(run_name="iris-training-run") as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run {run_id}")

            # Log params
            logger.info("[STEP 1] Logging hyperparameters...")
            mlflow.log_params(params)

            # Train
            logger.info("[STEP 2] Training RandomForestClassifier...")
            self.model = RandomForestClassifier(**params)
            self.model.fit(X_train, y_train)
            logger.info("✓ Training complete")

            # Evaluate on training set
            logger.info("[STEP 3] Evaluating on training data...")
            y_pred_train = self.model.predict(X_train)
            train_metrics = {
                'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
                'train_precision': float(precision_score(y_train, y_pred_train, average='weighted')),
                'train_recall': float(recall_score(y_train, y_pred_train, average='weighted')),
                'train_f1': float(f1_score(y_train, y_pred_train, average='weighted'))
            }

            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"✓ {metric_name}: {metric_value:.4f}")

            # Save model file locally and log as artifact
            logger.info("[STEP 4] Saving model...")
            model_path = self.models_dir / "model.pkl"
            joblib.dump(self.model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="models")
            logger.info(
                f"✓ Model saved to {model_path} and logged as artifact")

            # Also log the model using mlflow.sklearn so it appears in MLflow Models
            try:
                mlflow.sklearn.log_model(
                    self.model, artifact_path="sklearn-model")
                logger.info(
                    "✓ Model logged to MLflow Model Registry (artifact path 'sklearn-model')")
            except Exception as e:
                logger.warning(f"Could not mlflow.sklearn.log_model(): {e}")

            # Tags
            mlflow.set_tag("model_type", "RandomForestClassifier")
            mlflow.set_tag("dataset", "iris")
            mlflow.set_tag("status", "training")
            mlflow.set_tag("run_host", os.getenv("HOSTNAME", "local"))

        logger.info("\n" + "=" * 60)
        logger.info("✓ TRAINING COMPLETE")
        logger.info("=" * 60)

        return self.model

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels

        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        logger.info("=" * 60)
        logger.info("EVALUATING MODEL")
        logger.info("=" * 60)

        y_pred = self.model.predict(X_test)

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1': float(f1_score(y_test, y_pred, average='weighted'))
        }

        # Log evaluation metrics to MLflow under a new child run (or same experiment)
        with mlflow.start_run(nested=True):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
                logger.info(f"✓ test_{metric_name}: {metric_value:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")

        logger.info("\n" + "=" * 60)
        logger.info("✓ EVALUATION COMPLETE")
        logger.info("=" * 60)

        return metrics

    def save_metadata(self, metrics: Dict) -> Dict:
        """
        Save model metadata

        Args:
            metrics (dict): Model performance metrics

        Returns:
            dict: Saved metadata
        """
        logger.info("\n[STEP 5] Saving model metadata...")

        metadata = {
            'model_name': 'RandomForestClassifier',
            'framework': 'scikit-learn',
            'version': '1.0.0',
            'dataset': 'iris',
            'metrics': metrics,
            'feature_count': 4,
            'feature_names': [
                'sepal length (cm)',
                'sepal width (cm)',
                'petal length (cm)',
                'petal width (cm)'
            ],
            'classes': [0, 1, 2],
            'class_names': ['setosa', 'versicolor', 'virginica'],
            'created_at': datetime.now().isoformat()
        }

        metadata_path = self.models_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Metadata saved to {metadata_path}")
        mlflow.log_artifact(str(metadata_path), artifact_path="metadata")

        return metadata


def main():
    logger.info("\n" + "=" * 60)
    logger.info("MLOPS TRAINING PIPELINE")
    logger.info("=" * 60)

    # load data
    X_train = pd.read_csv("data/processed/X_train.csv").values
    X_test = pd.read_csv("data/processed/X_test.csv").values
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    logger.info(f"✓ Loaded training data: {X_train.shape}")
    logger.info(f"✓ Loaded test data: {X_test.shape}")

    # instantiate trainer (this sets tracking URI from env var if present)
    trainer = MLOpsTrainer(experiment_name="iris-classification")

    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }

    trainer.train(X_train, y_train, params)
    metrics = trainer.evaluate(X_test, y_test)
    metadata = trainer.save_metadata(metrics)

    logger.info("\n" + "=" * 60)
    logger.info("✓ TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nFinal Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    return trainer, metrics


if __name__ == "__main__":
    trainer, metrics = main()
