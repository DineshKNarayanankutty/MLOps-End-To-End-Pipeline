"""
MLOps Pipeline - Core ML Modules

This package contains the complete machine learning pipeline including
data loading, preprocessing, model training, inference, and monitoring.

Modules:
    - data_loader: Load and validate datasets
    - preprocessing: Feature scaling and transformation
    - training: Model training with MLflow tracking
    - drift_detector: Data and model drift detection
    - model_registry: MLflow model versioning and promotion
    - metrics_exporter: Prometheus metrics export
"""

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.training import MLOpsTrainer
from src.drift_detector import DriftDetector
from src.model_registry import ModelRegistry

__version__ = "1.0.0"
__author__ = "MLOps Engineer-DKN"
__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "MLOpsTrainer",
    "DriftDetector",
    "ModelRegistry",
]