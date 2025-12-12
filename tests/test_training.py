"""
Unit Tests for Model Training Module

Tests cover:
    - Model training functionality
    - Model evaluation
    - MLflow integration
    - Metadata persistence
"""

import pytest
import numpy as np
from src.training import MLOpsTrainer
from sklearn.ensemble import RandomForestClassifier


class TestTrainerInitialization:
    """Test MLOpsTrainer initialization"""
    
    def test_init_default_experiment(self):
        """Test initialization with default experiment"""
        trainer = MLOpsTrainer()
        assert trainer.experiment_name == "iris-classification"
        assert trainer.model is None
    
    def test_init_custom_experiment(self):
        """Test initialization with custom experiment"""
        trainer = MLOpsTrainer(experiment_name="custom-exp")
        assert trainer.experiment_name == "custom-exp"


class TestModelTraining:
    """Test model training"""
    
    def test_train_model(self, X_y_train_test):
        """Test successful model training"""
        X_train, _, y_train, _ = X_y_train_test
        
        trainer = MLOpsTrainer()
        model = trainer.train(X_train, y_train)
        
        assert model is not None
        assert isinstance(model, RandomForestClassifier)
        assert trainer.model is not None
    
    def test_train_with_custom_params(self, X_y_train_test):
        """Test training with custom parameters"""
        X_train, _, y_train, _ = X_y_train_test
        
        params = {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 42
        }
        
        trainer = MLOpsTrainer()
        model = trainer.train(X_train, y_train, params)
        
        assert model.n_estimators == 50
        assert model.max_depth == 5


class TestModelEvaluation:
    """Test model evaluation"""
    
    def test_evaluate_model(self, X_y_train_test):
        """Test model evaluation"""
        X_train, X_test, y_train, y_test = X_y_train_test
        
        trainer = MLOpsTrainer()
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
    
    def test_metrics_in_valid_range(self, X_y_train_test):
        """Test that metrics are in valid range"""
        X_train, X_test, y_train, y_test = X_y_train_test
        
        trainer = MLOpsTrainer()
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        for metric_name, metric_value in metrics.items():
            assert 0 <= metric_value <= 1.0, f"{metric_name} out of range"


class TestModelMetadata:
    """Test model metadata persistence"""
    
    def test_save_metadata(self, X_y_train_test):
        """Test saving model metadata"""
        X_train, X_test, y_train, y_test = X_y_train_test
        
        trainer = MLOpsTrainer()
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        metadata = trainer.save_metadata(metrics)
        
        assert metadata['model_name'] == 'RandomForestClassifier'
        assert 'version' in metadata
        assert 'created_at' in metadata
    
    def test_metadata_file_created(self, X_y_train_test):
        """Test that metadata file is created"""
        X_train, X_test, y_train, y_test = X_y_train_test
        
        trainer = MLOpsTrainer()
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        metadata = trainer.save_metadata(metrics)
        
        metadata_path = trainer.models_dir / "model_metadata.json"
        assert metadata_path.exists()