"""
Unit Tests for Data Preprocessing Module

Tests cover:
    - Feature scaling with StandardScaler
    - Scaler fit and transform operations
    - Statistics calculation
    - Scaler persistence
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocessing import DataPreprocessor
import joblib


class TestDataPreprocessorInitialization:
    """Test DataPreprocessor initialization"""
    
    def test_init(self):
        """Test successful initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler is not None
        assert preprocessor.feature_names is None
    
    def test_models_directory_creation(self):
        """Test that models directory is created"""
        preprocessor = DataPreprocessor()
        assert preprocessor.models_dir.exists()


class TestScalerFitTransform:
    """Test scaler fitting and transformation"""
    
    def test_fit_transform(self, X_y_train_test):
        """Test fitting scaler on training data"""
        X_train, _, _, _ = X_y_train_test
        X_train_df = pd.DataFrame(
            X_train,
            columns=['f1', 'f2', 'f3', 'f4']
        )
        
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.fit_transform(X_train_df)
        
        assert X_scaled.shape == X_train.shape
        assert preprocessor.scaler is not None
    
    def test_scaled_features_have_unit_variance(self, X_y_train_test):
        """Test that scaled features have ~unit variance"""
        X_train, _, _, _ = X_y_train_test
        X_train_df = pd.DataFrame(
            X_train,
            columns=['f1', 'f2', 'f3', 'f4']
        )
        
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.fit_transform(X_train_df)
        
        # Check that std is close to 1
        stds = np.std(X_scaled, axis=0)
        assert np.allclose(stds, 1.0, atol=0.1)
    
    def test_scaled_features_centered_at_zero(self, X_y_train_test):
        """Test that scaled features are centered at ~0"""
        X_train, _, _, _ = X_y_train_test
        X_train_df = pd.DataFrame(
            X_train,
            columns=['f1', 'f2', 'f3', 'f4']
        )
        
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.fit_transform(X_train_df)
        
        means = np.mean(X_scaled, axis=0)
        assert np.allclose(means, 0.0, atol=0.1)


class TestScalerTransform:
    """Test scaler transform on new data"""
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error"""
        preprocessor = DataPreprocessor()
        X = np.random.randn(10, 4)
        
        # Should raise ValueError since scaler not fitted
        with pytest.raises((ValueError, AttributeError)):
            preprocessor.transform(pd.DataFrame(X))
    
    def test_transform_after_fit(self, X_y_train_test):
        """Test transform on new data after fitting"""
        X_train, X_test, _, _ = X_y_train_test
        X_train_df = pd.DataFrame(
            X_train,
            columns=['f1', 'f2', 'f3', 'f4']
        )
        X_test_df = pd.DataFrame(
            X_test,
            columns=['f1', 'f2', 'f3', 'f4']
        )
        
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(X_train_df)
        
        X_test_scaled = preprocessor.transform(X_test_df)
        
        assert X_test_scaled.shape == X_test.shape


class TestFeatureStatistics:
    """Test statistics calculation"""
    
    def test_get_feature_stats(self, X_y_train_test):
        """Test feature statistics calculation"""
        X_train, _, _, _ = X_y_train_test
        X_train_df = pd.DataFrame(
            X_train,
            columns=['f1', 'f2', 'f3', 'f4']
        )
        
        preprocessor = DataPreprocessor()
        stats = preprocessor.get_feature_stats(X_train_df)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert stats['shape'] == X_train.shape
    
    def test_get_scaled_stats(self, X_y_train_test):
        """Test scaled data statistics"""
        X_train, _, _, _ = X_y_train_test
        X_train_df = pd.DataFrame(
            X_train,
            columns=['f1', 'f2', 'f3', 'f4']
        )
        
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.fit_transform(X_train_df)
        stats = preprocessor.get_scaled_stats(X_scaled)
        
        # Means should be ~0
        assert all(abs(m) < 0.1 for m in stats['mean'])
        # Stds should be ~1
        assert all(0.9 < s < 1.1 for s in stats['std'])


class TestScalerPersistence:
    """Test scaler saving and loading"""
    
    def test_scaler_saved_to_file(self, X_y_train_test):
        """Test that scaler is saved to file"""
        X_train, _, _, _ = X_y_train_test
        X_train_df = pd.DataFrame(
            X_train,
            columns=['f1', 'f2', 'f3', 'f4']
        )
        
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(X_train_df)
        
        scaler_path = preprocessor.models_dir / "scaler.pkl"
        assert scaler_path.exists()
    
    def test_load_scaler(self, X_y_train_test):
        """Test loading pre-fitted scaler"""
        X_train, _, _, _ = X_y_train_test
        X_train_df = pd.DataFrame(
            X_train,
            columns=['f1', 'f2', 'f3', 'f4']
        )
        
        # Fit and save scaler
        preprocessor1 = DataPreprocessor()
        preprocessor1.fit_transform(X_train_df)
        
        # Load scaler in new instance
        preprocessor2 = DataPreprocessor()
        preprocessor2.load_scaler()
        
        assert preprocessor2.scaler is not None