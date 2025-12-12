"""
Pytest Configuration and Fixtures

Provides reusable fixtures for testing across the entire test suite.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_iris_data():
    """Create sample Iris dataset for testing"""
    np.random.seed(42)
    n_samples = 100
    
    X = np.random.randn(n_samples, 4) * [0.8, 0.4, 1.8, 0.8] + [5.8, 3.0, 3.8, 1.2]
    y = np.random.randint(0, 3, n_samples)
    
    feature_names = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, X, y


@pytest.fixture
def X_y_train_test(sample_iris_data):
    """Split sample data into train and test sets"""
    df, X, y = sample_iris_data
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def preprocessor():
    """Initialize preprocessor fixture"""
    from src.preprocessing import DataPreprocessor
    return DataPreprocessor()


@pytest.fixture
def trained_model(X_y_train_test):
    """Create a trained model for testing"""
    from src.training import MLOpsTrainer
    from sklearn.ensemble import RandomForestClassifier
    
    X_train, _, y_train, _ = X_y_train_test
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model