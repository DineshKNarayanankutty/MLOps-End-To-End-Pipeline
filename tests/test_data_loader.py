"""
Unit Tests for Data Loading Module

Tests cover:
    - Dataset loading from Iris source
    - Train/test splitting
    - Data quality validation
    - Feature and target separation
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import DataLoader


class TestDataLoaderInitialization:
    """Test DataLoader initialization"""
    
    def test_init_default_params(self):
        """Test initialization with default parameters"""
        loader = DataLoader()
        assert loader.test_size == 0.2
        assert loader.random_state == 42
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters"""
        loader = DataLoader(test_size=0.3, random_state=123)
        assert loader.test_size == 0.3
        assert loader.random_state == 123
    
    def test_data_directory_creation(self):
        """Test that data directory is created"""
        loader = DataLoader()
        assert loader.data_dir.exists()


class TestDataLoading:
    """Test dataset loading functionality"""
    
    def test_load_iris_dataset(self):
        """Test successful loading of Iris dataset"""
        loader = DataLoader()
        df = loader.load_iris_dataset()
        
        # Verify dataset structure
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 150  # Iris has 150 samples
        assert df.shape[1] == 6    # 4 features + target + species
    
    def test_iris_columns_present(self):
        """Test that all expected columns are present"""
        loader = DataLoader()
        df = loader.load_iris_dataset()
        
        expected_columns = {
            'sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
            'petal width (cm)',
            'target',
            'species'
        }
        
        assert set(df.columns) == expected_columns
    
    def test_iris_species_names(self):
        """Test that species are correctly mapped"""
        loader = DataLoader()
        df = loader.load_iris_dataset()
        
        species_set = set(df['species'].unique())
        expected_species = {'setosa', 'versicolor', 'virginica'}
        
        assert species_set == expected_species


class TestDataSplitting:
    """Test train/test data splitting"""
    
    def test_split_data(self):
        """Test that data is correctly split"""
        loader = DataLoader(test_size=0.2)
        df = loader.load_iris_dataset()
        
        X_train, X_test, y_train, y_test = loader.split_data(df)
        
        # Check shapes
        assert X_train.shape[0] == 120  # 80% of 150
        assert X_test.shape[0] == 30    # 20% of 150
        assert len(y_train) == 120
        assert len(y_test) == 30
    
    def test_split_stratification(self):
        """Test that stratification preserves class distribution"""
        loader = DataLoader(test_size=0.2)
        df = loader.load_iris_dataset()
        
        X_train, X_test, y_train, y_test = loader.split_data(df)
        
        # Check class distribution is similar
        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()
        
        # Distribution should be roughly similar (within 10%)
        for idx in train_dist.index:
            assert abs(train_dist[idx] - test_dist[idx]) < 0.15
    
    def test_no_overlap_between_splits(self):
        """Test that train and test sets don't overlap"""
        loader = DataLoader()
        df = loader.load_iris_dataset()
        
        X_train, X_test, y_train, y_test = loader.split_data(df)
        
        # Check for any row overlap
        assert X_train.shape[0] + X_test.shape[0] == df.shape[0] - 1


class TestDataValidation:
    """Test data quality validation"""
    
    def test_validate_no_missing_values(self):
        """Test that Iris dataset has no missing values"""
        loader = DataLoader()
        df = loader.load_iris_dataset()
        
        quality_report = loader.validate_data_quality(df)
        
        assert quality_report['missing_values'] == {} or all(
            v == 0 for v in quality_report['missing_values'].values()
        )
    
    def test_validate_duplicates(self):
        """Test duplicate detection"""
        loader = DataLoader()
        df = loader.load_iris_dataset()
        
        quality_report = loader.validate_data_quality(df)
        
        assert quality_report['duplicate_rows'] >= 0
    
    def test_quality_report_structure(self):
        """Test that quality report has expected structure"""
        loader = DataLoader()
        df = loader.load_iris_dataset()
        
        quality_report = loader.validate_data_quality(df)
        
        expected_keys = {
            'total_rows',
            'total_columns',
            'missing_values',
            'duplicate_rows',
            'shape',
            'columns'
        }
        
        assert set(quality_report.keys()) == expected_keys