"""
Unit Tests for FastAPI Endpoints

Tests cover:
    - Health check endpoint
    - Prediction endpoint
    - Batch prediction
    - Model info endpoint
    - Error handling
"""

import pytest
import json
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_status(self, client):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_response(self, client):
        """Test health endpoint response structure"""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"


class TestPredictionEndpoint:
    """Test prediction endpoint"""
    
    def test_predict_single_sample(self, client):
        """Test single sample prediction"""
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
    
    def test_predict_valid_range(self, client):
        """Test prediction with valid range"""
        payload = {
            "sepal_length": 6.0,
            "sepal_width": 3.0,
            "petal_length": 3.5,
            "petal_width": 1.0
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["prediction"] in [0, 1, 2]  # Valid iris classes
        assert 0 <= data["confidence"] <= 1.0


class TestBatchPredictionEndpoint:
    """Test batch prediction endpoint"""
    
    def test_batch_predict(self, client):
        """Test batch prediction"""
        payload = {
            "samples": [
                [5.1, 3.5, 1.4, 0.2],
                [6.2, 2.9, 4.3, 1.3],
                [7.1, 3.0, 5.9, 2.1]
            ]
        }
        
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3


class TestModelInfoEndpoint:
    """Test model info endpoint"""
    
    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "framework" in data


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_payload(self, client):
        """Test with invalid payload"""
        payload = {
            "invalid_field": 1.0
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code in [422, 400]
    
    def test_missing_fields(self, client):
        """Test with missing required fields"""
        payload = {
            "sepal_length": 5.1
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code in [422, 400]
    
    def test_invalid_data_type(self, client):
        """Test with invalid data type"""
        payload = {
            "sepal_length": "invalid",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code in [422, 400]