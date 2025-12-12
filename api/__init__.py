"""
FastAPI Application Module

REST API endpoints for model serving and monitoring.

Features:
    - /predict: Single and batch predictions
    - /health: Service health check
    - /metrics: Prometheus metrics endpoint
    - /model/info: Model information and metadata
"""

from api.main import app

__version__ = "1.0.0"
__all__ = ["app"]