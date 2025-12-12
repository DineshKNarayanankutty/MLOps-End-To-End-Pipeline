import os
import time
import logging
import json
import joblib
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest

# ============================================================================
# CONFIGURATION
# ============================================================================
# Load paths from Env Vars (12-Factor App) or default to local folders
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
SCALER_PATH = Path(os.getenv("SCALER_PATH", "models/scaler.pkl"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "models/model_metadata.json"))

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mlops-api")

# Global storage for artifacts (Dictionary acts as an in-memory cache)
ml_artifacts: Dict[str, Any] = {}

# ============================================================================
# METRICS
# ============================================================================
prediction_count = Counter(
    'predictions_total', 
    'Total predictions made', 
    ['model_version', 'status']
)
prediction_latency = Histogram(
    'prediction_latency_seconds', 
    'Time taken for prediction', 
    ['model_version']
)
prediction_errors = Counter(
    'prediction_errors_total', 
    'Total errors encountered', 
    ['error_type']
)

# ============================================================================
# LIFESPAN MANAGER
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown logic.
    Loads models into memory once when the app starts.
    """
    logger.info("--- STARTUP: Loading model artifacts ---")
    
    # Validation: Ensure files exist
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        logger.error(f"CRITICAL: Model files not found at {MODEL_PATH} or {SCALER_PATH}")
        # In Kubernetes, we might want to crash here so the pod restarts, 
        # but for now we set a flag.
        ml_artifacts["ready"] = False
    else:
        try:
            ml_artifacts["model"] = joblib.load(MODEL_PATH)
            ml_artifacts["scaler"] = joblib.load(SCALER_PATH)
            
            # Load metadata if available
            if METADATA_PATH.exists():
                with open(METADATA_PATH) as f:
                    ml_artifacts["metadata"] = json.load(f)
            else:
                ml_artifacts["metadata"] = {"version": "1.0.0", "type": "unknown"}
            
            ml_artifacts["ready"] = True
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            ml_artifacts["ready"] = False
            
    yield
    
    # Shutdown logic
    logger.info("--- SHUTDOWN: Cleaning up resources ---")
    ml_artifacts.clear()

# ============================================================================
# APP & MODELS
# ============================================================================
app = FastAPI(
    title="MLOps Production API",
    description="Secure, monitored API for Iris Classification",
    version="2.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    sepal_length: float = Field(..., ge=0, le=15, description="Sepal Length in cm")
    sepal_width: float = Field(..., ge=0, le=15, description="Sepal Width in cm")
    petal_length: float = Field(..., ge=0, le=15, description="Petal Length in cm")
    petal_width: float = Field(..., ge=0, le=15, description="Petal Width in cm")

class PredictionResponse(BaseModel):
    species: str
    confidence: float
    model_version: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    ready: bool

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Liveness probe for Kubernetes"""
    is_ready = ml_artifacts.get("ready", False)
    return {
        "status": "healthy" if is_ready else "degraded",
        "ready": is_ready
    }

@app.get("/metrics")
def metrics():
    """Exposes Prometheus metrics"""
    return PlainTextResponse(generate_latest())

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Standard function (not async) to ensure it runs in a threadpool 
    and doesn't block the event loop during CPU-heavy inference.
    """
    if not ml_artifacts.get("ready"):
        raise HTTPException(status_code=503, detail="Model service not ready")

    start_time = time.time()
    version = ml_artifacts["metadata"].get("version", "unknown")

    try:
        # 1. Preprocess
        input_data = np.array([[
            request.sepal_length, 
            request.sepal_width, 
            request.petal_length, 
            request.petal_width
        ]])
        
        scaler = ml_artifacts["scaler"]
        model = ml_artifacts["model"]
        
        scaled_data = scaler.transform(input_data)

        # 2. Inference
        prediction_idx = model.predict(scaled_data)[0]
        probs = model.predict_proba(scaled_data)[0]
        confidence = float(np.max(probs))

        # 3. Postprocess
        species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        species = species_map.get(int(prediction_idx), "unknown")

        # 4. Record Metrics
        duration = time.time() - start_time
        prediction_latency.labels(model_version=version).observe(duration)
        prediction_count.labels(model_version=version, status="success").inc()

        logger.info(f"Pred: {species} | Conf: {confidence:.2f}")

        return {
            "species": species,
            "confidence": confidence,
            "model_version": version,
            "processing_time_ms": round(duration * 1000, 2)
        }

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        prediction_count.labels(model_version=version, status="error").inc()
        prediction_errors.labels(error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail="Internal prediction error")

if __name__ == "__main__":
    import uvicorn
    # Use workers=1 for dev, increase for production
    uvicorn.run(app, host="0.0.0.0", port=8000)