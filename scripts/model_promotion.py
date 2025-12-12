"""
Model Registry and Promotion Script
Register and transition models in MLflow
"""

import mlflow
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Register and promote model"""
    
    logger.info("=" * 60)
    logger.info("MODEL REGISTRATION & PROMOTION")
    logger.info("=" * 60)
    
    registry = ModelRegistry()
    
    # Note: Get run_id from MLflow UI or latest run
    # For this example, we'll get the latest run
    
    try:
        # Get latest run
        logger.info("\nFetching latest MLflow run...")
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("iris-classification")
        
        if not experiment:
            logger.error("Experiment 'iris-classification' not found")
            return False
        
        runs = client.search_runs(experiment.experiment_id, max_results=1)
        
        if not runs:
            logger.error("No runs found in experiment")
            return False
        
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        
        logger.info(f"Latest run ID: {run_id}")
        
        # Register model
        logger.info("\nRegistering model...")
        model_info = registry.register_model(
            model_name="iris-classifier",
            run_id=run_id,
            description="Iris flower classification model"
        )
        
        logger.info(f"✓ Model registered: {model_info}")
        
        # Transition to staging
        logger.info("\nTransitioning to Staging...")
        registry.transition_model_stage(
            model_name="iris-classifier",
            version=int(model_info['version']),
            stage="Staging"
        )
        
        logger.info("\n✓ Model ready in Staging")
        logger.info("After validation, promote to Production:")
        logger.info(f"  registry.transition_model_stage('iris-classifier', {model_info['version']}, 'Production')")
        
        return True
        
    except Exception as e:
        logger.error(f"Model promotion failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)