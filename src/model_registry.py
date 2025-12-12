"""
MLflow Model Registry Module
Register, version, and manage models in MLflow
"""

import mlflow
import mlflow.sklearn
import logging
from typing import Dict, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelRegistry:
    """Manage model versioning and promotion"""
    
    def __init__(self):
        """Initialize model registry"""
        logger.info("ModelRegistry initialized")
        
    def register_model(self, model_name: str, run_id: str, description: str = "") -> Dict:
        """
        Register model to MLflow Model Registry
        
        Args:
            model_name (str): Name for the registered model
            run_id (str): MLflow run ID containing the model
            description (str): Model description
            
        Returns:
            dict: Model version information
        """
        logger.info(f"Registering model: {model_name}")
        
        # Get run URI
        run_uri = f"runs:/{run_id}/model"
        
        try:
            # Register model
            model_version = mlflow.register_model(
                model_uri=run_uri,
                name=model_name
            )
            
            logger.info(f"✓ Model registered successfully")
            logger.info(f"  Name: {model_version.name}")
            logger.info(f"  Version: {model_version.version}")
            logger.info(f"  Stage: {model_version.current_stage}")
            
            # Save registration info
            registry_info = {
                'model_name': model_version.name,
                'version': model_version.version,
                'stage': model_version.current_stage,
                'run_id': run_id,
                'description': description
            }
            
            # Persist registry info
            registry_path = Path("models") / "model_registry.json"
            with open(registry_path, 'w') as f:
                json.dump(registry_info, f, indent=2)
            
            return registry_info
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def transition_model_stage(self, model_name: str, version: int, stage: str) -> Dict:
        """
        Transition model to a new stage
        
        Args:
            model_name (str): Name of registered model
            version (int): Model version
            stage (str): Target stage ('Staging', 'Production', 'Archived')
            
        Returns:
            dict: Updated model information
        """
        valid_stages = ['Staging', 'Production', 'Archived']
        
        if stage not in valid_stages:
            raise ValueError(f"Stage must be one of {valid_stages}")
        
        logger.info(f"Transitioning {model_name} v{version} to {stage}...")
        
        try:
            client = mlflow.tracking.MlflowClient()
            
            model_version = client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"✓ Model transitioned successfully")
            logger.info(f"  Name: {model_version.name}")
            logger.info(f"  Version: {model_version.version}")
            logger.info(f"  New Stage: {model_version.current_stage}")
            
            return {
                'model_name': model_version.name,
                'version': model_version.version,
                'stage': model_version.current_stage
            }
            
        except Exception as e:
            logger.error(f"Failed to transition model: {e}")
            raise
    
    def get_latest_production_model(self, model_name: str) -> Optional[Dict]:
        """
        Get the latest production model
        
        Args:
            model_name (str): Name of registered model
            
        Returns:
            dict: Model information or None
        """
        logger.info(f"Getting latest production model: {model_name}")
        
        try:
            client = mlflow.tracking.MlflowClient()
            
            versions = client.get_latest_versions(model_name, stages=['Production'])
            
            if versions:
                model = versions[0]
                logger.info(f"✓ Found production model v{model.version}")
                
                return {
                    'model_name': model.name,
                    'version': model.version,
                    'stage': model.current_stage,
                    'run_id': model.run_id,
                    'status': model.status
                }
            else:
                logger.info("No production model found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get production model: {e}")
            return None
    
    def list_model_versions(self, model_name: str) -> list:
        """
        List all versions of a registered model
        
        Args:
            model_name (str): Name of registered model
            
        Returns:
            list: List of model versions
        """
        logger.info(f"Listing versions for {model_name}...")
        
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name)
            
            version_list = []
            for model in versions:
                version_list.append({
                    'version': model.version,
                    'stage': model.current_stage,
                    'status': model.status,
                    'run_id': model.run_id
                })
            
            logger.info(f"✓ Found {len(version_list)} versions")
            
            return version_list
            
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []


def main():
    logger.info("=" * 60)
    logger.info("MODEL REGISTRY TEST")
    logger.info("=" * 60)
    
    registry = ModelRegistry()
    
    logger.info("\nNote: Run training.py first to create a run")
    logger.info("Then use register_model() with the run_id")


if __name__ == "__main__":
    main()