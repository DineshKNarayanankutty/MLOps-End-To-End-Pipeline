"""
Complete training pipeline script
Orchestrates all steps from data loading to model deployment
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.training import MLOpsTrainer
from src.drift_detector import DriftDetector
from src.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Execute complete ML pipeline"""
    
    logger.info("=" * 80)
    logger.info("COMPLETE MLOps TRAINING PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Step 1: Load Data
        logger.info("\n[STEP 1] Loading data...")
        loader = DataLoader(test_size=0.2, random_state=42)
        df = loader.load_iris_dataset()
        quality_report = loader.validate_data_quality(df)
        X_train, X_test, y_train, y_test = loader.split_data(df)
        logger.info("✓ Data loaded successfully")
        
        # Step 2: Preprocessing
        logger.info("\n[STEP 2] Preprocessing data...")
        preprocessor = DataPreprocessor()
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        train_stats = preprocessor.get_feature_stats(X_train)
        logger.info("✓ Preprocessing complete")
        
        # Step 3: Training
        logger.info("\n[STEP 3] Training model...")
        trainer = MLOpsTrainer(experiment_name="iris-classification")
        model = trainer.train(X_train_scaled, y_train.values)
        logger.info("✓ Training complete")
        
        # Step 4: Evaluation
        logger.info("\n[STEP 4] Evaluating model...")
        metrics = trainer.evaluate(X_test_scaled, y_test.values)
        trainer.save_metadata(metrics)
        logger.info("✓ Evaluation complete")
        
        # Step 5: Drift Detection Setup
        logger.info("\n[STEP 5] Setting up drift detection...")
        detector = DriftDetector(train_stats, threshold=0.05)
        logger.info("✓ Drift detector ready")
        
        # Step 6: Model Registry (Optional - requires MLflow to be running)
        logger.info("\n[STEP 6] Model registry preparation...")
        registry = ModelRegistry()
        logger.info("✓ Model registry ready (use after training)")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ COMPLETE PIPELINE SUCCESS")
        logger.info("=" * 80)
        
        logger.info("\nNext steps:")
        logger.info("1. Start MLflow server: mlflow ui")
        logger.info("2. Register model: python scripts/model_promotion.py")
        logger.info("3. Start API: python api/main.py")
        logger.info("4. Deploy: docker-compose up -d")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)