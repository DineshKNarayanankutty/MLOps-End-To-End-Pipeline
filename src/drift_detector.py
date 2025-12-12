"""
Drift Detection Module
Detects data drift and model drift using statistical tests
"""

import numpy as np
from scipy import stats
import pandas as pd
import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DriftDetector:
    """Detect data and model drift"""
    
    def __init__(self, reference_stats: Dict, threshold: float = 0.05):
        """
        Initialize drift detector
        
        Args:
            reference_stats (dict): Statistics from reference/training data
            threshold (float): Significance level for drift detection
        """
        self.reference_stats = reference_stats
        self.threshold = threshold
        logger.info(f"DriftDetector initialized with threshold={threshold}")
        
    def detect_drift(self, new_data: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Detect data drift using Kolmogorov-Smirnov test
        
        Args:
            new_data (np.ndarray): New data to check for drift
            feature_names (list): Names of features
            
        Returns:
            dict: Drift detection results
        """
        drift_results = {
            'has_drift': False,
            'features_drifted': [],
            'drift_scores': {},
            'timestamp': str(pd.Timestamp.now())
        }
        
        logger.info(f"Detecting drift in {new_data.shape[0]} samples...")
        
        for i, feature_name in enumerate(feature_names):
            # Get reference statistics
            ref_mean = self.reference_stats['mean'][feature_name]
            ref_std = self.reference_stats['std'][feature_name]
            ref_min = self.reference_stats['min'][feature_name]
            ref_max = self.reference_stats['max'][feature_name]
            
            # Get new data feature
            new_feature = new_data[:, i]
            new_mean = new_feature.mean()
            new_std = new_feature.std()
            new_min = new_feature.min()
            new_max = new_feature.max()
            
            # Kolmogorov-Smirnov test
            ref_dist = np.random.normal(ref_mean, ref_std, 1000)
            statistic, p_value = stats.ks_2samp(new_feature, ref_dist)
            
            # Mean shift detection
            mean_diff = abs(new_mean - ref_mean) / (ref_std + 1e-6)
            
            drift_results['drift_scores'][feature_name] = {
                'ks_statistic': float(statistic),
                'ks_p_value': float(p_value),
                'mean_shift': float(mean_diff),
                'ref_mean': float(ref_mean),
                'new_mean': float(new_mean),
                'ref_range': [float(ref_min), float(ref_max)],
                'new_range': [float(new_min), float(new_max)],
                'drifted': p_value < self.threshold or mean_diff > 2.0
            }
            
            if drift_results['drift_scores'][feature_name]['drifted']:
                drift_results['has_drift'] = True
                drift_results['features_drifted'].append(feature_name)
                logger.warning(f"Drift detected in {feature_name} (p={p_value:.4f}, mean_shift={mean_diff:.4f})")
        
        logger.info(f"✓ Drift detection complete | has_drift={drift_results['has_drift']}")
        
        return drift_results
    
    def get_drift_summary(self, drift_results: Dict) -> Dict:
        """
        Get summary of drift detection results
        
        Args:
            drift_results (dict): Results from detect_drift()
            
        Returns:
            dict: Summary statistics
        """
        total_features = len(drift_results['drift_scores'])
        drifted_count = len(drift_results['features_drifted'])
        drift_percentage = (drifted_count / total_features * 100) if total_features > 0 else 0
        
        summary = {
            'has_drift': drift_results['has_drift'],
            'total_features': total_features,
            'drifted_features': drifted_count,
            'drift_percentage': drift_percentage,
            'drifted_feature_names': drift_results['features_drifted'],
            'timestamp': drift_results['timestamp']
        }
        
        return summary


def main():
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION TEST")
    logger.info("=" * 60)
    
    # Simulate reference stats
    reference_stats = {
        'mean': {
            'feature_1': 5.0,
            'feature_2': 3.0,
            'feature_3': 3.5,
            'feature_4': 1.0
        },
        'std': {
            'feature_1': 0.8,
            'feature_2': 0.4,
            'feature_3': 1.8,
            'feature_4': 0.8
        },
        'min': {
            'feature_1': 4.0,
            'feature_2': 2.0,
            'feature_3': 1.0,
            'feature_4': 0.1
        },
        'max': {
            'feature_1': 8.0,
            'feature_2': 4.5,
            'feature_3': 7.0,
            'feature_4': 2.5
        }
    }
    
    # Create drift detector
    detector = DriftDetector(reference_stats, threshold=0.05)
    
    # Test data without drift
    logger.info("\n[TEST 1] Testing data WITHOUT drift...")
    normal_data = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [5.2, 3.2, 1.5, 0.2]
    ])
    
    results = detector.detect_drift(
        normal_data,
        ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    )
    summary = detector.get_drift_summary(results)
    logger.info(f"Results: {summary}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ DRIFT DETECTION TEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()