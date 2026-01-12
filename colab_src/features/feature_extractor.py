"""
Unified feature extraction pipeline.
Orchestrates HRV, morphology, and context feature extraction.
Supports extraction with clinical context from ground truth dataset.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from .hrv_features import HRVFeatureExtractor
from .morphology_features import MorphologyFeatureExtractor
from .clinical_context import ClinicalContextEncoder

logger = logging.getLogger(__name__)


class UnifiedFeatureExtractor:
    """
    Unified pipeline for extracting all features from PPG signals.
    
    Total features: 30 (HRV) + 6 (morphology) + 6 (context) = 42 features
    
    Supports two extraction modes:
    1. extract_single/extract_batch: Classic mode with optional demographics
    2. extract_with_ground_truth: Uses demographics from signal_clinical_integrated_dataset
    """
    
    def __init__(self, sampling_rate=125):
        self.fs = sampling_rate
        self.hrv_extractor = HRVFeatureExtractor(sampling_rate)
        self.morph_extractor = MorphologyFeatureExtractor(sampling_rate)
        self.context_encoder = ClinicalContextEncoder()
    
    def extract_single(self, signal: np.ndarray, 
                      age: float = None, bmi: float = None, sex: int = None,
                      cci_score: float = None) -> Dict[str, float]:
        """
        Extract all features from a single PPG segment.
        
        Args:
            signal: PPG signal (75,000 samples @ 125 Hz)
            age: Patient age (years)
            bmi: Body mass index
            sex: Binary sex (0=Female, 1=Male)
            cci_score: Charlson Comorbidity Index
            
        Returns:
            Dictionary with all extracted features
        """
        # Extract HRV features (includes peak detection)
        hrv_features = self.hrv_extractor.extract_all(signal)
        
        # Extract morphology features (needs peaks from HRV)
        peak_data = self.hrv_extractor.extract_peaks(signal)
        if peak_data is not None:
            morph_features = self.morph_extractor.extract_all(
                peak_data['cleaned_signal'], 
                peak_data['peaks']
            )
        else:
            # Fill with NaN if peak detection failed
            morph_features = {f: np.nan for f in self.morph_extractor.get_feature_names()}
        
        # Extract context features (now with real demographics if provided)
        context_features = self.context_encoder.extract_all(hrv_features, age, bmi, sex)
        
        # Add comorbidity if provided
        if cci_score is not None:
            context_features['cci_score'] = cci_score
        
        # Combine all features
        all_features = {**hrv_features, **morph_features, **context_features}
        
        return all_features
    
    def extract_batch(self, signals: np.ndarray, show_progress: bool = True) -> pd.DataFrame:
        """
        Extract features from multiple signals.
        
        Args:
            signals: Array of PPG signals (n_segments, 75000)
            show_progress: Show progress bar
            
        Returns:
            DataFrame with features (n_segments, 42+)
        """
        from tqdm import tqdm
        
        features_list = []
        
        iterator = tqdm(signals, desc="Extracting features") if show_progress else signals
        
        for signal in iterator:
            features = self.extract_single(signal)
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Extracted {len(features_df)} feature vectors")
        
        return features_df
    
    def extract_with_ground_truth(
        self,
        signals: np.ndarray,
        ground_truth_dataset: pd.DataFrame,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Extract features for all signals using clinical context from ground truth dataset.
        
        This method integrates real demographics and clinical data.
        No placeholders - all clinical context is actual MIMIC data.
        
        Args:
            signals: Array of PPG signals (n_segments, 75000)
            ground_truth_dataset: DataFrame with [segment_id, age, sex, bmi, cci_score, ...]
            show_progress: Show progress bar
            
        Returns:
            DataFrame with extracted features (n_segments, 42+)
        """
        from tqdm import tqdm
        
        logger.info(f"🔍 Extracting features with clinical ground truth ({len(signals)} segments)")
        
        features_list = []
        
        iterator = tqdm(range(len(signals)), desc="Extracting features") if show_progress else range(len(signals))
        
        for idx in iterator:
            signal = signals[idx]
            metadata = ground_truth_dataset.iloc[idx]
            
            # Extract features with demographic context
            features = self.extract_single(
                signal,
                age=metadata.get('age'),
                sex=metadata.get('sex'),
                bmi=metadata.get('bmi'),
                cci_score=metadata.get('cci_score')
            )
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Extracted {len(features_df)} feature vectors with ground truth")
        logger.info(f"   Features: {features_df.shape[1]} columns")
        logger.info(f"   Completeness: {(1 - features_df.isnull().sum().sum() / (features_df.shape[0] * features_df.shape[1])) * 100:.1f}%")
        
        return features_df
    
    def get_feature_names(self) -> List[str]:
        """Return ordered list of all feature names."""
        return (
            self.hrv_extractor.get_feature_names() +
            self.morph_extractor.get_feature_names() +
            self.context_encoder.get_feature_names() +
            ['cci_score']  # Added for ground truth extraction
        )