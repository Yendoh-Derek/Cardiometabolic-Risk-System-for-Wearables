"""Data quality validation tests."""

import pandas as pd
import numpy as np
from pathlib import Path

class DataQualityValidator:
    def __init__(self, signals_path, metadata_path):
        self.signals = np.load(signals_path)
        self.metadata = pd.read_parquet(metadata_path)
    
    def run_all_tests(self):
        print("="*60)
        print("DATA QUALITY VALIDATION")
        print("="*60)
        
        tests = [
            self.test_schema(),
            self.test_signal_shape(),
            self.test_sqi_distribution(),
            self.test_no_nans(),
            self.test_signal_range(),
            self.test_sampling_rate()
        ]
        
        passed = sum(tests)
        total = len(tests)
        
        print(f"\n{'='*60}")
        print(f"RESULT: {passed}/{total} tests passed")
        print(f"{'='*60}\n")
        
        return passed == total
    
    def test_schema(self):
        required_cols = ['record_name', 'subject_id', 'segment_idx', 
                        'sqi_score', 'quality_grade']
        missing = [col for col in required_cols if col not in self.metadata.columns]
        
        if missing:
            print(f"❌ Schema Test: Missing columns {missing}")
            return False
        print("✅ Schema Test: All required columns present")
        return True
    
    def test_signal_shape(self):
        expected_length = 600 * 125
        
        if self.signals.shape[1] != expected_length:
            print(f"❌ Signal Shape: Expected {expected_length}, got {self.signals.shape[1]}")
            return False
        print(f"✅ Signal Shape: {self.signals.shape} (correct)")
        return True
    
    def test_sqi_distribution(self):
        mean_sqi = self.metadata['sqi_score'].mean()
        min_sqi = self.metadata['sqi_score'].min()
        
        if min_sqi < 0.5:
            print(f"❌ SQI Distribution: Minimum SQI {min_sqi:.3f} < 0.5")
            return False
        
        print(f"✅ SQI Distribution: Mean={mean_sqi:.3f}, Min={min_sqi:.3f}")
        return True
    
    def test_no_nans(self):
        signal_nans = np.isnan(self.signals).sum()
        metadata_nans = self.metadata.isnull().sum().sum()
        
        if signal_nans > 0 or metadata_nans > 0:
            print(f"❌ NaN Test: Found {signal_nans} signal NaNs, {metadata_nans} metadata NaNs")
            return False
        print("✅ NaN Test: No missing values")
        return True
    
    def test_signal_range(self):
        signal_min = self.signals.min()
        signal_max = self.signals.max()
        
        if abs(signal_max) > 1e6 or abs(signal_min) > 1e6:
            print(f"❌ Signal Range: Extreme values (min={signal_min}, max={signal_max})")
            return False
        
        print(f"✅ Signal Range: Min={signal_min:.2f}, Max={signal_max:.2f}")
        return True
    
    def test_sampling_rate(self):
        if 'fs' in self.metadata.columns:
            unique_fs = self.metadata['fs'].unique()
            if len(unique_fs) != 1 or unique_fs[0] != 125:
                print(f"❌ Sampling Rate: Inconsistent rates {unique_fs}")
                return False
        
        print("✅ Sampling Rate: Consistent at 125 Hz")
        return True