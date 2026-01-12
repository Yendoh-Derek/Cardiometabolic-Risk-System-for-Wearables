"""
Clinical context encoding (age, BMI, demographics).
Placeholder for now - will integrate with ICD-9 data later.
"""

import numpy as np
from typing import Dict

class ClinicalContextEncoder:
    """
    Encode clinical context features.
    
    For Sprint 2: Placeholder features (all zeros)
    For Sprint 3+: Will integrate actual demographics and ICD-9 codes
    
    Features:
    - Age, BMI, Sex (3 features)
    - Interaction terms: age×SDNN, BMI×LF/HF, RMSSD/SDNN (3 features)
    
    Total: 6 context features
    """
    
    def __init__(self):
        pass
    
    def extract_all(self, hrv_features: Dict[str, float], 
                   age: float = None, bmi: float = None, sex: int = None) -> Dict[str, float]:
        """
        Extract clinical context features.
        
        Args:
            hrv_features: Already extracted HRV features (for interactions)
            age: Patient age (years) - placeholder for now
            bmi: Body mass index - placeholder for now
            sex: Binary encoding (0=F, 1=M) - placeholder for now
            
        Returns:
            Dictionary with 6 context features
        """
        features = {}
        
        # Placeholder: Set to reasonable defaults
        # Will be replaced with actual data in later sprints
        features['age'] = age if age is not None else 50.0  # Mean age placeholder
        features['bmi'] = bmi if bmi is not None else 25.0  # Mean BMI placeholder
        features['sex'] = sex if sex is not None else 0.5   # Unknown = 0.5
        
        # Interaction terms (combine demographics with HRV)
        sdnn = hrv_features.get('sdnn', np.nan)
        lf_hf = hrv_features.get('lf_hf_ratio', np.nan)
        rmssd = hrv_features.get('rmssd', np.nan)
        
        features['age_x_sdnn'] = features['age'] * sdnn if not np.isnan(sdnn) else np.nan
        features['bmi_x_lf_hf'] = features['bmi'] * lf_hf if not np.isnan(lf_hf) else np.nan
        features['rmssd_over_sdnn'] = rmssd / sdnn if (not np.isnan(rmssd) and not np.isnan(sdnn) and sdnn > 0) else np.nan
        
        return features
    
    def get_feature_names(self) -> list:
        """Return list of all feature names in order."""
        return [
            'age', 'bmi', 'sex',
            'age_x_sdnn', 'bmi_x_lf_hf', 'rmssd_over_sdnn'
        ]