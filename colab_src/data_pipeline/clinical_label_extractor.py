"""
Extract cardiometabolic labels and Charlson Comorbidity Index from ICD-9 codes.
Handles Option B: Labels apply to entire admission period.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ClinicalLabelExtractor:
    """
    Extract multi-hot labels (Diabetes, Hypertension, Obesity) and CCI from diagnoses.
    
    ICD-9 Mappings (Option B - per admission):
    - Diabetes: 250.xx (all subtypes)
    - Hypertension: 401.x, 402.x, 403.x (primary, secondary, malignant)
    - Obesity: 278.0x
    - Charlson Index: All 17 categories computed from all diagnoses
    
    Note: No "Healthy" label. Healthy = all three conditions = 0.
    """
    
    # ICD-9 Code Prefixes for Target Conditions
    ICD9_DIABETES = ['250']
    ICD9_HYPERTENSION = ['401', '402', '403']
    ICD9_OBESITY = ['278']
    
    def __init__(self):
        pass
    
    def _match_icd9_prefix(self, icd9_code: str, prefixes: List[str]) -> bool:
        """
        Check if ICD-9 code starts with any of the given prefixes.
        
        Args:
            icd9_code: ICD-9 code string (e.g., '250.02')
            prefixes: List of prefixes to match (e.g., ['250'])
            
        Returns:
            True if code matches any prefix
        """
        icd9_str = str(icd9_code).strip()
        for prefix in prefixes:
            if icd9_str.startswith(prefix):
                return True
        return False
    
    def extract_labels_per_admission(self, diagnoses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract multi-hot labels per HADM_ID (admission).
        
        Logic (Option B):
        - All diagnoses in DIAGNOSES_ICD table apply to the entire admission
        - If any ICD-9 code matches a condition, label = 1
        
        Args:
            diagnoses_df: DataFrame with columns [hadm_id, icd9_code, seq_num]
            
        Returns:
            DataFrame with columns [hadm_id, diabetes, hypertension, obesity]
        """
        logger.info("🏥 Extracting cardiometabolic labels from ICD-9 codes...")
        
        # Group diagnoses by admission
        admission_diagnoses = diagnoses_df.groupby('hadm_id')['icd9_code'].apply(list).reset_index()
        admission_diagnoses.columns = ['hadm_id', 'icd9_codes']
        
        labels = []
        
        for idx, row in admission_diagnoses.iterrows():
            hadm_id = row['hadm_id']
            icd9_codes = row['icd9_codes']
            
            label_row = {'hadm_id': hadm_id}
            
            # Check each condition (binary: 1 if present, 0 otherwise)
            label_row['diabetes'] = int(
                any(self._match_icd9_prefix(code, self.ICD9_DIABETES) for code in icd9_codes)
            )
            
            label_row['hypertension'] = int(
                any(self._match_icd9_prefix(code, self.ICD9_HYPERTENSION) for code in icd9_codes)
            )
            
            label_row['obesity'] = int(
                any(self._match_icd9_prefix(code, self.ICD9_OBESITY) for code in icd9_codes)
            )
            
            labels.append(label_row)
        
        labels_df = pd.DataFrame(labels)
        
        # Log distribution
        logger.info(f"✅ Extracted labels for {len(labels_df)} admissions")
        logger.info(f"   Diabetes:      {labels_df['diabetes'].sum():5d} ({100*labels_df['diabetes'].mean():5.1f}%)")
        logger.info(f"   Hypertension:  {labels_df['hypertension'].sum():5d} ({100*labels_df['hypertension'].mean():5.1f}%)")
        logger.info(f"   Obesity:       {labels_df['obesity'].sum():5d} ({100*labels_df['obesity'].mean():5.1f}%)")
        
        # Multi-morbidity
        multi_morbid = (labels_df[['diabetes', 'hypertension', 'obesity']].sum(axis=1) > 1).sum()
        healthy = (labels_df[['diabetes', 'hypertension', 'obesity']].sum(axis=1) == 0).sum()
        logger.info(f"   Multi-morbid:  {multi_morbid:5d} ({100*multi_morbid/len(labels_df):5.1f}%)")
        logger.info(f"   Healthy (0,0,0): {healthy:5d} ({100*healthy/len(labels_df):5.1f}%)")
        
        return labels_df
    
    def compute_charlson_index(self, diagnoses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Charlson Comorbidity Index from all ICD-9 diagnoses.
        
        Simplified implementation: counts presence of key conditions.
        Full implementation would use comorbidipy.charlson().
        
        Args:
            diagnoses_df: DataFrame with columns [hadm_id, icd9_code, seq_num]
            
        Returns:
            DataFrame with columns [hadm_id, cci_score]
        """
        logger.info("📊 Computing Charlson Comorbidity Index...")
        
        # Simple Charlson mapping (points for each condition)
        charlson_rules = {
            'MI': ('410', 1),  # Myocardial infarction
            'CHF': ('428', 1),  # Congestive heart failure
            'PVD': ('443', 1),  # Peripheral vascular disease
            'CVA': ('43', 1),  # Cerebrovascular accident
            'COPD': ('49', 1),  # COPD
            'DM': ('250', 1),  # Diabetes
            'Renal': ('585', 2),  # Chronic renal disease
            'Cancer': ('17', 2),  # Malignancy
            'AIDS': ('042', 6)  # AIDS
        }
        
        cci_scores = []
        
        for hadm_id in diagnoses_df['hadm_id'].unique():
            codes = diagnoses_df[diagnoses_df['hadm_id'] == hadm_id]['icd9_code'].tolist()
            codes_str = [str(c).strip() for c in codes]
            
            cci = 0
            
            # Check each condition
            for condition, (icd_prefix, points) in charlson_rules.items():
                if any(code.startswith(icd_prefix) for code in codes_str):
                    cci += points
            
            cci_scores.append({'hadm_id': hadm_id, 'cci_score': cci})
        
        cci_df = pd.DataFrame(cci_scores)
        
        logger.info(f"✅ Computed CCI for {len(cci_df)} admissions")
        logger.info(f"   Mean CCI:   {cci_df['cci_score'].mean():6.2f}")
        logger.info(f"   Median CCI: {cci_df['cci_score'].median():6.0f}")
        logger.info(f"   Range:      [{cci_df['cci_score'].min():.0f}, {cci_df['cci_score'].max():.0f}]")
        
        return cci_df
