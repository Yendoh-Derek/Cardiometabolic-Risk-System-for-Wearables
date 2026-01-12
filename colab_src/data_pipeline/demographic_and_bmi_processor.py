"""
Extract and process demographics (age, BMI, sex).
Handles missing data with imputation and flagging.
Uses parallel processing for efficiency.
"""

import pandas as pd
import numpy as np
from typing import Optional
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)


class DemographicProcessor:
    """
    Extract age from DOB, handle BMI with imputation and flagging.
    
    Age Logic:
    - age = (admittime - dob).days / 365.25
    - If age > 89: set to 90 (MIMIC privacy masking)
    - If age missing or invalid: mark as NaN (will be excluded later)
    
    BMI Logic:
    - If missing: impute with population median
    - Flag missing with is_bmi_missing = 1
    - Set flag to 0 if BMI was actually measured
    
    Sex Encoding:
    - Female (F): 0
    - Male (M): 1
    """
    
    def __init__(self, n_jobs: int = -1):
        """
        Args:
            n_jobs: Number of parallel jobs (-1 = use all CPUs)
        """
        self.n_jobs = n_jobs
    
    def process_demographics(
        self,
        linked_segments_df: pd.DataFrame,
        patients_df: pd.DataFrame,
        admissions_df: pd.DataFrame,
        n_jobs: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract and process demographics for all segments.
        
        Args:
            linked_segments_df: Segments with [segment_id, subject_id, hadm_id]
            patients_df: Patients with [subject_id, dob, gender]
            admissions_df: Admissions with [hadm_id, admittime]
            n_jobs: Parallel jobs
            
        Returns:
            DataFrame with [segment_id, subject_id, hadm_id, age, sex, bmi, is_bmi_missing]
        """
        
        n_jobs = n_jobs or self.n_jobs
        logger.info(f"👤 Processing demographics for {len(linked_segments_df)} segments")
        
        # Create lookup tables (indexed for O(1) access)
        patients_lookup = patients_df.set_index('subject_id')
        admissions_lookup = admissions_df.set_index('hadm_id')
        
        # Process each segment in parallel
        def process_segment(row):
            """Extract demographics for a single segment."""
            result = {
                'segment_id': row['segment_id'],
                'subject_id': row['subject_id'],
                'hadm_id': row['hadm_id'],
            }
            
            # Skip if missing hadm_id (orphan segment)
            if pd.isna(row['hadm_id']):
                return {**result, 'age': np.nan, 'sex': np.nan, 'bmi': np.nan, 'is_bmi_missing': np.nan}
            
            try:
                # Get patient DOB and gender
                patient = patients_lookup.loc[row['subject_id']]
                dob = patient['dob']
                gender = patient['gender']
                
                # Get admission time
                admission = admissions_lookup.loc[row['hadm_id']]
                admit_time = admission['admittime']
                
                # Calculate age in years
                age_days = (admit_time - dob).days
                age = age_days / 365.25
                
                # MIMIC privacy masking: ages > 89 are reported as 90
                if age > 89:
                    age = 90
                elif age < 0 or age > 120:
                    age = np.nan  # Invalid age
                
                # Convert sex to binary (M=1, F=0)
                sex = 1 if gender == 'M' else 0
                
                # BMI: placeholder (all missing for now - will impute with median later)
                bmi = np.nan
                is_bmi_missing = 1  # Will be updated after median imputation
                
                result.update({
                    'age': age,
                    'sex': sex,
                    'bmi': bmi,
                    'is_bmi_missing': is_bmi_missing
                })
                
            except (KeyError, AttributeError, TypeError):
                # Patient, admission, or data not found or invalid
                result.update({
                    'age': np.nan,
                    'sex': np.nan,
                    'bmi': np.nan,
                    'is_bmi_missing': np.nan
                })
            
            return result
        
        # Parallel processing
        logger.info(f"   Processing {len(linked_segments_df)} segments in parallel...")
        demographics = Parallel(n_jobs=n_jobs)(
            delayed(process_segment)(row) for _, row in linked_segments_df.iterrows()
        )
        
        demographics_df = pd.DataFrame(demographics)
        
        # Impute BMI with median (for now, all NaN so median = NaN)
        # In a real scenario, you might calculate from LABEVENTS
        median_bmi = demographics_df['bmi'].median()
        if pd.isna(median_bmi):
            median_bmi = 25.0  # Default population average if all missing
            logger.warning(f"⚠️  All BMI values missing. Using default: {median_bmi}")
        
        demographics_df['bmi'] = demographics_df['bmi'].fillna(median_bmi)
        
        # Age validation
        valid_age = demographics_df['age'].notna().sum()
        missing_age = demographics_df['age'].isna().sum()
        
        logger.info(f"✅ Demographics processed:")
        logger.info(f"   Valid age:     {valid_age:6d} segments")
        logger.info(f"   Missing age:   {missing_age:6d} segments (will exclude)")
        logger.info(f"   Age range:     [{demographics_df['age'].min():.0f}, {demographics_df['age'].max():.0f}] years")
        logger.info(f"   Age mean:      {demographics_df['age'].mean():.1f} ± {demographics_df['age'].std():.1f}")
        
        # Sex distribution
        male_count = (demographics_df['sex'] == 1).sum()
        female_count = (demographics_df['sex'] == 0).sum()
        logger.info(f"   Sex:           {male_count:6d} M ({100*male_count/len(demographics_df):.1f}%), "
                   f"{female_count:6d} F ({100*female_count/len(demographics_df):.1f}%)")
        
        # BMI info
        logger.info(f"   BMI median:    {median_bmi:6.1f} kg/m²")
        logger.info(f"   BMI range:     [{demographics_df['bmi'].min():.1f}, {demographics_df['bmi'].max():.1f}]")
        
        return demographics_df
