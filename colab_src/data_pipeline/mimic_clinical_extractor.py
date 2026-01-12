"""
Load, validate, and structure MIMIC-III clinical tables.
Handles missing data, type conversions, and basic validation.

Caching: Intermediate tables cached to avoid repeated CSV loading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MIMICClinicalExtractor:
    """
    Extract and validate MIMIC-III clinical tables.
    
    Loads: PATIENTS, ADMISSIONS, DIAGNOSES_ICD
    Outputs: Cleaned dataframes with standardized schemas
    
    Tables:
    - patients_df: [subject_id, dob, gender]
    - admissions_df: [subject_id, hadm_id, admittime, dischtime]
    - diagnoses_df: [hadm_id, icd9_code, seq_num]
    """
    
    def __init__(self, mimic_clinical_dir: Path, cache_dir: Optional[Path] = None):
        """
        Args:
            mimic_clinical_dir: Path to folder with PATIENTS.csv, ADMISSIONS.csv, DIAGNOSES_ICD.csv
            cache_dir: Optional path to cache intermediate dataframes
        """
        self.clinical_dir = Path(mimic_clinical_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.patients_df = None
        self.admissions_df = None
        self.diagnoses_df = None
        
        logger.info(f"MIMICClinicalExtractor initialized")
        logger.info(f"  Clinical dir: {self.clinical_dir}")
        logger.info(f"  Cache dir: {self.cache_dir}")
    
    def load_clinical_tables(self, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load MIMIC clinical tables with caching.
        
        Args:
            use_cache: If True, check cache before loading CSVs
            
        Returns:
            Tuple of (patients_df, admissions_df, diagnoses_df)
        """
        cache_file = self.cache_dir / 'mimic_clinical_tables.pkl'
        
        # Try cache first
        if use_cache and cache_file.exists():
            logger.info(f"📦 Loading from cache: {cache_file}")
            cached_data = pd.read_pickle(cache_file)
            self.patients_df = cached_data['patients']
            self.admissions_df = cached_data['admissions']
            self.diagnoses_df = cached_data['diagnoses']
            
            logger.info(f"✅ Loaded from cache:")
            logger.info(f"   PATIENTS: {len(self.patients_df)} records")
            logger.info(f"   ADMISSIONS: {len(self.admissions_df)} records")
            logger.info(f"   DIAGNOSES_ICD: {len(self.diagnoses_df)} records")
            
            return self.patients_df, self.admissions_df, self.diagnoses_df
        
        # Load from CSV
        logger.info(f"📥 Loading MIMIC clinical tables from: {self.clinical_dir}")
        
        # Load PATIENTS
        logger.info("  Loading PATIENTS.csv...")
        patients_path = self.clinical_dir / 'PATIENTS.csv'
        self.patients_df = pd.read_csv(patients_path, usecols=['subject_id', 'dob', 'gender'])
        self.patients_df['dob'] = pd.to_datetime(self.patients_df['dob'])
        
        # Load ADMISSIONS
        logger.info("  Loading ADMISSIONS.csv...")
        admissions_path = self.clinical_dir / 'ADMISSIONS.csv'
        self.admissions_df = pd.read_csv(
            admissions_path,
            usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime']
        )
        self.admissions_df['admittime'] = pd.to_datetime(self.admissions_df['admittime'])
        self.admissions_df['dischtime'] = pd.to_datetime(self.admissions_df['dischtime'])
        
        # Load DIAGNOSES_ICD
        logger.info("  Loading DIAGNOSES_ICD.csv...")
        diagnoses_path = self.clinical_dir / 'DIAGNOSES_ICD.csv'
        self.diagnoses_df = pd.read_csv(
            diagnoses_path,
            usecols=['hadm_id', 'icd9_code', 'seq_num']
        )
        
        # Cache for future runs
        if use_cache:
            logger.info(f"💾 Caching tables...")
            pd.to_pickle({
                'patients': self.patients_df,
                'admissions': self.admissions_df,
                'diagnoses': self.diagnoses_df
            }, cache_file)
            logger.info(f"   Cached to: {cache_file}")
        
        logger.info(f"✅ Clinical tables loaded:")
        logger.info(f"   PATIENTS: {len(self.patients_df)} records")
        logger.info(f"   ADMISSIONS: {len(self.admissions_df)} records")
        logger.info(f"   DIAGNOSES_ICD: {len(self.diagnoses_df)} records")
        
        return self.patients_df, self.admissions_df, self.diagnoses_df
    
    def validate_clinical_tables(self) -> bool:
        """
        Validate data quality and log issues.
        
        Returns:
            True if all validations pass, False otherwise
        """
        if self.patients_df is None or self.admissions_df is None or self.diagnoses_df is None:
            raise RuntimeError("Clinical tables not loaded. Call load_clinical_tables() first.")
        
        logger.info("\n🔍 Validating clinical tables...")
        issues = []
        
        # Check PATIENTS
        if self.patients_df['dob'].isnull().sum() > 0:
            msg = f"⚠️  PATIENTS.dob: {self.patients_df['dob'].isnull().sum()} nulls"
            issues.append(msg)
            logger.warning(msg)
        
        if self.patients_df['gender'].isnull().sum() > 0:
            msg = f"⚠️  PATIENTS.gender: {self.patients_df['gender'].isnull().sum()} nulls"
            issues.append(msg)
            logger.warning(msg)
        
        # Check ADMISSIONS
        if self.admissions_df['admittime'].isnull().sum() > 0:
            msg = f"⚠️  ADMISSIONS.admittime: {self.admissions_df['admittime'].isnull().sum()} nulls"
            issues.append(msg)
            logger.warning(msg)
        
        if self.admissions_df['dischtime'].isnull().sum() > 0:
            msg = f"⚠️  ADMISSIONS.dischtime: {self.admissions_df['dischtime'].isnull().sum()} nulls"
            issues.append(msg)
            logger.warning(msg)
        
        # Check DIAGNOSES_ICD
        if self.diagnoses_df['icd9_code'].isnull().sum() > 0:
            msg = f"⚠️  DIAGNOSES_ICD.icd9_code: {self.diagnoses_df['icd9_code'].isnull().sum()} nulls"
            issues.append(msg)
            logger.warning(msg)
        
        # Check for data consistency
        unique_subjects_patients = self.patients_df['subject_id'].nunique()
        unique_subjects_admissions = self.admissions_df['subject_id'].nunique()
        
        if unique_subjects_admissions > unique_subjects_patients:
            msg = f"⚠️  ADMISSIONS has subjects not in PATIENTS ({unique_subjects_admissions} vs {unique_subjects_patients})"
            issues.append(msg)
            logger.warning(msg)
        
        if len(issues) == 0:
            logger.info("✅ All validations passed")
            return True
        else:
            logger.warning(f"❌ {len(issues)} validation issues found")
            return False
    
    def get_summary_statistics(self) -> dict:
        """Return summary statistics about loaded tables."""
        if self.patients_df is None:
            raise RuntimeError("Clinical tables not loaded.")
        
        return {
            'n_patients': len(self.patients_df),
            'n_admissions': len(self.admissions_df),
            'n_diagnosis_records': len(self.diagnoses_df),
            'n_unique_patients_in_admissions': self.admissions_df['subject_id'].nunique(),
            'n_unique_diagnoses': self.diagnoses_df['icd9_code'].nunique(),
            'date_range_admissions': (
                self.admissions_df['admittime'].min(),
                self.admissions_df['admittime'].max()
            )
        }
