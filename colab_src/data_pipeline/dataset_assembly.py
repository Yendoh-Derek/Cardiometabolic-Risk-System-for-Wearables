"""
Assemble final signal-clinical integrated dataset.
Merge all components and validate completeness.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatasetAssembler:
    """
    Merge all intermediate datasets into one cohesive parquet.
    Final validation and schema enforcement.
    
    Output Schema:
    segment_id, subject_id, hadm_id, record_name, channel_name,
    sqi_score, snr_db, perfusion_index, quality_grade,
    age, sex, bmi, is_bmi_missing, cci_score,
    diabetes, hypertension, obesity
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Args:
            output_dir: Directory for output parquet files
        """
        self.output_dir = Path(output_dir) if output_dir else Path('data/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def assemble(
        self,
        signal_metadata_df: pd.DataFrame,
        linked_segments_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        demographics_df: pd.DataFrame,
        cci_df: pd.DataFrame,
        output_filename: str = 'signal_clinical_integrated_dataset.parquet'
    ) -> pd.DataFrame:
        """
        Merge all components into final dataset.
        
        Args:
            signal_metadata_df: Segments with SQI scores (from Sprint 1)
            linked_segments_df: Segments with hadm_id (from linker)
            labels_df: Labels per hadm_id (diabetes, hypertension, obesity)
            demographics_df: Demographics per segment (age, sex, bmi, flags)
            cci_df: Charlson Index per hadm_id
            output_filename: Output parquet filename
            
        Returns:
            Final assembled dataset
        """
        
        logger.info(f"\n🔀 ASSEMBLING FINAL DATASET")
        logger.info(f"   Input segments: {len(signal_metadata_df)}")
        
        # Start with signal metadata
        dataset = signal_metadata_df.copy()
        
        # Merge hadm_id from linker
        dataset = dataset.merge(
            linked_segments_df[['segment_id', 'hadm_id']],
            on='segment_id',
            how='left'
        )
        logger.info(f"   After hadm_id merge: {len(dataset)} segments")
        
        # Merge labels (by hadm_id)
        dataset = dataset.merge(
            labels_df,
            on='hadm_id',
            how='left'
        )
        logger.info(f"   After labels merge: {len(dataset)} segments")
        
        # Merge demographics (by segment_id)
        dataset = dataset.merge(
            demographics_df[['segment_id', 'age', 'sex', 'bmi', 'is_bmi_missing']],
            on='segment_id',
            how='left'
        )
        logger.info(f"   After demographics merge: {len(dataset)} segments")
        
        # Merge CCI (by hadm_id)
        dataset = dataset.merge(
            cci_df,
            on='hadm_id',
            how='left'
        )
        logger.info(f"   After CCI merge: {len(dataset)} segments")
        
        # Remove rows with missing critical data
        logger.info(f"\n📋 QUALITY FILTERING")
        logger.info(f"   Before filtering: {len(dataset)} rows")
        
        # Mandatory columns
        dataset = dataset[dataset['age'].notna()]
        logger.info(f"   After age filter: {len(dataset)} rows")
        
        dataset = dataset[dataset['hadm_id'].notna()]
        logger.info(f"   After hadm_id filter: {len(dataset)} rows")
        
        dataset = dataset[dataset['diabetes'].notna()]
        logger.info(f"   After labels filter: {len(dataset)} rows")
        
        dataset = dataset[dataset['sex'].notna()]
        logger.info(f"   After sex filter: {len(dataset)} rows")
        
        # Verify schema
        expected_columns = {
            'segment_id', 'subject_id', 'hadm_id', 'record_name', 'channel_name',
            'sqi_score', 'snr_db', 'perfusion_index', 'quality_grade',
            'age', 'sex', 'bmi', 'is_bmi_missing', 'cci_score',
            'diabetes', 'hypertension', 'obesity'
        }
        
        actual_columns = set(dataset.columns)
        missing_cols = expected_columns - actual_columns
        extra_cols = actual_columns - expected_columns
        
        if missing_cols:
            logger.warning(f"⚠️  Missing columns: {missing_cols}")
        
        if extra_cols:
            logger.info(f"   Extra columns (kept): {extra_cols}")
        
        # Log quality report
        self._log_quality_report(dataset)
        
        # Save to parquet
        output_path = self.output_dir / output_filename
        dataset.to_parquet(output_path, index=False)
        logger.info(f"\n✅ Dataset saved to: {output_path}")
        logger.info(f"   Size: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
        
        return dataset
    
    def _log_quality_report(self, dataset: pd.DataFrame):
        """Log comprehensive data quality report."""
        
        logger.info(f"\n📊 DATASET QUALITY REPORT")
        logger.info(f"   Total segments:        {len(dataset):,}")
        logger.info(f"   Unique subjects:       {dataset['subject_id'].nunique():,}")
        logger.info(f"   Unique admissions:     {dataset['hadm_id'].nunique():,}")
        
        # Demographics
        logger.info(f"\n   👤 DEMOGRAPHICS:")
        age_stats = dataset['age'].describe()
        logger.info(f"      Age (years):")
        logger.info(f"        mean ± std:      {age_stats['mean']:.1f} ± {age_stats['std']:.1f}")
        logger.info(f"        range:           [{age_stats['min']:.0f}, {age_stats['max']:.0f}]")
        logger.info(f"        median:          {dataset['age'].median():.0f}")
        
        male = (dataset['sex'] == 1).sum()
        female = (dataset['sex'] == 0).sum()
        logger.info(f"      Sex:")
        logger.info(f"        Male:            {male:6d} ({100*male/len(dataset):5.1f}%)")
        logger.info(f"        Female:          {female:6d} ({100*female/len(dataset):5.1f}%)")
        
        bmi_stats = dataset['bmi'].describe()
        logger.info(f"      BMI (kg/m²):")
        logger.info(f"        mean ± std:      {bmi_stats['mean']:.1f} ± {bmi_stats['std']:.1f}")
        logger.info(f"        range:           [{bmi_stats['min']:.1f}, {bmi_stats['max']:.1f}]")
        logger.info(f"        imputed:         {dataset['is_bmi_missing'].sum():6d} ({100*dataset['is_bmi_missing'].mean():5.1f}%)")
        
        logger.info(f"      Comorbidity (CCI):")
        cci_stats = dataset['cci_score'].describe()
        logger.info(f"        mean ± std:      {cci_stats['mean']:.2f} ± {cci_stats['std']:.2f}")
        logger.info(f"        range:           [{cci_stats['min']:.0f}, {cci_stats['max']:.0f}]")
        logger.info(f"        median:          {dataset['cci_score'].median():.0f}")
        
        # Labels
        logger.info(f"\n   🏥 CARDIOMETABOLIC LABELS (MULTI-HOT):")
        logger.info(f"      Diabetes:")
        logger.info(f"        positive:        {dataset['diabetes'].sum():6d} ({100*dataset['diabetes'].mean():5.1f}%)")
        logger.info(f"      Hypertension:")
        logger.info(f"        positive:        {dataset['hypertension'].sum():6d} ({100*dataset['hypertension'].mean():5.1f}%)")
        logger.info(f"      Obesity:")
        logger.info(f"        positive:        {dataset['obesity'].sum():6d} ({100*dataset['obesity'].mean():5.1f}%)")
        
        # Multi-morbidity
        condition_count = dataset[['diabetes', 'hypertension', 'obesity']].sum(axis=1)
        multi_morbid = (condition_count > 1).sum()
        healthy = (condition_count == 0).sum()
        single_condition = (condition_count == 1).sum()
        
        logger.info(f"      Morbidity patterns:")
        logger.info(f"        Healthy (0,0,0):    {healthy:6d} ({100*healthy/len(dataset):5.1f}%)")
        logger.info(f"        Single condition:   {single_condition:6d} ({100*single_condition/len(dataset):5.1f}%)")
        logger.info(f"        Multi-morbid (>1):  {multi_morbid:6d} ({100*multi_morbid/len(dataset):5.1f}%)")
        
        # Signal quality
        logger.info(f"\n   📡 SIGNAL QUALITY:")
        logger.info(f"      SQI score:")
        sqi_stats = dataset['sqi_score'].describe()
        logger.info(f"        mean ± std:      {sqi_stats['mean']:.3f} ± {sqi_stats['std']:.3f}")
        logger.info(f"        range:           [{sqi_stats['min']:.3f}, {sqi_stats['max']:.3f}]")
        
        logger.info(f"      SNR (dB):")
        snr_stats = dataset['snr_db'].describe()
        logger.info(f"        mean ± std:      {snr_stats['mean']:.1f} ± {snr_stats['std']:.1f}")
        
        quality_dist = dataset['quality_grade'].value_counts()
        logger.info(f"      Quality grade distribution:")
        for grade in ['Excellent', 'Good', 'Fair', 'Poor']:
            if grade in quality_dist.index:
                count = quality_dist[grade]
                logger.info(f"        {grade:10s}:      {count:6d} ({100*count/len(dataset):5.1f}%)")
        
        # Validation checks
        logger.info(f"\n   ✓ VALIDATION CHECKS:")
        checks_pass = True
        
        # Check 1: Minimum subjects
        if dataset['subject_id'].nunique() >= 20:
            logger.info(f"      ✅ Subjects: {dataset['subject_id'].nunique()} >= 20")
        else:
            logger.warning(f"      ❌ Subjects: {dataset['subject_id'].nunique()} < 20")
            checks_pass = False
        
        # Check 2: Minimum segments
        if len(dataset) >= 200:
            logger.info(f"      ✅ Segments: {len(dataset)} >= 200")
        else:
            logger.warning(f"      ❌ Segments: {len(dataset)} < 200")
            checks_pass = False
        
        # Check 3: Label diversity
        for condition in ['diabetes', 'hypertension', 'obesity']:
            count = dataset[condition].sum()
            if count >= 10:
                logger.info(f"      ✅ {condition}: {count} positive examples")
            else:
                logger.warning(f"      ❌ {condition}: {count} positive examples (need >= 10)")
                checks_pass = False
        
        if checks_pass:
            logger.info(f"\n   ✅ ALL VALIDATION CHECKS PASSED!")
        else:
            logger.warning(f"\n   ⚠️  SOME VALIDATION CHECKS FAILED")
