"""
Link waveform segments to clinical admissions (Option B: Same HADM_ID).
Handles temporal validation and parallel matching with rate limiting.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from joblib import Parallel, delayed
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class WaveformClinicalLinker:
    """
    Link signal segments to hospital admissions.
    
    Strategy (Option B):
    - Each segment is linked to an admission by SUBJECT_ID
    - If signal timestamp available: validate temporal match (admittime ≤ signal_time ≤ dischtime)
    - If timestamp missing: use latest admission (conservative approach)
    
    Uses parallel processing with joblib for efficient matching.
    """
    
    def __init__(self, n_jobs: int = -1, batch_size: int = 1000):
        """
        Args:
            n_jobs: Number of parallel jobs (-1 = use all CPUs)
            batch_size: Process segments in batches to reduce memory spike
        """
        self.n_jobs = n_jobs
        self.batch_size = batch_size
    
    def link_segments_to_admissions(
        self,
        signal_metadata_df: pd.DataFrame,
        admissions_df: pd.DataFrame,
        n_jobs: Optional[int] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Link each signal segment to its clinical admission.
        
        Args:
            signal_metadata_df: Segments with [segment_id, subject_id, ...]
            admissions_df: Admissions with [subject_id, hadm_id, admittime, dischtime]
            n_jobs: Override default n_jobs for this call
            show_progress: Show progress bar
            
        Returns:
            signal_metadata_df with added 'hadm_id' column
        """
        
        n_jobs = n_jobs or self.n_jobs
        logger.info(f"🔗 Linking {len(signal_metadata_df)} segments to admissions (n_jobs={n_jobs})")
        
        # Prepare lookup dictionary: subject_id -> sorted admissions
        admissions_by_subject = {}
        for subject_id in admissions_df['subject_id'].unique():
            subject_admissions = admissions_df[
                admissions_df['subject_id'] == subject_id
            ].sort_values('admittime')
            
            admissions_by_subject[subject_id] = subject_admissions.to_dict('records')
        
        # Process in batches to control memory usage
        result_df = signal_metadata_df.copy()
        n_batches = (len(signal_metadata_df) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"   Processing {n_batches} batches of size {self.batch_size}...")
        
        hadm_ids = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(signal_metadata_df))
            
            batch = signal_metadata_df.iloc[start_idx:end_idx]
            
            def link_segment(row):
                """Link a single segment to an admission."""
                subject_id = row['subject_id']
                
                if subject_id not in admissions_by_subject:
                    return None  # Subject not in clinical data
                
                subject_admissions = admissions_by_subject[subject_id]
                
                # If no timestamp: use latest admission (conservative)
                if 'signal_time' not in row or pd.isna(row.get('signal_time')):
                    return subject_admissions[-1]['hadm_id']
                
                # If timestamp exists: find matching admission
                signal_time = row['signal_time']
                
                for admission in subject_admissions:
                    if admission['admittime'] <= signal_time <= admission['dischtime']:
                        return admission['hadm_id']
                
                # Signal outside admission windows - use latest (conservative)
                logger.warning(
                    f"Segment {row['segment_id']}: signal_time {signal_time} "
                    f"outside admission windows, using latest admission"
                )
                return subject_admissions[-1]['hadm_id']
            
            # Parallel linking within batch
            batch_hadm_ids = Parallel(n_jobs=n_jobs)(
                delayed(link_segment)(row) for _, row in batch.iterrows()
            )
            
            hadm_ids.extend(batch_hadm_ids)
        
        # Add hadm_id to result
        result_df['hadm_id'] = hadm_ids
        
        # Count results
        linked = result_df['hadm_id'].notna().sum()
        unlinked = result_df['hadm_id'].isna().sum()
        
        logger.info(f"✅ Linking complete:")
        logger.info(f"   Linked:   {linked:6d} segments")
        logger.info(f"   Unlinked: {unlinked:6d} segments (orphans)")
        
        return result_df
