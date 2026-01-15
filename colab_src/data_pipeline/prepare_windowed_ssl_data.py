"""
Convert Phase 5A windowed data into SSL training/validation metadata.

Creates ssl_pretraining_data.parquet and ssl_validation_data.parquet
that reference window indices instead of segment indices.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_windowed_ssl_data(
    data_dir: Path,
    output_dir: Path = None
):
    """
    Convert mimic_windows_metadata.parquet into windowed SSL train/val split.
    
    Args:
        data_dir: Directory containing mimic_windows_metadata.parquet
        output_dir: Output directory (defaults to data_dir)
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load window metadata
    windows_meta_path = data_dir / "mimic_windows_metadata.parquet"
    if not windows_meta_path.exists():
        raise FileNotFoundError(f"Window metadata not found: {windows_meta_path}")
    
    logger.info(f"Loading window metadata: {windows_meta_path}")
    windows_df = pd.read_parquet(windows_meta_path)
    logger.info(f"  Loaded: {len(windows_df)} windows")
    
    # Verify required columns
    required_cols = ['window_id', 'source_signal_id', 'subject_id']
    missing = [col for col in required_cols if col not in windows_df.columns]
    if missing:
        raise ValueError(
            f"Window metadata missing columns: {missing}. "
            f"Available: {list(windows_df.columns)}"
        )
    
    # Get unique subjects and their statistics
    all_subjects = windows_df['subject_id'].unique()
    logger.info(f"\nTotal unique subjects in windows: {len(all_subjects)}")
    
    # Try to load existing train/val split from notebook execution
    # Fall back to quality-based split if metadata files are empty
    train_meta_path = Path(data_dir) / "ssl_pretraining_data.parquet"
    val_meta_path = Path(data_dir) / "ssl_validation_data.parquet"
    
    train_subjects = set()
    val_subjects = set()
    
    if train_meta_path.exists():
        train_old = pd.read_parquet(train_meta_path)
        if len(train_old) > 0 and 'subject_id' in train_old.columns:
            train_subjects = set(train_old['subject_id'].unique())
            logger.info(f"Loaded train subjects from existing metadata: {len(train_subjects)}")
    
    if val_meta_path.exists():
        val_old = pd.read_parquet(val_meta_path)
        if len(val_old) > 0 and 'subject_id' in val_old.columns:
            val_subjects = set(val_old['subject_id'].unique())
            logger.info(f"Loaded val subjects from existing metadata: {len(val_subjects)}")
    
    # If split not found in metadata files, use quality-based heuristic
    # (117 high-quality train, 13 low-quality val)
    if not train_subjects or not val_subjects:
        logger.info(f"\nMissing or empty split metadata. Using quality-based heuristic...")
        
        # Group by subject and calculate average SQI
        subject_stats = windows_df.groupby('subject_id').agg({
            'sqi_score': 'mean',
            'window_id': 'count'
        }).rename(columns={'window_id': 'window_count'})
        subject_stats = subject_stats.sort_values('sqi_score', ascending=False)
        
        # Take top 117 subjects (highest SQI) for train, rest for val
        num_subjects = len(subject_stats)
        num_train = max(int(num_subjects * 0.9), num_subjects - 13)  # ~90% train, ~10% val
        
        train_subjects = set(subject_stats.index[:num_train])
        val_subjects = set(subject_stats.index[num_train:])
        
        logger.info(f"Split based on SQI:")
        logger.info(f"  Train: {len(train_subjects)} subjects (highest quality)")
        logger.info(f"  Val:   {len(val_subjects)} subjects (lowest quality)")
    
    # Filter windows by subject split
    train_df = windows_df[windows_df['subject_id'].isin(train_subjects)].copy()
    val_df = windows_df[windows_df['subject_id'].isin(val_subjects)].copy()
    
    logger.info(f"\nSplit distribution:")
    logger.info(f"  Train: {len(train_df)} windows from {len(train_df['subject_id'].unique())} subjects")
    logger.info(f"  Val:   {len(val_df)} windows from {len(val_df['subject_id'].unique())} subjects")
    
    # Create metadata for training with window indices
    # The dataloader will use 'global_segment_idx' to index into mimic_windows.npy
    train_meta = pd.DataFrame({
        'global_segment_idx': train_df['window_id'].values,
        'window_id': train_df['window_id'].values,
        'source_signal_id': train_df['source_signal_id'].values,
        'subject_id': train_df['subject_id'].values,
    })
    
    val_meta = pd.DataFrame({
        'global_segment_idx': val_df['window_id'].values,
        'window_id': val_df['window_id'].values,
        'source_signal_id': val_df['source_signal_id'].values,
        'subject_id': val_df['subject_id'].values,
    })
    
    # Add optional columns from original window metadata
    optional_cols = ['sqi_score', 'start_sample', 'snr_db']
    for col in optional_cols:
        if col in windows_df.columns:
            train_meta[col] = train_df[col].values
            val_meta[col] = val_df[col].values
    
    # Save metadata
    train_out_path = output_dir / "ssl_pretraining_data.parquet"
    val_out_path = output_dir / "ssl_validation_data.parquet"
    
    train_meta.to_parquet(train_out_path)
    val_meta.to_parquet(val_out_path)
    
    logger.info(f"\n✅ SSL training metadata created:")
    logger.info(f"  {train_out_path} ({len(train_meta)} rows)")
    logger.info(f"  {val_out_path} ({len(val_meta)} rows)")
    
    # Verify the array dimensions match
    windows_array_path = data_dir / "mimic_windows.npy"
    if windows_array_path.exists():
        windows_memmap = np.load(windows_array_path, mmap_mode='r')
        max_idx_train = train_meta['global_segment_idx'].max()
        max_idx_val = val_meta['global_segment_idx'].max()
        max_idx = max(max_idx_train, max_idx_val)
        
        logger.info(f"\n✅ Validation:")
        logger.info(f"  Windows array shape: {windows_memmap.shape}")
        logger.info(f"  Max window index in metadata: {max_idx}")
        
        if max_idx >= windows_memmap.shape[0]:
            raise ValueError(
                f"Window index out of bounds: max={max_idx}, "
                f"but array has {windows_memmap.shape[0]} rows"
            )
        logger.info(f"  ✓ All indices within bounds")
    
    return train_meta, val_meta


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prepare_windowed_ssl_data.py <data_dir>")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    prepare_windowed_ssl_data(data_dir)
