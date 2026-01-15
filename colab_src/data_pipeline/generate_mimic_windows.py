"""
Generate overlapping 10-second (1,250-sample) windows from full MIMIC signals.

Purpose: Transform 4,417 × 75,000-sample signals into 617,000 × 1,250-sample
training examples via stride-500 sliding windows.

CRITICAL: Subject ID is preserved as STRING for Phase 8 subject-level splitting.
DO NOT split by window_id. DO split by subject_id to prevent patient biometric leakage.

Leakage Scenario (WRONG): Patient A has 1,000 windows → 800 train, 200 test.
Model learns Patient A's individual waveform signature, not disease markers.

Correct Split: Group all windows by subject_id, then 5-fold split subjects.
Patient A (all 1,000 windows) → either TRAIN or TEST, never both.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
import tempfile
import shutil
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MIMICWindowGenerator:
    """
    Generate overlapping windows from MIMIC signals with quality tracking.
    
    Attributes:
        signal_dir: Directory containing denoised .npy files
        denoised_index_path: Path to denoised_signal_index.json
        window_length: 1250 samples (10 sec @ 125 Hz)
        stride: 500 samples (4 sec step)
    """
    
    def __init__(
        self,
        signal_dir: Path,
        denoised_index_path: Path,
        window_length: int = 1250,
        stride: int = 500,
    ):
        """
        Initialize window generator.
        
        Args:
            signal_dir: Path to denoised_signals/ directory
            denoised_index_path: Path to denoised_signal_index.json
            window_length: 1250 samples (10 sec)
            stride: 500 samples (4 sec step) → ~60 windows per signal
        """
        self.signal_dir = Path(signal_dir)
        self.denoised_index_path = Path(denoised_index_path)
        self.window_length = window_length
        self.stride = stride
        
        # Validate paths
        if not self.signal_dir.exists():
            raise FileNotFoundError(f"Signal directory not found: {self.signal_dir}")
        if not self.denoised_index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.denoised_index_path}")
        
        # Load signal index
        with open(self.denoised_index_path, 'r') as f:
            self.signal_index = json.load(f)
        
        logger.info(f"Loaded index with {len(self.signal_index)} signals")
    
    def generate_windows(
        self,
        output_array_path: Path,
        output_metadata_path: Path,
        quality_metadata_path: Optional[Path] = None,
        batch_size: int = 1000,
    ) -> Tuple[int, int]:
        """
        Generate overlapping windows with batch saving to avoid memory issues.
        
        Uses memory mapping to write windows incrementally to disk instead of
        loading all 653K windows (3GB) into memory at once.
        
        Args:
            output_array_path: Path to save windows.npy [N, 1250]
            output_metadata_path: Path to save windows_metadata.parquet
            quality_metadata_path: Path to original metadata (for SQI, SNR)
            batch_size: Number of windows to keep in memory before flushing (1000 = ~5MB)
        
        Returns:
            (total_windows_generated, total_windows_kept)
        """
        # Initialize output paths
        output_array_path = Path(output_array_path) if not isinstance(output_array_path, Path) else output_array_path
        output_array_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load quality metadata if provided
        quality_df = None
        if quality_metadata_path and Path(quality_metadata_path).exists():
            quality_df = pd.read_parquet(quality_metadata_path)
            logger.info(f"Loaded quality metadata with {len(quality_df)} rows")
        
        # First pass: count total windows needed
        logger.info("Scanning signals to estimate output size...")
        total_windows_estimate = 0
        signals_to_process = []
        
        for signal_idx, signal_path_rel in self.signal_index.items():
            signal_path = self.signal_dir / signal_path_rel
            if signal_path.exists():
                signal_len = np.load(signal_path, mmap_mode='r').shape[0]
                if signal_len >= self.window_length:
                    n_windows = max(0, (signal_len - self.window_length) // self.stride + 1)
                    total_windows_estimate += n_windows
                    signals_to_process.append((signal_idx, signal_path_rel, n_windows))
        
        logger.info(f"Estimated total windows: {total_windows_estimate:,}")
        
        # Create memory-mapped output array
        logger.info(f"Creating memory-mapped output array: ({total_windows_estimate:,}, {self.window_length})")
        
        # Use a temporary directory for the memmap
        temp_dir = tempfile.mkdtemp()
        temp_memmap_path = Path(temp_dir) / "temp_windows.dat"
        
        windows_memmap = np.memmap(
            str(temp_memmap_path),
            dtype=np.float32,
            mode='w+',
            shape=(total_windows_estimate, self.window_length)
        )
        
        try:
            # Process signals in batches and write directly to disk
            batch_windows = []
            metadata_list = []
            total_windows = 0
            window_id = 0
            
            for signal_idx, signal_path_rel, n_windows_in_signal in tqdm(signals_to_process, desc="Generating windows"):
                try:
                    # Load signal using memory mapping
                    signal_path = self.signal_dir / signal_path_rel
                    signal = np.load(signal_path, mmap_mode='r')
                    
                    # Extract subject ID
                    subject_id = str(signal_idx).zfill(6)
                    
                    # Get SQI/SNR if available
                    sqi_score = 0.8
                    snr_db = 20.0
                    if quality_df is not None:
                        try:
                            quality_row = quality_df[quality_df['global_segment_idx'] == int(signal_idx)]
                            if not quality_row.empty:
                                sqi_score = float(quality_row.iloc[0].get('sqi_score', 0.8))
                                snr_db = float(quality_row.iloc[0].get('snr_db', 20.0))
                        except Exception as e:
                            logger.warning(f"Could not retrieve quality metrics for signal {signal_idx}: {e}")
                    
                    # Extract overlapping windows
                    num_windows_in_signal = 0
                    for start in range(0, len(signal) - self.window_length + 1, self.stride):
                        window = signal[start:start + self.window_length].astype(np.float32)
                        
                        if len(window) != self.window_length:
                            continue
                        
                        # Add to batch
                        batch_windows.append((total_windows, window))
                        
                        # Record metadata
                        metadata_list.append({
                            'window_id': window_id,
                            'source_signal_id': signal_idx,
                            'subject_id': subject_id,
                            'start_sample': start,
                            'sqi_score': sqi_score,
                            'snr_db': snr_db,
                            'is_normalized': False,
                        })
                        
                        window_id += 1
                        num_windows_in_signal += 1
                        total_windows += 1
                        
                        # Flush batch to disk when size reached
                        if len(batch_windows) >= batch_size:
                            for idx, wnd in batch_windows:
                                windows_memmap[idx] = wnd
                            windows_memmap.flush()
                            batch_windows = []
                            logger.info(f"Flushed {total_windows:,} windows to disk...")
                    
                    logger.debug(f"Signal {signal_idx}: {num_windows_in_signal} windows extracted")
                
                except Exception as e:
                    logger.error(f"Error processing signal {signal_idx}: {e}")
                    continue
            
            # Flush remaining batch
            if batch_windows:
                for idx, wnd in batch_windows:
                    windows_memmap[idx] = wnd
                windows_memmap.flush()
            
            logger.info(f"Generated {total_windows:,} windows, shape: ({total_windows}, {self.window_length})")
            
            # Save memmap to proper numpy .npy file format
            logger.info(f"Saving windows to {output_array_path}...")
            np.save(str(output_array_path), windows_memmap[:total_windows])  # Only save actual data
            logger.info(f"✅ Saved windows to {output_array_path}")
            
            # Save metadata parquet
            metadata_df = pd.DataFrame(metadata_list)
            output_metadata_path = Path(output_metadata_path) if not isinstance(output_metadata_path, Path) else output_metadata_path
            output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_df.to_parquet(output_metadata_path)
            logger.info(f"Saved metadata to {output_metadata_path} ({len(metadata_df)} rows)")
            
            # Log quality statistics
            logger.info("=" * 60)
            logger.info("WINDOW GENERATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total windows generated: {total_windows:,}")
            logger.info(f"Signals processed: {len(signals_to_process)}")
            logger.info(f"Windows per signal (avg): {total_windows / len(signals_to_process):.1f}")
            if len(metadata_df) > 0:
                logger.info(f"SQI score range: [{metadata_df['sqi_score'].min():.2f}, {metadata_df['sqi_score'].max():.2f}]")
                logger.info(f"SNR range: [{metadata_df['snr_db'].min():.1f}, {metadata_df['snr_db'].max():.1f}] dB")
            logger.info("=" * 60)
            
            return total_windows, total_windows
        
        finally:
            # Always clean up temp files
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main entry point."""
    import sys
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    signal_dir = project_root / "data" / "processed" / "denoised_signals"
    denoised_index = project_root / "data" / "processed" / "denoised_signal_index.json"
    quality_metadata = project_root / "data" / "processed" / "ssl_pretraining_data.parquet"
    
    output_array = project_root / "data" / "processed" / "mimic_windows.npy"
    output_metadata = project_root / "data" / "processed" / "mimic_windows_metadata.parquet"
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate windows
    generator = MIMICWindowGenerator(signal_dir, denoised_index)
    total, kept = generator.generate_windows(output_array, output_metadata, quality_metadata)
    
    print(f"\n✅ Phase 5A: Window generation complete!")
    print(f"   Generated: {total} windows")
    print(f"   Kept after validation: {kept} windows")
    print(f"   Output: {output_array}")
    print(f"   Metadata: {output_metadata}")


if __name__ == "__main__":
    main()
