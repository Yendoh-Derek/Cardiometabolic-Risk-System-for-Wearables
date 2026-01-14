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
    ) -> Tuple[int, int]:
        """
        Generate all overlapping windows from MIMIC signals.
        
        Args:
            output_array_path: Path to save windows.npy [N, 1250]
            output_metadata_path: Path to save windows_metadata.parquet
            quality_metadata_path: Path to original metadata (for SQI, SNR)
        
        Returns:
            (total_windows_generated, total_windows_after_filtering)
        """
        windows_list = []
        metadata_list = []
        
        # Load quality metadata if provided
        quality_df = None
        if quality_metadata_path and Path(quality_metadata_path).exists():
            quality_df = pd.read_parquet(quality_metadata_path)
            logger.info(f"Loaded quality metadata with {len(quality_df)} rows")
        
        total_windows = 0
        window_id = 0
        
        # Iterate through all signals
        for signal_idx, signal_path_rel in tqdm(self.signal_index.items(), desc="Generating windows"):
            try:
                # Load signal
                signal_path = self.signal_dir / signal_path_rel
                if not signal_path.exists():
                    logger.warning(f"Signal file not found: {signal_path}")
                    continue
                
                signal = np.load(signal_path)  # [75000]
                
                # Validate signal
                if len(signal) < self.window_length:
                    logger.warning(f"Signal {signal_idx} too short ({len(signal)} < {self.window_length}), skipping")
                    continue
                
                # Extract subject ID from metadata if available
                subject_id = str(signal_idx).zfill(6)  # e.g., "000000", "004416"
                
                # Get SQI/SNR if available
                sqi_score = 0.8  # Default
                snr_db = 20.0  # Default
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
                    
                    # Validate window
                    if len(window) != self.window_length:
                        continue
                    
                    windows_list.append(window)
                    
                    # Record metadata (STRING subject_id for Phase 8 compatibility)
                    metadata_list.append({
                        'window_id': window_id,
                        'source_signal_id': signal_idx,
                        'subject_id': subject_id,  # STRING: e.g., "000000"
                        'start_sample': start,
                        'sqi_score': sqi_score,
                        'snr_db': snr_db,
                        'is_normalized': False,  # Will be normalized in DataLoader
                    })
                    
                    window_id += 1
                    num_windows_in_signal += 1
                    total_windows += 1
                
                logger.debug(f"Signal {signal_idx}: {num_windows_in_signal} windows extracted")
            
            except Exception as e:
                logger.error(f"Error processing signal {signal_idx}: {e}")
                continue
        
        # Stack windows into array
        windows_array = np.array(windows_list, dtype=np.float32)  # [N, 1250]
        logger.info(f"Generated {windows_array.shape[0]} windows, shape: {windows_array.shape}")
        
        # Save windows array
        output_array_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_array_path, windows_array)
        logger.info(f"Saved windows to {output_array_path}")
        
        # Save metadata parquet
        metadata_df = pd.DataFrame(metadata_list)
        output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_parquet(output_metadata_path)
        logger.info(f"Saved metadata to {output_metadata_path} ({len(metadata_df)} rows)")
        
        # Log quality statistics
        logger.info("=" * 60)
        logger.info("WINDOW GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total windows generated: {total_windows}")
        logger.info(f"Signals processed: {len(metadata_list) / max(1, (total_windows / num_windows_in_signal)) if total_windows > 0 else 0:.0f}")
        logger.info(f"Windows per signal (avg): {total_windows / len(self.signal_index):.1f}")
        logger.info(f"SQI score range: [{metadata_df['sqi_score'].min():.2f}, {metadata_df['sqi_score'].max():.2f}]")
        logger.info(f"SNR range: [{metadata_df['snr_db'].min():.1f}, {metadata_df['snr_db'].max():.1f}] dB")
        logger.info("=" * 60)
        
        return total_windows, len(windows_array)


def main():
    """Main entry point."""
    import sys
    from pathlib import Path
    
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
