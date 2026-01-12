"""
Complete signal processing pipeline.
Modular processing functions for batch execution.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
from pathlib import Path

class SignalProcessingPipeline:
    """
    End-to-end pipeline: Download → Resample → Filter → Denoise → Segment → SQI
    """
    
    def __init__(self, ingestor, sqi_engine, ppg_filter, denoiser, segmenter):
        self.ingestor = ingestor
        self.sqi_engine = sqi_engine
        self.filter = ppg_filter
        self.denoiser = denoiser
        self.segmenter = segmenter
    
    def process_single_record(self, record_name: str, min_sqi: float = 0.5) -> List[Dict]:
        """Process a single record through full pipeline."""

        # Download signal
        signal_data = self.ingestor.download_signal(record_name)
        if signal_data is None:
            return []

        raw = signal_data["signal"]
        fs_orig = signal_data["fs"]

        # Resample if needed
        if fs_orig != 125:
            raw = self.filter.resample_signal(raw, fs_orig, 125)

        # Filter
        filtered = self.filter.apply(raw)

        # Denoise
        denoised = self.denoiser.denoise(filtered)

        # Segment with SQI gating
        segments_sqi = self.segmenter.segment_with_sqi(
            denoised, self.sqi_engine, min_sqi=min_sqi
        )

        # Package results
        results = []
        for idx, (seg, sqi) in enumerate(segments_sqi):
            results.append(
                {
                    "record_name": record_name,
                    "subject_id": signal_data["subject_id"],
                    "segment_idx": idx,
                    "signal": seg,
                    "fs": 125,
                    "sqi_score": sqi["sqi_score"],
                    "quality_grade": sqi["quality_grade"],
                    "snr_db": sqi["snr_db"],
                    "perfusion_index": sqi["perfusion_index"],
                    "channel_name": signal_data["channel_name"],
                }
            )

        return results
    
    def process_batch(self, record_names: List[str], min_sqi: float = 0.5,
                     show_progress: bool = True) -> List[Dict]:
        """
        Process multiple records.
        
        Args:
            record_names: List of record paths
            min_sqi: Minimum SQI threshold
            show_progress: Show progress bar
            
        Returns:
            List of all accepted segments
        """
        all_segments = []
        failed_records = []
        
        iterator = tqdm(record_names, desc="Processing records") if show_progress else record_names
        
        for record_name in iterator:
            try:
                segments = self.process_single_record(record_name, min_sqi=min_sqi)
                all_segments.extend(segments)
            except Exception as e:
                failed_records.append({'record': record_name, 'error': str(e)})
                if len(failed_records) <= 5:
                    print(f"\n⚠️  Failed {record_name}: {str(e)[:80]}")
        
        print(f"\n✅ Batch processing complete")
        print(f"   Records processed: {len(record_names)}")
        print(f"   Segments accepted: {len(all_segments)}")
        print(f"   Failed records: {len(failed_records)}")
        
        return all_segments
    
    def save_processed_data(self, segments: List[Dict], output_dir: Path,
                           prefix: str = 'processed') -> Dict[str, str]:
        """
        Save processed signals and metadata to disk.
        
        Args:
            segments: List of processed segment dicts
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            Dict with paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if len(segments) == 0:
            print("⚠️  No segments to save")
            return {}
        
        # Separate signals from metadata
        signals = np.array([seg['signal'] for seg in segments])
        metadata = pd.DataFrame([
            {k: v for k, v in seg.items() if k != 'signal'} 
            for seg in segments
        ])
        
        # Save
        signals_path = output_dir / f'{prefix}_signals.npy'
        metadata_path = output_dir / f'{prefix}_metadata.parquet'
        
        np.save(signals_path, signals)
        metadata.to_parquet(metadata_path, index=False, compression='snappy')
        
        print(f"\n💾 Saved processed data:")
        print(f"   Signals: {signals_path}")
        print(f"   Metadata: {metadata_path}")
        print(f"   Shape: {signals.shape}")
        print(f"   Mean SQI: {metadata['sqi_score'].mean():.3f}")
        
        return {
            'signals_path': str(signals_path),
            'metadata_path': str(metadata_path),
            'n_segments': len(segments)
        }
    
    def process_single_record_top_k(self, record_name: str, min_sqi: float = 0.5, 
                                     top_k: int = 3) -> List[Dict]:
        """
        Process a single record and return only top-K highest SQI segments.
        """
        # Get all segments using existing method
        all_segments = self.process_single_record(record_name, min_sqi=min_sqi)
        
        if len(all_segments) == 0:
            return []
        
        # Sort by SQI score descending and take top-K
        sorted_segments = sorted(all_segments, key=lambda x: x['sqi_score'], reverse=True)
        top_segments = sorted_segments[:top_k]
        
        # Re-index segments
        for idx, seg in enumerate(top_segments):
            seg['segment_idx'] = idx
        
        return top_segments

    def process_batch_parallel(self, record_names: List[str], min_sqi: float = 0.5,
                               top_k: int = 3, max_workers: int = 4,
                               show_progress: bool = True) -> List[Dict]:
        """
        Process multiple records IN PARALLEL with top-K SQI filtering.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_segments = []
        failed_records = []
        
        # Wrapper function for parallel execution
        def process_wrapper(record_name):
            try:
                return self.process_single_record_top_k(record_name, min_sqi=min_sqi, top_k=top_k)
            except Exception as e:
                return {'error': record_name, 'message': str(e)}
        
        # Parallel processing with ThreadPoolExecutor (better for I/O-bound tasks like downloads)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_wrapper, rec): rec for rec in record_names}
            
            iterator = tqdm(as_completed(futures), total=len(record_names), 
                           desc="Processing records") if show_progress else as_completed(futures)
            
            for future in iterator:
                result = future.result()
                
                # Check for errors
                if isinstance(result, dict) and 'error' in result:
                    failed_records.append(result)
                    if len(failed_records) <= 5:
                        print(f"\n⚠️  Failed {result['error']}: {result['message'][:80]}")
                else:
                    all_segments.extend(result)
        
        print(f"\n✅ Parallel batch processing complete")
        print(f"   Records processed: {len(record_names)}")
        print(f"   Segments accepted: {len(all_segments)} (top-{top_k} per record)")
        print(f"   Failed records: {len(failed_records)}")
        if len(record_names) > len(failed_records):
            print(f"   Avg segments/record: {len(all_segments)/(len(record_names)-len(failed_records)):.1f}")
        
        return all_segments

    def load_and_filter_batches(self, checkpoint_dir, signals_dir, top_k_per_record=3, min_sqi=0.7):
        """
        Memory-efficient streaming assembly with incremental filtering.
        Processes batches one at a time to avoid RAM overflow.
        """
        from pathlib import Path
        import pandas as pd
        import numpy as np
        import gc
        
        checkpoint_dir = Path(checkpoint_dir)
        signals_dir = Path(signals_dir)
        
        checkpoint_files = sorted(checkpoint_dir.glob('batch_*_metadata.parquet'))
        signals_files = sorted(signals_dir.glob('batch_*_signals.npy'))
        
        print(f"\n🔍 Found {len(checkpoint_files)} checkpoint files")
        print(f"🔍 Found {len(signals_files)} signal files")
        
        # Check for mismatches
        checkpoint_nums = {int(f.stem.split('_')[1]) for f in checkpoint_files}
        signal_nums = {int(f.stem.split('_')[1]) for f in signals_files}
        
        if checkpoint_nums != signal_nums:
            missing_signals = checkpoint_nums - signal_nums
            missing_metadata = signal_nums - checkpoint_nums
            
            if missing_signals:
                print(f"⚠️  Missing signal files for batches: {sorted(missing_signals)}")
            if missing_metadata:
                print(f"⚠️  Missing metadata files for batches: {sorted(missing_metadata)}")
            
            valid_nums = checkpoint_nums & signal_nums
            checkpoint_files = [f for f in checkpoint_files if int(f.stem.split('_')[1]) in valid_nums]
            signals_files = [f for f in signals_files if int(f.stem.split('_')[1]) in valid_nums]
            
            print(f"✅ Proceeding with {len(checkpoint_files)} matching batches")
        
        # ===== PHASE 1: BUILD RECORD-TO-SEGMENTS INDEX (MEMORY EFFICIENT) =====
        print(f"\n📊 Phase 1: Building record index (lightweight)...")
        
        # Collect all record-level statistics without loading signals
        record_segments = {}  # {record_name: [(batch_num, batch_idx, sqi_score), ...]}
        
        for ckpt_file in checkpoint_files:
            try:
                batch_num = int(ckpt_file.stem.split('_')[1])
                metadata = pd.read_parquet(ckpt_file)
                
                # Index each segment by record
                for idx, row in metadata.iterrows():
                    record_name = row['record_name']
                    if record_name not in record_segments:
                        record_segments[record_name] = []
                    
                    record_segments[record_name].append({
                        'batch_num': batch_num,
                        'batch_idx': idx,
                        'sqi_score': row['sqi_score']
                    })
            
            except Exception as e:
                print(f"   ❌ Failed to index {ckpt_file.stem}: {str(e)[:80]}")
                continue
        
        print(f"   ✅ Indexed {len(record_segments)} unique records")
        
        # ===== PHASE 2: SELECT TOP-K SEGMENTS PER RECORD =====
        print(f"\n🎯 Phase 2: Selecting top-{top_k_per_record} segments per record...")
        
        segments_to_keep = []  # [(batch_num, batch_idx), ...]
        
        for record_name, segments in record_segments.items():
            # Sort by SQI and take top-K
            top_segments = sorted(segments, key=lambda x: x['sqi_score'], reverse=True)[:top_k_per_record]
            
            # Keep only if above SQI threshold
            for seg in top_segments:
                if seg['sqi_score'] >= min_sqi:
                    segments_to_keep.append((seg['batch_num'], seg['batch_idx']))
        
        print(f"   ✅ Selected {len(segments_to_keep)} segments total")
        
        # Group by batch for efficient loading
        batch_indices = {}  # {batch_num: [indices]}
        for batch_num, batch_idx in segments_to_keep:
            if batch_num not in batch_indices:
                batch_indices[batch_num] = []
            batch_indices[batch_num].append(batch_idx)
        
        print(f"   📦 Segments span {len(batch_indices)} batches")
        
        # ===== PHASE 3: STREAM AND EXTRACT SELECTED SEGMENTS =====
        print(f"\n📥 Phase 3: Streaming selected segments...")
        
        filtered_metadata_list = []
        filtered_signals_list = []
        
        for batch_num in sorted(batch_indices.keys()):
            indices = batch_indices[batch_num]
            
            # Load only this batch
            ckpt_file = checkpoint_dir / f'batch_{batch_num:03d}_metadata.parquet'
            sig_file = signals_dir / f'batch_{batch_num:03d}_signals.npy'
            
            try:
                metadata = pd.read_parquet(ckpt_file)
                signals = np.load(sig_file)
                
                # Extract only selected indices
                selected_metadata = metadata.iloc[indices].copy()
                selected_signals = signals[indices]
                
                filtered_metadata_list.append(selected_metadata)
                filtered_signals_list.append(selected_signals)
                
                print(f"   ✅ Batch {batch_num:03d}: extracted {len(indices)} segments")
                
                # CRITICAL: Free memory immediately
                del metadata, signals, selected_metadata, selected_signals
                gc.collect()
            
            except Exception as e:
                print(f"   ❌ Failed batch {batch_num:03d}: {str(e)[:80]}")
                continue
        
        # ===== PHASE 4: FINAL ASSEMBLY =====
        print(f"\n🔗 Phase 4: Final assembly...")
        
        filtered_metadata = pd.concat(filtered_metadata_list, ignore_index=True)
        filtered_signals = np.concatenate(filtered_signals_list, axis=0)
        
        # Clear intermediate data
        del filtered_metadata_list, filtered_signals_list, record_segments, batch_indices
        gc.collect()
        
        # Reset indices
        filtered_metadata['global_segment_idx'] = range(len(filtered_metadata))
        filtered_metadata = filtered_metadata.reset_index(drop=True)
        
        # Statistics
        stats = {
            'total_segments': len(filtered_signals),
            'unique_subjects': filtered_metadata['subject_id'].nunique(),
            'unique_records': filtered_metadata['record_name'].nunique(),
            'mean_sqi': filtered_metadata['sqi_score'].mean(),
            'median_sqi': filtered_metadata['sqi_score'].median(),
            'min_sqi': filtered_metadata['sqi_score'].min(),
            'max_sqi': filtered_metadata['sqi_score'].max(),
            'size_mb': filtered_signals.nbytes / 1e6,
            'reduction_pct': 100.0  # Will calculate if we had raw count
        }
        
        print(f"\n📊 Final dataset statistics:")
        print(f"   Total segments: {stats['total_segments']:,}")
        print(f"   Unique subjects: {stats['unique_subjects']}")
        print(f"   Unique records: {stats['unique_records']}")
        print(f"   Mean SQI: {stats['mean_sqi']:.3f}")
        print(f"   Size: {stats['size_mb']:.1f} MB")
        
        return {
            'signals': filtered_signals,
            'metadata': filtered_metadata,
            'stats': stats
        }
