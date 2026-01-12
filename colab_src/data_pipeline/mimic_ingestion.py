"""
Optimized MIMIC-III Matched Subset data ingestion with parallel processing.
Key optimization: Use rdheader() for metadata extraction (header-only, no signal download).
"""

import wfdb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from pathlib import Path
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class MIMICDataIngestor:
    def __init__(
        self, database="mimic3wdb-matched", version="1.0", cache_dir="data/cache"
    ):
        self.database = database
        self.version = version
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.target_fs = 125
        self.base_url = f"https://physionet.org/files/{database}/{version}"
        
        # Rate limiting components
        self.rate_limiter = Lock()
        self.last_request_time = [0]
    
    def _rate_limited_request(self, func, *args, **kwargs):
        """
        Ensure minimum 0.05s between requests (20 req/sec max).
        Prevents overwhelming PhysioNet servers with parallel requests.
        """
        with self.rate_limiter:
            elapsed = time.time() - self.last_request_time[0]
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)
            result = func(*args, **kwargs)
            self.last_request_time[0] = time.time()
            return result
    
    def _read_header_with_retry(self, record_name: str, pn_dir: str, max_retries: int = 3):
        """
        Read WFDB header with exponential backoff retry logic.
        
        Args:
            record_name: Record name without path
            pn_dir: PhysioNet directory path
            max_retries: Maximum number of retry attempts
            
        Returns:
            WFDB header object
            
        Raises:
            Exception if all retries fail
        """
        for attempt in range(max_retries):
            try:
                # Rate-limited request
                return self._rate_limited_request(
                    wfdb.rdheader, 
                    record_name, 
                    pn_dir=pn_dir
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(wait)
                else:
                    raise  # Re-raise after final attempt

    def get_all_record_files(self, limit_subjects=None) -> List[str]:
        """
        Two-level RECORDS file reading.
        Returns full paths: p00/p000020/p000020-2183-04-28-17-47
        """
        cache_file = self.cache_dir / "all_record_files.json"

        if cache_file.exists():
            import json
            with open(cache_file, "r") as f:
                record_files = json.load(f)
            print(f"📦 Loaded {len(record_files)} record files from cache")
            return record_files

        print(f"🔍 Step 1: Getting subject directories from main RECORDS...")

        try:
            response = requests.get(f"{self.base_url}/RECORDS", timeout=30)
            response.raise_for_status()
            subject_dirs = [
                line.strip().rstrip("/")
                for line in response.text.strip().split("\n")
                if line.strip() and "/" in line
            ]
            print(f"✅ Found {len(subject_dirs)} subject directories")
        except Exception as e:
            print(f"❌ Failed to download main RECORDS: {e}")
            return []

        if limit_subjects:
            subject_dirs = subject_dirs[:limit_subjects]
            print(f"📊 Limited to first {limit_subjects} subjects")

        print(f"\n🔍 Step 2: Getting record files from each subject...")

        all_record_files = []
        failed_subjects = 0

        for subject_dir in tqdm(subject_dirs, desc="Scanning subjects"):
            try:
                response = requests.get(f"{self.base_url}/{subject_dir}/RECORDS", timeout=10)
                response.raise_for_status()
                record_names = [
                    line.strip()
                    for line in response.text.strip().split("\n")
                    if line.strip()
                ]
                for rec_name in record_names:
                    full_path = f"{subject_dir}/{rec_name}"
                    all_record_files.append(full_path)
                time.sleep(0.1)
            except Exception as e:
                failed_subjects += 1
                if failed_subjects <= 3:
                    print(f"\n⚠️  Failed {subject_dir}: {str(e)[:60]}")
                continue

        print(f"\n✅ Discovery complete")
        print(f"   Total records found: {len(all_record_files)}")
        print(f"   Failed subjects: {failed_subjects}/{len(subject_dirs)}")

        if len(all_record_files) > 0:
            import json
            with open(cache_file, "w") as f:
                json.dump(all_record_files, f)
            print(f"💾 Cached to {cache_file}")

        return all_record_files

    def _parse_record_path(self, full_path: str) -> tuple:
        """Parse full path into components for WFDB."""
        parts = full_path.split("/")
        if len(parts) == 3:
            block, subject_id, record_name = parts
            subject_dir = f"{block}/{subject_id}"
        elif len(parts) == 2:
            subject_dir, record_name = parts
            subject_id = parts[0]
        else:
            subject_dir = "/".join(parts[:-1])
            record_name = parts[-1]
            subject_id = parts[1] if len(parts) > 1 else parts[0]
        
        pn_dir = f"{self.database}/{self.version}/{subject_dir}"
        return record_name, pn_dir, subject_id

    def _extract_single_metadata(self, full_path: str) -> Optional[Dict]:
        """
        Extract metadata from single record using HEADER-ONLY read.
        KEY OPTIMIZATION: rdheader() downloads only ~1KB header, not full signal.
        Includes retry logic and rate limiting for reliability.
        """
        # Skip numerics files
        if full_path.endswith('n'):
            return None
            
        try:
            record_name, pn_dir, subject_id = self._parse_record_path(full_path)
            
            # KEY CHANGE: Use rdheader() with retry logic and rate limiting
            # This downloads ONLY the header file (~1KB) not the signal data (~10-100MB)
            header = self._read_header_with_retry(record_name, pn_dir)
            
            duration_sec = header.sig_len / header.fs if header.fs > 0 else 0
            duration_min = duration_sec / 60
            
            # Check for PPG signals
            sig_names = [s.lower() for s in (header.sig_name or [])]
            has_ppg = any("pleth" in s or "ppg" in s for s in sig_names)
            has_ecg = any("ecg" in s or "ii" in s or "avf" in s for s in sig_names)
            
            return {
                "record_name": full_path,
                "subject_id": subject_id,
                "duration_min": duration_min,
                "duration_sec": duration_sec,
                "has_ppg": has_ppg,
                "has_ecg": has_ecg,
                "sampling_rate": header.fs,
                "n_channels": header.n_sig,
                "signal_names": ",".join(header.sig_name) if header.sig_name else "",
                "base_datetime": str(header.base_datetime) if header.base_datetime else None,
            }
        except Exception as e:
            return {
                "record_name": full_path,
                "error": str(e)[:200],
                "failed": True
            }

    def stream_records_metadata_parallel(
        self, 
        limit_subjects=None,
        max_workers: int = 16,
        checkpoint_every: int = 1000
    ) -> pd.DataFrame:
        """
        OPTIMIZED: Parallel metadata extraction using header-only reads.
        Includes rate limiting (20 req/sec) and retry logic with exponential backoff.
        Expected speedup: 10-20x over sequential rdrecord() approach.
        
        Args:
            limit_subjects: Number of subjects to process (None = all)
            max_workers: Number of parallel workers (recommended: 8-16)
            checkpoint_every: Save progress every N records
            
        Returns:
            DataFrame with metadata for all records
        """
        cache_file = self.cache_dir / "record_metadata.parquet"
        
        if cache_file.exists():
            print(f"📦 Loading cached metadata from {cache_file}")
            return pd.read_parquet(cache_file)
        
        # Get all record files
        record_files = self.get_all_record_files(limit_subjects=limit_subjects)
        if len(record_files) == 0:
            return pd.DataFrame()
        
        print(f"\n🚀 Starting parallel metadata extraction")
        print(f"   Workers: {max_workers}")
        print(f"   Records: {len(record_files)}")
        print(f"   Rate limit: 20 req/sec (with exponential backoff retry)")
        print(f"   Method: rdheader() [header-only, ~1KB per record]\n")
        
        metadata = []
        failed_count = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_record = {
                executor.submit(self._extract_single_metadata, rec): rec 
                for rec in record_files
            }
            
            with tqdm(total=len(record_files), desc="Extracting metadata") as pbar:
                for i, future in enumerate(as_completed(future_to_record), 1):
                    try:
                        result = future.result(timeout=60)  # Increased timeout for retries
                        if result and not result.get('failed'):
                            metadata.append(result)
                        elif result and result.get('failed'):
                            failed_count += 1
                    except Exception as e:
                        failed_count += 1
                    
                    pbar.update(1)
                    
                    # Checkpoint
                    if i % checkpoint_every == 0:
                        df_checkpoint = pd.DataFrame(metadata)
                        checkpoint_path = self.cache_dir / "metadata_checkpoint.parquet"
                        df_checkpoint.to_parquet(checkpoint_path, index=False)
                        
                        elapsed = time.time() - start_time
                        rate = i / elapsed
                        remaining = (len(record_files) - i) / rate / 60
                        pbar.set_postfix({
                            'rate': f'{rate:.1f} rec/s',
                            'failed': failed_count,
                            'eta': f'{remaining:.1f}m'
                        })
        
        if len(metadata) == 0:
            print(f"\n❌ No records successfully processed")
            return pd.DataFrame()
        
        df = pd.DataFrame(metadata)
        
        # Save cache
        df.to_parquet(cache_file, index=False)
        
        elapsed_min = (time.time() - start_time) / 60
        print(f"\n✅ Metadata extraction complete")
        print(f"   Total time: {elapsed_min:.1f} minutes")
        print(f"   Successful: {len(df)}/{len(record_files)}")
        print(f"   Failed: {failed_count}")
        print(f"   Rate: {len(record_files)/elapsed_min:.0f} records/min")
        print(f"\n📊 Signal availability:")
        print(f"   PPG: {df['has_ppg'].sum()} / {len(df)} records")
        print(f"   ECG: {df['has_ecg'].sum()} / {len(df)} records")
        
        return df

    def stream_records_metadata(self, limit=None, limit_subjects=None) -> pd.DataFrame:
        """
        DEPRECATED: Use stream_records_metadata_parallel() instead.
        This method is kept for backward compatibility but redirects to parallel version.
        """
        print("⚠️  Using legacy method - redirecting to optimized parallel version")
        return self.stream_records_metadata_parallel(limit_subjects=limit_subjects)

    def filter_usable_records(
        self, 
        metadata_df: pd.DataFrame, 
        min_duration_min=10
    ) -> pd.DataFrame:
        """Filter records meeting quality criteria."""
        if len(metadata_df) == 0:
            return pd.DataFrame()

        filtered = metadata_df[
            (metadata_df["duration_min"] >= min_duration_min) &
            (metadata_df["has_ppg"] == True)
        ].copy()

        print(f"\n✅ Filtered: {len(filtered)}/{len(metadata_df)} usable records")
        if len(filtered) > 0:
            print(f"\n📊 Duration: Mean={filtered['duration_min'].mean():.1f} min")
            print(f"📊 Sampling rates:\n{filtered['sampling_rate'].value_counts()}")

        return filtered

    def download_signal(self, full_path: str, channel="PLETH") -> Optional[Dict]:
        """Download single record's PPG signal."""
        try:
            record_name, pn_dir, subject_id = self._parse_record_path(full_path)
            record = wfdb.rdrecord(record_name, pn_dir=pn_dir)

            channel_idx = None
            for i, sig_name in enumerate(record.sig_name or []):
                if any(kw in sig_name.lower() for kw in ["pleth", "ppg", "photopleth"]):
                    channel_idx = i
                    break

            if channel_idx is None:
                return None

            return {
                "signal": record.p_signal[:, channel_idx],
                "fs": record.fs,
                "record_name": full_path,
                "subject_id": subject_id,
                "base_datetime": record.base_datetime,
                "channel_name": record.sig_name[channel_idx],
                "units": record.units[channel_idx],
            }
        except Exception as e:
            print(f"❌ Error downloading {full_path}: {str(e)[:100]}")
            return None

    def test_connection(self) -> bool:
        """Test connection by reading subject RECORDS files."""
        try:
            print("🔍 Testing MIMIC-III Matched Subset access...")
            response = requests.get(f"{self.base_url}/RECORDS", timeout=30)
            response.raise_for_status()
            
            subject_dirs = [
                line.strip().rstrip("/")
                for line in response.text.strip().split("\n")
                if line.strip() and "/" in line
            ]
            
            if len(subject_dirs) == 0:
                return False
            
            print(f"✅ Found {len(subject_dirs)} subjects")
            test_subject = subject_dirs[0]
            
            response = requests.get(f"{self.base_url}/{test_subject}/RECORDS", timeout=10)
            response.raise_for_status()
            
            record_names = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
            if len(record_names) == 0:
                return False
            
            full_path = f"{test_subject}/{record_names[0]}"
            record_name, pn_dir, subject_id = self._parse_record_path(full_path)
            
            header = wfdb.rdheader(record_name, pn_dir=pn_dir)
            print(f"✅ Successfully read header")
            print(f"   Signals: {header.sig_name}")
            print(f"   Duration: {header.sig_len / header.fs / 60:.1f} min")
            
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False