"""
PPG dataset loader for self-supervised learning.

Loads signal metadata and waveforms from parquet files with lazy loading.
Supports Colab (via Drive) and local paths with automatic detection.
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List
import logging

logger = logging.getLogger(__name__)


def collate_fn_skip_none(batch: List) -> Tuple:
    """
    Custom collate function that skips samples returning None.
    Used for quality/SQI filtering where some samples may be excluded.
    
    Args:
        batch: List of (signal, target) tuples, some may be (None, None)
    
    Returns:
        (signals_tensor, targets_tensor) with None samples removed
    """
    # Filter out None samples
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    
    if len(batch) == 0:
        # Return empty tensors if all samples were filtered
        return torch.zeros(0, 1, 1250), torch.zeros(0, 1, 1250)
    
    # Stack remaining samples
    signals = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    
    return signals, targets


class PPGDataset(Dataset):
    """
    PyTorch Dataset for PPG signals with lazy loading.
    
    Features:
    - Lazy loads waveforms (memory efficient)
    - Supports both numpy arrays and per-segment files
    - Colab-aware path handling
    - Optional augmentation on-the-fly
    """
    
    def __init__(
        self,
        metadata_path: Union[str, Path],
        signal_array_path: Optional[Union[str, Path]] = None,
        denoised_index_path: Optional[Union[str, Path]] = None,
        signal_dir: Optional[Union[str, Path]] = None,
        augmentation=None,
        normalize: bool = True,
        device: str = 'cpu',
        load_in_memory: bool = False,
        normalize_per_window: bool = True,  # Phase 5A: Z-score normalization
        normalization_epsilon: float = 1e-8,  # Phase 5A: epsilon for variance
        min_std_threshold: float = 1e-5,  # Phase 5A: drop dead sensors
        sqi_threshold: float = 0.4,  # Phase 5A: filter by quality
    ):
        """
        Initialize PPG dataset.
        
        Args:
            metadata_path: Path to metadata parquet file
            signal_array_path: Path to signal numpy array (optional)
            denoised_index_path: Path to denoised signal index JSON
            signal_dir: Directory containing individual signal files
            augmentation: Augmentation callable (optional)
            normalize: Whether to normalize signals to [0, 1] (legacy, prefer per-window)
            device: Device for data ('cpu' or 'cuda')
            load_in_memory: Load all signals into memory
            normalize_per_window: Phase 5A - per-window Z-score normalization
            normalization_epsilon: Epsilon for variance calculation
            min_std_threshold: Skip windows with std < this (dead sensors)
            sqi_threshold: Skip windows with SQI < this (quality filtering)
        """
        self.metadata_path = Path(metadata_path)
        
        # Validate metadata file exists
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        # Load metadata with error handling
        try:
            self.metadata_df = pd.read_parquet(self.metadata_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet metadata from {self.metadata_path}: {e}")
        
        # Validate required columns
        required_columns = ['global_segment_idx']
        missing_columns = [col for col in required_columns if col not in self.metadata_df.columns]
        if missing_columns:
            raise ValueError(
                f"Metadata missing required columns: {missing_columns}. "
                f"Available columns: {list(self.metadata_df.columns)}"
            )
        
        self.device = device
        self.augmentation = augmentation
        self.normalize = normalize
        self.load_in_memory = load_in_memory
        
        # Phase 5A: New normalization parameters
        self.normalize_per_window = normalize_per_window
        self.normalization_epsilon = normalization_epsilon
        self.min_std_threshold = min_std_threshold
        self.sqi_threshold = sqi_threshold
        
        # Handle paths
        self.project_root = self.metadata_path.parent.parent.parent
        self.signal_array_path = signal_array_path
        self.denoised_index_path = Path(denoised_index_path) if denoised_index_path else None
        self.signal_dir = Path(signal_dir) if signal_dir else None
        
        # Load signal array if available
        self.signal_array = None
        if signal_array_path and Path(signal_array_path).exists():
            try:
                self.signal_array = np.load(signal_array_path)
                logger.info(f"Loaded signal array: {self.signal_array.shape}")
            except Exception as e:
                logger.warning(f"Failed to load signal array: {e}")
        
        # Load denoised signal index if available
        self.denoised_index = {}
        if self.denoised_index_path and self.denoised_index_path.exists():
            try:
                with open(self.denoised_index_path, 'r') as f:
                    self.denoised_index = json.load(f)
                logger.info(f"Loaded denoised signal index: {len(self.denoised_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load denoised index: {e}")
        
        # Load all signals if requested
        self.signals_cache = {}
        self.denoised_cache = {}
        if load_in_memory:
            self._load_all_signals()
        
        logger.info(f"PPGDataset initialized: {len(self)} samples from {self.metadata_path}")
    
    def _load_all_signals(self):
        """Pre-load all signals into memory."""
        logger.info("Loading all signals into memory...")
        for i in range(len(self)):
            self.signals_cache[i] = self._load_signal(i)
            self.denoised_cache[i] = self._load_denoised(i)
    
    def _load_signal(self, idx: int) -> np.ndarray:
        """Load raw signal for a single segment."""
        if idx in self.signals_cache:
            return self.signals_cache[idx]
        
        # Try signal array first
        if self.signal_array is not None:
            return self.signal_array[idx].astype(np.float32)
        
        # Try per-segment file
        segment_id = self.metadata_df.iloc[idx]['global_segment_idx']
        
        if self.signal_dir and self.signal_dir.exists():
            signal_file = self.signal_dir / f"{segment_id:06d}.npy"
            if signal_file.exists():
                return np.load(signal_file).astype(np.float32)
        
        # Try batch files
        batch_dir = self.project_root / "data" / "processed" / "signal_batches"
        if batch_dir.exists():
            batch_files = list(batch_dir.glob("batch_*.npy"))
            if batch_files:
                # This is a fallback; in practice, we'd need proper indexing
                raise NotImplementedError("Batch file loading requires proper indexing")
        
        raise FileNotFoundError(f"Signal not found for segment {segment_id}")
    
    def _load_denoised(self, idx: int) -> np.ndarray:
        """Load denoised (ground truth) signal."""
        if idx in self.denoised_cache:
            return self.denoised_cache[idx]
        
        segment_id = self.metadata_df.iloc[idx]['global_segment_idx']
        segment_id_str = str(int(segment_id))
        
        if segment_id_str in self.denoised_index:
            denoised_path = self.project_root / self.denoised_index[segment_id_str]
            if denoised_path.exists():
                return np.load(denoised_path).astype(np.float32)
        
        # If denoised not available, return original (will learn identity)
        return self._load_signal(idx)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with Phase 5A enhancements.
        
        Args:
            idx: Sample index
        
        Returns:
            (augmented_signal, denoised_signal) — both normalized to (μ=0, σ=1)
        
        Phase 5A Additions:
            - Per-window Z-score normalization (prevents sensor artifact learning)
            - Dead sensor filtering (skip if std < 1e-5)
            - SQI filtering (skip if SQI < threshold)
        """
        # Load signals
        raw_signal = self._load_signal(idx)
        denoised_signal = self._load_denoised(idx)
        
        # Phase 5A: Check SQI threshold (quality filtering)
        if self.sqi_threshold > 0 and 'sqi_score' in self.metadata_df.columns:
            sqi = self.metadata_df.iloc[idx]['sqi_score']
            if sqi < self.sqi_threshold:
                # Return None to signal DataLoader to skip this sample
                return None, None
        
        # Phase 5A: Per-window Z-score normalization (critical for PPG)
        # Standardizes μ=0, σ=1 regardless of sensor pressure/hardware
        if self.normalize_per_window:
            raw_signal = self._normalize_per_window(raw_signal)
            denoised_signal = self._normalize_per_window(denoised_signal)
            
            # Dead sensor filtering: skip if std < threshold
            if raw_signal is None or denoised_signal is None:
                return None, None
        elif self.normalize:
            # Legacy: min-max normalization
            raw_signal = self._normalize(raw_signal)
            denoised_signal = self._normalize(denoised_signal)
        
        # Apply augmentation
        if self.augmentation is not None:
            raw_signal = self.augmentation.compose(raw_signal)
        
        # Convert to torch and reshape
        raw_signal = torch.from_numpy(raw_signal).float().unsqueeze(0)  # (1, length)
        denoised_signal = torch.from_numpy(denoised_signal).float().unsqueeze(0)  # (1, length)
        
        return raw_signal, denoised_signal
    
    @staticmethod
    def _normalize(signal: np.ndarray) -> np.ndarray:
        """Normalize signal to [0, 1]."""
        signal_min = signal.min()
        signal_max = signal.max()
        if signal_max > signal_min:
            return (signal - signal_min) / (signal_max - signal_min)
        else:
            return signal - signal_min
    
    def _normalize_per_window(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """
        Phase 5A: Per-window Z-score normalization.
        
        Formula: (x - μ) / (σ + ε) where ε = 1e-8 (inside denominator)
        
        Returns:
            Normalized signal (μ≈0, σ≈1), or None if std < min_std_threshold (dead sensor)
        """
        mean = signal.mean()
        std = signal.std()
        
        # Dead sensor check: skip if std < threshold
        if std < self.min_std_threshold:
            logger.debug(f"Dead sensor detected (std={std:.2e} < {self.min_std_threshold:.2e}), skipping window")
            return None
        
        # Z-score with epsilon inside denominator (gradient-safe)
        normalized = (signal - mean) / (std + self.normalization_epsilon)
        
        return normalized.astype(np.float32)


def create_dataloaders(
    train_metadata_path: Union[str, Path],
    val_metadata_path: Union[str, Path],
    test_metadata_path: Optional[Union[str, Path]] = None,
    signal_array_path: Optional[Union[str, Path]] = None,
    signal_dir: Optional[Union[str, Path]] = None,
    denoised_index_path: Optional[Union[str, Path]] = None,
    augmentation=None,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: str = 'cpu',
    load_in_memory: bool = False,
    normalize_per_window: bool = True,
    normalization_epsilon: float = 1e-8,
    min_std_threshold: float = 1e-5,
    sqi_threshold_train: float = 0.4,
    sqi_threshold_eval: float = 0.7,
):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_metadata_path: Path to training metadata parquet
        val_metadata_path: Path to validation metadata parquet
        test_metadata_path: Path to test metadata parquet (optional)
        signal_array_path: Path to signal array numpy file
        signal_dir: Directory containing individual signal files
        denoised_index_path: Path to denoised signal index JSON
        augmentation: Augmentation callable
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
        device: Device ('cpu' or 'cuda')
        load_in_memory: Load all signals into memory
    
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders
    """
    dataloaders = {}
    
    # Training set (with augmentation)
    train_dataset = PPGDataset(
        metadata_path=train_metadata_path,
        signal_array_path=signal_array_path,
        signal_dir=signal_dir,
        denoised_index_path=denoised_index_path,
        augmentation=augmentation,
        normalize=True,
        device=device,
        load_in_memory=load_in_memory,
        normalize_per_window=normalize_per_window,
        normalization_epsilon=normalization_epsilon,
        min_std_threshold=min_std_threshold,
        sqi_threshold=sqi_threshold_train,
    )
    
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for stability
        collate_fn=collate_fn_skip_none,  # Skip quality-filtered samples
    )
    
    # Validation set (no augmentation)
    val_dataset = PPGDataset(
        metadata_path=val_metadata_path,
        signal_array_path=signal_array_path,
        signal_dir=signal_dir,
        denoised_index_path=denoised_index_path,
        augmentation=None,
        normalize=True,
        device=device,
        load_in_memory=load_in_memory,
        normalize_per_window=normalize_per_window,
        normalization_epsilon=normalization_epsilon,
        min_std_threshold=min_std_threshold,
        sqi_threshold=sqi_threshold_eval,
    )
    
    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_skip_none,  # Skip quality-filtered samples
    )
    
    # Test set (optional, no augmentation)
    if test_metadata_path:
        test_dataset = PPGDataset(
            metadata_path=test_metadata_path,
            signal_array_path=signal_array_path,
            signal_dir=signal_dir,
            denoised_index_path=denoised_index_path,
            augmentation=None,
            normalize=True,
            device=device,
            load_in_memory=load_in_memory,
            normalize_per_window=normalize_per_window,
            normalization_epsilon=normalization_epsilon,
            min_std_threshold=min_std_threshold,
            sqi_threshold=sqi_threshold_eval,
        )
        
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn_skip_none,  # Skip quality-filtered samples
        )
    
    logger.info(f"DataLoaders created:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val:   {len(val_dataset)} samples")
    if test_metadata_path:
        logger.info(f"  Test:  {len(test_dataset)} samples")
    
    return dataloaders
