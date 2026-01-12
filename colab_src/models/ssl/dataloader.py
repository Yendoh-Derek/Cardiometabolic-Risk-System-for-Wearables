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
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize PPG dataset.
        
        Args:
            metadata_path: Path to metadata parquet file
            signal_array_path: Path to signal numpy array (optional)
            denoised_index_path: Path to denoised signal index JSON
            signal_dir: Directory containing individual signal files
            augmentation: Augmentation callable (optional)
            normalize: Whether to normalize signals to [0, 1]
            device: Device for data ('cpu' or 'cuda')
            load_in_memory: Load all signals into memory
        """
        # Load metadata
        self.metadata_df = pd.read_parquet(metadata_path)
        self.metadata_path = Path(metadata_path)
        self.device = device
        self.augmentation = augmentation
        self.normalize = normalize
        self.load_in_memory = load_in_memory
        
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
        
        logger.info(f"PPGDataset initialized with {len(self)} samples")
    
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
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            (augmented_signal, denoised_signal)
        """
        # Load signals
        raw_signal = self._load_signal(idx)
        denoised_signal = self._load_denoised(idx)
        
        # Normalize
        if self.normalize:
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


def create_dataloaders(
    train_metadata_path: Union[str, Path],
    val_metadata_path: Union[str, Path],
    test_metadata_path: Optional[Union[str, Path]] = None,
    signal_array_path: Optional[Union[str, Path]] = None,
    denoised_index_path: Optional[Union[str, Path]] = None,
    augmentation=None,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: str = 'cpu',
    load_in_memory: bool = False,
):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_metadata_path: Path to training metadata parquet
        val_metadata_path: Path to validation metadata parquet
        test_metadata_path: Path to test metadata parquet (optional)
        signal_array_path: Path to signal array numpy file
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
        denoised_index_path=denoised_index_path,
        augmentation=augmentation,
        normalize=True,
        device=device,
        load_in_memory=load_in_memory,
    )
    
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for stability
    )
    
    # Validation set (no augmentation)
    val_dataset = PPGDataset(
        metadata_path=val_metadata_path,
        signal_array_path=signal_array_path,
        denoised_index_path=denoised_index_path,
        augmentation=None,
        normalize=True,
        device=device,
        load_in_memory=load_in_memory,
    )
    
    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    # Test set (optional, no augmentation)
    if test_metadata_path:
        test_dataset = PPGDataset(
            metadata_path=test_metadata_path,
            signal_array_path=signal_array_path,
            denoised_index_path=denoised_index_path,
            augmentation=None,
            normalize=True,
            device=device,
            load_in_memory=load_in_memory,
        )
        
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    logger.info(f"DataLoaders created:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val:   {len(val_dataset)} samples")
    if test_metadata_path:
        logger.info(f"  Test:  {len(test_dataset)} samples")
    
    return dataloaders
