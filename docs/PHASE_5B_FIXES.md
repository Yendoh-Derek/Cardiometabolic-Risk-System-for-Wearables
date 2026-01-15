# Phase 5B: Training Data Fixes

## Problem

Training failed with shape mismatch:

```
RuntimeError: The size of tensor a (1250) must match the size of tensor b (75000)
```

**Root Cause**: Dataloader was loading full denoised signals (75,000 samples) instead of windowed data (1,250 samples) from Phase 5A.

## Solution

### 1. Updated Training Script (`colab_src/models/ssl/train.py`)

- Now passes `windows_array_path` pointing to `mimic_windows.npy` to the dataloader
- Changed from `signal_array_path=None` to `signal_array_path=windows_array_path`
- This tells the dataloader to use the windowed array instead of per-segment files

### 2. Created Window Metadata Converter (`colab_src/data_pipeline/prepare_windowed_ssl_data.py`)

- Converts `mimic_windows_metadata.parquet` (653,716 windows with window IDs)
- Generates new SSL training metadata with window indices as `global_segment_idx`
- Splits windows into train (651K) and val (1.9K) based on quality heuristic
- Output: `ssl_pretraining_data.parquet` and `ssl_validation_data.parquet`

### 3. Enhanced Dataloader (`colab_src/models/ssl/dataloader.py`)

- Added `collate_fn_skip_none()` custom collate function
- Handles quality-filtered samples that return `None` gracefully
- Applied to all dataloaders (train, val, test)

## Verification

**Local Testing** (test_dataloader_windows.py):

- ✅ Loads 1,250-sample windows correctly
- ✅ Batch shapes: (batch_size, 1, 1250)
- ✅ Quality filtering works without crashing
- ✅ Both train and val loaders verified

## Data Pipeline

```
Phase 5A:
  denoised_signals (4,417 files, 75K samples each)
        ↓
  mimic_windows.npy (653,716 × 1,250 windows)
  mimic_windows_metadata.parquet

Phase 5B (FIXED):
  mimic_windows.npy + window metadata
        ↓
  PPGDataset (with signal_array_path=mimic_windows.npy)
        ↓
  Batches: (batch_size, 1, 1250)
        ↓
  Training: Encoder-Decoder SSL
```

## Ready for Colab

All fixes have been:

- ✅ Implemented in source code
- ✅ Tested locally for correctness
- ✅ Ready to push to GitHub and use in Colab

**Next Steps**:

1. Git pull latest changes in Colab notebook
2. Clear old checkpoints (checkpoint cleanup cell)
3. Run Phase 5B training with corrected dataloader
