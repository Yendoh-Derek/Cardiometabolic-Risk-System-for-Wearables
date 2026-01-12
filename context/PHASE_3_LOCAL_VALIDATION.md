# Phase 3: Local CPU Validation

## Overview

Phase 3 is a **pilot training run** on CPU to validate the end-to-end SSL pipeline before deploying to Colab GPU.

**Goal**: Train encoder-decoder for 2-3 epochs on Phase 0 data to verify:

- ✅ Data loading works correctly
- ✅ Training loop executes without errors
- ✅ Loss decreases (convergence signal)
- ✅ Memory usage is acceptable
- ✅ Checkpoints save/load properly

**Duration**: ~30-60 minutes (2-3 epochs on CPU)

**Status**: Ready to start

---

## Prerequisites

All complete from Phase 0 and Phase 1:

- ✅ Phase 0 data: 4,417 signals split into train/val/test

  - `data/processed/ssl_pretraining_data.parquet` (4,133 training)
  - `data/processed/ssl_validation_data.parquet` (200 validation)
  - `data/processed/ssl_test_data.parquet` (84 test)
  - `data/processed/denoised_signals/` (4,417 .npy files)

- ✅ Phase 1 components: All 9 SSL modules ready

  - Encoder, Decoder, Loss functions, Augmentation, DataLoader
  - All 39 tests passing

- ✅ Environment: Python + PyTorch installed and configured

---

## Running Phase 3

### Quick Start (Default: 2 epochs)

```bash
python phase3_local_validation.py
```

This will:

- Train for 2 epochs on CPU
- Use batch size 8
- Log metrics after each epoch
- Save checkpoints and metrics

### Custom Configuration

```bash
# Train for 5 epochs
python phase3_local_validation.py --epochs 5

# Use larger batch size (if memory allows)
python phase3_local_validation.py --batch-size 16

# Force GPU (if available)
python phase3_local_validation.py --device cuda

# Full example
python phase3_local_validation.py --epochs 3 --batch-size 8 --device auto
```

### Expected Output

```
================================================================================
Phase 3: Local CPU Validation - SSL Pretraining Pilot Run
================================================================================
Device: cpu

Config loaded from defaults
  Model: latent_dim=512, bottleneck=768
  Training: batch_size=8, epochs=2
  Loss: MSE=0.5, SSIM=0.3, FFT=0.2

Initializing models...
Encoder parameters: 254,976
Decoder parameters: 254,976
Total parameters: 509,952

Loading Phase 0 data...
Train samples: 4133
Val samples: 200
Test samples: 84

Initializing trainer...

Starting training for 2 epochs...
================================================================================

Epoch 1/2
  Batch 10/517: Loss = 0.452341
  Batch 20/517: Loss = 0.389023
  ...
Train Loss: 0.325642
Val Loss:   0.298574
Best Val:   0.298574
Checkpoint saved to checkpoints/phase3/checkpoint_epoch_01.pt

Epoch 2/2
  Batch 10/517: Loss = 0.289234
  Batch 20/517: Loss = 0.276123
  ...
Train Loss: 0.268342
Val Loss:   0.251234
Best Val:   0.251234
Checkpoint saved to checkpoints/phase3/checkpoint_epoch_02.pt

================================================================================
Training Complete!
================================================================================

Training Summary:
  Train Loss: 0.268342 → 0.325642 (decreasing ✅)
  Val Loss:   0.251234 → 0.298574 (decreasing ✅)
  Best Val:   0.251234

Checkpoints saved to: checkpoints/phase3/
```

---

## What Gets Saved

### Checkpoints

```
checkpoints/phase3/
├── checkpoint_best.pt          # Best validation loss model
├── checkpoint_epoch_01.pt      # After epoch 1
├── checkpoint_epoch_02.pt      # After epoch 2
└── metrics.json                # Training metrics summary
```

Each checkpoint contains:

- Encoder state dict
- Decoder state dict
- Optimizer state
- Training/validation loss history

### Metrics

`metrics.json` contains:

```json
{
  "train_losses": [0.325642, 0.268342],
  "val_losses": [0.298574, 0.251234],
  "best_val_loss": 0.251234,
  "timestamp": "2024-01-15T14:23:45.123456",
  "device": "cpu",
  "epochs": 2,
  "batch_size": 8,
  "config": {
    "model": {
      "latent_dim": 512,
      "bottleneck_dim": 768
    },
    "loss_weights": {
      "mse": 0.5,
      "ssim": 0.3,
      "fft": 0.2
    }
  }
}
```

---

## Success Criteria

Phase 3 is successful if ALL of these are true:

- ✅ **No errors**: Training completes without exceptions
- ✅ **Convergence**: Loss decreases from epoch 1 to epoch 2
  - Expected: ~10-20% reduction in validation loss
- ✅ **Memory**: CPU memory usage stays < 5GB
  - Monitor with `task manager` or `top`
- ✅ **Speed**: Epoch completes in < 5 minutes
  - With 4,133 training samples + 200 val samples
- ✅ **Checkpoints**: All saves complete without IO errors
- ✅ **Reproducibility**: Can load checkpoint and resume training

### Expected Loss Values

With Phase 0 data and Phase 1 components:

- **Initial train loss**: 0.35-0.45 (first batch)
- **End of epoch 1**: 0.25-0.35 (after 517 batches)
- **End of epoch 2**: 0.20-0.30 (further convergence)

If loss increases or stays flat → indicates data/model mismatch

---

## Troubleshooting

### Out of Memory

If you get OOM error:

```bash
# Reduce batch size
python phase3_local_validation.py --batch-size 4
```

### Slow Training

CPU training on 4,133 samples will be slow. Expected timeline:

- Epoch 1: 2-3 minutes
- Epoch 2: 2-3 minutes
- Total: 5-10 minutes for 2 epochs

To speed up locally:

```bash
# Use fewer samples (data-level, not recommended)
# Or just accept the timeline for this pilot run
```

### Data Not Found

If you see `FileNotFoundError` for Phase 0 data:

```
Check that these exist:
- data/processed/ssl_pretraining_data.parquet
- data/processed/ssl_validation_data.parquet
- data/processed/denoised_signals/*.npy
```

If missing, run Phase 0 again:

```bash
jupyter notebook notebooks/05_ssl_data_preparation.ipynb
```

### Import Errors

If you see import errors:

```bash
# Make sure colab_src is in Python path
# The script adds it automatically, but verify:
python -c "import sys; print(sys.path)"
```

---

## Next Steps After Phase 3

Once Phase 3 is successful:

1. **Review metrics**: Check convergence curves and loss values
2. **Create validation notebook**: `notebooks/06_phase3_validation.ipynb`
   - Load checkpoint_best.pt
   - Evaluate on test set
   - Visualize latent representations
3. **Estimate training time**: Scale up to full 50 epochs on Colab T4
   - 2 epochs on CPU ≈ X minutes
   - 50 epochs on T4 ≈ Y minutes (much faster)
4. **Push to GitHub**: Commit Phase 0-3 outputs
5. **Deploy to Colab**: Upload checkpoints and run full training

---

## Architecture Summary

For reference during Phase 3:

**Encoder** (ResNetEncoder):

- Input: [B, 75000] (PPG signal)
- Output: [B, 512] (latent representation)
- Processing: 1D convolutions with stride-2 downsampling

**Decoder** (ResNetDecoder):

- Input: [B, 512] (latent)
- Output: [B, 75000] (reconstructed signal)
- Processing: Transpose convolutions to upsample

**Loss Function** (SSLLoss):

- MSE: 0.5 weight (reconstruction error)
- SSIM: 0.3 weight (perceptual similarity)
- FFT: 0.2 weight (frequency domain)
- Total: Weighted composite loss

**Augmentation** (PPGAugmentation):

- Temporal shift: ±10%
- Amplitude scale: 0.85-1.15×
- Baseline wander: 0.2 Hz sinusoid
- Noise injection: 40% probability, SNR-matched

---

## Configuration Details

Current default config:

```yaml
model:
  in_channels: 1 # PPG is 1D signal
  latent_dim: 512 # Bottleneck output dimension
  bottleneck_dim: 768 # Internal bottleneck width
  num_blocks: 4 # ResNet blocks
  base_filters: 64 # Initial conv filters
  max_filters: 256 # Maximum filters in conv layers

training:
  batch_size: 8
  num_epochs: 50 # Full training, but Phase 3 uses --epochs 2
  learning_rate: 0.001
  weight_decay: 0.0001
  gradient_accumulation_steps: 4

loss:
  mse_weight: 0.5
  ssim_weight: 0.3
  fft_weight: 0.2

augmentation:
  temporal_shift_range: 0.1
  amplitude_scale_range: [0.85, 1.15]
  baseline_wander_freq: 0.2
  noise_prob: 0.4
```

---

## Files Involved

**Phase 3 Script**:

- `phase3_local_validation.py` - Main training script (this directory)

**Phase 0 Dependencies**:

- `data/processed/ssl_pretraining_data.parquet` - Training metadata
- `data/processed/ssl_validation_data.parquet` - Validation metadata
- `data/processed/denoised_signals/` - Ground truth signals (4,417 files)

**Phase 1 Dependencies**:

- `colab_src/models/ssl/config.py` - SSLConfig
- `colab_src/models/ssl/encoder.py` - ResNetEncoder
- `colab_src/models/ssl/decoder.py` - ResNetDecoder
- `colab_src/models/ssl/losses.py` - SSLLoss, component losses
- `colab_src/models/ssl/augmentation.py` - PPGAugmentation
- `colab_src/models/ssl/dataloader.py` - PPGDataset, create_dataloaders

**Output**:

- `checkpoints/phase3/checkpoint_best.pt` - Best model
- `checkpoints/phase3/metrics.json` - Training metrics

---

## Version Info

- **PyTorch**: 2.0+
- **Python**: 3.8+
- **CUDA**: Not required (CPU validation)
- **Project**: cardiometabolic-risk-colab
- **Phase**: 3/5

---

Created: 2024
Phase Status: Ready to Execute
