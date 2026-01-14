# Phase 5B: Quick Reference - Window Generation & Training

## Status: Ready to Execute on Colab T4 GPU

---

## Step 1: Verify Phase 5A Files (5 minutes)

```bash
# Check all critical files exist
ls -la colab_src/models/ssl/encoder.py
ls -la colab_src/models/ssl/decoder.py
ls -la colab_src/models/ssl/config.py
ls -la colab_src/models/ssl/losses.py
ls -la colab_src/models/ssl/dataloader.py
ls -la colab_src/models/ssl/trainer.py
ls -la colab_src/data_pipeline/generate_mimic_windows.py
ls -la configs/ssl_pretraining.yaml
```

## Step 2: Generate Windows (10 minutes on Colab)

```python
from colab_src.data_pipeline.generate_mimic_windows import MIMICWindowGenerator
from pathlib import Path

# Initialize generator
generator = MIMICWindowGenerator(
    denoised_index_path="data/processed/denoised_signal_index.json",
    output_dir="data/processed/",
    window_length=1250,
    stride=500,  # 50% overlap
    sqi_scores_path="data/processed/quality_metadata.parquet"  # optional
)

# Generate windows (produces mimic_windows.npy + mimic_windows_metadata.parquet)
windows, metadata = generator.generate_windows()

print(f"✅ Generated {len(windows)} windows")
print(f"✅ Window shape: {windows.shape}")
print(f"✅ Metadata columns: {metadata.columns.tolist()}")
```

**Expected Output:**

- `data/processed/mimic_windows.npy` [617,000 × 1,250] (Float32)
- `data/processed/mimic_windows_metadata.parquet` with columns:
  - window_id (unique ID)
  - source_signal_id (which of 4,417 signals)
  - subject_id (STRING, e.g., "00001")
  - start_sample (position in original signal)
  - sqi_score (quality metric 0-1)
  - snr_db (signal-to-noise ratio)

---

## Step 3: Run 50-Epoch Training (2.5 hours on Colab T4)

```python
from colab_src.models.ssl.config import SSLConfig
from colab_src.models.ssl.trainer import SSLTrainer
from pathlib import Path

# Load config (includes all critical fixes)
config = SSLConfig.from_yaml("configs/ssl_pretraining.yaml")

# Update for Colab (optional)
config.is_colab = True
config.device = "cuda"
config.checkpoint_dir = Path("/content/checkpoints/ssl")
config.data.data_dir = Path("/content/")  # if data mounted elsewhere

# Initialize trainer
trainer = SSLTrainer(config)

# Train with auto-checkpoint recovery
history = trainer.fit(
    train_loader=None,      # Will load from config.data.train_data_path
    val_loader=None,        # Will load from config.data.val_data_path
    epochs=config.training.num_epochs  # 50
)

# Save final model
trainer.model.save("best_encoder.pt")
print("✅ Training complete! Model saved to best_encoder.pt")
```

**Expected Training Curve:**

- Epoch 1: loss ≈ 0.60
- Epoch 10: loss ≈ 0.40
- Epoch 20: loss ≈ 0.25 (plateau begins)
- Epoch 50: loss ≈ 0.20-0.25 (early stopping may trigger)

**If Colab Times Out:**

- Auto-checkpoint recovery activates automatically
- Resuming from latest checkpoint: `trainer.fit()` detects checkpoint and resumes from epoch+1
- No manual intervention needed!

---

## Step 4: Validate Results (5 minutes)

```python
# Check training artifacts
import json
from pathlib import Path

# Load training history
with open("checkpoints/ssl/training_history.json", "r") as f:
    history = json.load(f)

print(f"✅ Training epochs completed: {len(history['train_loss'])}")
print(f"✅ Final train loss: {history['train_loss'][-1]:.4f}")
print(f"✅ Final val loss: {history['val_loss'][-1]:.4f}")
print(f"✅ Best val loss: {history['best_val_loss']:.4f}")

# Verify model output shape
import torch
from colab_src.models.ssl.config import SSLConfig

config = SSLConfig.from_yaml("configs/ssl_pretraining.yaml")
encoder = config.create_encoder()
encoder.load_state_dict(torch.load("checkpoints/ssl/best_encoder.pt"))

# Test inference
test_input = torch.randn(4, 1, 1250)
with torch.no_grad():
    latent = encoder(test_input)

print(f"✅ Encoder output shape: {latent.shape}")
print(f"✅ Expected: torch.Size([4, 512])")
assert latent.shape == (4, 512), "Output shape mismatch!"
```

---

## Critical Hyperparameters (Don't Change Without Discussion)

| Parameter              | Value | Reason                                     |
| ---------------------- | ----- | ------------------------------------------ |
| `signal_length`        | 1,250 | 10 sec @ 125 Hz (cardiac dynamics)         |
| `num_blocks`           | 3     | Prevents over-compression of short signals |
| `batch_size`           | 128   | 10× faster epochs; fits T4 12GB VRAM       |
| `fft_pad_size`         | 2,048 | 67× faster than 131,072                    |
| `normalize_per_window` | true  | Removes sensor pressure artifacts          |
| `sqi_threshold_train`  | 0.4   | Lenient for SSL robustness                 |
| `sqi_threshold_eval`   | 0.7   | Strict for Phase 8 safety                  |
| `num_epochs`           | 50    | Convergence point for 1,250-sample signals |

---

## Checkpoint Auto-Recovery

If Colab session drops during training:

1. **Auto-detection:** `trainer.find_latest_checkpoint()` searches for `checkpoint_epoch_*.pt`
2. **Automatic Resume:** `fit()` loads latest checkpoint and resumes from epoch+1
3. **No Data Loss:** Training history + best model preserved
4. **Example:**
   ```
   Epoch 1-15: Completed, best_val_loss = 0.35
   [COLAB TIMEOUT]
   Epoch 16: Resuming from checkpoint_epoch_15.pt
   Epoch 16-50: Continue training
   ```

---

## Troubleshooting

### Issue: "No such file or directory: denoised_signal_index.json"

**Solution:** Ensure Phase 0-4 preprocessing complete. Check:

```bash
ls -la data/processed/denoised_signal_index.json
ls -la data/processed/denoised_signals/  # Should have 4,417 files
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch_size:

```python
config.training.batch_size = 64  # from 128
```

### Issue: "Training loss is NaN"

**Solution:** Check normalization:

```python
# Verify per-window normalization is working
from colab_src.models.ssl.dataloader import SSLDataLoader
loader = SSLDataLoader(config)
batch = next(iter(loader))
signals = batch['signal']
print(f"Signal mean: {signals.mean()}, std: {signals.std()}")  # Should be ~0, ~1
```

### Issue: "subject_id is integer, not string"

**Solution:** Check metadata parquet dtype:

```python
import pandas as pd
metadata = pd.read_parquet("data/processed/mimic_windows_metadata.parquet")
print(metadata['subject_id'].dtype)  # Should be 'object' (string)
```

---

## Success Criteria

✅ Window generation produces 617K samples  
✅ Training loss decreases: 0.6 → <0.25  
✅ Validation loss plateaus by epoch 20  
✅ No NaN/Inf during training  
✅ best_encoder.pt saved successfully  
✅ Checkpoint recovery works if interrupted  
✅ subject_id preserved as STRING in metadata

---

## Next Steps After Phase 5B

1. **Phase 6:** Extract 512-dim SSL embeddings
2. **Phase 7:** Combine with 37 classical features (515 total)
3. **Phase 8:** Transfer learning on VitalDB with **subject-level splits**

---

## Command Reference

```bash
# Run tests (local machine)
python -m pytest tests/test_phase5a_comprehensive.py -v

# Import and test encoder
python -c "
from colab_src.models.ssl.config import SSLConfig
config = SSLConfig.from_yaml('configs/ssl_pretraining.yaml')
encoder = config.create_encoder()
print(f'✅ Encoder initialized: {encoder}')
"

# Check config
python -c "
from colab_src.models.ssl.config import SSLConfig
config = SSLConfig.from_yaml('configs/ssl_pretraining.yaml')
print(f'signal_length: {config.data.signal_length}')
print(f'num_blocks: {config.model.num_blocks}')
print(f'batch_size: {config.training.batch_size}')
"
```

---

Ready for Phase 5B execution! 🚀
