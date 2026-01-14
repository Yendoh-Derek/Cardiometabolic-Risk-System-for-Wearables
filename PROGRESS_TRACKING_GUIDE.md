# Progress Tracking Guide for Phase 5B Training

## Overview

The `05_ssl_pretraining_colab.ipynb` notebook now includes comprehensive progress tracking for real-time monitoring of SSL pretraining.

---

## Features Implemented

### 1. **TrainingProgressTracker Class**

Tracks training metrics in real-time with ETAs:

```python
tracker = TrainingProgressTracker(
    total_epochs=50,
    samples_per_epoch=617000,  # Phase 5A windowed data
    batch_size=128
)

tracker.start()  # Start timer
tracker.epoch_start(epoch)  # Mark epoch start
tracker.log_batch(epoch, batch, loss)  # Log batch loss
tracker.epoch_end(epoch, train_loss, val_loss)  # Log epoch completion
tracker.summary()  # Print final summary
```

**Output Example:**

```
🎯 Training started: 2026-01-14 10:30:00
📊 Configuration:
   Total epochs: 50
   Samples/epoch: 617,000
   Batch size: 128
   Batches/epoch: 4,821

   Epoch  0 | Batch    0/4821 | Loss: 0.5238 | ETA: 45m23s
   Epoch  0 | Batch   50/4821 | Loss: 0.4892 | ETA: 38m12s
   ...
✅ Epoch 0/50 completed
   Time: 3m45s | Train loss: 0.3892 | Val loss: 0.3821 🌟 BEST
   Avg epoch time: 225.0s | ETA completion: 3h22m
```

### 2. **Real-Time Monitoring Cell**

Monitor training while it's running without restarting:

```python
def monitor_training_live(log_dir: Path, update_interval: int = 5):
    """
    Live monitoring of training_history.json
    Updates every 5 seconds (configurable)
    Shows loss curves, recent epochs, best model indicators
    """
```

**Features:**

- ✅ Auto-updates every 5 seconds
- ✅ Shows progress bar (████░░░░░░)
- ✅ Displays recent 5 epochs with best model indicators (🌟)
- ✅ Plots loss curves in real-time
- ✅ Shows loss improvement graph
- ✅ Press Ctrl+C to stop

**Output Example:**

```
📚 Training Progress (Last updated: 10:35:42)
======================================================================
Epochs completed: 15/50
Progress: ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
======================================================================

📈 Recent Epochs:
   🌟 Epoch 10: train_loss=0.3521 | val_loss=0.3445
      Epoch 11: train_loss=0.3489 | val_loss=0.3467
      Epoch 12: train_loss=0.3456 | val_loss=0.3498
      Epoch 13: train_loss=0.3421 | val_loss=0.3512
      Epoch 14: train_loss=0.3389 | val_loss=0.3521

📊 Summary:
   Best train loss: 0.3089
   Best val loss: 0.3445
   Best @ epoch: 10
```

### 3. **Comprehensive Visualization**

After training, generate a 6-panel analysis dashboard:

```python
# Plots generated:
1. Main loss curves (train + val) with best model marker
2. Loss improvement percentage over time
3. Train loss per epoch (bar chart)
4. Validation loss per epoch (bar chart)
5. Statistics panel (initial/final/min losses, reduction %)
6. Smoothed loss curves (5-epoch rolling average)
7. Convergence status (converged/converging/improving)
```

**Output:** `artifacts/training_analysis.png` with detailed metrics

---

## How to Use

### During Training

1. **Run training cell:**

   ```python
   # Runs training with progress tracking enabled
   result = subprocess.run(cmd, ...)
   tracker.summary()  # Prints final summary
   ```

2. **In a separate cell, run live monitoring:**

   ```python
   monitor_training_live(
       COLAB_DRIVE_PATH / "logs/ssl",
       update_interval=5  # Check every 5 seconds
   )
   ```

3. **Watch real-time updates** until Ctrl+C

### After Training

1. **View comprehensive analysis:**

   ```python
   # Automatically creates 6-panel visualization
   # Saved to artifacts/training_analysis.png
   # Shows convergence, improvement trends, statistics
   ```

2. **Check metrics:**
   - Training loss reduction percentage
   - Validation loss trajectory
   - Best model epoch
   - Convergence status

---

## Key Metrics Tracked

### Per-Epoch Metrics

- ✅ Train loss
- ✅ Validation loss
- ✅ Best model indicator (🌟)
- ✅ Epoch execution time
- ✅ ETA to completion

### Batch-Level Tracking

- ✅ Batch loss
- ✅ Time per batch
- ✅ ETA for epoch completion
- ✅ Logged every 50 batches

### Summary Statistics

- ✅ Total training time
- ✅ Average epoch time
- ✅ Loss improvement percentage
- ✅ Convergence rate
- ✅ Min/max/final losses

---

## Integration with Trainer

The notebook integrates with `colab_src/models/ssl/trainer.py`:

1. **Auto-checkpoint saves** training history to JSON:

   ```json
   {
     "train_loss": [0.5238, 0.4892, ...],
     "val_loss": [0.5201, 0.4856, ...],
     "best_val_loss": 0.3445,
     "training_time": 14400
   }
   ```

2. **Checkpoint auto-recovery** on timeout:

   - Saves every epoch
   - Resumes from latest checkpoint
   - Progress preserved

3. **Early stopping** based on patience:
   - Monitors validation loss
   - Stops after 15 epochs no improvement
   - Documented in final summary

---

## Example: Full Training Workflow

```python
# 1. Setup tracking
tracker = TrainingProgressTracker(
    total_epochs=50,
    samples_per_epoch=617000,
    batch_size=128
)

# 2. Start training with progress
tracker.start()
result = subprocess.run(training_cmd)
tracker.summary()  # Print final metrics

# 3. Live monitoring (run in separate cell during training)
monitor_training_live(
    COLAB_DRIVE_PATH / "logs/ssl",
    update_interval=5
)

# 4. Post-training analysis (automatic)
# Generates comprehensive visualization
# Shows convergence, improvements, statistics
```

---

## Expected Output Timeline

### Phase 5B Progress

```
10:30 - Training starts
        Batch loss displayed every 50 batches
        ETA updates for each epoch

10:35 - Epoch 0 complete (5 min)
        Train loss: 0.52 → 0.35
        Val loss: 0.51 → 0.34
        ETA: 3h40m

12:00 - Epoch 20 complete (3h30m elapsed)
        Train loss: 0.27
        Val loss: 0.28
        Best @ epoch 15

14:00 - Epoch 40 complete (3h30m)
        Train loss: 0.22
        Val loss: 0.23
        Convergence: STABLE

14:15 - Training complete (3h45m total)
        Final train loss: 0.21
        Final val loss: 0.23
        Improvement: 59% ✅
```

---

## Troubleshooting

### Issue: No output during training

**Solution:** Training runs in subprocess; check `logs/ssl/training.log`

### Issue: Live monitoring shows old data

**Solution:** Wait 5 seconds for auto-update (or refresh manually)

### Issue: History file not found

**Solution:** Ensure `logs/ssl/` directory exists and training is saving checkpoints

### Issue: ETA keeps changing

**Solution:** This is normal during early epochs; stabilizes after epoch 5

---

## Configuration

### Customize Update Interval

```python
# Update every 10 seconds instead of 5
monitor_training_live(
    COLAB_DRIVE_PATH / "logs/ssl",
    update_interval=10
)
```

### Customize Progress Tracker

```python
tracker = TrainingProgressTracker(
    total_epochs=100,  # Adjust epochs
    samples_per_epoch=617000,  # Or 4,417 for old data
    batch_size=64  # Or other batch size
)
```

---

## Files Generated

| File                    | Purpose               | Location           |
| ----------------------- | --------------------- | ------------------ |
| `training_history.json` | Loss per epoch        | `logs/ssl/`        |
| `training_analysis.png` | 6-panel visualization | `artifacts/`       |
| `checkpoint_epoch_N.pt` | Model checkpoints     | `checkpoints/ssl/` |
| `best_encoder.pt`       | Best model            | `checkpoints/ssl/` |

---

## Summary

✅ **Real-time batch & epoch tracking**  
✅ **Live monitoring during training**  
✅ **ETA estimation (updates every epoch)**  
✅ **Automatic progress visualization**  
✅ **Convergence status indicators**  
✅ **Complete training history saved**  
✅ **Auto-checkpoint recovery on timeout**

Ready for Phase 5B on Colab T4 GPU! 🚀
