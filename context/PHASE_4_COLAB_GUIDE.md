# Phase 4 Implementation Guide: Running the Colab Notebook

**TL;DR**: Everything is ready. Open the Colab notebook and run cells in order. Checkpoints will be saved to Google Drive.

---

## Pre-Requisites ✅

Before running the Colab notebook, verify:

1. **GitHub repository exists** ✅

   - Your fork or copy should be at a GitHub URL
   - Must be public or have read access from Colab

2. **Local validation passed** ✅

   ```bash
   python validate_phase4.py
   # Expected: All 6 checks pass in <5 seconds
   ```

3. **Google Drive available** ✅

   - You have a Google account
   - Enough storage for checkpoints (~100MB-500MB per epoch)

4. **Colab access** ✅
   - Free tier is sufficient (T4 GPU, 12hr timeout)
   - Or Colab Pro (longer training, priority queue)

---

## Step-by-Step: Running Phase 5 Training

### **Step 1: Open Colab Notebook**

Option A: From GitHub

```
1. Go to: https://github.com/YOUR_ORG/cardiometabolic-risk-colab
2. Navigate to: notebooks/05_ssl_pretraining_colab.ipynb
3. Click "Open in Colab" button (appears in GitHub UI)
```

Option B: Direct Colab Link

```
https://colab.research.google.com/github/YOUR_ORG/cardiometabolic-risk-colab/blob/main/notebooks/05_ssl_pretraining_colab.ipynb
```

---

### **Step 2: Setup (Cells 1-4)**

**Cell 1**: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

- Click popup link to authorize
- Paste authorization code
- Expected: "Mounted at /content/drive"

**Cell 2**: Clone Repository

```python
# Clones from GitHub
# Expected: "✅ Repo already present" or "✅ Repo cloned"
```

**Cell 3**: Install Dependencies

```bash
!pip install -q -r requirements.txt
```

- Takes ~2 minutes
- Expected: "✅ Dependencies installed"

**Cell 4**: Verify GPU

```python
!nvidia-smi --query-gpu=name --format=csv,noheader
import torch
print(f"✅ GPU available: {torch.cuda.is_available()}")
```

- Expected: GPU name (e.g., "Tesla T4"), CUDA available

---

### **Step 3: Run Training (Cell 5)**

This is the long-running cell. Expected duration: **8–12 hours**

```python
# Creates output directory
# Prints config summary
# Starts training for 50 epochs
```

**What to expect**:

- Epoch 1: ~10 minutes (first epoch is slower due to setup)
- Epochs 2-50: ~9-10 minutes each
- **Total**: 8-12 hours

**Loss values over time**:

- Epoch 1: Total loss ~1.2-1.5
- Epoch 25: Total loss ~0.1-0.3 (learning)
- Epoch 50: Total loss ~0.01-0.05 (convergence)

**Handling timeouts**:

- Colab free tier: 12-hour timeout
- If training exceeds 12 hours: **Upgrade to Colab Pro** (unlimited)
- Or: Reduce epochs to 30 instead of 50

**Monitoring**:

- Checkpoints saved every epoch automatically
- Metrics saved to `metrics.json`
- Best checkpoint: `best_encoder.pt`

---

### **Step 4: Validate & Plot (Cells 6-7)**

**Cell 6**: Load best checkpoint and plot loss curves

```python
# Loads best_encoder.pt
# Plots training + validation loss
# Saves figure to Drive
```

**Cell 7**: Reconstruction quality check

```python
# Validates that latent→reconstruction works
# Checks loss values
```

---

## Troubleshooting

### **Problem: "Permission denied" when mounting Drive**

**Solution**: Click the authorization popup link and paste the code back

### **Problem: "ModuleNotFoundError: No module named 'torch'"**

**Solution**: Re-run Cell 3 (pip install -q -r requirements.txt)

### **Problem: GPU not available ("CUDA available: False")**

**Solution**:

1. Go to Runtime → Change runtime type
2. Select GPU (T4 recommended)
3. Restart kernel
4. Re-run cells

### **Problem: "Timeout" after 12 hours**

**Solution**:

- Colab free tier has 12-hour limit
- Upgrade to Colab Pro ($9.99/month)
- Or reduce epochs from 50 to 30 in Cell 5

### **Problem: "GitHub clone failed"**

**Solution**: Check that repo URL is correct and repository is public (or add SSH key)

### **Problem: Out of memory error**

**Solution**:

- Reduce batch size in config
- Or wait for free GPU to become available

---

## After Training Completes ✅

Once Cell 5 finishes:

1. **Checkpoints are in Drive** ✅

   - Location: `/content/drive/MyDrive/cardiometabolic-risk-colab/phase5_checkpoints/`
   - Files:
     - `best_encoder.pt` — Best checkpoint (smallest validation loss)
     - `latest_encoder.pt` — Last checkpoint
     - `training_metrics.json` — Loss curves
     - `loss_curves.png` — Plotted curves

2. **Verify training success** ✅

   - Check `loss_curves.png` for convergence
   - Final validation loss should be <0.01
   - SSIM should be >0.80

3. **Next phase: Phase 6** ⏭️
   - Extract embeddings + evaluate with linear probe
   - Create new Colab notebook or Python script
   - Load `best_encoder.pt` from Drive

---

## Alternative: Run Locally (CPU)

If you want to test without Colab:

```bash
# On your machine (requires GPU for reasonable speed)
python colab_src/models/ssl/train.py \
    --config configs/ssl_pretraining.yaml \
    --device cuda \
    --num_epochs 50 \
    --output_dir ./checkpoints/phase5
```

**Expected**:

- GPU: ~8-12 hours (same as Colab)
- CPU: ~48+ hours (much slower, not recommended)

---

## Key Files Used

| File                                          | Purpose                 |
| --------------------------------------------- | ----------------------- |
| `configs/ssl_pretraining.yaml`                | All hyperparameters     |
| `colab_src/models/ssl/train.py`               | Training entry point    |
| `colab_src/models/ssl/trainer.py`             | Training loop logic     |
| `colab_src/models/ssl/dataloader.py`          | Data loading            |
| `data/processed/ssl_pretraining_data.parquet` | Training data           |
| `data/processed/denoised_signals/`            | Denoised signal targets |

---

## Success Criteria (Phase 5 Gate)

Training is successful if:

1. ✅ Training completes without errors (50 epochs)
2. ✅ Validation loss <0.01 on final epoch
3. ✅ SSIM >0.80 on validation samples
4. ✅ Checkpoint saved to Drive
5. ✅ Loss curves show smooth convergence

---

## Next Steps After Phase 5 ⏭️

Once Phase 5 is complete:

### **Phase 6**: Linear Probe Evaluation

- Load `best_encoder.pt`
- Train linear classification head on 60 labeled samples
- Evaluate on 24 test samples
- Gate: AUROC >0.65 on ≥1 cardiometabolic condition

### **Phase 7**: Extract Embeddings

- Batch-process all 4,417 signals through encoder
- Extract 512-dim latent vectors
- Combine with 39 hand-crafted features

### **Phase 8**: XGBoost Downstream

- Train 4 classifiers (diabetes, hypertension, obesity, CCI)
- Evaluate on 24 test samples
- Export as pickle + ONNX for production API

---

## Getting Help

If something goes wrong:

1. **Check logs**: Cell outputs usually contain error details
2. **Verify GPU**: Run Cell 4 again to confirm CUDA
3. **Re-run setup**: Re-run Cells 1-4 if any import errors
4. **Read docstrings**: Code has comprehensive docstrings
5. **Check GitHub issues**: See if others had same problem

---

## Estimated Timeline

```
Setup (Cells 1-4):           ~5 minutes
Training (Cell 5):           ~8-12 hours ⏳
Validation (Cells 6-7):      ~1 minute
Checkpoints ready:           Done! ✅
```

---

## One More Thing

**Update GitHub URL before running**:

In the Colab notebook Cell 2, replace:

```python
repo_url = "https://github.com/YOUR_ORG/cardiometabolic-risk-colab.git"
```

With your actual GitHub URL. Then run the cell.

---

**Ready to train?** Open the Colab notebook and start! 🚀
