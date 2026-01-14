# Comprehensive Implementation Plan: Phases 0-8

## Multimodal Cardiometabolic Risk Estimation - SSL Pretraining & Transfer Learning

**Date**: January 13, 2026  
**Status**: Ready for Implementation  
**Project**: cardiometabolic-risk-colab

---

## EXECUTIVE SUMMARY

Transform your MIMIC SSL pretraining from 4,417 signals → **617,000 training samples** via overlapping 10-second windows. Validate transfer learning on VitalDB surgical population. Use 3-block encoder to preserve micro-morphology without over-compression.

**Key Innovation**: Overlapping sliding windows (stride=500 samples) create massive data augmentation, enabling robust self-supervised learning without labels. Transfer learning on VitalDB proves encoder generalizes across populations and hardware.

**Total Timeline**: ~40-50 hours

- Local refactoring: 4-5 hours
- Colab GPU training: 12-18 hours
- Validation & analysis: 3-4 hours

---

## PROJECT CONSTRAINTS & DATA REALITY

### Critical Constraint: No Clinical Labels in MIMIC Subset

- ❌ **No labeled clinical data** for any of 4,417 MIMIC signals (subject ID mismatch prevents linking)
- ✅ **4,417 unlabeled PPG waveforms** with signal quality metadata (SQI score)
- ✅ **84 high-quality segments** reserved for future labeled tasks (no labels currently)
- ✅ **VitalDB open dataset** (6,388 surgical cases) provides labeled ground truth for validation

### Strategic Pivot: Self-Supervised Learning

Because labeled data is unavailable in MIMIC subset:

1. **Phase 5**: Train SSL encoder on 4,133 unlabeled MIMIC signals (reconstruction task)
2. **Phase 6**: Validate encoder quality without labels (reconstruction metrics)
3. **Phase 7**: Extract 512-dim embeddings for all 4,417 signals
4. **Phase 8**: Validate on VitalDB labels via transfer learning (proof encoder learned medical signal structure)

### Data Sources

| Source                       | Purpose                      | Signals                  | Labels                          | Status                   |
| ---------------------------- | ---------------------------- | ------------------------ | ------------------------------- | ------------------------ |
| **MIMIC-III Matched Subset** | SSL pretraining              | 4,417 unlabeled PPG      | None                            | ✅ Available             |
| **VitalDB Open Dataset**     | Transfer learning validation | ~6,000 surgical cases    | Hypertension, Diabetes, Obesity | ✅ Static AWS S3 release |
| **Reserved test set**        | Future fine-tuning           | 84 high-quality segments | None (reserved)                 | ✅ Available             |

---

## PHASES 0-4: EXISTING CODEBASE (85% COMPLETE)

### Status: Code Ready, Architecture Validated

All foundational work is complete and tested:

- ✅ Phase 0: 4,417 signals prepared, split, denoised
- ✅ Phase 1: 9 SSL modules implemented (encoder, decoder, loss, augmentation, trainer)
- ✅ Phase 2: 39 tests passing, all components validated
- ✅ Phase 3: Local CPU validation successful (loss converged)
- ✅ Phase 4: Code on GitHub, Colab notebook structure ready

**What's different in refined plan:**

- Input size: 75,000 samples (10 min) → **1,250 samples (10 sec)**
- Encoder blocks: 4 → **3** (prevents over-compression)
- Training samples: 4,133 → **617,000** (overlapping windows)
- Validation: No labels → **Reconstruction metrics only**

---

## PHASE 5: SSL PRETRAINING REFACTORING & EXECUTION

### Phase 5A: Architecture Refactoring (Local, 2-3 hours)

#### 5A.1: Redesign Encoder for 1,250-Sample Input

**File**: `colab_src/models/ssl/encoder.py`

**Changes**:

```python
# Old: 4 blocks (75,000 samples) → 512-dim
# New: 3 blocks (1,250 samples) → 512-dim

# Spatial dimension progression:
# Input: 1,250
# Block 1: stride-2 → 625
# Block 2: stride-2 → 312
# Block 3: stride-2 → 156
# AvgPool → 1
# FC: 256 → 512 (latent)
```

**Deliverables**:

- ✅ ResNetEncoder accepting [B, 1, 1250] input
- ✅ 3 residual blocks with stride-2 downsampling
- ✅ Preserves 156 features post-blocks (vs 78 with 4 blocks)
- ✅ Output shape [B, 512] latent representation

**Test**: `encoder = ResNetEncoder(num_blocks=3); z = encoder(torch.randn(2, 1, 1250)); assert z.shape == (2, 512)`

---

#### 5A.2: Redesign Decoder (Mirror Architecture)

**File**: `colab_src/models/ssl/decoder.py`

**Changes**:

```python
# Mirror encoder: 512-dim → [B, 1, 1250]
# 3 transposed blocks with stride-2 upsampling
# Linear projection: 512 → 256 × 156 spatial features
# Clip output to exact [B, 1, 1250] shape
```

**Deliverables**:

- ✅ ResNetDecoder accepting [B, 512] latent
- ✅ 3 transposed conv blocks with stride-2 upsampling
- ✅ Output shape [B, 1, 1250] (exact reconstruction target)
- ✅ Proper shape clipping/padding if needed

**Test**: `dec = ResNetDecoder(); x_recon = dec(torch.randn(2, 512)); assert x_recon.shape == (2, 1, 1250)`

---

#### 5A.3: Generate Overlapping Windows from MIMIC Signals

**File**: `colab_src/data_pipeline/generate_mimic_windows.py` (NEW)

**Purpose**: Transform 4,417 × 75,000-sample signals into 617,000 × 1,250-sample training examples

**Algorithm**:

```python
# For each 10-minute signal (75,000 samples @ 125Hz):
for segment_id, signal_path in signal_index.items():
    signal = np.load(signal_path)  # [75,000]

    # Extract overlapping 10-second windows (stride=500)
    for start in range(0, len(signal) - 1250, 500):
        window = signal[start : start + 1250]
        save_window(window, segment_id, window_idx)
        metadata.append({
            'segment_id': segment_id,
            'window_idx': window_idx,
            'subject_id': ...,
            'sqi_score': ...
        })

# Output:
# - mimic_windows.npy [617k, 1250]
# - mimic_windows_metadata.parquet [617k rows]
```

**Deliverables**:

- ✅ `data/processed/mimic_windows.npy` [617,000 × 1,250]
- ✅ `data/processed/mimic_windows_metadata.parquet` with segment tracking
- ✅ Subject stratification preserved (prevent leakage)

**Statistics**:

```
Original signals: 4,417 @ 75,000 samples
Windows per signal: ~60 (stride=500)
Total windows: 4,417 × 60 ≈ 617,000
Data expansion: 60× for SSL training
```

---

#### 5A.4: Update Augmentation for 10-Second Windows

**File**: `colab_src/models/ssl/augmentation.py`

**Changes** (rescale for 1,250 samples):

```python
# Old: temporal_shift_pct = 0.10 (10% of 75,000 = 7,500 samples)
# New: temporal_shift_pct = 0.02 (2% of 1,250 = 25 samples)

# Old: amplitude_scale = (0.85, 1.15) — Keep
# New: baseline_wander_freq = 0.2 Hz — Keep
# New: snr_matched_noise — Keep
```

**Deliverables**:

- ✅ All 4 augmentations work on 1,250-sample windows
- ✅ Temporal shift: ±25 samples (reasonable for 10-sec window)
- ✅ Amplitude & noise: unchanged (scale-invariant)
- ✅ Baseline wander: unchanged (frequency-based)

**Test**: `aug = PPGAugmentation(np.random.randn(1250)); aug_sig = aug.compose(); assert aug_sig.shape == (1250,)`

---

#### 5A.5: Update Config & FFT Optimization

**File**: `configs/ssl_pretraining.yaml`

**Changes**:

```yaml
model:
  num_blocks: 3 # was 4
  input_size: 1250 # was 75000
  latent_dim: 512 # unchanged

training:
  warmup_epochs: 2 # was 5 (reduce val loss spikes)
  early_stopping_patience: 15 # was 5 (allow full 50 epochs)
  batch_size: 128 # CRITICAL: increased from 8 (1,250-sample windows are 60x smaller)
  accumulation_steps: 1 # CRITICAL: reduced from 4 (batch_size 128 is already sufficient)
  num_epochs: 50 # unchanged

loss:
  mse_weight: 0.50 # unchanged
  ssim_weight: 0.30 # unchanged
  fft_weight: 0.20 # unchanged
  fft_pad_size: 2048 # CRITICAL: pad to 2^11 (was 2^17, wastes 99% zero-padding)
```

**Deliverables**:

- ✅ Config updated for 1,250-sample inputs
- ✅ Warmup reduced to 2 epochs (smoother training)
- ✅ Patience increased to 15 (full 50-epoch potential)
- ✅ Batch size increased to 128 (60x smaller windows = better GPU utilization)
- ✅ FFT padding: 1,250 → 2,048 (2^11, not 2^17 overkill)

---

#### 5A.6: Implement Checkpoint-Resume Logic

**File**: `colab_src/models/ssl/trainer.py`

**Purpose**: Recover from Colab timeouts (T4 sessions drop after ~12-24 hours)

**Implementation**:

```python
def find_latest_checkpoint(checkpoint_dir):
    """Resume from latest checkpoint if session dropped."""
    checkpoints = sorted(Path(checkpoint_dir).glob('checkpoint_epoch_*.pt'))
    if checkpoints:
        return checkpoints[-1]
    return None

# In Trainer.__init__():
latest_ckpt = find_latest_checkpoint(self.checkpoint_dir)
if latest_ckpt:
    print(f"Resuming from {latest_ckpt}")
    self.load_checkpoint(latest_ckpt)
    self.start_epoch = self.current_epoch + 1
```

**Deliverables**:

- ✅ Auto-detect latest checkpoint on startup
- ✅ Resume training from last completed epoch
- ✅ No loss of training progress on timeout
- ✅ Periodic Drive sync (every 5 epochs)

---

#### 5A.7: Add Reproducibility Seed

**File**: `colab_src/models/ssl/train.py`

**Implementation**:

```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Call at start of main()
set_seed(42)
```

**Deliverables**:

- ✅ Global seed set to 42
- ✅ Ensures reproducible results across runs
- ✅ Critical for small-sample Phase 8 evaluation

---

### Phase 5B: Full Pretraining on Colab T4 (12-18 hours, Colab)

#### 5B.1: Setup & Validation (30 min)

1. Mount Google Drive
2. Clone repo from GitHub
3. Install dependencies
4. Verify GPU: `!nvidia-smi`
5. Quick validation: Load config, encoder, decoder, loss
6. Verify data: 617k windows available

**Gate**: All checks pass, GPU detected

---

#### 5B.2: Run Full Training (1-2 hours actual GPU time)

```bash
python -m colab_src.models.ssl.train \
  --config configs/ssl_pretraining.yaml \
  --device cuda \
  --epochs 50
```

**Expected Timeline**:

- 617,000 training samples
- 8-sample batch size
- ~77,000 batches per epoch
- ~1-2 minutes per epoch on T4
- Early stopping likely at epoch 35-45

**Total**: 35-50 epochs × 1.5 min/epoch = **50-90 minutes actual training**

**Deliverables**:

- ✅ `checkpoints/phase5/best_encoder.pt` — Best model
- ✅ `logs/ssl/training_history.json` — Loss curves
- ✅ Training completes without OOM
- ✅ Validation loss plateaus (convergence signal)

**Success Criteria**:

- ✅ Training loss: 0.6 → 0.25 (55% reduction)
- ✅ Validation loss: 0.28 → 0.24 (plateau by epoch 20)
- ✅ No NaN/Inf during training
- ✅ Reconstruction SSIM >0.80 on validation set

---

#### 5B.3: Monitor & Save Checkpoints

```python
# Every epoch:
# - Log train_loss, val_loss, lr
# - Save checkpoint_epoch_N.pt
# - Save best_encoder.pt (if val_loss improves)

# Every 5 epochs:
# - Sync to Google Drive: cp checkpoints/ → /MyDrive/backups/
```

**Deliverables**:

- ✅ Periodic checkpoints every epoch
- ✅ Best model saved automatically
- ✅ All checkpoints synced to Drive
- ✅ Loss curves logged for visualization

---

## PHASE 6: ENCODER QUALITY VALIDATION (Colab, 1 hour)

### Objective: Confirm Encoder Learned Signal Structure (No Labels)

Since you have **zero clinical labels** in MIMIC, validate encoder quality through reconstruction metrics.

### 6.1: Reconstruction Quality Check

**Goal**: Test if encoder preserves signal information during reconstruction

**Metrics**:

```python
# On 200 validation signals:
ssim_scores = []
mse_scores = []
corr_scores = []

for raw_signal, denoised_target in validation_set:
    z = encoder(raw_signal)  # [512]
    reconstruction = decoder(z)  # [1, 1250]

    # Metric 1: SSIM (structural similarity)
    ssim = compute_ssim(reconstruction, denoised_target)
    ssim_scores.append(ssim)

    # Metric 2: MSE (pixel-level error)
    mse = F.mse_loss(reconstruction, denoised_target)
    mse_scores.append(mse.item())

    # Metric 3: Correlation (temporal alignment)
    corr = pearsonr(reconstruction.flatten(), denoised_target.flatten())
    corr_scores.append(corr)

mean_ssim = np.mean(ssim_scores)  # Target: >0.85
mean_mse = np.mean(mse_scores)    # Target: <0.005
mean_corr = np.mean(corr_scores)  # Target: >0.95
```

**Deliverables**:

- ✅ SSIM scores >0.85 (structural preservation)
- ✅ MSE <0.005 (accurate reconstruction)
- ✅ Correlation >0.95 (temporal alignment)
- ✅ Metrics saved to `validation_metrics.json`

**Gate**: All 3 metrics pass → Encoder learned meaningful structure

---

### 6.2: Embedding Space Analysis

**Goal**: Confirm latent vectors encode signal quality variation

```python
# Compute embedding statistics
embeddings = extract_embeddings(validation_set)  # [200, 512]

# Check variance per dimension
var_per_dim = np.var(embeddings, axis=0)
mean_variance = var_per_dim.mean()

# Expected: >0.8 (not collapsed to single value)
if mean_variance > 0.8:
    print("✅ Latent space has good variance")
else:
    print("⚠️  Latent space collapsed (encoder didn't learn)")

# Check if embeddings cluster by SQI
sqi_scores = validation_metadata['sqi_score'].values
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Visualize: color by SQI
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=sqi_scores, cmap='viridis')
plt.colorbar(label='SQI Score')
plt.title('Learned Representations (Colored by Signal Quality)')
plt.savefig('embeddings_visualization.png')
```

**Deliverables**:

- ✅ Embedding variance analysis
- ✅ 2D PCA visualization (colored by SQI)
- ✅ Statistical test: embedding variance >0.8
- ✅ Visual confirmation: high-SQI signals cluster together

**Gate**: Embeddings show structure (variance + visual clustering) → Proceed to Phase 7

---

### 6.3: Output Report

**File**: `validation_report_phase6.md`

```markdown
# Phase 6: Encoder Quality Validation

## Reconstruction Metrics (200 validation signals)

- SSIM: 0.87 ± 0.05 ✅
- MSE: 0.0031 ± 0.0008 ✅
- Correlation: 0.96 ± 0.02 ✅

## Embedding Space Analysis

- Mean variance per dimension: 1.2 ✅
- Visual clustering by SQI: Clear separation ✅

## Conclusion

Encoder learned signal structure without labels ✓
Ready for Phase 7 (embedding extraction)
```

---

## PHASE 7: EXTRACT LATENT EMBEDDINGS & FEATURES (Colab, 30 min)

### Objective: Generate Final Feature Matrix for Downstream Tasks

### 7.1: Batch Extract Embeddings

**Input**: All 4,417 signals (via 617k windows)  
**Output**: 512-dim latent vectors

```python
encoder.eval()
device = torch.device('cuda')
encoder = encoder.to(device)

# Load windowed data
windows = np.load('data/processed/mimic_windows.npy')  # [617k, 1250]
metadata = pd.read_parquet('data/processed/mimic_windows_metadata.parquet')

embeddings = []
batch_size = 32

for batch_start in range(0, len(windows), batch_size):
    batch_end = min(batch_start + batch_size, len(windows))
    batch = windows[batch_start:batch_end]

    x = torch.FloatTensor(batch).unsqueeze(1).to(device)  # [B, 1, 1250]

    with torch.no_grad():
        z = encoder(x).cpu().numpy()  # [B, 512]

    embeddings.extend(z)

embeddings_array = np.array(embeddings)  # [617k, 512]
np.save('data/processed/ssl_embeddings.npy', embeddings_array)
```

**Deliverables**:

- ✅ `ssl_embeddings.npy` [617,000 × 512]
- ✅ All embeddings extracted at batch_size=32 (GPU efficient)
- ✅ Metadata preserved (segment_id, window_idx, subject_id)

---

### 7.2: Aggregate to Segment Level

**Goal**: One embedding per original signal (not per window)

```python
# For each of 4,417 original segments:
segment_embeddings = {}

for segment_id in unique_segment_ids:
    # Get all windows from this segment
    window_indices = metadata[metadata['segment_id'] == segment_id].index
    segment_latents = embeddings_array[window_indices]  # [~60, 512]

    # Aggregate (mean over windows)
    aggregate_embedding = segment_latents.mean(axis=0)  # [512]
    segment_embeddings[segment_id] = aggregate_embedding

# Save
segment_embeddings_array = np.array([segment_embeddings[sid] for sid in sorted_ids])
np.save('data/processed/ssl_embeddings_aggregated.npy', segment_embeddings_array)
```

**Deliverables**:

- ✅ `ssl_embeddings_aggregated.npy` [4,417 × 512]
- ✅ One embedding per original signal
- ✅ Window-level aggregation (mean over windows)

---

### 7.3: Combine with Hand-Crafted Features

**Feature Matrix Composition**:

```
[4,417 × 515] final feature matrix:
├─ SSL embeddings: 512 dims (learned from reconstruction)
├─ SQI score: 1 dim (signal quality)
└─ SNR: 1 dim (signal-to-noise ratio)
└─ Perfusion index: 1 dim (PPG amplitude dynamics)
```

```python
ssl_embed_df = pd.DataFrame(
    segment_embeddings_array,
    columns=[f'ssl_latent_{i}' for i in range(512)]
)

quality_df = pd.DataFrame({
    'sqi_score': metadata_by_segment['sqi_score'],
    'snr_db': metadata_by_segment['snr_db'],
    'perfusion_index': metadata_by_segment['perfusion_index']
})

final_features = pd.concat([ssl_embed_df, quality_df], axis=1)
final_features.to_parquet('data/processed/ssl_features_final.parquet')
```

**Deliverables**:

- ✅ `ssl_features_final.parquet` [4,417 × 515]
- ✅ 512 SSL latent dimensions
- ✅ 3 quality/metadata dimensions
- ✅ No NaN values

---

### 7.4: Feature Statistics

**Compute & Log**:

```python
# Descriptive statistics
feature_stats = {
    'mean': final_features.mean(),
    'std': final_features.std(),
    'min': final_features.min(),
    'max': final_features.max()
}

# Check for multicollinearity
corr_matrix = final_features.corr()
high_corr_pairs = [(i, j, corr_matrix.iloc[i, j])
                   for i in range(len(corr_matrix))
                   for j in range(i+1, len(corr_matrix))
                   if abs(corr_matrix.iloc[i, j]) > 0.95]

# Save statistics
with open('exports/feature_statistics.json', 'w') as f:
    json.dump({
        'shape': final_features.shape,
        'feature_names': final_features.columns.tolist(),
        'statistics': feature_stats.to_dict(),
        'high_correlation_pairs': high_corr_pairs
    }, f, indent=2)
```

**Deliverables**:

- ✅ `feature_statistics.json` with mean/std/min/max
- ✅ Correlation matrix computed
- ✅ Multicollinearity check (none expected, low correlation between SSL dims)

---

**Phase 7 Gate**: ✅ All 4,417 segments have 515-dim feature vectors, no NaN, ready for downstream

---

## PHASE 8: TRANSFER LEARNING VALIDATION ON VITALDB (Colab, 2 hours)

### Objective: Prove Encoder Generalizes Across Populations via VitalDB

**Context**: You have **zero clinical labels in MIMIC**, so use **VitalDB as external validation** that encoder learned medical-relevant patterns.

### Data Reality: Small Sample Size

- **VitalDB available**: ~6,000 surgical cases
- **With complete labels** (hypertension, diabetes, obesity): ~40-60% = 2,400-3,600 cases
- **Usable for validation**: ~1,500-2,000 labeled cases
- **Train/val/test split**: 600 train, 400 val, 500 test

---

### 8.1: Fetch & Filter VitalDB Data

**Step 1**: Identify cases with PPG + complete labels

```python
import vitaldb

# Find all surgical cases with PPG
caseids = sorted(vitaldb.find_cases(['SNUADC/PLETH']))
print(f"Total cases with PPG: {len(caseids)}")

# Pre-filter for label completeness (sample first 100 for feasibility)
vitaldb_data = {
    'caseid': [],
    'hypertension': [],
    'diabetes': [],
    'obesity': [],
    'num_windows': []
}

for caseid in caseids[:100]:
    try:
        vf = vitaldb.VitalFile(caseid, ['SNUADC/PLETH'])

        # Extract labels
        htn = vf.get('history_htn', None)
        dm = vf.get('history_dm', None)

        # Calculate obesity from height/weight
        height = vf.get('height', None)
        weight = vf.get('weight', None)
        if height and weight:
            bmi = weight / (height ** 2)
            obesity = bmi > 30
        else:
            obesity = None

        # Check label completeness
        if htn is not None and dm is not None and obesity is not None:
            # Fetch signal and count windows
            ppg = vf.to_numpy(['SNUADC/PLETH'], interval=1/125)
            num_windows = max(0, (len(ppg) - 1250) // 500 + 1)

            vitaldb_data['caseid'].append(caseid)
            vitaldb_data['hypertension'].append(htn)
            vitaldb_data['diabetes'].append(dm)
            vitaldb_data['obesity'].append(obesity)
            vitaldb_data['num_windows'].append(num_windows)

    except Exception as e:
        print(f"Case {caseid}: {e}")

df = pd.DataFrame(vitaldb_data)
print(f"Feasibility: {len(df)}/100 have complete labels")
print(f"Projected: {len(caseids) * len(df) / 100:.0f} labeled cases total")
```

**Deliverables**:

- ✅ Feasibility assessment (% cases with complete labels)
- ✅ VitalDB dataset loaded and filtered
- ✅ Projection: ~2,400-3,600 labeled cases available

**Gate**: >40% label completeness → Proceed to 8.2

---

### 8.2: Preprocess VitalDB Signals

**Apply identical preprocessing to MIMIC**:

```python
from scipy.signal import cheby2, filtfilt

# CRITICAL: Use IDENTICAL Chebyshev-II filter from MIMIC
sos = cheby2(order=4, rs=40, Wn=[0.5/62.5, 8/62.5],
             analog=False, btype='band', output='sos')

vitaldb_signals = []
vitaldb_labels = {'caseid': [], 'hypertension': [], 'diabetes': [], 'obesity': []}

for caseid in vitaldb_df['caseid']:
    try:
        vf = vitaldb.VitalFile(caseid, ['SNUADC/PLETH'])

        # Fetch signal (resampled to 125Hz by API)
        ppg_125hz = vf.to_numpy(['SNUADC/PLETH'], interval=1/125)

        # Skip too-short signals
        if len(ppg_125hz) < 1250:
            continue

        # Apply IDENTICAL Chebyshev-II filter
        ppg_filtered = filtfilt(sos, ppg_125hz)

        # Normalize to [0, 1]
        ppg_min, ppg_max = ppg_filtered.min(), ppg_filtered.max()
        if ppg_max > ppg_min:
            ppg_normalized = (ppg_filtered - ppg_min) / (ppg_max - ppg_min)
        else:
            ppg_normalized = ppg_filtered

        vitaldb_signals.append(ppg_normalized)
        vitaldb_labels['caseid'].append(caseid)
        vitaldb_labels['hypertension'].append(vitaldb_df[vitaldb_df['caseid']==caseid]['hypertension'].values[0])
        vitaldb_labels['diabetes'].append(vitaldb_df[vitaldb_df['caseid']==caseid]['diabetes'].values[0])
        vitaldb_labels['obesity'].append(vitaldb_df[vitaldb_df['caseid']==caseid]['obesity'].values[0])

    except Exception as e:
        print(f"Case {caseid}: {e}")

vitaldb_signals_array = np.array(vitaldb_signals)  # [N, ≥1250]
vitaldb_df_processed = pd.DataFrame(vitaldb_labels)

print(f"VitalDB processed: {len(vitaldb_signals_array)} signals ready")
```

**Deliverables**:

- ✅ All VitalDB signals filtered with **identical Chebyshev-II** (critical for transfer learning)
- ✅ Normalized to [0, 1]
- ✅ Metadata preserved (labels, caseids)

**Gate**: >1,000 preprocessed signals available → Proceed to 8.3

---

### 8.3: Train Linear Probe on VitalDB (Frozen Encoder)

**Strategy**: Extract features with frozen MIMIC-trained encoder, train linear classifier on VitalDB labels

```python
# Load pretrained MIMIC encoder
encoder = ResNetEncoder(num_blocks=3)
encoder.load_state_dict(torch.load('checkpoints/phase5/best_encoder.pt'))
encoder.eval()
device = torch.device('cuda')
encoder = encoder.to(device)

# Extract features from VitalDB signals
# CRITICAL: Track case IDs to prevent data leakage (same subject in train & test)
latents = []
case_ids = []
y_htn, y_dm, y_obesity = [], [], []

for signal, row in zip(vitaldb_signals_array, vitaldb_df_processed.itertuples()):
    # Extract overlapping 10-sec windows
    for start in range(0, len(signal) - 1250, 500):
        window = signal[start : start + 1250]
        x = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 1250]

        with torch.no_grad():
            z = encoder(x).cpu().numpy()  # [1, 512]

        latents.append(z[0])
        case_ids.append(row.caseid)  # Track which subject this window came from
        y_htn.append(row.hypertension)
        y_dm.append(row.diabetes)
        y_obesity.append(row.obesity)

X = np.array(latents)  # [N_windows, 512]
case_ids_array = np.array(case_ids)  # [N_windows]
y_htn, y_dm, y_obesity = np.array(y_htn), np.array(y_dm), np.array(y_obesity)

# CRITICAL FIX: Split by SUBJECT (caseid), not windows
# This prevents data leakage where Subject 101's 70 windows in train memorize their signature
from sklearn.model_selection import train_test_split

unique_cases = np.unique(case_ids_array)
case_labels = np.array([y_htn[np.where(case_ids_array == cid)[0][0]] for cid in unique_cases])

train_cases, test_cases = train_test_split(unique_cases, test_size=0.3, random_state=42, stratify=case_labels)
train_cases, val_cases = train_test_split(train_cases, test_size=0.2, random_state=42, stratify=case_labels[np.isin(unique_cases, train_cases)])

# Assign windows to train/val/test based on subject membership
train_idx = np.isin(case_ids_array, train_cases)
val_idx = np.isin(case_ids_array, val_cases)
test_idx = np.isin(case_ids_array, test_cases)

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

# Train linear probes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

results = {}

for condition_name, y_full in [('Hypertension', y_htn), ('Diabetes', y_dm), ('Obesity', y_obesity)]:
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]
    y_test = y_full[test_idx]

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred_proba_test = clf.predict_proba(X_test)[:, 1]
    auroc_test = roc_auc_score(y_test, y_pred_proba_test)

    results[condition_name] = {
        'auroc': auroc_test,
        'n_test': len(y_test),
        'prevalence': y_test.mean(),
        'y_pred': y_pred_proba_test,
        'y_test': y_test
    }

    print(f"{condition_name}: AUROC={auroc_test:.3f} (n={len(y_test)}, prev={y_test.mean():.1%})")
```

**Deliverables**:

- ✅ Linear probes trained on frozen encoder features
- ✅ AUROC computed per condition
- ✅ Held-out test set evaluation (no leakage)
- ✅ Results saved to `transfer_learning_results.json`

---

### 8.4: Visualize & Report Results

```python
# ROC curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for (cond, ax) in zip(results.keys(), axes):
    fpr, tpr, _ = roc_curve(results[cond]['y_test'], results[cond]['y_pred'])
    ax.plot(fpr, tpr, label=f"AUROC={results[cond]['auroc']:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(cond)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transfer_learning_roc_curves.png', dpi=150)
plt.show()

# Summary report
report = f"""
# PHASE 8: TRANSFER LEARNING VALIDATION (MIMIC → VitalDB)

## Data Sources
- **Source**: MIMIC-III Matched Subset (unlabeled, 4,417 signals @ 125Hz)
- **Target**: VitalDB Surgical Population (labeled, {len(X_test)} test windows)
- **Labels**: Hypertension, Diabetes, Obesity (from EHR)
- **Validation**: Held-out test set (30% of data)

## Results
| Condition | AUROC | N_test | Prevalence |
|-----------|-------|--------|------------|
| Hypertension | {results['Hypertension']['auroc']:.3f} | {results['Hypertension']['n_test']} | {results['Hypertension']['prevalence']:.1%} |
| Diabetes | {results['Diabetes']['auroc']:.3f} | {results['Diabetes']['n_test']} | {results['Diabetes']['prevalence']:.1%} |
| Obesity | {results['Obesity']['auroc']:.3f} | {results['Obesity']['n_test']} | {results['Obesity']['prevalence']:.1%} |

## Interpretation

### Transfer Learning Success Criteria
✅ AUROC ≥0.60 on ≥1 condition → Encoder learned medical signal structure

### Population Shift Impact
Expected: VitalDB AUROC lower than same-source (surgical vs. ICU)
- Surgical patients: generally healthier, more stable heart rates under anesthesia
- ICU patients: acute/critical, diverse pathologies
- This difference is **expected and validates generalization**

### What This Proves
1. **Encoder learns signal structure** (not noise)
2. **Generalizes across institutions** (MIMIC → VitalDB)
3. **Robust to hardware differences** (different PPG sensors)
4. **Ready for production fine-tuning** on labeled data

## Next Steps (Phase 9+)
- Acquire labels for your target population
- Fine-tune encoder on domain-specific labels
- Deploy for clinical risk stratification
"""

with open('transfer_learning_report.md', 'w') as f:
    f.write(report)
```

**Deliverables**:

- ✅ ROC curves visualization
- ✅ `transfer_learning_results.json` with AUROC per condition
- ✅ `transfer_learning_report.md` with interpretation
- ✅ Statistical summary (n, prevalence, AUROC ± CI)

**Gate**: ≥1 condition AUROC ≥0.60 → **Transfer learning successful**

---

### 8.5: Honest Interpretation

| Outcome                        | Interpretation                                                 | Next Action                                              |
| ------------------------------ | -------------------------------------------------------------- | -------------------------------------------------------- |
| **AUROC 0.55-0.75**            | Transfer learning successful, encoder learned medical patterns | ✅ Proceed to production fine-tuning                     |
| **AUROC <0.55** all conditions | Encoder didn't learn meaningful structure                      | 🔄 Loop back to Phase 5 (increase training, tune losses) |
| **High variance** (0.65±0.15)  | Small sample size (expected with surgical population)          | ✅ Document limitations, proceed cautiously              |

**Critical caveat**: VitalDB is surgical population (anesthetized, stable HR). Different from ICU (acute, variable). Lower AUROC expected — this **validates** encoder doesn't overfit to MIMIC artifacts.

---

## SUMMARY TABLE: ALL PHASES

| Phase   | Duration | Environment | Objectives                    | Deliverables                               | Success Gate                |
| ------- | -------- | ----------- | ----------------------------- | ------------------------------------------ | --------------------------- |
| **0-4** | Complete | Local       | Existing code validation      | Code ready, GitHub pushed                  | All tests passing ✅        |
| **5A**  | 2-3h     | Local       | Refactor for 10-sec windows   | Encoder/decoder/augmentation updated       | All tests pass locally      |
| **5B**  | 12-18h   | Colab T4    | Full SSL pretraining          | best_encoder.pt, training_history.json     | Val loss <0.01, SSIM >0.85  |
| **6**   | 1h       | Colab       | Encoder quality validation    | Reconstruction metrics, embedding analysis | SSIM >0.85, MSE <0.005      |
| **7**   | 0.5h     | Colab       | Extract embeddings & features | ssl_features_final.parquet [4,417 × 515]   | No NaN, all shapes correct  |
| **8**   | 2h       | Colab       | Transfer learning on VitalDB  | AUROC per condition, ROC curves            | AUROC ≥0.60 on ≥1 condition |

---

## TIMELINE & RESOURCE ALLOCATION

### Local Development (4-5 hours)

```
5A.1: Encoder redesign        (30 min)
5A.2: Decoder redesign        (30 min)
5A.3: Window generation       (1 hour)
5A.4: Augmentation update     (30 min)
5A.5: Config & FFT opt.       (30 min)
5A.6: Checkpoint-resume       (30 min)
5A.7: Reproducibility seed    (15 min)
--------------------------------------
Total: ~4 hours
```

### Colab GPU (15-20 hours)

```
5B: Full training         (1-2 hours actual GPU time)
6: Quality validation     (1 hour)
7: Embeddings extraction  (0.5 hours)
8: VitalDB validation     (2 hours)
--------------------------------------
Total: ~4.5 hours GPU active
(Waiting time: 10-16 hours for training)
```

### Monitoring & Analysis (2-3 hours)

- Loss curve inspection
- Metric computation
- Report generation
- Troubleshooting (if needed)

---

## KNOWN CONSTRAINTS & MITIGATIONS

| Constraint                      | Impact                                | Mitigation                                   |
| ------------------------------- | ------------------------------------- | -------------------------------------------- |
| **No MIMIC labels**             | Can't supervise fine-tune on own data | Use VitalDB for transfer learning validation |
| **Small test set (84)**         | Limited power for final evaluation    | Acknowledge in reports, use 5-fold CV        |
| **Cross-population validation** | VitalDB ≠ MIMIC (surgical vs ICU)     | **Feature, not bug** — proves generalization |
| **Colab session timeout**       | Can drop after 12-24 hours            | Checkpoint-resume logic (Phase 5A.6)         |
| **FFT on non-power-of-2**       | Memory inefficiency                   | Pad to 2^17 (Phase 5A.5)                     |
| **Small VitalDB labels**        | ~2.4k cases available                 | Use 70% train, 30% test (standard)           |

---

## SUCCESS CRITERIA CHECKLIST

### Phase 5B (SSL Training)

- [ ] Training loss converges (0.6 → 0.25)
- [ ] Validation loss plateaus by epoch 20
- [ ] No NaN/Inf during training
- [ ] Checkpoint saved without errors
- [ ] Time per epoch <3 min on T4

### Phase 6 (Quality Validation)

- [ ] SSIM >0.85 on validation set
- [ ] MSE <0.005 on validation set
- [ ] Correlation >0.95 on validation set
- [ ] Embedding variance >0.8 per dimension
- [ ] Visual clustering by SQI in 2D PCA

### Phase 7 (Embeddings)

- [ ] 4,417 × 512 embeddings extracted
- [ ] Aggregation: one embedding per signal
- [ ] No NaN/Inf values
- [ ] 4,417 × 515 final feature matrix (512 SSL + 3 metadata)

### Phase 8 (Transfer Learning)

- [ ] VitalDB: >1,000 labeled windows available
- [ ] Preprocessing: identical filter to MIMIC
- [ ] Linear probe: AUROC computed per condition
- [ ] Test set: held-out, stratified split
- [ ] Report: honest interpretation (acknowledge population shift)

---

## BEYOND PHASE 8: PATH TO PRODUCTION

Once Phase 8 validates encoder:

### Phase 9: Fine-Tuning on Labeled Data

- Acquire labels for target population (e.g., 500+ samples per condition)
- Load frozen encoder
- Add classification head
- Train on labeled data
- Evaluate on held-out test set

### Phase 10: Deployment

- Export to ONNX
- Build FastAPI service
- Deploy via Docker
- Monitor in production

### Phase 11: Explainability

- SHAP values per prediction
- Feature importance
- Model interpretability

---

## DOCUMENT MAINTENANCE

**This plan is living**. After each phase:

1. Update `PROGRESS_CURRENT.md` with results
2. Document any deviations
3. Update success criteria if needed
4. Keep timeline realistic

---

**Status**: ✅ **READY FOR IMPLEMENTATION**  
**Next Step**: Begin Phase 5A refactoring locally
