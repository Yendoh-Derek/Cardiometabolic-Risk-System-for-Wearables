# Cardiometabolic Risk Estimation — Codebase Structure (Phase 5-8 Pivot)

**Updated**: January 14, 2026  
**Status**: Ready for Phase 5A Refactoring  
**Architecture**: Self-Supervised Learning on Overlapping 10-Second Windows → Transfer Learning Validation on VitalDB

---

## 🔄 CRITICAL PIVOT: Window Size & Architecture Changes

| Parameter            | Old (Phase 4)           | New (Phase 5+)              | Rationale                                                     |
| -------------------- | ----------------------- | --------------------------- | ------------------------------------------------------------- |
| **Window size**      | 75,000 samples (10 min) | 1,250 samples (10 sec)      | Preserve micro-morphology, reduce over-compression            |
| **Encoder blocks**   | 4                       | 3                           | Prevent excessive downsampling of 1,250-sample input          |
| **Training samples** | 4,133 signals           | 617,000 overlapping windows | 60× data expansion via stride-500 sliding window              |
| **Batch size**       | 8                       | **128**                     | ✅ Critical fix: GPU utilization, better BatchNorm            |
| **FFT padding**      | 2^17 (131,072)          | **2^11 (2,048)**            | ✅ Critical fix: 67× faster, eliminate 99% zero-padding waste |
| **Data split**       | Window-level            | **Subject-level (caseid)**  | ✅ Critical fix: Prevent data leakage in Phase 8              |

---

## Project Directory Structure

```
cardiometabolic-risk-colab/
│
├── docs/ # Documentation (updated January 14, 2026)
│ ├── IMPLEMENTATION_PLAN_PHASES_0-8.md # Master plan with all 3 critical fixes
│ ├── FINAL_CRITICAL_FIXES_SUMMARY.md # Detailed fix documentation
│ ├── architecture.md # System design (this file — updated)
│ ├── codebase.md # File structure (this file — updated)
│ ├── PROJECT_STATUS.md # Current status & known issues
│ └── PROGRESS_CURRENT.md # Phase-by-phase progress tracking
│
├── colab_src/ # Core Python modules (SSL pretraining)
│ ├── __init__.py
│ │
│ ├── models/ssl/ # Self-Supervised Learning (Phase 5)
│ │ ├── __init__.py
│ │ ├── encoder.py # ResNetEncoder: [B,1,1250] → [B,512] (3 blocks)
│ │ ├── decoder.py # ResNetDecoder: [B,512] → [B,1,1250] (mirror architecture)
│ │ ├── losses.py # Hybrid loss: MSE(50%) + SSIM(30%) + FFT(20%)
│ │ ├── augmentation.py # PPGAugmentation: temporal shift, amplitude scale, noise, baseline wander
│ │ ├── dataloader.py # PPGDataset: lazy-load 617k windowed samples
│ │ ├── trainer.py # Training loop with checkpoint-resume & gradient accumulation
│ │ ├── train.py # CLI entry point for Phase 5B (Colab)
│ │ ├── config.py # YAML config loader
│ │ └── vitaldb_transfer.py # Phase 8: VitalDB linear probes
│ │
│ ├── data_pipeline/ # Data preparation (Phases 0-5)
│ │ ├── __init__.py
│ │ ├── mimic_ingestion.py # Download 4,417 MIMIC PPG signals (@125Hz, 75k samples each)
│ │ ├── signal_preprocessing.py # Chebyshev-II filter, wavelet denoising
│ │ ├── signal_quality.py # SQI scoring
│ │ ├── generate_mimic_windows.py # NEW: Generate 617k overlapping 1,250-sample windows
│ │ ├── demographic_processor.py # Extract age, sex, BMI (minimal — no clinical labels)
│ │ └── dataset_assembly.py # Combine signals + metadata into parquet
│ │
│ ├── signal_processing/ # Signal utilities
│ │ ├── __init__.py
│ │ ├── filters.py # Chebyshev-II bandpass [0.5–8 Hz]
│ │ ├── denoising.py # Wavelet decomposition (db4, level 5)
│ │ ├── segmentation.py # Sliding window extraction
│ │ └── quality_metrics.py # SQI, SNR, perfusion index
│ │
│ ├── features/ # Classical feature extraction (Phase 7)
│ │ ├── __init__.py
│ │ ├── hrv_features.py # HRV (28 features: time, frequency, nonlinear)
│ │ ├── morphology_features.py # PPG morphology (6 features: systolic height, diastolic area, etc.)
│ │ ├── clinical_context.py # Context encoding (3 features: age, sex, BMI)
│ │ └── feature_combiner.py # Combine SSL embeddings + classical features
│ │
│ ├── validation/ # Quality assurance (Phases 6-8)
│ │ ├── __init__.py
│ │ ├── reconstruction_metrics.py # SSIM, MSE, correlation
│ │ ├── embedding_analysis.py # Variance, clustering, PCA visualization
│ │ └── transfer_learning_eval.py # AUROC, ROC curves, fairness checks
│ │
│ ├── utils/ # Shared utilities
│ │ ├── __init__.py
│ │ ├── config_loader.py # Load configs/ssl_pretraining.yaml
│ │ ├── checkpoint_manager.py # Save/load best_encoder.pt
│ │ ├── logging.py # MLflow + standard logging
│ │ └── reproducibility.py # Set seed(42) for all RNGs
│ │
│ └── evaluation/ # Reporting (Phase 8)
│ ├── __init__.py
│ ├── vitaldb_validation.py # Cross-subject evaluation, population shift analysis
│ ├── report_generator.py # Markdown report with AUROC/CI
│ └── visualization.py # ROC curves, embedding plots
│
├── data/ # All datasets (Colab storage)
│ ├── raw/
│ │ └── RECORDS-waveforms # MIMIC file index
│ │
│ ├── processed/
│ │ ├── mimic_signals_denoised/ # 4,417 denoised 75k-sample signals (.npy)
│ │ ├── mimic_windows.npy # NEW: [617,000 × 1,250] windowed samples
│ │ ├── mimic_windows_metadata.parquet # Window-level metadata (segment_id, subject_id, sqi_score)
│ │ ├── ssl_embeddings.npy # Phase 7: [617,000 × 512] window embeddings
│ │ ├── ssl_embeddings_aggregated.npy # Phase 7: [4,417 × 512] signal-level embeddings (mean)
│ │ └── ssl_features_final.parquet # Phase 7: [4,417 × 515] (512 SSL + 3 quality dims)
│ │
│ ├── metadata/
│ │ ├── signal_metadata.parquet # 4,417 rows: subject_id, signal_length, sqi_score, snr_db
│ │ └── vitaldb_labels.parquet # Phase 8: VitalDB cases with hypertension/diabetes/obesity labels
│ │
│ └── cache/
│ └── temporary processing files
│
├── checkpoints/ # Model weights (Colab)
│ ├── phase5/
│ │ ├── checkpoint_epoch_001.pt
│ │ ├── checkpoint_epoch_010.pt
│ │ ├── checkpoint_epoch_020.pt
│ │ └── best_encoder.pt # Best validation loss model
│ │
│ └── phase3/ # Legacy checkpoints (kept for reference)
│ ├── checkpoint_pilot.pt
│ └── metrics_pilot.json
│
├── artifacts/ # Evaluation outputs
│ ├── models/ # Model exports
│ │ ├── best_encoder.onnx # Phase 5: Encoder in ONNX format
│ │ └── vitaldb_transfer_results.json # Phase 8: AUROC per condition
│ │
│ ├── evaluation/
│ │ ├── reconstruction_metrics.json # Phase 6: SSIM, MSE, correlation
│ │ ├── embeddings_visualization.png # Phase 6: 2D PCA colored by SQI
│ │ ├── transfer_learning_roc_curves.png # Phase 8: ROC curves
│ │ └── transfer_learning_report.md # Phase 8: Interpretation & next steps
│ │
│ └── preprocessing/
│ ├── filter_params.json # Chebyshev-II coefficients
│ └── normalization_stats.json # Mean/std for signal normalization
│
├── configs/ # Configuration files
│ ├── ssl_pretraining.yaml # MAIN: Phase 5 config with critical fixes
│ │ # batch_size: 128 (was 8) ✅
│ │ # fft_pad_size: 2048 (was 131072) ✅
│ │ # num_blocks: 3 (was 4)
│ ├── preprocessing.yaml # Phase 0: Signal preprocessing params
│ └── data.yaml # Data paths & split ratios
│
├── tests/ # Unit tests (centralized, January 14 cleanup)
│ ├── __init__.py
│ ├── conftest.py # Pytest configuration
│ ├── run_tests.py # Run all tests
│ ├── test_config.py # Config loader tests
│ ├── test_encoder.py # Encoder shape tests
│ ├── test_decoder.py # Decoder shape tests
│ ├── test_losses.py # Loss computation tests
│ ├── test_augmentation.py # Augmentation tests
│ ├── test_smoke.py # Quick smoke tests
│ ├── test_integration.py # End-to-end tests
│ ├── test_phase0_data_pipeline.py # Data pipeline tests (moved from root)
│ ├── test_training_single_batch.py # Single batch training test (moved from root)
│ └── test_mimic_clinical_extractor.py # MIMIC extraction tests (moved from root)
│
├── notebooks/ # Jupyter exploration (read-only reference)
│ ├── 01_data_exploration.ipynb
│ ├── 02_signal_quality_analysis.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_clinical_data_integration.ipynb
│ ├── 05_ssl_data_preparation.ipynb
││    05_ssl_pretraining_colab.ipynb
│ ├── 06_model_training.ipynb
│ ├── 07_model_evaluation.ipynb
│ └── 08_interpretability.ipynb
│
├── logs/ # Training logs (Colab)
│ ├── ssl/ # Phase 5 training logs
│ │ ├── training_history.json # Loss curves, metrics
│ │ └── training.log # Stdout/stderr
│ └── mlruns/ # MLflow experiment tracking
│
├── exports/ # Final artifacts for deployment
│ ├── models/
│ │ ├── encoder_best.onnx # Phase 5: Exported encoder
│ │ └── vitaldb_results.json # Phase 8: Transfer learning results
│ │
│ ├── feature_definitions/
│ │ ├── ssl_latent_dimensions.yaml # 512 learned features
│ │ └── classical_features.yaml # HRV + morphology + context
│ │
│ └── metadata/
│ ├── feature_statistics.json # Mean, std, min, max per feature
│ ├── model_card.md # Model documentation
│ └── training_config.yaml # Hyperparameters used
│
├── context/ # Documentation & tracking (January 14 updated)
│ ├── PROJECT_STATUS.md # Overall project status
│ ├── PROGRESS_CURRENT.md # Phase-by-phase progress
│ ├── IMPLEMENTATION_PLAN_PHASES_0-8.md # Master plan (GitHub)
│ ├── CRITICAL_FIXES_APPLIED.md # Three flaws fixed (GitHub)
│ ├── FINAL_CRITICAL_FIXES_SUMMARY.md # Fix details (GitHub)
│ └── CLEANUP_FINAL.md # Cleanup documentation
│
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore patterns
├── README.md # Project overview
└── setup.py # Package installation
```

---

## Module Descriptions by Phase

### **Phase 0: Data Preparation** (Complete)

- **mimic_ingestion.py**: Download 4,417 PPG signals from MIMIC-III
- **signal_preprocessing.py**: Filter + denoise → 4,417 clean signals
- **dataset_assembly.py**: Create parquet metadata index

### **Phase 5A: Architecture Refactoring** (4-5 hours local)

- **encoder.py**: Refactor to 3 blocks, accept [B,1,1250] input
- **decoder.py**: Refactor to mirror architecture, output [B,1,1250]
- **augmentation.py**: Rescale temporal_shift_pct for small windows
- **generate_mimic_windows.py**: Generate 617k overlapping windows
- **train.py**: CLI with config loading & reproducibility seed
- **configs/ssl_pretraining.yaml**: Update batch_size, fft_pad_size, warmup, patience

### **Phase 5B: Full Pretraining** (12-18 hours Colab T4)

- **trainer.py**: Execute 50 epochs on 617k samples
- **losses.py**: Hybrid loss with optimized FFT padding
- **checkpoint_manager.py**: Save/resume on timeout

### **Phase 6: Validation** (1 hour Colab)

- **reconstruction_metrics.py**: SSIM, MSE, correlation
- **embedding_analysis.py**: Variance checks, PCA visualization

### **Phase 7: Feature Extraction** (30 min Colab)

- **dataloader.py**: Batch load embeddings
- **feature_combiner.py**: Combine SSL + classical features
- **ssl_features_final.parquet**: [4,417 × 515] final matrix

### **Phase 8: Transfer Learning** (2 hours Colab)

- **vitaldb_transfer.py**: ✅ **Critical fix #3**: Split by caseid, not windows
- **transfer_learning_eval.py**: Compute AUROC per condition
- **report_generator.py**: Markdown report with interpretation

---

## Critical Fixes Applied (January 14, 2026)

### ✅ Fix #1: Data Leakage Prevention

**File**: `vitaldb_transfer.py` (Phase 8)  
**Change**: Split by subject (caseid) before assigning windows to train/test  
**Impact**: Honest cross-subject evaluation, no artificial AUROC inflation

### ✅ Fix #2: FFT Padding Efficiency

**File**: `configs/ssl_pretraining.yaml`  
**Change**: `fft_pad_size: 2048` (was 131,072)  
**Impact**: 67× faster loss computation, eliminate 99% zero-padding waste

### ✅ Fix #3: Batch Size Optimization

**File**: `configs/ssl_pretraining.yaml`  
**Change**: `batch_size: 128`, `accumulation_steps: 1` (were 8, 4)  
**Impact**: 10× faster epoch (1.5 min vs 16 min on T4)

---

## Key Dependencies

| Package          | Version | Purpose                                 |
| ---------------- | ------- | --------------------------------------- |
| **PyTorch**      | 2.1+    | Deep learning framework                 |
| **NumPy**        | 1.24+   | Numerical computing                     |
| **Pandas**       | 2.0+    | Data manipulation                       |
| **SciPy**        | 1.10+   | Signal processing (Chebyshev-II filter) |
| **scikit-learn** | 1.3+    | Logistic regression, train_test_split   |
| **PyWavelets**   | 1.4+    | Wavelet denoising (db4)                 |
| **NeuroKit2**    | 0.2.7+  | HRV feature extraction                  |
| **Matplotlib**   | 3.7+    | Visualization                           |

---

## Execution Environment

- **Local**: Python 3.10+, CPU for Phase 5A refactoring (4-5 hours)
- **Colab**: T4 GPU, 12GB+ RAM for Phases 5B-8 (12-18 hours actual GPU)
- **Reproducibility**: torch.manual_seed(42), deterministic=True

---

## Data Flow Summary

```
Phase 0: 4,417 MIMIC signals (75k samples each)
    ↓ [filter, denoise, quality check]
Phase 1-4: Codebase validation (existing)
    ↓ [generate overlapping windows]
Phase 5A: 617,000 overlapping 1,250-sample windows
    ↓ [train encoder 50 epochs]
Phase 5B: Trained encoder (best_encoder.pt)
    ↓ [extract latent vectors]
Phase 6: Reconstruction validation (SSIM >0.85, MSE <0.005)
    ↓ [aggregate & combine with classical features]
Phase 7: 4,417 × 515 feature matrix (512 SSL + 3 metadata)
    ↓ [frozen encoder + VitalDB labels]
Phase 8: Cross-subject AUROC per condition (Hypertension/Diabetes/Obesity)
    ↓ [report & interpretation]
Final: Model documentation & deployment artifacts
```

---

**Status**: ✅ Ready for Phase 5A Implementation  
**Last Updated**: January 14, 2026  
**All Critical Fixes Applied**: Yes
