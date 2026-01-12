Multimodal Cardiometabolic Risk Estimation

PROJECT OVERVIEW
Goal: Build a deep learning system to predict cardiometabolic risk (diabetes, hypertension, dyslipidemia, obesity) from wearable PPG (photoplethysmography) signals.
Data Source: MIMIC-III Waveform Database Matched Subset (PhysioNet)
Approach: Hybrid architecture combining:

Hand-crafted physiological features (HRV, pulse morphology)
Deep learning features (1D-ResNet encoder)
Clinical context (demographics, ICD-9 codes)
SQI-gated fusion for robust predictions

Target Output: Multi-label Regression for each of the (5 conditions) with SHAP explainability

PROJECT STRUCTURE
cardiometabolic-risk-colab/
├── notebooks/ # Jupyter notebooks (Colab execution)
│ ├── 01_data_exploration.ipynb ✅ COMPLETE
│ ├── 02_signal_quality_analysis.ipynb ✅ COMPLETE
│ ├── 03_feature_engineering.ipynb ✅ COMPLETE
│ ├── 04_baseline_analysis.ipynb 🔄 Pending
│ ├── 05_pretraining_cnn.ipynb ⏳ PENDING
│ └── 06_model_training.ipynb ⏳ PENDING
│ │── 07_model_evaluation.ipynb ⏳ PENDING
│ │── 08_interpretability.ipynb ⏳ PENDING
│
├── data/
│ ├── raw/ # Empty (signals streamed from PhysioNet)
│ ├── processed/ # Preprocessed signals (2.2 GB)This my change after my scale up.
│ ├── features/ # Feature matrices (1.2 MB)
│ └── metadata/ # Record catalogs, logs
│
├── colab_src/ # Python modules (production code)
│ ├── data_pipeline/ ✅ mimic_ingestion.py, processing_pipeline.py, clinical_linker.py
│ ├── signal_processing/ ✅ quality.py, filters.py, denoising.py, segmentation.py
│ ├── features/ ✅ hrv_features.py, morphology_features.py, clinical_context.py, feature_extractor.py
│ ├── models/ ⏳ cnn_encoder.py, xgboost_classifier.py, fusion_model.py
│ ├── evaluation/ ⏳ clinical_metrics.py, calibration.py, interpretability.py
│ ├── utils/ ✅ experiment_tracker.py, visualization.py
│ └── validation/ ✅ data_quality_tests.py
│
├── artifacts/ # Trained models & pipelines (for API export)
├── exports/ # API-ready artifacts (ONNX, pickles)
├── configs/ # Hydra YAML configs
└── logs/mlruns/ # MLflow experiment tracking

SPRINT 1: DATA PIPELINE & SIGNAL PROCESSING ✅ COMPLETE
Objectives

Discover usable PPG(Pleth) records from MIMIC-III Matched Subset
Implement signal quality assessment (SQI)
Build preprocessing pipeline (filtering, denoising, segmentation)
Convert to Parquet format for efficient storage

Deliverables

✅ data/processed/sprint1_signals.npy (3,660 segments × 75,000 samples)
✅ data/processed/sprint1_metadata.parquet (segment-level metadata)
✅ All signal processing modules functional

Technical Implementation
Key Modules:

mimic_ingestion.py: Two-level RECORDS file reading

Streams subject directories
Downloads individual record headers/signals
Handles MIMIC's nested structure: block/subject/record

processing_pipeline.py: End-to-end signal processing

Resampling to 125 Hz
Chebyshev-II bandpass filter (0.5-8 Hz)
Wavelet denoising (db4, level 5)
10-minute windowing with SQI gating

quality.py: Signal Quality Index (SQI) computation

SNR, zero-crossing rate, perfusion index
Flatline & saturation detection
Composite score 0-1 scale

Preprocessing Parameters:
yamlsampling_rate: 125 Hz
filter: Chebyshev-II (0.5-8 Hz, order 4)
denoising: Wavelet db4, level 5, soft thresholding
segmentation: 10-min windows (75,000 samples), no overlap
min_sqi: 0.5 (segments below threshold rejected)
Results Achieved

Subjects scanned: 500
Records discovered: ~50,000

Constraints & Learnings

Colab T4 GPU: 12-hour session limit, requires checkpointing
WFDB API quirk: Must use rdrecord(physical=False) for metadata (rdheader returns None for sig_name)
MIMIC structure: RECORDS files list directories, not files; must read subject-level RECORDS
NaN corruption: ~10-20% of records contain NaN values, must pre-filter

SPRINT 2: FEATURE ENGINEERING ✅ COMPLETE
Objectives

Extract physiological features from preprocessed signals
Implement HRV analysis (time/frequency/nonlinear domains)
Extract PPG morphology features
Save feature matrix for model training

Deliverables

✅ data/features/feature_matrix.parquet (3,660 × 37 features)
✅ All feature extraction modules functional
✅ 99.97% feature completeness (only 1 segment incomplete)

Feature Breakdown
Total: 37 features (originally 42, dropped 5 problematic ones)

HRV Time-Domain (12 features):

mean_rr, sdnn, rmssd, sdsd, pnn50, pnn20
mean_hr, max_hr, min_hr, hr_std, tri_index, tinn

HRV Frequency-Domain (10 features):

vlf_power, lf_power, hf_power, total_power
lf_hf_ratio, lf_norm, hf_norm
vlf_peak, lf_peak, hf_peak

HRV Nonlinear (8 features):

sampen, apen (entropy)
dfa_alpha1 (fractal scaling)
sd1, sd2, sd1_sd2_ratio (Poincaré plot)
corr_dim (correlation dimension)
Note: dfa_alpha2 dropped - 100% NaN (NeuroKit2 bug)

PPG Morphology (6 features):

pulse_amplitude, pulse_width, crest_time
stiffness_index, area_under_curve
Note: reflection_index dropped - constant value (placeholder)

Clinical Context (6 features - PLACEHOLDERS):

age, bmi, sex (all set to defaults: 50, 25, 0.5) My next session should replace these placeholders to the clinical ICD codes
age_x_sdnn, bmi_x_lf_hf, rmssd_over_sdnn
Note: Will be replaced with actual demographics in Sprint 4

Technical Implementation
Key Modules:

hrv_features.py: Uses NeuroKit2 for peak detection

Extracts RR intervals from PPG peaks
Computes time/freq/nonlinear features
Handles edge cases (minimum 30 peaks required)

morphology_features.py: Pulse wave analysis

Per-pulse feature extraction
Averages across all pulses in 10-min window

feature_extractor.py: Unified pipeline

Orchestrates all extractors
Batch processing with progress tracking
Handles NaN gracefully

Results Achieved

Features extracted: 135,420 total feature values
Completeness: 3,659/3,660 segments 100% complete
Quality: 0 constant features, 0 infinite values
Correlation: High correlations within domains (expected)

Known Issues

dfa_alpha2 always NaN: NeuroKit2 known issue, dropped from feature set
Clinical features are placeholders: Need integration with MIMIC clinical database

Constraints

NeuroKit2 dependency: Required for HRV analysis
Peak detection threshold: Minimum 30 peaks needed (~50 bpm × 10 min)
Memory: Feature extraction done in-memory (manageable at 3,660 segments)

. FEATURE SELECTION: MULTICOLLINEARITY REMOVAL
6.1 Manual Drops (Domain Knowledge)
Dropped Features (9 total):
FeatureReasonKept AlternativesdsdDerived from rmssdrmssdtinnLess informativesdnnpnn20Lower threshold versionpnn50total_powerSum of VLF+LF+HFIndividual bandsvlf_powerLess clinical relevancelf_power, hf_powerlf_norm, hf_normRedundant normalizationlf_hf_ratiovlf_peakNoisy, unreliablelf_peak, hf_peak
Rationale: Correlation block (r > 0.9) indicates variance inflation.

6.2 VIF Analysis
Added: Variance Inflation Factor computation
pythonfrom statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
Threshold: Drop features with VIF > 10.

7. CONFIGURATION CHANGES
   7.1 Processing Parameters
   yaml# configs/preprocessing.yaml
   filtering:
   lowcut: 0.5 # Hz (preserve respiratory sinus arrhythmia)
   highcut: 8.0 # Hz (remove high-freq noise)

segmentation:
window_sec: 600 # 10 minutes (was 15, reduced for yield)

quality:
min_sqi: 0.5 # Accept "Good" quality (was 0.7)

SPRINT 3: MODEL TRAINING 🔄 IN PROGRESS - BLOCKED
Objectives

Generate/link clinical outcome labels (ICD-9 codes)
Implement subject-wise train/test split
Train baseline models (XGBoost)
Train CNN encoder (1D-ResNet) on raw signals
Implement SQI-gated fusion model

Current Blocker
Minimum 20-30 subjects needed for proper validation
Planned Deliverables

⏳ artifacts/models/xgboost/\*.pkl (multi-label classifiers)
⏳ artifacts/models/cnn_encoder/weights.pth (1D-ResNet)
⏳ artifacts/models/fusion/hybrid_model.pkl (gated fusion)
⏳ Training reports with AUPRC, calibration curves

Technical Approach
Model Architecture:

Classical Baseline:

XGBoost per condition (5 separate models)
Input: hand-crafted features
Loss: BCEWithLogitsLoss with scale_pos_weight

CNN Encoder:

1D-ResNet (4 residual blocks)
Input: Raw signals (75,000 samples)
Output: 256-dim embeddings
Pre-training: Temporal contrastive learning (windows from same patient)

Hybrid Fusion:

SQI-gated weighting: weight = sigmoid(0.3 + 0.7 × SQI)
Combines classical + CNN outputs
Higher SQI → trust CNN more; lower SQI → trust classical more

Label strategy MIMIC Clinical Integration (proper approach)
Download ADMISSIONS.csv, DIAGNOSES_ICD.csv
Link via SUBJECT_ID and temporal alignment
Extract ICD-9 codes: 250.xx (diabetes), 401.x (hypertension), 272.x (dyslipidemia), 278.0x (obesity)
Compute Charlson Comorbidity Index using comorbidipy
NB: The MIMIC subset data on PhysioNet is open access
Training should be optimized for to run T4 GPU and on a RAM of 12GB. Use patch processing and paralle processing if applicable to reduce the burden.

Training Configuration:
yamlvalidation:
strategy: GroupKFold # Subject-wise splits
n_splits: 5
test_size: 0.2

xgboost:
n_estimators: 150
max_depth: 6
learning_rate: 0.05
objective: binary:logistic
eval_metric: aucpr

cnn:
batch_size: 32
epochs: 50
learning_rate: 0.001
optimizer: Adam
loss: FocalLoss (alpha=0.25, gamma=2.0) # For class imbalance
Critical Requirements

Subject-wise splitting: NEVER split same subject across train/test

python from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=metadata['subject_id']))

```

2. **Stratified sampling:** Ensure all subjects represented in splits

3. **Metric priority:** AUPRC > Accuracy (for imbalanced classes)

### **Constraints**
- **Minimum subjects:** 20-30 for GroupKFold CV
- **Colab GPU limits:** T4, 12-hour sessions
- **Class imbalance:** Expected 10-30% positive rate per condition
- **Training time:** CNN ~2-4 hours, XGBoost ~10 minutes

---

 CLINICAL DATA INTEGRATION** ⏳ **PENDING**

### **Objectives**
1. Download MIMIC-III Clinical Database tables
2. Link waveform records to patient diagnoses
3. Extract cardiometabolic ICD-9 codes
4. Compute Charlson Comorbidity Index
5. Create multi-hot label vectors

### **Planned Deliverables**
- ⏳ `data/metadata/clinical_labels.parquet` (segment-level labels)
- ⏳ `colab_src/data_pipeline/clinical_linker.py` (complete implementation)
- ⏳ ICD-9 code mappings and validation reports

### **Technical Approach**

**Data Required:**
```

MIMIC-III Clinical Database tables:

- ADMISSIONS.csv (hospital admission records)
- DIAGNOSES_ICD.csv (ICD-9 diagnosis codes)
- PATIENTS.csv (demographics: age, sex)
- (Optional) LABEVENTS.csv (BMI calculation from height/weight)
  Label Generation Logic:
  python# Target ICD-9 codes
  target_codes = {
  'diabetes': ['250.00', '250.02', ...], # Type 2 DM
  'hypertension': ['401.9', '402.xx', ...],
  'dyslipidemia': ['272.0', '272.4', ...],
  'obesity': ['278.01', '278.02', ...]
  }

# Temporal alignment

# Only use diagnoses ACTIVE at time of waveform recording

signal_time = record.base_datetime
admission_time = ADMISSIONS[SUBJECT_ID]['ADMITTIME']
valid = (admission_time <= signal_time <= discharge_time)

# Multi-hot encoding

labels = {
'diabetes': 1 if any diabetes ICD-9 else 0,
'hypertension': 1 if any HTN ICD-9 else 0,
'dyslipidemia': 1 if any lipid ICD-9 else 0,
'obesity': 1 if any obesity ICD-9 else 0,
'healthy': 1 if NO cardiometabolic codes else 0
}
Charlson Index:
pythonimport comorbidipy
cci_score = comorbidipy.charlson(icd9_codes, assign_icd_version=9)
Constraints

PhysioNet access required(Open access): Clinical database separate from waveforms
Temporal alignment critical: Avoid future leakage
Data linkage: Not all waveforms have clinical data (Matched Subset ~30% coverage)
ICD-9 specificity: Use 5-digit codes for accurate severity (e.g., 250.00 vs 250.92)

SPRINT 4: MODEL EVALUATION & INTERPRETABILITY ⏳ PENDING
Objectives

Comprehensive model evaluation (AUPRC, calibration, fairness)
SHAP-based interpretability
Generate clinical validation reports
Uncertainty quantification

Planned Deliverables

⏳ artifacts/evaluation/metrics_report.json
⏳ artifacts/evaluation/calibration_curves.pkl
⏳ artifacts/explainability/shap_explainer.pkl
⏳ Clinical validation report (PDF/HTML)

Technical Approach
Evaluation Metrics:
python# Primary metrics (per condition)

- AUPRC (Area Under Precision-Recall Curve)
- AUROC (Area Under ROC Curve)
- F2-score (recall-weighted, prefer false positives over false negatives)
- Sensitivity, Specificity, PPV, NPV

# Calibration

- Expected Calibration Error (ECE < 0.10)
- Reliability diagrams (predicted vs actual probabilities)

# Fairness

- Performance stratified by age, sex, BMI
- Bias detection using demographic parity difference
  SHAP Interpretability:
  python# Global feature importance
  shap_explainer = shap.TreeExplainer(xgboost_model)
  shap_values = shap_explainer.shap_values(X_test)

# Local explanations (per prediction)

shap.force_plot(explainer.expected_value, shap_values[i], X_test.iloc[i])
Uncertainty Estimation:

MC Dropout for neural models
Ensemble variance across model families
Prediction intervals (95% confidence)

Success Criteria

AUPRC >0.75 for high-risk conditions
AUPRC >0.65 for healthy controls (10% prevalence)
ECE <0.10 (well-calibrated)
No >20% performance gap across demographic groups

SPRINT 5: API DEPLOYMENT ⏳ PENDING
Objectives

Export trained models to API-compatible formats
Build FastAPI endpoints
Implement real-time inference pipeline
Deploy with Docker

Planned Deliverables

⏳ exports/models/\*.onnx (ONNX format for API)
⏳ FastAPI application (separate codebase)
⏳ Docker container with all dependencies
⏳ API documentation (OpenAPI spec)

Technical Approach
Model Export:
python# PyTorch → ONNX
torch.onnx.export(cnn_encoder, dummy_input, 'cnn_encoder.onnx')

# XGBoost → Pickle

joblib.dump(xgb_models, 'xgboost_models.pkl')

# Preprocessing pipelines

joblib.dump(sqi_engine, 'sqi_engine.pkl')

```

**API Endpoints:**
```

POST /api/v1/predict

- Input: PPG signal (75,000 samples), metadata
- Output: Risk scores, confidence, SHAP values

GET /api/v1/baseline/{user_id}

- Output: User's historical baseline features

POST /api/v1/baseline/update

- Update user's baseline with new recording
  API will be developed in VS Code (separate from Colab notebooks)

CURRENT STATUS & IMMEDIATE PRIORITIES
✅ Completed (Ready for Production)

Signal processing pipeline (SQI, filtering, denoising, segmentation)
Feature extraction (37 physiological features)
Data validation and quality assurance
Experiment tracking (MLflow)

1. Why 10-minute windows?

Minimum for reliable HRV frequency analysis (needs ≥5 minutes)
Trade-off: Longer = better features, but fewer segments
10 min = sweet spot for clinical HRV standards

2. Why SQI gating at 0.5?

Empirically determined threshold
Rejects ~30% of segments but ensures high-quality features
Can be lowered to 0.3-0.4 if more data needed

3. Why hybrid (classical + CNN)?

Classical features: Interpretable, robust to noise
CNN features: Captures subtle morphology patterns
Fusion: Best of both worlds, SQI-weighted confidence

4. Why subject-wise splitting?

Critical: Same person in train & test = data leakage
Patient-specific patterns (sensor placement, physiology) would inflate performance
Real-world deployment: Model sees NEW patients, not new recordings of known patients

5. Why multi-label (not multi-class)?

Patients often have multiple conditions (diabetes + hypertension)
Multi-label: Each condition predicted independently
Multi-class would force single-condition assumption (unrealistic)

DEPENDENCIES & SETUP
Python Environment
bash# Core dependencies
pip install wfdb 4.1.2 neurokit2 pandas2.2.2 numpy scipy scikit-learn
pip install torch torchvision torchaudio # PyTorch
pip install xgboost lightgbm
pip install mlflow pywavelets hydra-core
pip install shap plotly seaborn

# Clinical tools

pip install comorbidipy # Charlson index
Data Access

PhysioNet account required (free access)
Datasets needed:

MIMIC-III Waveform Database Matched Subset (for waveforms)
MIMIC-III Clinical Database (for ICD-9 codes, demographics)

Compute Requirements

Colab T4 GPU: Sufficient for current scale
Scaling to 10K+ segments: May need Colab Pro or local GPU
RAM: 12 GB minimum (current dataset fits in memory in batches)

KNOWN ISSUES & WORKAROUNDS

1. WFDB API Quirks
   Issue: rdheader() returns None for sig_name
   Workaround: Use rdrecord(physical=False) for metadata
2. NeuroKit2 dfa_alpha2 Bug
   Issue: Always returns NaN
   Workaround: Dropped from feature set (37 instead of 42 features)
3. Colab Session Limits
   Issue: 12-hour disconnects
   Workaround: Checkpoint every 50-100 records, resume-capable pipelines
4. MIMIC PPG Scarcity
   Issue: Only ~0.35% of records have PPG
   Workaround: Pre-filter with metadata streaming before downloading signals
5. NaN Corruption
   Issue: ~10-20% of downloaded signals contain NaN
   Workaround: Pre-validation cell before batch processing

TESTING & VALIDATION
Unit Tests Needed (Not Yet Implemented)
python# tests/test_signal_processing.py
def test_sqi_computation():
"""Verify SQI scores in valid range [0, 1]"""

def test_feature_extraction():
"""Ensure all features are numeric, no nested arrays"""

def test_subject_wise_split():
"""Verify no subject appears in both train and test"""
Integration Tests
python# tests/test_pipeline.py
def test_end_to_end_pipeline():
"""Single record → features → prediction"""
Before proceeding, clarify:

QUICK START GUIDE
To Resume Work

# 2. Navigate to project

# 3. Check current status

ls -lh data/processed/ # Should see sprint1_signals.npy (2.2 GB)
ls -lh data/features/ # Should see feature_matrix.parquet (1.2 MB)

# 4. Load latest data

import pandas as pd
features = pd.read_parquet('data/features/feature_matrix.parquet')
print(f"Current dataset: {len(features)} segments, {features['subject_id'].nunique()} subjects")
To Continue to Sprint 3:

CONTACT & RESOURCES

MLflow UI: logs/mlruns/ (view with mlflow ui)
Documentation: Check each module's docstrings
Hydra configs: configs/\*.yaml for parameter tuning

Key Files to Understand:

colab_src/data_pipeline/mimic_ingestion.py - Data loading logic
colab_src/signal_processing/quality.py - SQI computation
colab_src/features/feature_extractor.py - Feature engineering
notebooks/01_data_exploration.ipynb - Data pipeline walkthrough
