To ensure the success of the Multimodal Cardiometabolic Risk Estimation project, I have customized your reasoning and planning framework. Use these tailored instructions to guide every phase of development—from raw signal ingestion in MIMIC-III to clinical explainability.

1. Logical Dependencies & Clinical Constraints
   1.1 Physiological Fidelity First: Resolve conflicts in favor of clinical validity over raw accuracy. An 85% accurate model that uses physiologically sound features is superior to a 95% "black box" that might be picking up on hospital-specific noise (e.g., ventilator frequencies). 1.2 The Dependency Chain: 1. Signal Quality Index (SQI) must gate all downstream tasks. 2. Metadata Matching (MIMIC Clinical <-> Waveform) must be verified before feature extraction to prevent patient-record mismatch. 3. Subject-Wise Splitting is a mandatory prerequisite for evaluation to prevent data leakage. 1.3 Order of Operations: Reorder tasks to ensure Baseline Personalization (Longitudinal data) is established before calculating "Risk Deltas."

2. Risk Assessment (Bio-Signal Focus)
   2.1 False Negative Risk: In cardiometabolic screening, missing a high-risk patient is a critical failure. Evaluate the consequences of signal "smoothing"—did the filter remove a relevant arrhythmia or dicrotic notch? 2.2 Data Privacy: Ensure that while joining the Matched Subset, no Protected Health Information (PHI) is exposed during logging or experiment tracking (W&B/MLflow). 2.3 Sensor Bias: Assess the risk of the model performing differently across skin tones (Melanin levels) due to PPG physics—this must be a primary check in the fairness.py layer.

3. Abductive Reasoning: "Signal-to-Symptom"
   3.1 Root Cause Analysis of Noise: If the model fails on a segment, look beyond "motion." Is the issue sensor saturation, low perfusion (cold extremities), or a genuine physiological event like peripheral vasoconstriction? 3.2 Hypothesis Exploration: If HRV proxies are impaired, test multiple hypotheses:

Hypothesis A: Autonomic imbalance (The intended target).

Hypothesis B: Algorithmic artifact (Ectopic beats misidentified as normal RR-intervals).

Hypothesis C: Metadata conflict (The patient was on Beta-blockers, suppressing HRV).

4. Outcome Evaluation & Adaptability
   4.1 Model Drift vs. Physiological Shift: If performance drops during the "Monitoring" phase, determine if the model is drifting or if the user population’s health baseline has shifted (e.g., a seasonal change in activity levels). 4.2 Feedback Loops: If SHAP values indicate that "Age" is the only contributing factor, the plan must pivot to de-biasing the encoders to force them to look at the PPG morphology.

5. Information Availability (The MIMIC Context)
   5.1 Clinical Grounding: Cross-reference every "Deep Feature" with established medical literature. Does the 1D-CNN latent space correlate with Pulse Wave Velocity (PWV)? 5.2 Tool Utilization: Use wfdb for signal integrity, NeuroKit2 for feature gold-standards, and Great Expectations to enforce the "Data Contract" for the Matched Subset.

6. Precision and Grounding
   6.1 Clinical Codes: Verify all "Risk" labels by quoting exact ICD-9/10 codes from the DIAGNOSES_ICD table. A "Diabetes" label is only as good as the code that generated it. 6.2 Unit Standardization: Ensure all signals are normalized to the same sampling frequency (e.g., 125Hz) before being fed into the cnn_encoder.py.

7. Completeness & Persistence
   7.1 Exhaustive Feature Search: Do not conclude a feature set is "optimal" until Time, Frequency, and Non-Linear (Entropy) domains have been evaluated against the cardiometabolic target. 7.2 Intelligent Retry: If the clinical_linker.py fails to join a waveform to a patient record, do not just drop the data. Investigate the HADM_ID (Hospital Admission ID) to see if the record exists in a different version of the subset.

Implementation Roadmap Checklist
[ ] Phase 1: Build the clinical_linker.py to join MIMIC Waveforms + Clinical Tables. This is already simplified since I'm using the MIMIC-III Waveform Database Matched Subset. This's because the full MIMIC-III Waveform Database offers maximal signal volume with limited clinical linkage, while the MIMIC-III Waveform Database Matched Subset provides fewer waveforms but reliable patient-level metadata, making the matched subset more suitable for supervised machine learning and clinically contextualized PPG analysis.

[ ] Phase 2: Implement the SQI-gate in quality.py to prevent "garbage-in."

[ ] Phase 3: Train the cnn_encoder.py (Unsupervised/Self-supervised) on the full Waveform set.

[ ] Phase 4: Train the xgboost_head.py using the Matched Subset labels.

[ ] Phase 5: Run Decision Curve Analysis in evaluation/ to prove clinical utility.
I started this project in colab and I'm continuing it here in vscode.
