cardiometabolic-risk-colab/
│
├── notebooks/ # Jupyter notebooks for exploration & experiments
│ ├── 01_data_exploration.ipynb
│ ├── 02_signal_quality_analysis.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_model_training.ipynb
│ ├── 05_model_evaluation.ipynb
│ └── 06_interpretability.ipynb
│
├── data/ # All datasets (Colab storage)
│ ├── raw/ # Downloaded MIMIC signals (.npy)
│ ├── processed/ # Filtered & windowed signals (.parquet, .npy)
│ ├── metadata/ # Demographics, ICD-9 codes, record info (.csv)
│ └── cache/ # Temporary processing files
│
├── artifacts/ # Trained models & pipelines
│ ├── models/ # Trained model weights
│ │ ├── cnn_encoder/
│ │ │ ├── pretrained_weights.pth
│ │ │ └── finetuned_weights.pth
│ │ ├── xgboost/
│ │ │ ├── diabetes_model.pkl
│ │ │ ├── hypertension_model.pkl
│ │ │ └── [other conditions].pkl
│ │ └── fusion/
│ │ └── hybrid_fusion_v1.pkl
│ │
│ ├── preprocessing/ # Preprocessing pipelines
│ │ ├── sqi_engine.pkl
│ │ ├── ppg_filter.pkl
│ │ ├── wavelet_denoiser.pkl
│ │ └── scaler_params.pkl
│ │
│ ├── features/ # Feature engineering artifacts
│ │ ├── feature_definitions.yaml # Feature catalog
│ │ ├── population_baselines.pkl # Stratified baselines
│ │ └── baseline_selector.pkl
│ │
│ ├── explainability/ # Interpretability models
│ │ ├── shap_explainer_classical.pkl
│ │ └── shap_explainer_fusion.pkl
│ │
│ └── evaluation/ # Model performance reports
│ ├── metrics_report.json
│ ├── calibration_curves.pkl
│ └── confusion_matrices.npy
│
├── colab_src/ # Python modules (Colab execution)
│ ├── **init**.py
│ │
│ ├── data_pipeline/ # Data ingestion & processing
│ │ ├── **init**.py
│ │ ├── mimic_ingestion.py # WFDB streaming & download
│ │ ├── clinical_linker.py # ICD-9 code matching
│ │ └── parquet_converter.py
│ │
│ ├── signal_processing/ # Signal preprocessing
│ │ ├── **init**.py
│ │ ├── quality.py # SQI engine
│ │ ├── filters.py # Bandpass filtering
│ │ ├── denoising.py # Wavelet denoising
│ │ └── segmentation.py # Windowing
│ │
│ ├── features/ # Feature engineering
│ │ ├── **init**.py
│ │ ├── hrv_features.py # Time/Freq/Nonlinear HRV
│ │ ├── morphology_features.py # PPG morphology
│ │ ├── clinical_context.py # Age/BMI/CCI encoding
│ │ ├── baseline_selector.py # Longitudinal baseline logic
│ │ └── feature_extractor.py # Unified extraction pipeline
│ │
│ ├── models/ # Model architectures
│ │ ├── **init**.py
│ │ ├── cnn_encoder.py # 1D-ResNet implementation
│ │ ├── xgboost_classifier.py # Multi-label XGBoost
│ │ ├── fusion_model.py # SQI-gated fusion
│ │ └── losses.py # Focal loss, custom losses
│ │
│ ├── training/ # Training orchestration
│ │ ├── **init**.py
│ │ ├── pretrain_cnn.py # Contrastive pre-training
│ │ ├── train_classical.py # XGBoost training
│ │ ├── train_fusion.py # Hybrid model training
│ │ └── utils.py # DataLoaders, callbacks
│ │
│ ├── evaluation/ # Model evaluation
│ │ ├── **init**.py
│ │ ├── clinical_metrics.py # AUPRC, sensitivity, NPV
│ │ ├── calibration.py # Reliability diagrams
│ │ ├── fairness.py # Bias detection
│ │ └── interpretability.py # SHAP wrappers
│ │
│ ├── validation/ # Data quality checks
│ │ ├── **init**.py
│ │ ├── data_quality_tests.py
│ │ └── stationarity_check.py
│ │
│ └── utils/ # Shared utilities
│ ├── **init**.py
│ ├── experiment_tracker.py # MLflow integration
│ ├── config_loader.py # Hydra config management
│ └── visualization.py # Plotting helpers
│
├── exports/ # API-ready artifacts (final export)
│ ├── models/ # Serialized models for deployment
│ │ ├── cnn_encoder.onnx # ONNX format for API
│ │ ├── xgboost_models.pkl
│ │ └── fusion_model.pkl
│ │
│ ├── preprocessing/ # Preprocessing for API inference
│ │ ├── sqi_engine.pkl
│ │ ├── filter_params.json
│ │ └── baseline_selector.pkl
│ │
│ ├── feature_definitions/ # Feature catalog for API
│ │ └── features.yaml
│ │
│ └── metadata/ # Model cards & documentation
│ ├── model_card.json # Model metadata
│ ├── feature_importance.json
│ └── performance_metrics.json
│
├── configs/ # Hydra configuration files
│ ├── config.yaml # Main config
│ ├── data/
│ │ ├── mimic.yaml
│ │ └── processing.yaml
│ ├── models/
│ │ ├── cnn.yaml
│ │ ├── xgboost.yaml
│ │ └── fusion.yaml
│ ├── training/
│ │ └── default.yaml
│ └── experiment/
│ └── ablation.yaml
│
├── logs/ # Experiment tracking
│ ├── mlruns/ # MLflow artifacts
│ └── tensorboard/ # TensorBoard logs
│
├── requirements.txt # Colab dependencies
├── setup_colab.sh # Colab initialization script
└── README.md # Project documentation
