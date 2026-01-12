# Cardiometabolic Risk from PPG Signals

## Completion Status

- Phase 0–3: ✅ COMPLETE (Data prep + SSL modules + pilot training)
- Phase 4: ✅ COMPLETE (Colab deployment setup)
- Phase 5–8: 🚀 READY (Colab GPU training → production API)

### Phase 0-3 Outputs

- **Data**: 4,417 PPG signals (train: 4,133, val: 200, test: 84)
- **SSL Modules**: 9 production-ready modules (encoder, decoder, losses, augmentation, trainer)
- **Tests**: 39 passing, 0 failures
- **Validation**: Pilot training 1 epoch, loss 6.88→2.06 (70% convergence ✓)

### Phase 4 Deliverables

- **Validation Script**: `validate_phase4.py` — Quick module checks (<5 seconds)
- **Colab Notebook**: `notebooks/05_ssl_pretraining_colab.ipynb` — Single-page orchestration
- **Utils Helper**: `colab_src/utils/colab_utils.py` — Optional Colab detection
- **Documentation**: Updated README with Phase 5-8 roadmap

### Phase 5-8 Roadmap

| Phase | Goal                                  | Duration | Gate                                        |
| ----- | ------------------------------------- | -------- | ------------------------------------------- |
| **5** | Full 50-epoch training on Colab T4    | 8–12h    | Val loss <0.01, SSIM >0.80                  |
| **6** | Linear probe evaluation               | 1–2h     | AUROC >0.65 on ≥1 cardiometabolic condition |
| **7** | Extract embeddings + crafted features | 30min    | 551-dim feature matrix ready                |
| **8** | XGBoost downstream models             | 2–3h     | AUROC ≥0.70 on ≥2 conditions                |

---

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/cardiometabolic-risk-colab
cd cardiometabolic-risk-colab

# Install dependencies
pip install -r requirements.txt

# Run Phase 4 validation (quick check, <5 seconds)
python validate_phase4.py
```

### Colab Training

1. Open [Colab Notebook](notebooks/05_ssl_pretraining_colab.ipynb)
2. Run cells in order:
   - Mount Google Drive
   - Clone repository
   - Install dependencies
   - Verify GPU
   - Train (50 epochs, ~8-12 hours)
3. Download checkpoints from Drive

---

## Project Structure

```
cardiometabolic-risk-colab/
├── colab_src/
│   ├── models/ssl/              # SSL modules (Phase 1)
│   │   ├── config.py            # YAML config loader
│   │   ├── encoder.py           # ResNet encoder (2.8M params)
│   │   ├── decoder.py           # ResNet decoder (1.2M params)
│   │   ├── losses.py            # Multi-loss (MSE+SSIM+FFT)
│   │   ├── augmentation.py      # Label-free augmentations
│   │   ├── dataloader.py        # Lazy-loading dataset
│   │   ├── trainer.py           # Training loop with grad accumulation
│   │   ├── train.py             # CLI entry point
│   │   └── __init__.py
│   └── utils/
│       └── colab_utils.py       # Colab helper functions
├── configs/
│   └── ssl_pretraining.yaml     # Full config (model, loss, training)
├── data/
│   ├── raw/                     # MIMIC CSV files
│   ├── processed/               # Parquets + denoised signals
│   ├── cache/                   # Cached metadata
│   └── metadata/                # Dataset statistics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_quality_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_clinical_data_integration.ipynb
│   └── 05_ssl_pretraining_colab.ipynb   # Phase 5 entry point
├── checkpoints/                 # Model checkpoints
│   ├── phase3/                  # Phase 3 pilot checkpoint
│   └── phase5/                  # Phase 5 full training (will be created)
├── logs/                        # MLflow logs
├── tests/                       # 39 passing tests
├── context/                     # Technical documentation
├── validate_phase4.py           # Phase 4 validation script
├── requirements.txt             # Locked dependencies
└── README.md
```

---

## Architecture Overview

### Model

- **Encoder**: 1D ResNet, 75K→512-dim latent, 2.8M params
- **Decoder**: Transposed ResNet, 512→75K, 1.2M params
- **Total**: 4.0M parameters

### Loss Function

$$L_{total} = 0.50 \cdot L_{MSE} + 0.30 \cdot L_{SSIM} + 0.20 \cdot L_{FFT}$$

- **MSE**: Pixel-level temporal reconstruction
- **SSIM**: Structural similarity (morphology preservation)
- **FFT**: Frequency domain alignment (heart rate preservation)

### Training

- **Batch size**: 8 (with 4× gradient accumulation = effective 32)
- **Optimizer**: Adam, LR=0.001
- **Mixed precision**: FP16 forward/backward, FP32 optimizer
- **Epochs**: 50
- **Data**: 4,133 training signals (10 min each @ 125 Hz)

---

## Documentation

- **Architecture**: [context/architecture.md](context/architecture.md) — System design, MIMIC integration
- **Phase 1 Implementation**: [context/phase_1_implementation.md](context/phase_1_implementation.md) — Technical specs
- **Phase 1 Checklist**: [context/phase_1_checklist.md](context/phase_1_checklist.md) — Verification tests
- **Phase 1 Review**: [context/PHASE_1_REVIEW.md](context/PHASE_1_REVIEW.md) — Code quality, performance analysis
- **Phase 0 Review**: [PHASE0_IMPLEMENTATION_REVIEW.md](PHASE0_IMPLEMENTATION_REVIEW.md) — Data pipeline design
- **Implementation Index**: [IMPLEMENTATION_INDEX.md](IMPLEMENTATION_INDEX.md) — Complete file reference

---

## Performance Predictions

### Phase 5: Full Training (50 epochs)

- **Hardware**: Colab T4 GPU (16GB VRAM)
- **Duration**: 8–12 hours (≈10 min/epoch)
- **Memory**: Peak ~4–5 GB
- **Expected losses**:
  - MSE: 0.005–0.01 (pixel-level error)
  - SSIM: 0.80–0.90 (structural similarity)
  - Total: 0.01–0.02

### Phase 6: Linear Probe

- **Validation samples**: 24 (holdout test set)
- **Expected AUROC**: >0.65 on ≥1 cardiometabolic condition (gate)

### Phase 8: XGBoost Downstream

- **Training samples**: 60 (labeled test set)
- **Expected AUROC**: ≥0.70 on ≥2 conditions
- **Features**: 551-dim (512 latent + 39 crafted)

---

## Next Steps

1. **Phase 5**: Run full training in Colab

   - Expected: 8–12 hours on T4
   - Output: `best_encoder.pt` checkpoint

2. **Phase 6**: Linear probe evaluation

   - Quick validation of learned representations
   - Gate: AUROC >0.65

3. **Phase 7**: Embeddings + features

   - Extract 512-dim latent vectors for all 4,417 samples
   - Combine with 39 hand-crafted features

4. **Phase 8**: XGBoost models
   - Train 4 downstream classifiers (diabetes, hypertension, obesity, CCI)
   - Evaluate on 24-sample test set
   - Export as pickle + ONNX for production API

---

## Citation

If you use this codebase, please cite:

```bibtex
@dataset{cardiometabolic_ppg_2026,
  title={Self-Supervised Learning for Cardiometabolic Risk from PPG Signals},
  author={Your Name},
  year={2026},
  note={Based on MIMIC-III Matched Subset}
}
```

---

## License

MIT License — See LICENSE file

---

## Questions?

See [context/](context/) directory for detailed technical documentation.
