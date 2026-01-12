# Cardiometabolic Risk from PPG Signals

## Completion Status

- Phase 0–3: ✅ COMPLETE (Data prep + SSL modules + pilot training)
- Phase 4–8: ⏳ IN PROGRESS (GitHub/Colab deployment → production API)

### Phase 0-3 Outputs

- Data: 4,417 PPG signals (train: 4,133, val: 200, test: 84)
- SSL Modules: 9 production-ready modules (encoder, decoder, losses, augmentation)
- Tests: 39 passing, 0 failures
- Pilot: 1-epoch training, loss 6.88→2.06 (70% convergence)

### Phase 4-8 Roadmap

1. Phase 4: Push to GitHub, create Colab notebook
2. Phase 5: Full 50-epoch training on Colab T4
3. Phase 6: Linear probe evaluation (gate: AUROC >0.65)
4. Phase 7: Extract embeddings + crafted features
5. Phase 8: Train XGBoost downstream models (production API)
