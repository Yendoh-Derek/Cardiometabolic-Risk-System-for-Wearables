"""
Self-supervised learning modules for PPG signal pretraining.

Package structure:
- config.py: Configuration and hyperparameter management
- encoder.py: ResNet encoder architecture
- decoder.py: ResNet decoder architecture
- losses.py: Multi-loss training function (MSE + SSIM + FFT)
- augmentation.py: Signal augmentation pipeline
- dataloader.py: PPG dataset loader
- trainer.py: Training loop with gradient accumulation
- train.py: Main entry point
"""

__version__ = "0.1.0"
