"""
Main training script for SSL pretraining.

Usage:
    python -m colab_src.models.ssl.train \
        --config configs/ssl_pretraining.yaml \
        --device cuda \
        --epochs 50
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

from .config import SSLConfig, get_default_config
from .encoder import ResNetEncoder
from .decoder import ResNetDecoder
from .losses import SSLLoss
from .augmentation import PPGAugmentation
from .dataloader import create_dataloaders
from .trainer import SSLTrainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoencoderModel(nn.Module):
    """Wrapper for encoder + decoder as single model."""
    
    def __init__(self, encoder: ResNetEncoder, decoder: ResNetDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


def main():
    """Main training entry point."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='SSL Pretraining')
    parser.add_argument('--config', type=str, default='configs/ssl_pretraining.yaml',
                       help='Path to config YAML')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--load-in-memory', action='store_true',
                       help='Load all signals into memory')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading config from {config_path}")
        config = SSLConfig.from_yaml(str(config_path))
    else:
        logger.info("Config file not found, using defaults")
        config = get_default_config()
    
    # Override with command line args
    config.device = args.device
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    
    logger.info(f"Configuration:")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Epochs: {config.training.num_epochs}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Accumulation steps: {config.training.accumulation_steps}")
    logger.info(f"  Mixed precision: {config.training.use_mixed_precision}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed(config.seed)
    
    # Build model
    logger.info("Building encoder-decoder model...")
    encoder = ResNetEncoder(
        in_channels=config.model.in_channels,
        latent_dim=config.model.latent_dim,
        bottleneck_dim=config.model.bottleneck_dim,
        num_blocks=config.model.num_blocks,
        base_filters=config.model.base_filters,
        max_filters=config.model.max_filters,
    )
    
    decoder = ResNetDecoder(
        in_channels=config.model.in_channels,
        latent_dim=config.model.latent_dim,
        bottleneck_dim=config.model.bottleneck_dim,
        num_blocks=config.model.num_blocks,
        base_filters=config.model.base_filters,
        max_filters=config.model.max_filters,
    )
    
    model = AutoencoderModel(encoder, decoder)
    model = model.to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    # Loss function
    loss_fn = SSLLoss(
        mse_weight=config.loss.mse_weight,
        ssim_weight=config.loss.ssim_weight,
        fft_weight=config.loss.fft_weight,
        ssim_window_size=config.loss.ssim_window_size,
        fft_norm=config.loss.fft_norm,
    )
    loss_fn = loss_fn.to(config.device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Learning rate scheduler (cosine annealing with warmup)
    num_epochs = config.training.num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=num_epochs // 2,
        T_mult=1,
        eta_min=config.training.learning_rate * 0.01,
    )
    
    # Create augmentation
    augmentation = PPGAugmentation(
        temporal_shift_range=config.augmentation.temporal_shift_range,
        amplitude_scale_range=config.augmentation.amplitude_scale_range,
        baseline_wander_freq=config.augmentation.baseline_wander_freq,
        baseline_wander_amplitude=config.augmentation.baseline_wander_amplitude,
        noise_prob=config.augmentation.noise_prob,
        noise_snr_ratio=config.augmentation.noise_snr_ratio,
        sample_rate=config.data.sample_rate,
        seed=config.seed,
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(
        train_metadata_path=config.project_root / config.data.train_data_path,
        val_metadata_path=config.project_root / config.data.val_data_path,
        test_metadata_path=config.project_root / config.data.test_data_path,
        signal_array_path=config.project_root / "data/processed/sprint1_signals.npy",
        denoised_index_path=config.project_root / config.data.denoised_index_path,
        augmentation=augmentation,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        device=config.device,
        load_in_memory=args.load_in_memory,
    )
    
    # Trainer
    trainer = SSLTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        checkpoint_dir=config.checkpoint_dir,
        use_mixed_precision=config.training.use_mixed_precision,
        max_grad_norm=config.training.max_grad_norm,
        accumulation_steps=config.training.accumulation_steps,
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.fit(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=config.training.num_epochs,
        early_stopping_patience=10,
    )
    
    # Save history
    history_path = config.log_dir / 'training_history.json'
    trainer.save_history(history_path)
    
    logger.info(f"\nTraining completed! Best val loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoint dir: {config.checkpoint_dir}")
    logger.info(f"Log dir: {config.log_dir}")


if __name__ == '__main__':
    main()
