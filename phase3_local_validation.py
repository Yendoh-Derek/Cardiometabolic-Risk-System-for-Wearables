"""
Phase 3: Local CPU Validation - SSL Pretraining Pilot Run

Goal: Verify end-to-end training pipeline on CPU with Phase 0 data
Duration: 2-3 epochs (< 1 hour on CPU)

This script:
1. Loads Phase 0 data (train/val/test splits + denoised signals)
2. Creates dataloaders with augmentation
3. Trains encoder-decoder on local CPU
4. Validates after each epoch
5. Saves checkpoints and metrics
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))

from models.ssl.config import SSLConfig
from models.ssl.encoder import ResNetEncoder
from models.ssl.decoder import ResNetDecoder
from models.ssl.losses import SSLLoss
from models.ssl.augmentation import PPGAugmentation
from models.ssl.dataloader import PPGDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleSSLTrainer:
    """Simplified SSL trainer for Phase 3 validation."""
    
    def __init__(self, encoder, decoder, loss_fn, config, device='cpu'):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.loss_fn = loss_fn.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = optim.Adam(
            params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.encoder.train()
        self.decoder.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (raw_signal, denoised_signal) in enumerate(train_loader):
            raw_signal = raw_signal.to(self.device)
            denoised_signal = denoised_signal.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Reconstruct raw signal
            latent = self.encoder(raw_signal)
            recon = self.decoder(latent)
            
            # Compute loss (reconstruct raw from denoised as reference)
            # This is self-supervised: no labels, just reconstruction quality
            loss = self.loss_fn(recon, raw_signal)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log every 20 batches (less chatty, faster output)
            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"Loss = {loss.item():.6f}",
                    flush=True
                )
        
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.encoder.eval()
        self.decoder.eval()
        
        epoch_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for raw_signal, denoised_signal in val_loader:
                raw_signal = raw_signal.to(self.device)
                denoised_signal = denoised_signal.to(self.device)
                
                # Forward pass
                latent = self.encoder(raw_signal)
                recon = self.decoder(latent)
                
                # Compute loss
                loss = self.loss_fn(recon, raw_signal)
                epoch_loss += loss.item()
                num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Save checkpoint if best
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best')
        
        return avg_loss
    
    def save_checkpoint(self, tag='latest'):
        """Save model checkpoint."""
        checkpoint_dir = PROJECT_ROOT / 'checkpoints' / 'phase3'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_{tag}.pt'
        
        torch.save({
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_phase0_data(batch_size=8, num_workers=0, subset_fraction=0.1):
    """Load Phase 0 data splits."""
    data_dir = PROJECT_ROOT / 'data' / 'processed'
    
    # Load metadata to check counts
    train_df = pd.read_parquet(data_dir / 'ssl_pretraining_data.parquet')
    val_df = pd.read_parquet(data_dir / 'ssl_validation_data.parquet')
    test_df = pd.read_parquet(data_dir / 'ssl_test_data.parquet')
    
    # For CPU pilot, use subset
    subset_size = max(1, int(len(train_df) * subset_fraction))
    train_df = train_df.iloc[:subset_size]
    val_df = val_df.iloc[:max(1, int(len(val_df) * subset_fraction))]
    
    signals_dir = data_dir / 'denoised_signals'
    denoised_index_path = data_dir / 'denoised_signal_index.json'
    
    logger.info(f"Train samples: {len(train_df)} (subset from 4133)")
    logger.info(f"Val samples: {len(val_df)} (subset from 200)")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Create datasets WITHOUT augmentation for speed (validation only, no augmentation)
    train_dataset = PPGDataset(
        metadata_path=str(data_dir / 'ssl_pretraining_data.parquet'),
        denoised_index_path=str(denoised_index_path),
        signal_dir=str(signals_dir),
        augmentation=None,  # Skip augmentation for faster training
        normalize=True,
        load_in_memory=False
    )
    # Subset the dataset
    train_dataset.metadata_df = train_df
    
    val_dataset = PPGDataset(
        metadata_path=str(data_dir / 'ssl_validation_data.parquet'),
        denoised_index_path=str(denoised_index_path),
        signal_dir=str(signals_dir),
        augmentation=None,  # No augmentation on validation
        normalize=True,
        load_in_memory=False
    )
    # Subset the dataset
    val_dataset.metadata_df = val_df
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep 0 for Windows compatibility
        pin_memory=False  # CPU only
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader, train_df, val_df


def main(args):
    """Run Phase 3 training."""
    logger.info("=" * 80)
    logger.info("Phase 3: Local CPU Validation - SSL Pretraining Pilot Run")
    logger.info("=" * 80)
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Device: {device}")
    
    # Load configuration
    config = SSLConfig()
    logger.info(f"Config loaded from defaults")
    logger.info(f"  Model: latent_dim={config.model.latent_dim}, bottleneck={config.model.bottleneck_dim}")
    logger.info(f"  Training: batch_size={config.training.batch_size}, epochs={config.training.num_epochs}")
    logger.info(f"  Loss: MSE={config.loss.mse_weight}, SSIM={config.loss.ssim_weight}, FFT={config.loss.fft_weight}")
    
    # Initialize models
    logger.info("\nInitializing models...")
    encoder = ResNetEncoder(
        in_channels=config.model.in_channels,
        latent_dim=config.model.latent_dim,
        bottleneck_dim=config.model.bottleneck_dim,
        num_blocks=config.model.num_blocks,
        base_filters=config.model.base_filters,
        max_filters=config.model.max_filters
    )
    
    decoder = ResNetDecoder(
        in_channels=config.model.in_channels,
        latent_dim=config.model.latent_dim,
        bottleneck_dim=config.model.bottleneck_dim,
        num_blocks=config.model.num_blocks,
        base_filters=config.model.base_filters,
        max_filters=config.model.max_filters
    )
    
    loss_fn = SSLLoss(
        mse_weight=config.loss.mse_weight,
        ssim_weight=config.loss.ssim_weight,
        fft_weight=config.loss.fft_weight
    )
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    logger.info(f"Encoder parameters: {encoder_params:,}")
    logger.info(f"Decoder parameters: {decoder_params:,}")
    logger.info(f"Total parameters: {encoder_params + decoder_params:,}")
    
    # Load data
    logger.info("\nLoading Phase 0 data...")
    train_loader, val_loader, train_df, val_df = load_phase0_data(
        batch_size=args.batch_size,
        num_workers=0,  # CPU validation, keep num_workers=0
        subset_fraction=0.1  # Use 10% of data for fast pilot run
    )
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = SimpleSSLTrainer(
        encoder=encoder,
        decoder=decoder,
        loss_fn=loss_fn,
        config=config,
        device=device
    )
    
    # Training loop
    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info("=" * 80)
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        logger.info(f"Train Loss: {train_loss:.6f}")
        
        # Validate
        val_loss = trainer.validate(val_loader)
        logger.info(f"Val Loss:   {val_loss:.6f}")
        logger.info(f"Best Val:   {trainer.best_val_loss:.6f}")
        
        # Save checkpoint
        trainer.save_checkpoint(f'epoch_{epoch + 1:02d}')
    
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    
    # Save training metrics
    metrics = {
        'train_losses': [float(l) for l in trainer.train_losses],
        'val_losses': [float(l) for l in trainer.val_losses],
        'best_val_loss': float(trainer.best_val_loss),
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'config': {
            'model': {
                'latent_dim': config.model.latent_dim,
                'bottleneck_dim': config.model.bottleneck_dim,
            },
            'loss_weights': {
                'mse': config.loss.mse_weight,
                'ssim': config.loss.ssim_weight,
                'fft': config.loss.fft_weight,
            },
        }
    }
    
    metrics_path = PROJECT_ROOT / 'checkpoints' / 'phase3' / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Summary
    logger.info("\nTraining Summary:")
    logger.info(f"  Train Loss: {trainer.train_losses[-1]:.6f} → {trainer.train_losses[0]:.6f}")
    logger.info(f"  Val Loss:   {trainer.val_losses[-1]:.6f} → {trainer.val_losses[0]:.6f}")
    logger.info(f"  Best Val:   {trainer.best_val_loss:.6f}")
    logger.info(f"\nCheckpoints saved to: {PROJECT_ROOT / 'checkpoints' / 'phase3'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: Local CPU Validation - SSL Pretraining Pilot Run"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Number of training epochs (default: 1)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training (default: 8)'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for training (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
