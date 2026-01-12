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
import numpy as np
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
    parser = argparse.ArgumentParser(
        description='SSL Pretraining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Local development (auto-detect device):
    python -m colab_src.models.ssl.train --config configs/ssl_pretraining.yaml --epochs 50
  
  Colab (auto-detect device, uses GPU if available):
    python -m colab_src.models.ssl.train --config configs/ssl_pretraining.yaml \\
        --data-dir /content/drive/MyDrive/cardiometabolic-risk-colab/data/processed \\
        --epochs 50
  
  Force CPU:
    python -m colab_src.models.ssl.train --config configs/ssl_pretraining.yaml \\
        --device cpu --epochs 50
        """
    )
    parser.add_argument('--config', type=str, default='configs/ssl_pretraining.yaml',
                       help='Path to config YAML')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory (Colab: /content/drive/MyDrive/.../data/processed)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto (default: auto-detect GPU if available, else CPU)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--load-in-memory', action='store_true',
                       help='Load all signals into memory')
    
    args = parser.parse_args()
    
    try:
        # Load config
        config_path = Path(args.config)
        if config_path.exists():
            logger.info(f"Loading config from {config_path}")
            config = SSLConfig.from_yaml(str(config_path))
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config = get_default_config()
        
        # Override with command line args
        config.training.num_epochs = args.epochs
        config.training.batch_size = args.batch_size
        
        # CRITICAL: Auto-detect device with fallback to CPU
        # If user requests CUDA but it's not available, fall back to CPU
        if args.device == 'cuda':
            if torch.cuda.is_available():
                config.device = 'cuda'
                logger.info(f"✅ CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
            else:
                config.device = 'cpu'
                logger.warning(f"⚠️  CUDA requested but not available, falling back to CPU")
        elif args.device == 'cpu':
            config.device = 'cpu'
            logger.info(f"✅ Using CPU")
        else:
            # Auto-detect (default)
            config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if torch.cuda.is_available():
                logger.info(f"✅ Auto-detected GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info(f"✅ Auto-detected CPU (no GPU available)")
        
        # CRITICAL: Override data directory if specified (Colab use case)
        if args.data_dir:
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            config.data_dir = data_dir
            logger.info(f"Data directory overridden: {data_dir}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Configuration:")
        logger.info(f"  Device:              {config.device}")
        logger.info(f"  Data dir:            {config.data_dir or config.project_root / 'data/processed'}")
        logger.info(f"  Epochs:              {config.training.num_epochs}")
        logger.info(f"  Batch size:          {config.training.batch_size}")
        logger.info(f"  Accumulation steps:  {config.training.accumulation_steps}")
        logger.info(f"  Mixed precision:     {config.training.use_mixed_precision}")
        logger.info(f"{'='*60}\n")
        
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
            T_0=max(1, num_epochs // 2),
            T_mult=1,
            eta_min=config.training.learning_rate * 0.01,
        )
        
        # Create augmentation
        np.random.seed(config.seed)
        augmentation = PPGAugmentation(
            temporal_shift_range=config.augmentation.temporal_shift_range,
            amplitude_scale_range=tuple(config.augmentation.amplitude_scale_range) 
                if isinstance(config.augmentation.amplitude_scale_range, (list, tuple))
                else config.augmentation.amplitude_scale_range,
            baseline_wander_freq=config.augmentation.baseline_wander_freq,
            baseline_wander_amplitude=config.augmentation.baseline_wander_amplitude,
            noise_prob=config.augmentation.noise_prob,
            noise_snr_ratio=config.augmentation.noise_snr_ratio,
            sample_rate=config.data.sample_rate,
        )
        
        # Determine data paths
        if config.data_dir:
            data_base = config.data_dir
        else:
            data_base = config.project_root / "data/processed"
        
        train_meta_path = data_base / "ssl_pretraining_data.parquet"
        val_meta_path = data_base / "ssl_validation_data.parquet"
        test_meta_path = data_base / "ssl_test_data.parquet"
        denoised_index_path = data_base / "denoised_signal_index.json"
        
        # Validate data paths exist
        logger.info(f"\nValidating data paths:")
        if not train_meta_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_meta_path}")
        logger.info(f"  ✓ Train: {train_meta_path}")
        
        if not val_meta_path.exists():
            raise FileNotFoundError(f"Validation data not found: {val_meta_path}")
        logger.info(f"  ✓ Val:   {val_meta_path}")
        
        if not test_meta_path.exists():
            logger.warning(f"  ⚠ Test not found: {test_meta_path} (will skip test)")
            test_meta_path = None
        else:
            logger.info(f"  ✓ Test:  {test_meta_path}")
        
        if denoised_index_path.exists():
            logger.info(f"  ✓ Denoised index: {denoised_index_path}")
        else:
            logger.warning(f"  ⚠ Denoised index not found: {denoised_index_path}")
        
        # Create dataloaders
        logger.info(f"\nCreating dataloaders...")
        dataloaders = create_dataloaders(
            train_metadata_path=train_meta_path,
            val_metadata_path=val_meta_path,
            test_metadata_path=test_meta_path,
            signal_array_path=None,
            signal_dir=data_base / "denoised_signals",
            denoised_index_path=denoised_index_path,
            augmentation=augmentation,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            device=config.device,
            load_in_memory=args.load_in_memory,
        )
        
        logger.info(f"DataLoaders created successfully\n")
        
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
        logger.info(f"{'='*60}")
        logger.info(f"🚀 Starting training loop ({config.training.num_epochs} epochs)")
        logger.info(f"{'='*60}\n")
        history = trainer.fit(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            num_epochs=config.training.num_epochs,
            early_stopping_patience=10,
        )
        
        # Save history
        history_path = config.log_dir / 'training_history.json'
        trainer.save_history(history_path)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"   Best validation loss: {trainer.best_val_loss:.6f}")
        logger.info(f"   Best epoch: {trainer.best_epoch}")
        logger.info(f"   Checkpoint dir: {config.checkpoint_dir}")
        logger.info(f"   Log dir: {config.log_dir}")
        logger.info(f"{'='*60}\n")
    
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}", exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}", exc_info=True)
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"❌ Runtime error (possible OOM or device issue): {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
