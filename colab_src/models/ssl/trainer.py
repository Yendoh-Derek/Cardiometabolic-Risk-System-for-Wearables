"""
Training loop for self-supervised learning.

Features:
- Gradient accumulation for large effective batch sizes
- Mixed precision training (FP16)
- Checkpointing with best model tracking
- Early stopping
- Learning rate scheduling with warmup
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SSLTrainer:
    """
    Trainer for self-supervised learning with denoising autoencoder.
    
    Handles:
    - Gradient accumulation
    - Mixed precision training
    - Checkpoint management
    - Early stopping
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler=None,
        device: str = 'cpu',
        checkpoint_dir: Path = None,
        use_mixed_precision: bool = True,
        max_grad_norm: float = 1.0,
        accumulation_steps: int = 1,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Encoder-Decoder model
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device ('cpu' or 'cuda')
            checkpoint_dir: Directory for saving checkpoints
            use_mixed_precision: Whether to use FP16 mixed precision
            max_grad_norm: Max gradient norm for clipping
            accumulation_steps: Gradient accumulation steps
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_mixed_precision = use_mixed_precision
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=use_mixed_precision)
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training DataLoader
        
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (raw_signal, target_signal) in enumerate(train_loader):
            raw_signal = raw_signal.to(self.device)
            target_signal = target_signal.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_mixed_precision):
                # Encode
                latent = self.model.encoder(raw_signal)
                # Decode
                reconstruction = self.model.decoder(latent)
                # Loss
                loss = self.loss_fn(reconstruction, target_signal)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.accumulation_steps
            
            # Backward pass
            self.scaler.scale(scaled_loss).backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                avg_loss = total_loss / num_batches
                logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {avg_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation DataLoader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for raw_signal, target_signal in val_loader:
                raw_signal = raw_signal.to(self.device)
                target_signal = target_signal.to(self.device)
                
                # Forward pass
                with autocast(enabled=self.use_mixed_precision):
                    latent = self.model.encoder(raw_signal)
                    reconstruction = self.model.decoder(latent)
                    loss = self.loss_fn(reconstruction, target_signal)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
    ) -> Dict:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        logger.info(f"Gradient accumulation: {self.accumulation_steps} steps")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            logger.info(f"Epoch {epoch + 1}/{num_epochs}:")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Val Loss:   {val_metrics['loss']:.4f}")
            logger.info(f"  LR:         {current_lr:.2e}")
            
            # Store history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['lr'].append(current_lr)
            
            # Early stopping and checkpointing
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best checkpoint
                self.save_checkpoint(epoch + 1, is_best=True)
                logger.info(f"  ✓ Best model updated (loss: {self.best_val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"  Patience: {patience_counter}/{early_stopping_patience}")
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Best epoch: {self.best_epoch} (val_loss: {self.best_val_loss:.4f})")
        
        return self.training_history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    def save_history(self, path: Path):
        """Save training history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {path}")
