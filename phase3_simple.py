"""Phase 3 - Simplified version with minimal logging overhead."""
import sys
import os
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))

from models.ssl.config import SSLConfig
from models.ssl.encoder import ResNetEncoder
from models.ssl.decoder import ResNetDecoder
from models.ssl.losses import SSLLoss
from models.ssl.dataloader import PPGDataset

def main():
    print("=" * 80)
    print("Phase 3: Local CPU Validation - Simplified (10% Subset)")
    print("=" * 80)
    
    # Setup
    data_dir = PROJECT_ROOT / 'data' / 'processed'
    signals_dir = data_dir / 'denoised_signals'
    denoised_index_path = data_dir / 'denoised_signal_index.json'
    
    # Config
    print("\n[1] Loading config...", flush=True)
    config = SSLConfig()
    
    # Models
    print("[2] Initializing models (3.9M params)...", flush=True)
    encoder = ResNetEncoder(
        in_channels=config.model.in_channels,
        latent_dim=config.model.latent_dim,
        bottleneck_dim=config.model.bottleneck_dim
    )
    decoder = ResNetDecoder(
        in_channels=config.model.in_channels,
        latent_dim=config.model.latent_dim,
        bottleneck_dim=config.model.bottleneck_dim
    )
    loss_fn = SSLLoss(
        mse_weight=config.loss.mse_weight,
        ssim_weight=config.loss.ssim_weight,
        fft_weight=config.loss.fft_weight
    )
    
    # Data
    print("[3] Loading 10% subset of Phase 0 data...", flush=True)
    train_df = pd.read_parquet(data_dir / 'ssl_pretraining_data.parquet')
    val_df = pd.read_parquet(data_dir / 'ssl_validation_data.parquet')
    
    train_df = train_df.iloc[:int(len(train_df) * 0.1)]  # 413 samples
    val_df = val_df.iloc[:int(len(val_df) * 0.1)]  # 20 samples
    
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val: {len(val_df)} samples")
    
    train_dataset = PPGDataset(
        metadata_path=str(data_dir / 'ssl_pretraining_data.parquet'),
        denoised_index_path=str(denoised_index_path),
        signal_dir=str(signals_dir),
        augmentation=None,
        normalize=True,
        load_in_memory=False
    )
    train_dataset.metadata_df = train_df
    
    val_dataset = PPGDataset(
        metadata_path=str(data_dir / 'ssl_validation_data.parquet'),
        denoised_index_path=str(denoised_index_path),
        signal_dir=str(signals_dir),
        augmentation=None,
        normalize=True,
        load_in_memory=False
    )
    val_dataset.metadata_df = val_df
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Optimizer
    print("[4] Setting up optimizer...", flush=True)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.training.learning_rate
    )
    
    # Training
    print("[5] Starting training (1 epoch)...", flush=True)
    print("-" * 80)
    
    encoder.train()
    decoder.train()
    
    train_losses = []
    total_batches = len(train_loader)
    
    for batch_idx, (raw_signal, _) in enumerate(train_loader):
        optimizer.zero_grad()
        
        latent = encoder(raw_signal)
        recon = decoder(latent)
        loss = loss_fn(recon, raw_signal)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Log every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Batch {batch_idx + 1}/{total_batches}: "
                f"Loss = {loss.item():.6f}",
                flush=True
            )
    
    # Validation
    print(f"\nEpoch complete. Running validation...", flush=True)
    encoder.eval()
    decoder.eval()
    
    val_losses = []
    with torch.no_grad():
        for raw_signal, _ in val_loader:
            latent = encoder(raw_signal)
            recon = decoder(latent)
            loss = loss_fn(recon, raw_signal)
            val_losses.append(loss.item())
    
    print("-" * 80)
    print(f"\nResults:")
    print(f"  Train Loss (final): {train_losses[-1]:.6f}")
    print(f"  Train Loss (avg):   {sum(train_losses) / len(train_losses):.6f}")
    print(f"  Val Loss (avg):     {sum(val_losses) / len(val_losses):.6f}")
    
    # Save checkpoint
    checkpoint_dir = PROJECT_ROOT / 'checkpoints' / 'phase3'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, checkpoint_dir / 'checkpoint_phase3.pt')
    
    print(f"\nCheckpoint saved to {checkpoint_dir}/checkpoint_phase3.pt")
    print("=" * 80)
    print("✅ Phase 3 Validation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
