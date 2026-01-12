"""
Phase 3: Simplified Local Validation - Fast CPU Pilot Run
Ultra-minimal version to verify pipeline works end-to-end on CPU.
"""
import sys
from pathlib import Path
import json
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))

from models.ssl.config import SSLConfig
from models.ssl.encoder import ResNetEncoder
from models.ssl.decoder import ResNetDecoder
from models.ssl.losses import SSLLoss
from models.ssl.dataloader import PPGDataset

print("=" * 70)
print("PHASE 3: FAST CPU VALIDATION - MINIMAL PILOT RUN")
print("=" * 70)

# Setup
data_dir = PROJECT_ROOT / 'data' / 'processed'
signals_dir = data_dir / 'denoised_signals'
denoised_index = data_dir / 'denoised_signal_index.json'

# 1. Config
print("\n[1/6] Loading config...")
config = SSLConfig()
print(f"  ✓ Model: {config.model.latent_dim}D latent")

# 2. Models
print("[2/6] Initializing models...")
encoder = ResNetEncoder(
    in_channels=1, latent_dim=512, bottleneck_dim=768
).cpu()
decoder = ResNetDecoder(
    in_channels=1, latent_dim=512, bottleneck_dim=768
).cpu()
loss_fn = SSLLoss(mse_weight=0.5, ssim_weight=0.3, fft_weight=0.2).cpu()
print(f"  ✓ Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
print(f"  ✓ Decoder: {sum(p.numel() for p in decoder.parameters()):,} params")

# 3. Data - use only 50 samples for speed
print("[3/6] Loading data (50 samples)...")
dataset = PPGDataset(
    metadata_path=str(data_dir / 'ssl_pretraining_data.parquet'),
    denoised_index_path=str(denoised_index),
    signal_dir=str(signals_dir),
    augmentation=None,
    normalize=True
)
# Use first 50 samples only
train_subset = Subset(dataset, range(min(50, len(dataset))))
train_loader = DataLoader(train_subset, batch_size=8, shuffle=False, num_workers=0)
print(f"  ✓ {len(train_subset)} training samples")

# 4. Optimizer
print("[4/6] Setting up optimizer...")
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=0.001
)
print("  ✓ Adam optimizer ready")

# 5. Training (1 epoch)
print("[5/6] Training 1 epoch...")
encoder.train()
decoder.train()

train_losses = []
batch_count = 0

for batch_idx, (raw, denoised) in enumerate(train_loader):
    raw = raw.cpu()
    
    # Forward
    optimizer.zero_grad()
    z = encoder(raw)
    recon = decoder(z)
    loss = loss_fn(recon, raw)
    
    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()),
        max_norm=1.0
    )
    optimizer.step()
    
    train_losses.append(loss.item())
    batch_count += 1
    
    print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.6f}")

avg_loss = sum(train_losses) / len(train_losses)
print(f"  ✓ Epoch complete - Avg Loss: {avg_loss:.6f}")

# 6. Save checkpoint
print("[6/6] Saving checkpoint...")
checkpoint_dir = PROJECT_ROOT / 'checkpoints' / 'phase3'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'optimizer': optimizer.state_dict(),
    'losses': train_losses,
}, checkpoint_dir / 'checkpoint_pilot.pt')

metrics = {
    'timestamp': datetime.now().isoformat(),
    'device': 'cpu',
    'epochs': 1,
    'batches': batch_count,
    'train_losses': train_losses,
    'avg_loss': float(avg_loss),
    'encoder_params': sum(p.numel() for p in encoder.parameters()),
    'decoder_params': sum(p.numel() for p in decoder.parameters()),
}

with open(checkpoint_dir / 'metrics_pilot.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"  ✓ Checkpoint saved to {checkpoint_dir}")

# Summary
print("\n" + "=" * 70)
print("✅ PHASE 3 PILOT RUN COMPLETE!")
print("=" * 70)
print(f"Training Loss:     {train_losses[0]:.6f} → {train_losses[-1]:.6f}")
print(f"Average Loss:      {avg_loss:.6f}")
print(f"Batches Processed: {batch_count}")
print(f"Checkpoint:        {checkpoint_dir / 'checkpoint_pilot.pt'}")
print("\n✅ Pipeline validation successful - ready for Colab GPU training!")
