"""Quick Phase 3 test - verify one training batch works."""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))

from models.ssl.config import SSLConfig
from models.ssl.encoder import ResNetEncoder
from models.ssl.decoder import ResNetDecoder
from models.ssl.losses import SSLLoss
from models.ssl.dataloader import PPGDataset

# Setup
data_dir = PROJECT_ROOT / 'data' / 'processed'
signals_dir = data_dir / 'denoised_signals'
denoised_index_path = data_dir / 'denoised_signal_index.json'

print("[1/5] Loading config...", flush=True)
config = SSLConfig()

print("[2/5] Initializing models...", flush=True)
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

print("[3/5] Loading dataset...", flush=True)
dataset = PPGDataset(
    metadata_path=str(data_dir / 'ssl_pretraining_data.parquet'),
    denoised_index_path=str(denoised_index_path),
    signal_dir=str(signals_dir),
    augmentation=None,
    normalize=True
)

loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

print("[4/5] Setting up optimizer...", flush=True)
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=config.training.learning_rate
)

print("[5/5] Running one training batch...", flush=True)
encoder.train()
decoder.train()

raw, denoised = next(iter(loader))
print(f"  Batch shapes: raw {raw.shape}, denoised {denoised.shape}", flush=True)

optimizer.zero_grad()
latent = encoder(raw)
print(f"  Latent shape: {latent.shape}", flush=True)

recon = decoder(latent)
print(f"  Recon shape: {recon.shape}", flush=True)

loss = loss_fn(recon, raw)
print(f"  Loss: {loss.item():.6f}", flush=True)

loss.backward()
print(f"  Backward pass complete", flush=True)

optimizer.step()
print(f"  Optimizer step complete", flush=True)

print("\n✅ ONE BATCH TRAINING SUCCESSFUL!")
print(f"   Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
print(f"   Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
print(f"   Loss value: {loss.item():.6f}")
