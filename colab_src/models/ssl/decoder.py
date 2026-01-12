"""
ResNet decoder architecture for self-supervised learning.

1D Transposed Convolutional ResNet, mirrors encoder.
Input: (batch, latent_dim) -> Output: (batch, 1, 75000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TransposedResidualBlock(nn.Module):
    """1D Transposed Residual block with stride."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        
        # Main path: transpose conv + regular conv
        self.conv_t = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, output_padding=stride - 1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1,
                                  stride=stride, output_padding=stride - 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = self.skip(x)
        
        out = self.conv_t(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class ResNetDecoder(nn.Module):
    """
    1D ResNet decoder for PPG signal reconstruction.
    
    Mirrors the encoder architecture:
    - MLP projection: latent_dim -> bottleneck_dim -> latent_dim -> final_channels
    - 4x TransposedResidualBlocks (stride-2 upsampling) with decreasing channels
    - ConvTranspose1d(channels, 1, kernel=7, stride=2) output
    
    Input:  (batch, latent_dim)
    Output: (batch, 1, 75000)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 512,
        bottleneck_dim: int = 768,
        num_blocks: int = 4,
        base_filters: int = 32,
        max_filters: int = 512,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_blocks = num_blocks
        self.base_filters = base_filters
        self.max_filters = max_filters
        
        # Compute filter sizes for decoder (mirror of encoder)
        self.filter_sizes = [base_filters * (2 ** min(i, num_blocks - 1)) for i in range(num_blocks)]
        self.filter_sizes = [min(f, max_filters) for f in self.filter_sizes]
        self.final_channels = self.filter_sizes[-1]
        
        # MLP head to expand from latent space (mirror of encoder)
        # latent_dim -> bottleneck_dim -> latent_dim -> final_channels
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, self.final_channels),
        )
        
        # Transposed residual blocks with stride-2 upsampling
        # Reverse order: from highest channels to base
        self.blocks = nn.ModuleList()
        
        for i in range(num_blocks - 1, -1, -1):
            in_ch = self.filter_sizes[i]
            out_ch = self.filter_sizes[i - 1] if i > 0 else base_filters
            stride = 2  # Always stride-2 for upsampling
            
            self.blocks.append(
                TransposedResidualBlock(in_ch, out_ch, stride=stride)
            )
        
        # Final upsampling conv layer
        self.conv_out = nn.ConvTranspose1d(
            base_filters, in_channels, kernel_size=7,
            stride=2, padding=3, output_padding=1, bias=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Latent embedding (batch, latent_dim)
        
        Returns:
            Reconstructed signal (batch, 1, 75000)
        """
        # MLP projection from latent to spatial
        x = self.mlp(x)  # (batch, final_channels)
        
        # Reshape to add spatial dimension for upsampling
        # Need spatial dimension S such that S * 2^(num_blocks) * 2 = 75000
        # For num_blocks=4: S * 16 * 2 = S * 32 = 75000 => S = 2344
        initial_spatial_size = 75000 // (2 ** (self.num_blocks + 1))
        if 75000 % (2 ** (self.num_blocks + 1)) != 0:
            initial_spatial_size += 1
        
        x = x.unsqueeze(-1)  # (batch, final_channels, 1)
        
        # Upsample from 1 to initial_spatial_size using interpolation
        x = F.interpolate(x, size=initial_spatial_size, mode='linear', align_corners=False)
        
        # Transposed residual blocks (upsampling with stride-2)
        for block in self.blocks:
            x = block(x)
        
        # Final conv output
        x = self.conv_out(x)  # (batch, 1, ~75000)
        
        # Ensure correct output length (should be 75000)
        # Clip or pad to desired length
        if x.size(-1) > 75000:
            x = x[..., :75000]
        elif x.size(-1) < 75000:
            x = F.pad(x, (0, 75000 - x.size(-1)))
        
        return x
