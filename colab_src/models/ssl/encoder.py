"""
ResNet encoder architecture for self-supervised learning.

1D Convolutional ResNet with stride-2 blocks (no max pooling).
Input: (batch, 1, 75000) -> Output: (batch, latent_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """1D Residual block with stride."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class ResNetEncoder(nn.Module):
    """
    1D ResNet encoder for PPG signals.
    
    Architecture:
    - Conv1d(1, 32, kernel=7, stride=2) + BN + ReLU
    - 4x ResidualBlocks (stride-2) with increasing channels
    - Global average pooling
    - Linear layers to latent_dim
    - Bottleneck projection: latent_dim -> bottleneck_dim -> latent_dim
    
    Input:  (batch, 1, 75000)
    Output: (batch, latent_dim)
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
        
        # Initial conv layer
        self.conv1 = nn.Conv1d(
            in_channels, base_filters, kernel_size=7, 
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks with stride-2 (each reduces length by 2)
        self.blocks = nn.ModuleList()
        in_ch = base_filters
        
        for i in range(num_blocks):
            out_ch = min(base_filters * (2 ** (i + 1)), max_filters)
            stride = 2  # Always stride-2 for dimensionality reduction
            
            self.blocks.append(
                ResidualBlock(in_ch, out_ch, stride=stride)
            )
            in_ch = out_ch
        
        # Final channels after all blocks
        self.final_channels = in_ch
        
        # Global average pooling reduces (batch, channels, L) to (batch, channels)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # MLP head to latent space with bottleneck projection
        # latent_dim -> bottleneck_dim -> latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.final_channels, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input signal (batch, 1, 75000)
        
        Returns:
            Latent embedding (batch, latent_dim)
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = self.avgpool(x)  # (batch, channels, 1)
        x = x.view(x.size(0), -1)  # (batch, channels)
        
        # MLP projection to latent space
        x = self.mlp(x)
        
        return x
    
    def compute_flops(self, signal_length: int = 75000) -> dict:
        """Estimate FLOPs for a single forward pass."""
        # This is approximate
        # Conv1d flops: kernel_size * input_len * in_ch * out_ch
        
        # Initial conv: 7 * (75000/2) * 1 * 32 = ~8.4M FLOPs
        initial_flops = 7 * (signal_length // 2) * 1 * self.conv1.out_channels
        
        # Blocks: ~reduction by 2x for each block
        block_flops = 0
        in_ch = self.conv1.out_channels
        current_len = signal_length // 2
        
        for block in self.blocks:
            # Each block: 2 Conv1d layers
            current_len = current_len // 2
            out_ch = block.conv1.out_channels
            # Conv1d + Conv1d per block
            block_flops += 3 * current_len * in_ch * out_ch  # First conv
            block_flops += 3 * current_len * out_ch * out_ch  # Second conv
            in_ch = out_ch
        
        # MLP flops: Linear layer multiplications
        mlp_flops = (self.final_channels + self.latent_dim + self.bottleneck_dim) * self.latent_dim
        
        total_flops = initial_flops + block_flops + mlp_flops
        
        return {
            'initial_conv': initial_flops,
            'residual_blocks': block_flops,
            'mlp': mlp_flops,
            'total': total_flops,
        }
