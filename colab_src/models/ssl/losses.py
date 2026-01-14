"""
Multi-loss training function for self-supervised PPG reconstruction.

Loss components:
1. MSE loss (0.50): Temporal fidelity
2. SSIM loss (0.30): Structural similarity
3. FFT loss (0.20): Frequency domain alignment (magnitude + phase)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss."""
    
    def __init__(self, window_size: int = 11, reduction: str = 'mean'):
        super().__init__()
        self.window_size = window_size
        self.reduction = reduction
        
        # Create Gaussian kernel for SSIM
        kernel = self._create_gaussian_kernel(window_size)
        self.register_buffer('kernel', kernel)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    
    @staticmethod
    def _create_gaussian_kernel(window_size: int) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        sigma = window_size / 6.0
        x = torch.arange(window_size).float() - (window_size - 1) / 2.0
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
        return gauss / gauss.sum()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss.
        
        Args:
            pred: Predicted signal (batch, 1, length)
            target: Target signal (batch, 1, length)
        
        Returns:
            SSIM loss (scalar)
        """
        # Ensure 3D input
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)
        
        # Compute local means
        kernel = self.kernel.view(1, 1, -1)
        
        # Pad signal
        pad_size = self.window_size // 2
        pred_padded = F.pad(pred, (pad_size, pad_size), mode='reflect')
        target_padded = F.pad(target, (pad_size, pad_size), mode='reflect')
        
        mu1 = F.conv1d(pred_padded, kernel)
        mu2 = F.conv1d(target_padded, kernel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances
        sigma1_sq = F.conv1d(pred_padded ** 2, kernel) - mu1_sq
        sigma2_sq = F.conv1d(target_padded ** 2, kernel) - mu2_sq
        sigma12 = F.conv1d(pred_padded * target_padded, kernel) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        if self.reduction == 'mean':
            return 1.0 - ssim_map.mean()
        elif self.reduction == 'min':
            return 1.0 - ssim_map.min()
        else:
            return 1.0 - ssim_map


class FFTLoss(nn.Module):
    """Frequency domain loss using real FFT with configurable padding."""
    
    def __init__(self, norm: str = 'ortho', fft_pad_size: int = 2048):
        super().__init__()
        self.norm = norm
        self.fft_pad_size = fft_pad_size  # Phase 5A: padding for efficiency (2048 vs 131072)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute FFT loss (magnitude + phase alignment).
        
        Args:
            pred: Predicted signal (batch, 1, length)
            target: Target signal (batch, 1, length)
        
        Returns:
            FFT loss (scalar)
        """
        # Remove channel dimension if present
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        # Compute FFT with padding (Phase 5A: 2048 instead of 131072)
        pred_fft = torch.fft.rfft(pred, n=self.fft_pad_size, norm=self.norm)
        target_fft = torch.fft.rfft(target, n=self.fft_pad_size, norm=self.norm)
        
        # Magnitude and phase
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # Magnitude loss (L2)
        mag_loss = F.mse_loss(pred_mag, target_mag)
        
        # Phase loss (cosine distance)
        phase_loss = 1.0 - torch.cos(pred_phase - target_phase).mean()
        
        # Combined
        fft_loss = mag_loss + phase_loss
        
        return fft_loss


class SSLLoss(nn.Module):
    """
    Multi-loss for self-supervised PPG reconstruction.
    
    L_total = w_mse * L_mse + w_ssim * L_ssim + w_fft * L_fft
    """
    
    def __init__(
        self,
        mse_weight: float = 0.50,
        ssim_weight: float = 0.30,
        fft_weight: float = 0.20,
        ssim_window_size: int = 11,
        fft_norm: str = 'ortho',
        fft_pad_size: int = 2048,  # Phase 5A: critical fix (was 131072, now 2048)
    ):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.fft_weight = fft_weight
        
        # Verify weights sum to 1.0
        total_weight = mse_weight + ssim_weight + fft_weight
        assert abs(total_weight - 1.0) < 1e-6, \
            f"Loss weights must sum to 1.0, got {total_weight}"
        
        # Initialize individual losses
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=ssim_window_size)
        self.fft_loss = FFTLoss(norm=fft_norm, fft_pad_size=fft_pad_size)  # Phase 5A: pass fft_pad_size
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]] | torch.Tensor:
        """
        Compute combined SSL loss.
        
        Args:
            pred: Predicted reconstruction (batch, 1, length) or (batch, length)
            target: Target/denoised signal (batch, 1, length) or (batch, length)
            return_components: If True, return loss dict
        
        Returns:
            If return_components=False: Total loss (scalar)
            If return_components=True: (total_loss, loss_dict)
        """
        # Ensure consistent shapes
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)
        
        # Compute individual losses
        mse_loss_val = self.mse_loss(pred, target)
        ssim_loss_val = self.ssim_loss(pred, target)
        fft_loss_val = self.fft_loss(pred, target)
        
        # Weighted combination
        total_loss = (
            self.mse_weight * mse_loss_val +
            self.ssim_weight * ssim_loss_val +
            self.fft_weight * fft_loss_val
        )
        
        if return_components:
            loss_dict = {
                'total': total_loss.item(),
                'mse': mse_loss_val.item(),
                'ssim': ssim_loss_val.item(),
                'fft': fft_loss_val.item(),
            }
            return total_loss, loss_dict
        else:
            return total_loss
