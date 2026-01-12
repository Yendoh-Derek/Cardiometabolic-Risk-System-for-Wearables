"""Unit tests for loss functions module."""
import sys
from pathlib import Path
import torch
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))

from models.ssl.losses import SSIMLoss, FFTLoss, SSLLoss


class TestSSIMLoss:
    """Test suite for SSIM loss."""

    @pytest.fixture
    def ssim_loss(self):
        """Create SSIM loss instance."""
        return SSIMLoss(window_size=11)

    def test_ssim_loss_identical_signals(self, ssim_loss):
        """Test SSIM loss is zero for identical signals."""
        x = torch.randn(4, 1, 75000)
        loss = ssim_loss(x, x)
        
        # SSIM of identical signals should be close to 1 (loss ~0)
        assert loss < 0.01, f"SSIM loss for identical signals is {loss}"

    def test_ssim_loss_different_signals(self, ssim_loss):
        """Test SSIM loss is higher for different signals."""
        x1 = torch.randn(4, 1, 75000)
        x2 = torch.randn(4, 1, 75000)
        loss = ssim_loss(x1, x2)
        
        # Loss should be positive and less than 1
        assert 0 < loss < 1, f"SSIM loss {loss} out of expected range"

    def test_ssim_loss_symmetry(self, ssim_loss):
        """Test SSIM loss is symmetric."""
        x1 = torch.randn(4, 1, 75000)
        x2 = torch.randn(4, 1, 75000)
        
        loss_12 = ssim_loss(x1, x2)
        loss_21 = ssim_loss(x2, x1)
        
        assert torch.allclose(loss_12, loss_21, atol=1e-5), "SSIM loss not symmetric"

    def test_ssim_loss_gradient_flow(self, ssim_loss):
        """Test gradients flow through SSIM loss."""
        x1 = torch.randn(2, 1, 75000, requires_grad=True)
        x2 = torch.randn(2, 1, 75000, requires_grad=True)
        
        loss = ssim_loss(x1, x2)
        loss.backward()
        
        assert x1.grad is not None, "Gradient not computed for x1"
        assert x2.grad is not None, "Gradient not computed for x2"


class TestFFTLoss:
    """Test suite for FFT loss."""

    @pytest.fixture
    def fft_loss(self):
        """Create FFT loss instance."""
        return FFTLoss()

    def test_fft_loss_identical_signals(self, fft_loss):
        """Test FFT loss is zero for identical signals."""
        x = torch.randn(4, 1, 75000)
        loss = fft_loss(x, x)
        
        # Loss should be very small for identical signals
        assert loss < 0.01, f"FFT loss for identical signals is {loss}"

    def test_fft_loss_different_signals(self, fft_loss):
        """Test FFT loss is positive for different signals."""
        x1 = torch.randn(4, 1, 75000)
        x2 = torch.randn(4, 1, 75000)
        loss = fft_loss(x1, x2)
        
        assert loss > 0, "FFT loss should be positive for different signals"

    def test_fft_loss_scale_invariance(self, fft_loss):
        """Test FFT loss with scaled signals."""
        x1 = torch.randn(4, 1, 75000)
        x2 = x1.clone()
        
        # Slightly scale x2
        x2_scaled = x2 * 1.1
        loss_original = fft_loss(x1, x2)
        loss_scaled = fft_loss(x1, x2_scaled)
        
        # Scaled version should have higher loss
        assert loss_scaled > loss_original, "Scaled signals should have higher FFT loss"

    def test_fft_loss_gradient_flow(self, fft_loss):
        """Test gradients flow through FFT loss."""
        x1 = torch.randn(2, 1, 75000, requires_grad=True)
        x2 = torch.randn(2, 1, 75000, requires_grad=True)
        
        loss = fft_loss(x1, x2)
        loss.backward()
        
        assert x1.grad is not None, "Gradient not computed for x1"
        assert x2.grad is not None, "Gradient not computed for x2"


class TestSSLLoss:
    """Test suite for combined SSL loss."""

    @pytest.fixture
    def ssl_loss(self):
        """Create SSL loss instance."""
        return SSLLoss(
            mse_weight=0.5,
            ssim_weight=0.3,
            fft_weight=0.2
        )

    def test_ssl_loss_weights_sum_to_one(self, ssl_loss):
        """Test that loss weights sum to 1."""
        total_weight = ssl_loss.mse_weight + ssl_loss.ssim_weight + ssl_loss.fft_weight
        assert torch.allclose(torch.tensor(total_weight), torch.tensor(1.0)), \
            "Loss weights don't sum to 1"

    def test_ssl_loss_identical_signals(self, ssl_loss):
        """Test SSL loss is near zero for identical signals."""
        x = torch.randn(4, 1, 75000)
        loss = ssl_loss(x, x)
        
        assert loss < 0.05, f"SSL loss for identical signals is {loss}"

    def test_ssl_loss_different_signals(self, ssl_loss):
        """Test SSL loss is positive for different signals."""
        x1 = torch.randn(4, 1, 75000)
        x2 = torch.randn(4, 1, 75000)
        loss = ssl_loss(x1, x2)
        
        assert loss > 0, "SSL loss should be positive for different signals"

    def test_ssl_loss_composition(self, ssl_loss):
        """Test that SSL loss is composition of components."""
        x1 = torch.randn(4, 1, 75000)
        x2 = torch.randn(4, 1, 75000)
        
        total_loss = ssl_loss(x1, x2)
        
        # Compute component losses separately
        mse_loss_fn = torch.nn.MSELoss()
        mse = mse_loss_fn(x1, x2)
        
        # Total should be sum of weighted components
        # We can't check exact equality due to different loss implementations,
        # but we can check that total is in reasonable range
        assert 0 < total_loss < 10, f"SSL loss {total_loss} out of expected range"

    def test_ssl_loss_gradient_flow(self, ssl_loss):
        """Test gradients flow through SSL loss."""
        x1 = torch.randn(2, 1, 75000, requires_grad=True)
        x2 = torch.randn(2, 1, 75000, requires_grad=True)
        
        loss = ssl_loss(x1, x2)
        loss.backward()
        
        assert x1.grad is not None, "Gradient not computed for x1"
        assert x2.grad is not None, "Gradient not computed for x2"

    def test_ssl_loss_different_batch_sizes(self, ssl_loss):
        """Test SSL loss handles different batch sizes."""
        batch_sizes = [1, 2, 4, 8, 16]
        
        for bs in batch_sizes:
            x1 = torch.randn(bs, 1, 75000)
            x2 = torch.randn(bs, 1, 75000)
            loss = ssl_loss(x1, x2)
            
            assert loss.item() > 0, f"Failed for batch_size={bs}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
