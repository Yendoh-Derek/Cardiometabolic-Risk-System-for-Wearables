"""Unit tests for decoder module."""
import sys
from pathlib import Path
import torch
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))

from models.ssl.decoder import ResNetDecoder


class TestResNetDecoder:
    """Test suite for ResNetDecoder."""

    @pytest.fixture
    def decoder(self):
        """Create decoder instance."""
        return ResNetDecoder(in_channels=1, latent_dim=512, bottleneck_dim=768)

    def test_decoder_initialization(self, decoder):
        """Test decoder initialization."""
        assert decoder.in_channels == 1
        assert decoder.latent_dim == 512

    def test_decoder_forward_pass(self, decoder):
        """Test decoder forward pass with valid input."""
        batch_size = 4
        x = torch.randn(batch_size, 512)
        
        output = decoder(x)
        assert output.shape == (batch_size, 1, 75000)

    def test_decoder_output_shape(self, decoder):
        """Test decoder output shape for various batch sizes."""
        batch_sizes = [1, 2, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 512)
            output = decoder(x)
            
            assert output.shape == (batch_size, 1, 75000), f"Failed for batch_size={batch_size}"

    def test_decoder_gradient_flow(self, decoder):
        """Test that gradients flow through decoder."""
        x = torch.randn(4, 512, requires_grad=True)
        output = decoder(x)
        loss = output.mean()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None, "Input gradient not computed"
        for param in decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradient not computed for some parameters"

    def test_decoder_reconstruction_smooth(self, decoder):
        """Test decoder produces smooth reconstructions."""
        decoder.eval()
        
        with torch.no_grad():
            x = torch.randn(2, 512)
            output = decoder(x)
            
            # Check that output is not sparse (has values throughout)
            sparsity = (output.abs() < 1e-6).float().mean()
            assert sparsity < 0.8, "Reconstruction too sparse"

    def test_decoder_parameter_count(self, decoder):
        """Test decoder has reasonable parameter count."""
        total_params = sum(p.numel() for p in decoder.parameters())
        # Decoder should have similar parameters to encoder (~1.5M)
        assert 1e6 < total_params < 2e6, f"Parameter count {total_params} outside expected range"

    def test_decoder_training_mode(self, decoder):
        """Test decoder training and eval modes."""
        x = torch.randn(2, 512)
        
        # Eval mode
        decoder.eval()
        with torch.no_grad():
            out1 = decoder(x)
            out2 = decoder(x)
        assert torch.allclose(out1, out2), "Decoder not deterministic in eval mode"

    def test_decoder_device_compatibility(self, decoder):
        """Test decoder can handle different devices."""
        if torch.cuda.is_available():
            decoder_gpu = decoder.to('cuda')
            x_gpu = torch.randn(2, 512, device='cuda')
            output = decoder_gpu(x_gpu)
            assert output.device.type == 'cuda'


class TestDecoderInputValidation:
    """Test input validation for decoder."""

    def test_decoder_wrong_latent_dim(self):
        """Test decoder expects correct latent dimension."""
        decoder = ResNetDecoder(in_channels=1, latent_dim=512)
        
        # Wrong latent dimension
        x_wrong = torch.randn(2, 256)  # Wrong dimension
        
        try:
            output = decoder(x_wrong)
            # May fail with dimension mismatch
            assert False, "Should have raised error for wrong input dimension"
        except RuntimeError:
            # Expected
            pass

    def test_decoder_1d_input(self):
        """Test decoder rejects 1D input."""
        decoder = ResNetDecoder(in_channels=1, latent_dim=512)
        
        x_1d = torch.randn(512)  # Missing batch dimension
        
        try:
            output = decoder(x_1d)
            # May work if unsqueezed, but check dimensions
            assert False, "Should have raised error for 1D input"
        except (RuntimeError, IndexError):
            # Expected
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
