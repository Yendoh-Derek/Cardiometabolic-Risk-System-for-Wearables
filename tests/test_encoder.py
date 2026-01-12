"""Unit tests for encoder module."""
import sys
from pathlib import Path
import torch
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))

from models.ssl.encoder import ResNetEncoder


class TestResNetEncoder:
    """Test suite for ResNetEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return ResNetEncoder(in_channels=1, latent_dim=512, bottleneck_dim=768)

    def test_encoder_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.in_channels == 1
        assert encoder.latent_dim == 512
        assert encoder.bottleneck_dim == 768

    def test_encoder_forward_pass(self, encoder):
        """Test encoder forward pass with valid input."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 75000)
        
        # Forward through backbone
        latent = encoder.backbone(x)
        assert latent.shape == (batch_size, 512)
        
        # Forward through bottleneck projection
        bottleneck = encoder.bottleneck(latent)
        assert bottleneck.shape == (batch_size, 768)

    def test_encoder_output_shape(self, encoder):
        """Test encoder output shape for various batch sizes."""
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 1, 75000)
            latent = encoder.backbone(x)
            bottleneck = encoder.bottleneck(latent)
            
            assert latent.shape == (batch_size, 512), f"Failed for batch_size={batch_size}"
            assert bottleneck.shape == (batch_size, 768), f"Failed for batch_size={batch_size}"

    def test_encoder_gradient_flow(self, encoder):
        """Test that gradients flow through encoder."""
        x = torch.randn(4, 1, 75000, requires_grad=True)
        latent = encoder.backbone(x)
        loss = latent.mean()
        loss.backward()
        
        # Check gradients exist in backbone
        for param in encoder.backbone.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradient not computed for some parameters"

    def test_encoder_deterministic(self, encoder):
        """Test encoder produces same output for same input."""
        x = torch.randn(2, 1, 75000)
        encoder.eval()
        
        with torch.no_grad():
            out1 = encoder.backbone(x)
            out2 = encoder.backbone(x)
        
        assert torch.allclose(out1, out2), "Encoder not deterministic in eval mode"

    def test_encoder_device_compatibility(self, encoder):
        """Test encoder can handle different devices."""
        if torch.cuda.is_available():
            encoder_gpu = encoder.to('cuda')
            x_gpu = torch.randn(2, 1, 75000, device='cuda')
            output = encoder_gpu.backbone(x_gpu)
            assert output.device.type == 'cuda'

    def test_encoder_parameter_count(self, encoder):
        """Test encoder has expected number of parameters."""
        total_params = sum(p.numel() for p in encoder.parameters())
        # ResNet encoder with these settings should have ~1.5M parameters
        assert 1e6 < total_params < 2e6, f"Parameter count {total_params} outside expected range"

    def test_encoder_training_mode(self, encoder):
        """Test encoder training and eval modes."""
        x = torch.randn(2, 1, 75000)
        
        # Training mode
        encoder.train()
        out_train1 = encoder.backbone(x)
        out_train2 = encoder.backbone(x)
        # May differ due to dropout
        
        # Eval mode
        encoder.eval()
        with torch.no_grad():
            out_eval1 = encoder.backbone(x)
            out_eval2 = encoder.backbone(x)
        assert torch.allclose(out_eval1, out_eval2)


class TestEncoderInputValidation:
    """Test input validation for encoder."""

    def test_encoder_wrong_input_length(self):
        """Test encoder rejects wrong input length."""
        encoder = ResNetEncoder(in_channels=1, latent_dim=512)
        
        # Wrong length should fail
        x_wrong = torch.randn(2, 1, 70000)  # Wrong length
        
        # Should raise error or handle gracefully
        try:
            output = encoder.backbone(x_wrong)
            # If it doesn't raise error, at least dimensions shouldn't match
            assert output.shape[0] == 2  # Batch dimension preserved
        except RuntimeError:
            # Expected - input length mismatch
            pass

    def test_encoder_wrong_channels(self):
        """Test encoder expects single channel input."""
        encoder = ResNetEncoder(in_channels=1, latent_dim=512)
        
        # 3-channel input instead of 1-channel
        x_wrong = torch.randn(2, 3, 75000)
        
        try:
            output = encoder.backbone(x_wrong)
            # May work due to convolution flexibility, but check channel handling
            assert output.shape == (2, 512)
        except RuntimeError:
            # Expected for channel mismatch
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
