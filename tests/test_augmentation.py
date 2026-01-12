"""Unit tests for augmentation module."""
import sys
from pathlib import Path
import torch
import pytest
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))

from models.ssl.augmentation import PPGAugmentation


class TestPPGAugmentation:
    """Test suite for PPG augmentation."""

    @pytest.fixture
    def augmentation(self):
        """Create augmentation instance."""
        return PPGAugmentation(
            temporal_shift_range=0.1,
            amplitude_scale_range=(0.85, 1.15),
            baseline_wander_freq=0.2,
            noise_prob=0.4
        )

    def test_augmentation_initialization(self, augmentation):
        """Test augmentation initialization."""
        assert augmentation.temporal_shift_range == 0.1
        assert augmentation.amplitude_scale_range == (0.85, 1.15)
        assert augmentation.baseline_wander_freq == 0.2
        assert augmentation.noise_prob == 0.4

    def test_temporal_shift_augmentation(self, augmentation):
        """Test temporal shift augmentation."""
        x = torch.randn(4, 75000)
        
        # Apply shift augmentation
        x_shifted = augmentation.shift(x)
        
        # Shape should be preserved
        assert x_shifted.shape == x.shape
        
        # Shifted version should be different from original
        assert not torch.allclose(x, x_shifted)

    def test_amplitude_scale_augmentation(self, augmentation):
        """Test amplitude scaling augmentation."""
        x = torch.randn(4, 75000)
        
        # Apply scaling augmentation
        x_scaled = augmentation.scale(x)
        
        # Shape should be preserved
        assert x_scaled.shape == x.shape
        
        # Scaled version should be different
        assert not torch.allclose(x, x_scaled)
        
        # Check that scaling is within range
        scale_factors = (x_scaled / (x + 1e-6)).abs().max(dim=1)[0]
        # Scale factors should be close to range [0.85, 1.15]
        assert (scale_factors > 0.8).all() and (scale_factors < 1.2).all()

    def test_baseline_wander_augmentation(self, augmentation):
        """Test baseline wander augmentation."""
        x = torch.randn(4, 75000)
        
        # Apply baseline wander
        x_baseline = augmentation.baseline_wander(x)
        
        # Shape should be preserved
        assert x_baseline.shape == x.shape
        
        # Should be different from original
        assert not torch.allclose(x, x_baseline)

    def test_noise_augmentation(self, augmentation):
        """Test noise augmentation."""
        x = torch.randn(4, 75000)
        
        # Apply noise
        x_noisy = augmentation.add_noise(x)
        
        # Shape should be preserved
        assert x_noisy.shape == x.shape
        
        # Noisy version should be different
        assert not torch.allclose(x, x_noisy)

    def test_augmentation_forward(self, augmentation):
        """Test forward method applies random augmentations."""
        x = torch.randn(4, 75000)
        
        # Apply augmentation
        x_aug = augmentation(x)
        
        # Shape should be preserved
        assert x_aug.shape == x.shape
        
        # Output should be different (with high probability)
        assert not torch.allclose(x, x_aug)

    def test_augmentation_probability(self):
        """Test augmentation respects probability settings."""
        # High probability augmentation
        aug_high = PPGAugmentation(temporal_shift_range=0.3)
        x = torch.randn(10, 75000)
        x_aug = aug_high(x)
        
        # Should be different due to augmentation
        assert not torch.allclose(x, x_aug)
        
        # Zero probability augmentation (no shift, no noise)
        aug_zero = PPGAugmentation(temporal_shift_range=0.0, noise_prob=0.0)
        x_aug_zero = aug_zero(x)
        
        # May be same or different due to other augmentations
        assert x_aug_zero.shape == x.shape

    def test_augmentation_batch_independence(self, augmentation):
        """Test that augmentation is applied independently to each batch."""
        x = torch.randn(4, 75000)
        
        # Apply augmentation multiple times
        x_aug1 = augmentation(x)
        x_aug2 = augmentation(x)
        
        # Results may differ due to randomness in augmentation
        # but shouldn't be identical
        assert not torch.allclose(x_aug1, x_aug2)

    def test_augmentation_numerical_stability(self, augmentation):
        """Test augmentation doesn't produce NaN or Inf."""
        x = torch.randn(4, 75000)
        
        for _ in range(10):
            x_aug = augmentation(x)
            assert not torch.isnan(x_aug).any(), "NaN values in augmented output"
            assert not torch.isinf(x_aug).any(), "Inf values in augmented output"

    def test_augmentation_preserves_dtype(self, augmentation):
        """Test augmentation preserves data type."""
        x = torch.randn(4, 75000, dtype=torch.float32)
        x_aug = augmentation(x)
        
        assert x_aug.dtype == torch.float32, "Data type not preserved"


class TestAugmentationExtremes:
    """Test augmentation with extreme values."""

    def test_augmentation_extreme_range(self):
        """Test augmentation with extreme shift range."""
        aug = PPGAugmentation(temporal_shift_range=0.5)  # 50% shift
        x = torch.randn(2, 75000)
        
        # Should still produce valid output
        x_aug = aug(x)
        assert not torch.isnan(x_aug).any()
        assert not torch.isinf(x_aug).any()

    def test_augmentation_zero_scale(self):
        """Test augmentation with zero amplitude scale."""
        aug = PPGAugmentation(amplitude_scale_range=(1.0, 1.0))  # No scaling
        x = torch.randn(2, 75000)
        
        # With no scaling, some scaling operations should not change
        x_aug = aug(x)
        assert not torch.isnan(x_aug).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
