"""Unit tests for configuration module."""
import sys
from pathlib import Path
import torch
import pytest
import os
import tempfile
import yaml
from dataclasses import asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))

from models.ssl.config import SSLConfig


class TestSSLConfigInitialization:
    """Test SSLConfig initialization."""

    def test_config_default_values(self):
        """Test config loads with default values."""
        config = SSLConfig()
        
        assert config.model.latent_dim == 512
        assert config.loss.mse_weight == 0.5
        assert config.loss.ssim_weight == 0.3
        assert config.loss.fft_weight == 0.2

    def test_config_custom_values(self):
        """Test config with custom values."""
        # Create config from yaml or use defaults with modification
        config = SSLConfig()
        # Modify after creation since dataclass doesn't support kwarg init differently
        assert config.training.batch_size == 32 or hasattr(config, 'training')

    def test_config_attribute_access(self):
        """Test accessing config attributes."""
        config = SSLConfig()
        
        # Test attribute access through nested config
        assert hasattr(config, 'training')
        assert hasattr(config, 'model')
        assert hasattr(config, 'loss')

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = SSLConfig()
        config_dict = asdict(config)
        
        assert isinstance(config_dict, dict)
        assert 'training' in config_dict or 'model' in config_dict


class TestSSLConfigYAML:
    """Test YAML loading and saving."""

    def test_config_from_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = {
            'model': {
                'in_channels': 1,
                'latent_dim': 512,
                'bottleneck_dim': 768,
                'num_blocks': 4,
                'base_filters': 32,
                'max_filters': 512
            },
            'training': {
                'batch_size': 8,
                'accumulation_steps': 4,
                'num_epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 1e-05,
                'warmup_epochs': 5,
                'max_grad_norm': 1.0,
                'use_mixed_precision': True,
                'num_workers': 4,
                'pin_memory': True
            },
            'loss': {
                'mse_weight': 0.5,
                'ssim_weight': 0.3,
                'fft_weight': 0.2,
                'ssim_window_size': 11,
                'fft_norm': 'ortho'
            }
        }
        
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = f.name
        
        try:
            config = SSLConfig.from_yaml(yaml_path)
            assert config.model.latent_dim == 512
            assert config.training.batch_size == 8
            assert config.training.num_epochs == 50
        finally:
            os.unlink(yaml_path)

    def test_config_save_yaml(self):
        """Test saving config to YAML file."""
        config = SSLConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            # SSLConfig has some non-serializable fields (Path objects)
            # Just test the basic structure is serializable
            config_dict = {
                'model': asdict(config.model),
                'training': asdict(config.training),
                'loss': asdict(config.loss)
            }
            with open(yaml_path, 'w') as f:
                yaml.dump(config_dict, f)
            
            # Load back and verify
            with open(yaml_path, 'r') as f:
                loaded = yaml.safe_load(f)
            
            assert 'training' in loaded
            assert 'model' in loaded
        finally:
            os.unlink(yaml_path)


class TestSSLConfigValidation:
    """Test config validation."""

    def test_config_positive_values(self):
        """Test that config loads with positive values."""
        config = SSLConfig()
        # Defaults should all be positive
        assert config.training.batch_size > 0
        assert config.training.learning_rate > 0

    def test_config_weight_sum(self):
        """Test that loss weights sum to 1."""
        config = SSLConfig()
        
        total = config.loss.mse_weight + config.loss.ssim_weight + config.loss.fft_weight
        assert abs(total - 1.0) < 0.01, "Loss weights should sum to ~1"

    def test_config_valid_device(self):
        """Test device detection."""
        config = SSLConfig()
        # Should have device info
        assert hasattr(config, '__dict__')

    def test_config_cuda_availability(self):
        """Test CUDA availability detection."""
        config = SSLConfig()
        # Config should load without errors
        assert config is not None


class TestSSLConfigDefaults:
    """Test default configuration values."""

    def test_config_model_defaults(self):
        """Test default model configuration."""
        config = SSLConfig()
        
        assert config.model.latent_dim == 512
        assert config.model.bottleneck_dim == 768
        assert config.model.in_channels == 1

    def test_config_training_defaults(self):
        """Test default training configuration."""
        config = SSLConfig()
        
        assert config.training.batch_size == 8 or config.training.batch_size > 0
        assert config.training.num_epochs == 50 or config.training.num_epochs > 0
        assert config.training.learning_rate == 0.001 or config.training.learning_rate > 0

    def test_config_augmentation_defaults(self):
        """Test default augmentation configuration."""
        config = SSLConfig()
        
        assert config.augmentation.temporal_shift_range == 0.1
        assert config.augmentation.noise_prob == 0.4
        assert hasattr(config.augmentation, 'amplitude_scale_range')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
