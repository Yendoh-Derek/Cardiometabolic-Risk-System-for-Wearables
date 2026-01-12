"""
Configuration module for SSL pretraining.

Handles environment detection (Colab vs local), YAML config loading, 
and hyperparameter management.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    train_data_path: str = "data/processed/ssl_pretraining_data.parquet"
    val_data_path: str = "data/processed/ssl_validation_data.parquet"
    test_data_path: str = "data/processed/ssl_test_data.parquet"
    denoised_index_path: str = "data/processed/denoised_signal_index.json"
    signal_length: int = 75000  # 10 min @ 125 Hz
    sample_rate: int = 125  # Hz


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    in_channels: int = 1
    latent_dim: int = 512
    bottleneck_dim: int = 768
    num_blocks: int = 4
    base_filters: int = 32
    max_filters: int = 512


@dataclass
class LossConfig:
    """Loss function weights and parameters."""
    mse_weight: float = 0.50
    ssim_weight: float = 0.30
    fft_weight: float = 0.20
    ssim_window_size: int = 11
    fft_norm: str = "ortho"  # or 'ortho'


@dataclass
class AugmentationConfig:
    """Signal augmentation configuration."""
    temporal_shift_range: float = 0.1  # ±10%
    amplitude_scale_range: tuple = (0.85, 1.15)
    baseline_wander_freq: float = 0.2  # Hz
    baseline_wander_amplitude: float = 0.05
    noise_prob: float = 0.4
    noise_snr_ratio: float = 0.8  # 80% of current SNR


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    batch_size: int = 8
    accumulation_steps: int = 4  # effective batch = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class SSLConfig:
    """Complete SSL configuration."""
    data: DataConfig = None
    model: ModelConfig = None
    loss: LossConfig = None
    augmentation: AugmentationConfig = None
    training: TrainingConfig = None
    
    # Environment
    project_root: Path = None
    data_dir: Path = None  # Override data directory (Colab use case)
    device: str = "cpu"
    is_colab: bool = False
    checkpoint_dir: Path = None
    log_dir: Path = None
    seed: int = 42
    
    def __post_init__(self):
        """Initialize defaults and set up paths."""
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.loss is None:
            self.loss = LossConfig()
        if self.augmentation is None:
            self.augmentation = AugmentationConfig()
        if self.training is None:
            self.training = TrainingConfig()
        
        # Auto-detect environment if not set
        if self.project_root is None:
            self.project_root = Path.cwd()
        
        # Detect Colab
        if self.is_colab is False:
            self.is_colab = self._detect_colab()
        
        # Detect device
        if self.device == "cpu":
            self.device = self._detect_device()
        
        # Setup directories
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "checkpoints" / "ssl"
        if self.log_dir is None:
            self.log_dir = self.project_root / "logs" / "ssl"
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _detect_colab() -> bool:
        """Detect if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _detect_device() -> str:
        """Detect available device (cuda or cpu)."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SSLConfig":
        """Load configuration from YAML file with error handling."""
        try:
            yaml_path_obj = Path(yaml_path)
            if not yaml_path_obj.exists():
                raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if config_dict is None:
                raise ValueError(f"Config file is empty: {yaml_path}")
            
            # Parse nested configs
            config = cls()
            
            if 'data' in config_dict:
                config.data = DataConfig(**config_dict['data'])
            if 'model' in config_dict:
                config.model = ModelConfig(**config_dict['model'])
            if 'loss' in config_dict:
                config.loss = LossConfig(**config_dict['loss'])
            if 'augmentation' in config_dict:
                config.augmentation = AugmentationConfig(**config_dict['augmentation'])
            if 'training' in config_dict:
                config.training = TrainingConfig(**config_dict['training'])
            if 'environment' in config_dict:
                env = config_dict['environment']
                config.is_colab = env.get('is_colab', config.is_colab)
                config.device = env.get('device', config.device)
                config.seed = env.get('seed', config.seed)
            
            return config
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load config: {e}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {yaml_path}: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error loading config from {yaml_path}: {e}")
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'loss': asdict(self.loss),
            'augmentation': asdict(self.augmentation),
            'training': asdict(self.training),
            'environment': {
                'is_colab': self.is_colab,
                'device': self.device,
                'seed': self.seed,
            }
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'loss': asdict(self.loss),
            'augmentation': asdict(self.augmentation),
            'training': asdict(self.training),
            'environment': {
                'is_colab': self.is_colab,
                'device': self.device,
                'seed': self.seed,
            }
        }


def get_default_config() -> SSLConfig:
    """Get default SSL configuration."""
    return SSLConfig()
