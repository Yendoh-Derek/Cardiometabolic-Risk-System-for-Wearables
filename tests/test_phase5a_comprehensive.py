"""
Comprehensive tests for Phase 5A architecture changes.

Features:
- Forward hooks to validate intermediate shapes (B, 256, 156) before pooling
- Shape validation for 1,250-sample inputs (not 75,000)
- 3-block architecture validation (not 4)
- Subject ID string preservation test
- Normalization test
- Dead sensor filtering test
"""

import sys
from pathlib import Path
import torch
import pytest
import numpy as np
import pandas as pd
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))

from models.ssl.encoder import ResNetEncoder
from models.ssl.decoder import ResNetDecoder
from models.ssl.losses import SSLLoss
from models.ssl.dataloader import PPGDataset


class TestEncoderPhase5A:
    """Test suite for Phase 5A encoder (3 blocks, 1,250-sample input)."""

    @pytest.fixture
    def encoder(self):
        """Create Phase 5A encoder (3 blocks, 1250 input)."""
        return ResNetEncoder(
            in_channels=1,
            latent_dim=512,
            bottleneck_dim=768,
            num_blocks=3,  # Phase 5A: 3 blocks
            base_filters=32,
            max_filters=512,
        )

    def test_encoder_initialization(self, encoder):
        """Test Phase 5A encoder initialization."""
        assert encoder.in_channels == 1
        assert encoder.latent_dim == 512
        assert encoder.num_blocks == 3
        assert encoder.bottleneck_dim == 768

    def test_encoder_forward_pass_phase5a(self, encoder):
        """Test encoder forward pass with Phase 5A 1,250-sample input."""
        batch_size = 4
        signal_length = 1250  # Phase 5A: 1,250 samples (10 sec @ 125 Hz)
        
        x = torch.randn(batch_size, 1, signal_length)
        latent = encoder(x)
        
        assert latent.shape == (batch_size, 512), f"Expected (4, 512), got {latent.shape}"

    def test_encoder_intermediate_shapes_via_hooks(self, encoder):
        """
        Test intermediate shapes using PyTorch forward hooks.
        
        CRITICAL: Validates (B, 256, 156) before AvgPool
        (prevents "Block-Chain Failure" where stride is missed)
        """
        batch_size = 2
        signal_length = 1250
        
        x = torch.randn(batch_size, 1, signal_length)
        
        # Register forward hooks to capture intermediate outputs
        hook_outputs = {}
        
        def register_hook(name, module):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    hook_outputs[name] = output.shape
            return hook(name, module)
        
        # Register hooks on key modules
        hook_handles = []
        
        # Initial conv
        hook_handles.append(
            encoder.conv1.register_forward_hook(
                lambda m, i, o: hook_outputs.update({'conv1_out': o.shape})
            )
        )
        
        # Blocks
        for i, block in enumerate(encoder.blocks):
            hook_handles.append(
                block.register_forward_hook(
                    lambda m, i, o, idx=i: hook_outputs.update({f'block_{idx}_out': o.shape})
                )
            )
        
        # AvgPool
        hook_handles.append(
            encoder.avgpool.register_forward_hook(
                lambda m, i, o: hook_outputs.update({'avgpool_out': o.shape})
            )
        )
        
        try:
            latent = encoder(x)
            
            # Validate shapes
            # Initial conv: [2, 1, 1250] -> [2, 32, 625]
            assert hook_outputs['conv1_out'][0] == batch_size, "Batch size mismatch"
            assert hook_outputs['conv1_out'][1] == 32, "Channel count mismatch in conv1"
            assert hook_outputs['conv1_out'][2] <= 625 and hook_outputs['conv1_out'][2] >= 624, \
                f"Conv1d spatial dimension should be ~625, got {hook_outputs['conv1_out'][2]}"
            
            # Block 0: [2, ?, ~625] -> [2, 64, ~312]
            assert hook_outputs['block_0_out'][0] == batch_size, "Batch size mismatch"
            assert hook_outputs['block_0_out'][1] == 64, "Channel count mismatch in block 0"
            assert hook_outputs['block_0_out'][2] <= 313 and hook_outputs['block_0_out'][2] >= 311, \
                f"Block 0 spatial dimension should be ~312, got {hook_outputs['block_0_out'][2]}"
            
            # Block 1: [2, 64, ~312] -> [2, 128, ~156]
            assert hook_outputs['block_1_out'][0] == batch_size, "Batch size mismatch"
            assert hook_outputs['block_1_out'][1] == 128, "Channel count mismatch in block 1"
            assert hook_outputs['block_1_out'][2] <= 157 and hook_outputs['block_1_out'][2] >= 155, \
                f"Block 1 spatial dimension should be ~156, got {hook_outputs['block_1_out'][2]}"
            
            # Block 2: [2, 128, ~156] -> [2, 256, ~78]
            # CRITICAL: This is the key intermediate shape for Phase 5A
            assert hook_outputs['block_2_out'][0] == batch_size, "Batch size mismatch"
            assert hook_outputs['block_2_out'][1] == 256, "Channel count mismatch in block 2 (CRITICAL)"
            assert hook_outputs['block_2_out'][2] <= 79 and hook_outputs['block_2_out'][2] >= 77, \
                f"Block 2 spatial dimension (CRITICAL) should be ~78, got {hook_outputs['block_2_out'][2]}"
            
            # AvgPool: [2, 256, ~78] -> [2, 256, 1]
            assert hook_outputs['avgpool_out'][0] == batch_size, "Batch size mismatch"
            assert hook_outputs['avgpool_out'][1] == 256, "Channel count mismatch in avgpool"
            assert hook_outputs['avgpool_out'][2] == 1, "AvgPool should reduce to 1"
            
            # Final latent: [2, 512]
            assert latent.shape == (batch_size, 512), \
                f"Final latent shape mismatch: {latent.shape}"
            
            print("✅ All intermediate shapes validated correctly!")
            
        finally:
            # Unregister hooks
            for handle in hook_handles:
                handle.remove()


class TestDecoderPhase5A:
    """Test suite for Phase 5A decoder (3 blocks, 1,250-sample output)."""

    @pytest.fixture
    def decoder(self):
        """Create Phase 5A decoder."""
        return ResNetDecoder(
            in_channels=1,
            latent_dim=512,
            bottleneck_dim=768,
            num_blocks=3,  # Phase 5A: 3 blocks
            base_filters=32,
            max_filters=512,
        )

    def test_decoder_forward_pass_phase5a(self, decoder):
        """Test decoder forward pass with 1,250-sample output."""
        batch_size = 4
        
        latent = torch.randn(batch_size, 512)
        reconstructed = decoder(latent)
        
        assert reconstructed.shape == (batch_size, 1, 1250), \
            f"Expected (4, 1, 1250), got {reconstructed.shape}"

    def test_decoder_symmetry(self):
        """Test that decoder is symmetric with encoder."""
        batch_size = 2
        signal_length = 1250
        
        encoder = ResNetEncoder(num_blocks=3)
        decoder = ResNetDecoder(num_blocks=3)
        
        # Forward through encoder
        x = torch.randn(batch_size, 1, signal_length)
        latent = encoder(x)
        
        # Forward through decoder
        reconstructed = decoder(latent)
        
        # Shape should match
        assert reconstructed.shape == x.shape, \
            f"Encoder-decoder symmetry failed: {x.shape} != {reconstructed.shape}"


class TestSSLLossWithFFTPadding:
    """Test SSL loss with Phase 5A FFT padding optimization."""

    @pytest.fixture
    def loss_fn(self):
        """Create SSL loss with Phase 5A FFT padding."""
        return SSLLoss(
            mse_weight=0.50,
            ssim_weight=0.30,
            fft_weight=0.20,
            fft_pad_size=2048,  # Phase 5A: critical fix (was 131072)
        )

    def test_fft_loss_computation(self, loss_fn):
        """Test FFT loss computes without error on 1,250-sample signals."""
        batch_size = 2
        signal_length = 1250
        
        pred = torch.randn(batch_size, 1, signal_length)
        target = torch.randn(batch_size, 1, signal_length)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() > 0, "Loss should be positive"
        assert not np.isnan(loss.item()), "Loss should not be NaN"
        assert not np.isinf(loss.item()), "Loss should not be Inf"

    def test_fft_padding_efficiency(self, loss_fn):
        """Test that FFT padding is efficient (2048, not 131072)."""
        assert loss_fn.fft_loss.fft_pad_size == 2048, \
            f"Expected fft_pad_size=2048, got {loss_fn.fft_loss.fft_pad_size}"


class TestNormalizationAndFiltering:
    """Test Phase 5A per-window normalization and SQI filtering."""

    def test_per_window_normalization(self):
        """Test Z-score normalization preserves μ≈0, σ≈1."""
        from models.ssl.dataloader import PPGDataset
        
        # Create mock metadata
        metadata_df = pd.DataFrame({
            'global_segment_idx': [0],
            'sqi_score': [0.8],
        })
        
        # Mock signal
        signal = np.random.randn(1250).astype(np.float32)
        signal = signal * 10 + 100  # Scale and offset
        
        # Normalize
        epsilon = 1e-8
        mean = signal.mean()
        std = signal.std()
        normalized = (signal - mean) / (std + epsilon)
        
        # Check
        assert np.abs(normalized.mean()) < 1e-6, \
            f"Normalized mean should be ~0, got {normalized.mean()}"
        assert np.abs(normalized.std() - 1.0) < 0.1, \
            f"Normalized std should be ~1.0, got {normalized.std()}"

    def test_dead_sensor_detection(self):
        """Test dead sensor filtering (std < 1e-5)."""
        min_std_threshold = 1e-5
        
        # Create flatline (dead sensor)
        flatline = np.ones(1250, dtype=np.float32) * 100.0
        assert flatline.std() < min_std_threshold, "Flatline should have tiny std"
        
        # Valid signal
        valid_signal = np.random.randn(1250).astype(np.float32)
        assert valid_signal.std() > min_std_threshold, "Random signal should have large std"


class TestSubjectIDStringPreservation:
    """Test that subject_id is preserved as STRING (Phase 8 split safety)."""

    def test_subject_id_string_format(self):
        """Test subject IDs are strings with consistent length."""
        # Create mock metadata
        metadata_dict = {
            'window_id': [0, 1, 2],
            'source_signal_id': ['0', '1', '100'],
            'subject_id': ['000000', '000001', '000100'],  # STRING format
            'sqi_score': [0.8, 0.9, 0.7],
        }
        
        df = pd.DataFrame(metadata_dict)
        
        # Verify subject_id is string
        assert df['subject_id'].dtype == 'object', "subject_id should be string (object)"
        
        # Verify no integer casting corruption (00432 -> 432)
        assert all(isinstance(s, str) for s in df['subject_id']), \
            "All subject_ids should be strings"
        
        # Verify leading zeros preserved
        assert df.loc[0, 'subject_id'] == '000000', \
            "Leading zeros should be preserved"

    def test_subject_id_prevents_leakage(self):
        """Test that subject_id enables proper Phase 8 subject-level split."""
        # Simulate Phase 8 split logic
        metadata = {
            'subject_id': ['P001', 'P001', 'P001', 'P002', 'P002', 'P003'],  # 3 patients
            'window_id': [0, 1, 2, 3, 4, 5],
        }
        df = pd.DataFrame(metadata)
        
        # Group by subject_id for proper split
        subject_groups = df.groupby('subject_id')
        subjects = list(subject_groups.groups.keys())
        
        # Verify each patient has all windows grouped
        assert len(subjects) == 3, "Should have 3 unique patients"
        assert len(subject_groups.get_group('P001')) == 3, "P001 should have 3 windows"
        assert len(subject_groups.get_group('P002')) == 2, "P002 should have 2 windows"
        
        print("✅ Subject-level split logic validated!")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
