"""
Signal augmentation pipeline for self-supervised learning.

Label-free augmentation strategies for PPG signals:
1. Temporal shifts (±5-10%): Jitter beat onsets
2. Amplitude scaling (0.85-1.15): Respects morphology, varies perfusion
3. Baseline wander injection (0.2 Hz): Simulates respiratory artifacts
4. SNR-matched noise: Adds noise at 80% of current SNR
"""

import numpy as np
import torch
from typing import Tuple, Optional


class PPGAugmentation:
    """
    Signal augmentation for PPG waveforms.
    
    All augmentations are label-free and preserve key morphological features
    (heart rate, pulse amplitude, crest time).
    """
    
    def __init__(
        self,
        temporal_shift_range: float = 0.10,
        amplitude_scale_range: Tuple[float, float] = (0.85, 1.15),
        baseline_wander_freq: float = 0.2,
        baseline_wander_amplitude: float = 0.05,
        noise_prob: float = 0.4,
        noise_snr_ratio: float = 0.8,
        sample_rate: int = 125,
        seed: Optional[int] = None,
    ):
        """Initialize augmentation pipeline."""
        self.temporal_shift_range = temporal_shift_range
        self.amplitude_scale_range = amplitude_scale_range
        self.baseline_wander_freq = baseline_wander_freq
        self.baseline_wander_amplitude = baseline_wander_amplitude
        self.noise_prob = noise_prob
        self.noise_snr_ratio = noise_snr_ratio
        self.sample_rate = sample_rate
        
        if seed is not None:
            np.random.seed(seed)
    
    def temporal_shift(self, signal: np.ndarray) -> np.ndarray:
        """Apply temporal shift via circular roll."""
        length = len(signal)
        max_shift = int(length * self.temporal_shift_range)
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(signal, shift) if shift != 0 else signal.copy()
    
    def amplitude_scaling(self, signal: np.ndarray) -> np.ndarray:
        """Apply random amplitude scaling."""
        scale = np.random.uniform(*self.amplitude_scale_range)
        return signal * scale
    
    def baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Inject baseline wander at 0.2 Hz."""
        length = len(signal)
        t = np.arange(length) / self.sample_rate
        wander = self.baseline_wander_amplitude * np.sin(2 * np.pi * self.baseline_wander_freq * t)
        wander += np.random.normal(0, self.baseline_wander_amplitude * 0.1, length)
        return signal + wander
    
    def snr_matched_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian noise matched to signal SNR."""
        signal_pp = np.max(signal) - np.min(signal)
        signal_rms = signal_pp / 4.0
        target_snr = signal_rms / signal.std() if signal.std() > 0 else 1.0
        target_snr = target_snr * self.noise_snr_ratio
        noise = np.random.normal(0, signal_rms / target_snr, len(signal))
        return signal + noise
    
    def compose(self, signal: np.ndarray, augment_prob: float = 1.0) -> np.ndarray:
        """Apply random composition of augmentations."""
        if np.random.random() > augment_prob:
            return signal.copy()
        
        augmented = self.amplitude_scaling(signal)
        if np.random.random() < 0.5:
            augmented = self.temporal_shift(augmented)
        if np.random.random() < 0.6:
            augmented = self.baseline_wander(augmented)
        if np.random.random() < self.noise_prob:
            augmented = self.snr_matched_noise(augmented)
        
        return augmented
import numpy as np
import torch
from typing import Tuple, Optional


class PPGAugmentation:
    """PPG signal augmentation pipeline."""
    
    def __init__(
        self,
        temporal_shift_range: float = 0.1,
        amplitude_scale_range: Tuple[float, float] = (0.85, 1.15),
        baseline_wander_freq: float = 0.2,
        baseline_wander_amplitude: float = 0.05,
        noise_prob: float = 0.4,
        noise_snr_ratio: float = 0.8,
        sample_rate: int = 125,
        random_state: Optional[int] = None,
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            temporal_shift_range: Range for temporal shifts (fraction, e.g., 0.1 = ±10%)
            amplitude_scale_range: Range for amplitude scaling
            baseline_wander_freq: Frequency of baseline wander (Hz)
            baseline_wander_amplitude: Amplitude of baseline wander (fraction of signal)
            noise_prob: Probability of adding noise (0-1)
            noise_snr_ratio: SNR ratio for added noise (0.8 = 80% of current SNR)
            sample_rate: Sampling rate (Hz)
            random_state: For reproducibility
        """
        self.temporal_shift_range = temporal_shift_range
        self.amplitude_scale_range = amplitude_scale_range
        self.baseline_wander_freq = baseline_wander_freq
        self.baseline_wander_amplitude = baseline_wander_amplitude
        self.noise_prob = noise_prob
        self.noise_snr_ratio = noise_snr_ratio
        self.sample_rate = sample_rate
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def temporal_shift(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply temporal shift by rolling the signal.
        
        Args:
            signal: Input signal (length,)
        
        Returns:
            Shifted signal
        """
        max_shift = int(len(signal) * self.temporal_shift_range)
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(signal, shift)
    
    def amplitude_scaling(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply amplitude scaling.
        
        Args:
            signal: Input signal (length,)
        
        Returns:
            Scaled signal
        """
        scale = np.random.uniform(*self.amplitude_scale_range)
        return signal * scale
    
    def baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """
        Add baseline wander (low-frequency oscillation).
        
        Args:
            signal: Input signal (length,)
        
        Returns:
            Signal with baseline wander added
        """
        duration = len(signal) / self.sample_rate
        t = np.linspace(0, duration, len(signal))
        
        # Generate sinusoidal baseline wander
        wander = self.baseline_wander_amplitude * np.sin(2 * np.pi * self.baseline_wander_freq * t)
        
        # Optionally add harmonics for more realistic respiratory artifact
        wander += 0.5 * self.baseline_wander_amplitude * np.sin(
            4 * np.pi * self.baseline_wander_freq * t + np.pi / 4
        )
        
        return signal + wander
    
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add SNR-matched Gaussian noise.
        
        Args:
            signal: Input signal (length,)
        
        Returns:
            Noisy signal
        """
        # Estimate current signal power
        signal_power = np.mean(signal ** 2)
        signal_rms = np.sqrt(signal_power)
        
        # Target SNR (80% of estimated current SNR)
        # Assume current SNR ~40 dB, so target ~39.2 dB
        target_snr_db = 10 * np.log10(signal_power) - 10 * np.log10(signal_rms * 0.1)
        target_snr_db *= self.noise_snr_ratio
        
        # Calculate noise power
        target_snr_linear = 10 ** (target_snr_db / 10)
        noise_power = signal_power / target_snr_linear
        
        # Generate and add Gaussian noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise
    
    def compose(self, signal: np.ndarray, apply_all: bool = True) -> np.ndarray:
        """
        Apply a composition of augmentations.
        
        Args:
            signal: Input signal (length,)
            apply_all: If True, apply all augmentations. If False, randomly select.
        
        Returns:
            Augmented signal
        """
        augmented = signal.copy()
        
        if apply_all:
            # Apply all augmentations
            augmented = self.temporal_shift(augmented)
            augmented = self.amplitude_scaling(augmented)
            augmented = self.baseline_wander(augmented)
            
            if np.random.rand() < self.noise_prob:
                augmented = self.add_noise(augmented)
        else:
            # Randomly select which augmentations to apply
            augmentations = [
                self.temporal_shift,
                self.amplitude_scaling,
                self.baseline_wander,
            ]
            
            # Apply 2-3 random augmentations
            num_to_apply = np.random.randint(2, 4)
            selected = np.random.choice(augmentations, num_to_apply, replace=False)
            
            for aug in selected:
                augmented = aug(augmented)
            
            # Add noise with probability
            if np.random.rand() < self.noise_prob:
                augmented = self.add_noise(augmented)
        
        return augmented
    
    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Allow use as callable."""
        return self.compose(signal)


class TorchAugmentation:
    """PyTorch-compatible augmentation wrapper."""
    
    def __init__(self, ppg_augmentation: PPGAugmentation):
        self.augmentation = ppg_augmentation
    
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to torch tensor.
        
        Args:
            signal: Input tensor (length,) or (1, length)
        
        Returns:
            Augmented tensor (same shape)
        """
        # Handle both 1D and 2D inputs
        if signal.dim() == 2:
            signal = signal.squeeze(0)
        
        # Convert to numpy, augment, convert back
        signal_np = signal.cpu().numpy()
        augmented_np = self.augmentation(signal_np)
        augmented_tensor = torch.from_numpy(augmented_np).float()
        
        return augmented_tensor
