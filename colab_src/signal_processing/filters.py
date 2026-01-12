"""Bandpass filtering for PPG signals."""

from scipy.signal import cheby2, filtfilt, resample
import numpy as np

class PPGFilter:
    def __init__(self, fs=125, lowcut=0.5, highcut=8.0, order=4, rs=40):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.rs = rs
        
        # Design filter coefficients
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = cheby2(order, rs, [low, high], btype='band')
    
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply zero-phase filtering (forward-backward)."""
        if len(signal) < 3 * max(len(self.a), len(self.b)):
            return signal
        
        filtered = filtfilt(self.b, self.a, signal)
        return filtered
    
    def resample_signal(self, signal: np.ndarray, original_fs: float, 
                       target_fs: float = 125) -> np.ndarray:
        """Resample signal to target frequency."""
        if original_fs == target_fs:
            return signal
        
        num_samples = int(len(signal) * target_fs / original_fs)
        resampled = resample(signal, num_samples)
        return resampled