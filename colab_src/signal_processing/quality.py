"""Signal Quality Index (SQI) assessment for PPG signals."""

import numpy as np
from typing import Dict

class SignalQualityAssessor:
    def __init__(self, fs=125, window_sec=10):
        self.fs = fs
        self.window_samples = int(window_sec * fs)
        
    def compute_snr(self, signal_segment: np.ndarray) -> float:
        """Estimate SNR using peak-to-peak amplitude vs noise floor."""
        signal_power = np.ptp(signal_segment)
        noise = np.diff(signal_segment)
        noise_power = np.std(noise)
        
        if noise_power == 0:
            return 0.0
        
        snr = 20 * np.log10(signal_power / noise_power)
        return max(0, snr)
    
    def compute_zero_crossing_rate(self, signal_segment: np.ndarray) -> float:
        """Count zero-crossings. Expected range for PPG: 1-3 Hz (60-180 bpm)."""
        zero_crossings = np.sum(np.diff(np.sign(signal_segment - np.mean(signal_segment))) != 0)
        zcr = zero_crossings / len(signal_segment) * self.fs
        
        # Normalize to 0-1 scale
        zcr_score = 1.0 if 1 <= zcr <= 3 else 0.5
        return zcr_score
    
    def detect_flatline(self, signal_segment: np.ndarray, threshold=0.01) -> bool:
        """Detect if signal is flatlined (sensor disconnect)."""
        std = np.std(signal_segment)
        return std < threshold
    
    def detect_saturation(self, signal_segment: np.ndarray, percentile=99) -> bool:
        """Detect if signal is saturated (clipping)."""
        threshold = np.percentile(np.abs(signal_segment), percentile)
        saturated_samples = np.sum(np.abs(signal_segment) >= threshold * 0.99)
        saturation_ratio = saturated_samples / len(signal_segment)
        return saturation_ratio > 0.05
    
    def compute_perfusion_index(self, signal_segment: np.ndarray) -> float:
        """Estimate perfusion quality (AC/DC ratio)."""
        ac_component = np.ptp(signal_segment)
        dc_component = np.mean(signal_segment)
        
        if dc_component == 0:
            return 0.0
        
        pi = (ac_component / dc_component) * 100
        return pi
    
    def compute_sqi(self, signal_segment: np.ndarray) -> Dict[str, float]:
        """Compute composite SQI score (0-1 scale)."""
        snr = self.compute_snr(signal_segment)
        zcr = self.compute_zero_crossing_rate(signal_segment)
        is_flatline = self.detect_flatline(signal_segment)
        is_saturated = self.detect_saturation(signal_segment)
        perfusion_idx = self.compute_perfusion_index(signal_segment)
        
        # Normalize SNR to 0-1 (expect 10-40 dB)
        snr_score = np.clip(snr / 40, 0, 1)
        
        # Perfusion score (expect 1-10%)
        perfusion_score = np.clip(perfusion_idx / 10, 0, 1)
        
        # Artifact penalties
        flatline_penalty = 0.0 if is_flatline else 1.0
        saturation_penalty = 0.0 if is_saturated else 1.0
        
        # Composite SQI (weighted average)
        sqi_score = (
            0.35 * snr_score +
            0.20 * zcr +
            0.20 * perfusion_score +
            0.15 * flatline_penalty +
            0.10 * saturation_penalty
        )
        
        return {
            'sqi_score': sqi_score,
            'snr_db': snr,
            'snr_score': snr_score,
            'zcr_score': zcr,
            'perfusion_index': perfusion_idx,
            'is_flatline': is_flatline,
            'is_saturated': is_saturated,
            'quality_grade': self._grade_quality(sqi_score)
        }
    
    def _grade_quality(self, sqi_score: float) -> str:
        if sqi_score >= 0.7:
            return "Excellent"
        elif sqi_score >= 0.5:
            return "Good"
        elif sqi_score >= 0.3:
            return "Fair"
        else:
            return "Poor"