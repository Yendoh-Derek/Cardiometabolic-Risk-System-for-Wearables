"""
PPG morphology feature extraction.
Analyzes pulse wave shape characteristics.
"""

import numpy as np
from scipy import signal
from typing import Dict

class MorphologyFeatureExtractor:
    """
    Extract PPG pulse wave morphology features.
    
    Features:
    - Pulse amplitude, width, area
    - Crest time, stiffness index
    - Reflection index
    
    Total: 6 morphology features
    """
    
    def __init__(self, sampling_rate=125):
        self.fs = sampling_rate
    
    def extract_pulse_features(self, ppg_signal: np.ndarray, peaks: np.ndarray) -> Dict[str, float]:
        """
        Extract features from individual pulses.
        
        Args:
            ppg_signal: Cleaned PPG signal
            peaks: Peak indices
            
        Returns:
            Dictionary with 6 morphology features
        """
        features = {}
        
        try:
            # Average pulse characteristics
            pulse_amplitudes = []
            pulse_widths = []
            crest_times = []
            pulse_areas = []
            
            for i in range(len(peaks) - 1):
                start = peaks[i]
                end = peaks[i + 1]
                pulse = ppg_signal[start:end]
                
                if len(pulse) < 10:  # Skip very short pulses
                    continue
                
                # Amplitude: Peak-to-trough difference
                amplitude = np.max(pulse) - np.min(pulse)
                pulse_amplitudes.append(amplitude)
                
                # Width: Time from onset to end (at 50% amplitude)
                threshold = np.min(pulse) + 0.5 * amplitude
                above_threshold = pulse > threshold
                if np.any(above_threshold):
                    width = np.sum(above_threshold) / self.fs * 1000  # ms
                    pulse_widths.append(width)
                
                # Crest time: Time from onset to peak
                peak_idx = np.argmax(pulse)
                crest_time = peak_idx / self.fs * 1000  # ms
                crest_times.append(crest_time)
                
                # Area under pulse
                area = np.trapz(pulse)
                pulse_areas.append(area)
            
            # Average features
            features['pulse_amplitude'] = np.mean(pulse_amplitudes) if pulse_amplitudes else np.nan
            features['pulse_width'] = np.mean(pulse_widths) if pulse_widths else np.nan
            features['crest_time'] = np.mean(crest_times) if crest_times else np.nan
            features['area_under_curve'] = np.mean(pulse_areas) if pulse_areas else np.nan
            
            # Stiffness Index (SI): Height / Crest Time
            features['stiffness_index'] = (features['pulse_amplitude'] / (features['crest_time'] / 1000)) if features['crest_time'] > 0 else np.nan
            
            # Reflection Index (RI): Simplified as ratio of diastolic peak to systolic peak
            # This is a simplified version - full calculation requires dicrotic notch detection
            features['reflection_index'] = 0.5  # Placeholder - would need more sophisticated pulse analysis
            
        except Exception as e:
            # Fill with NaN on error
            for key in ['pulse_amplitude', 'pulse_width', 'crest_time', 
                       'stiffness_index', 'reflection_index', 'area_under_curve']:
                features[key] = np.nan
        
        return features
    
    def extract_all(self, signal: np.ndarray, peaks: np.ndarray) -> Dict[str, float]:
        """
        Extract all morphology features.
        
        Args:
            signal: Cleaned PPG signal
            peaks: Peak indices from HRV extractor
            
        Returns:
            Dictionary with 6 morphology features
        """
        return self.extract_pulse_features(signal, peaks)
    
    def get_feature_names(self) -> list:
        """Return list of all feature names in order."""
        return [
            'pulse_amplitude', 'pulse_width', 'crest_time',
            'stiffness_index', 'reflection_index', 'area_under_curve'
        ]