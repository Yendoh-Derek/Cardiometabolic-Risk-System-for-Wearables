"""
Heart Rate Variability (HRV) feature extraction.
Extracts time-domain, frequency-domain, and nonlinear features from PPG signals.
"""

import numpy as np
import neurokit2 as nk
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class HRVFeatureExtractor:
    """
    Extract HRV features from PPG signals.
    
    Features extracted:
    - Time-domain: 12 features (RMSSD, SDNN, pNN50, etc.)
    - Frequency-domain: 10 features (LF, HF, LF/HF ratio, etc.)
    - Nonlinear: 8 features (entropy, DFA, Poincaré)
    
    Total: 30 HRV features
    """
    
    def __init__(self, sampling_rate=125):
        self.fs = sampling_rate
        
    def extract_peaks(self, signal: np.ndarray) -> Optional[Dict]:
        """
        Detect PPG peaks (systolic peaks).
        
        Returns:
            Dict with peaks and cleaned signal, or None if detection fails
        """
        try:
            # Clean signal first
            cleaned = nk.ppg_clean(signal, sampling_rate=self.fs)
            
            # Find peaks
            peaks, info = nk.ppg_peaks(cleaned, sampling_rate=self.fs)
            peak_indices = info['PPG_Peaks']
            
            # Need at least 30 peaks for HRV analysis
            if len(peak_indices) < 30:
                return None
            
            return {
                'peaks': peak_indices,
                'cleaned_signal': cleaned
            }
            
        except Exception as e:
            return None
    
    def compute_rr_intervals(self, peak_indices: np.ndarray) -> np.ndarray:
        """
        Compute RR intervals (inter-beat intervals) from peak indices.
        
        Returns:
            RR intervals in milliseconds
        """
        # Time between consecutive peaks
        rr_intervals = np.diff(peak_indices) / self.fs * 1000  # Convert to ms
        return rr_intervals
    
    def extract_time_domain(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain HRV features.
        
        Returns 12 features:
        - mean_rr, sdnn, rmssd, sdsd, pnn50, pnn20
        - mean_hr, max_hr, min_hr, hr_std, tri_index, tinn
        """
        features = {}
        
        try:
            # Basic statistics
            features['mean_rr'] = np.mean(rr_intervals)
            features['sdnn'] = np.std(rr_intervals, ddof=1)
            
            # RMSSD: Root mean square of successive differences
            diff_rr = np.diff(rr_intervals)
            features['rmssd'] = np.sqrt(np.mean(diff_rr ** 2))
            
            # SDSD: Standard deviation of successive differences
            features['sdsd'] = np.std(diff_rr, ddof=1)
            
            # pNN50: Percentage of successive RR intervals differing by > 50ms
            nn50 = np.sum(np.abs(diff_rr) > 50)
            features['pnn50'] = (nn50 / len(diff_rr)) * 100
            
            # pNN20: Percentage of successive RR intervals differing by > 20ms
            nn20 = np.sum(np.abs(diff_rr) > 20)
            features['pnn20'] = (nn20 / len(diff_rr)) * 100
            
            # Heart rate features
            hr = 60000 / rr_intervals  # Convert to bpm
            features['mean_hr'] = np.mean(hr)
            features['max_hr'] = np.max(hr)
            features['min_hr'] = np.min(hr)
            features['hr_std'] = np.std(hr, ddof=1)
            
            # Triangular index (TRI)
            hist, bin_edges = np.histogram(rr_intervals, bins=128)
            features['tri_index'] = len(rr_intervals) / np.max(hist) if np.max(hist) > 0 else 0
            
            # TINN: Baseline width of RR interval histogram
            # Simplified calculation
            features['tinn'] = np.ptp(rr_intervals)
            
        except Exception as e:
            # Fill with NaN on error
            for key in ['mean_rr', 'sdnn', 'rmssd', 'sdsd', 'pnn50', 'pnn20',
                       'mean_hr', 'max_hr', 'min_hr', 'hr_std', 'tri_index', 'tinn']:
                features[key] = np.nan
        
        return features
    
    def extract_frequency_domain(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency-domain HRV features using Welch's method.
        
        Returns 10 features:
        - vlf_power, lf_power, hf_power, total_power
        - lf_hf_ratio, lf_norm, hf_norm
        - vlf_peak, lf_peak, hf_peak
        """
        features = {}
        
        try:
            # Resample RR intervals to uniform time base (4 Hz)
            rr_times = np.cumsum(rr_intervals) / 1000  # Convert to seconds
            uniform_time = np.arange(0, rr_times[-1], 0.25)  # 4 Hz
            rr_interp = np.interp(uniform_time, rr_times, rr_intervals)
            
            # Compute PSD using Welch's method
            from scipy import signal
            freqs, psd = signal.welch(rr_interp, fs=4, nperseg=256)
            
            # Define frequency bands (Hz)
            vlf_band = (0.0033, 0.04)
            lf_band = (0.04, 0.15)
            hf_band = (0.15, 0.4)
            
            # Compute power in each band
            vlf_idx = np.logical_and(freqs >= vlf_band[0], freqs < vlf_band[1])
            lf_idx = np.logical_and(freqs >= lf_band[0], freqs < lf_band[1])
            hf_idx = np.logical_and(freqs >= hf_band[0], freqs < hf_band[1])
            
            features['vlf_power'] = np.trapz(psd[vlf_idx], freqs[vlf_idx])
            features['lf_power'] = np.trapz(psd[lf_idx], freqs[lf_idx])
            features['hf_power'] = np.trapz(psd[hf_idx], freqs[hf_idx])
            features['total_power'] = np.trapz(psd, freqs)
            
            # LF/HF ratio
            features['lf_hf_ratio'] = features['lf_power'] / features['hf_power'] if features['hf_power'] > 0 else 0
            
            # Normalized powers
            lf_hf_sum = features['lf_power'] + features['hf_power']
            features['lf_norm'] = (features['lf_power'] / lf_hf_sum) * 100 if lf_hf_sum > 0 else 0
            features['hf_norm'] = (features['hf_power'] / lf_hf_sum) * 100 if lf_hf_sum > 0 else 0
            
            # Peak frequencies
            features['vlf_peak'] = freqs[vlf_idx][np.argmax(psd[vlf_idx])] if np.any(vlf_idx) else 0
            features['lf_peak'] = freqs[lf_idx][np.argmax(psd[lf_idx])] if np.any(lf_idx) else 0
            features['hf_peak'] = freqs[hf_idx][np.argmax(psd[hf_idx])] if np.any(hf_idx) else 0
            
        except Exception as e:
            # Fill with NaN on error
            for key in ['vlf_power', 'lf_power', 'hf_power', 'total_power',
                       'lf_hf_ratio', 'lf_norm', 'hf_norm', 'vlf_peak', 'lf_peak', 'hf_peak']:
                features[key] = np.nan
        
        return features
    
    def extract_nonlinear(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Extract nonlinear HRV features.
        
        Returns 8 features:
        - sampen, apen (entropy measures)
        - dfa_alpha1, dfa_alpha2 (fractal scaling)
        - sd1, sd2, sd1_sd2_ratio (Poincaré plot)
        - corr_dim (correlation dimension)
        """
        features = {}
        
        try:
            # Sample Entropy
            features['sampen'] = nk.entropy_sample(rr_intervals, delay=1, dimension=2)[0]
            
            # Approximate Entropy
            features['apen'] = nk.entropy_approximate(rr_intervals, delay=1, dimension=2)[0]
            
            # DFA (Detrended Fluctuation Analysis)
            # Short-term scaling exponent (alpha1)
            dfa = nk.fractal_dfa(rr_intervals, multifractal=False)
            features['dfa_alpha1'] = dfa[0] if len(dfa) > 0 else np.nan
            features['dfa_alpha2'] = dfa[1] if len(dfa) > 1 else np.nan
            
            # Poincaré plot indices
            # SD1: Standard deviation of points perpendicular to line of identity
            diff_rr = np.diff(rr_intervals)
            features['sd1'] = np.std(diff_rr, ddof=1) / np.sqrt(2)
            
            # SD2: Standard deviation along line of identity
            sum_rr = rr_intervals[:-1] + rr_intervals[1:]
            features['sd2'] = np.sqrt(2 * np.std(rr_intervals, ddof=1)**2 - features['sd1']**2)
            
            # SD1/SD2 ratio
            features['sd1_sd2_ratio'] = features['sd1'] / features['sd2'] if features['sd2'] > 0 else 0
            
            # Correlation dimension (simplified)
            features['corr_dim'] = nk.fractal_correlation(rr_intervals)[0]
            
        except Exception as e:
            # Fill with NaN on error
            for key in ['sampen', 'apen', 'dfa_alpha1', 'dfa_alpha2',
                       'sd1', 'sd2', 'sd1_sd2_ratio', 'corr_dim']:
                features[key] = np.nan
        
        return features
    
    def extract_all(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract all HRV features from a PPG signal.
        
        Args:
            signal: PPG signal (75,000 samples @ 125 Hz = 10 minutes)
            
        Returns:
            Dictionary with 30 HRV features
        """
        # Detect peaks
        peak_data = self.extract_peaks(signal)
        
        if peak_data is None:
            # Return NaN for all features if peak detection fails
            return {f: np.nan for f in self.get_feature_names()}
        
        # Compute RR intervals
        rr_intervals = self.compute_rr_intervals(peak_data['peaks'])
        
        # Extract features from each domain
        time_features = self.extract_time_domain(rr_intervals)
        freq_features = self.extract_frequency_domain(rr_intervals)
        nonlinear_features = self.extract_nonlinear(rr_intervals)
        
        # Combine all features
        all_features = {**time_features, **freq_features, **nonlinear_features}
        
        return all_features
    
    def get_feature_names(self) -> list:
        """Return list of all feature names in order."""
        return [
            # Time-domain (12)
            'mean_rr', 'sdnn', 'rmssd', 'sdsd', 'pnn50', 'pnn20',
            'mean_hr', 'max_hr', 'min_hr', 'hr_std', 'tri_index', 'tinn',
            # Frequency-domain (10)
            'vlf_power', 'lf_power', 'hf_power', 'total_power',
            'lf_hf_ratio', 'lf_norm', 'hf_norm', 'vlf_peak', 'lf_peak', 'hf_peak',
            # Nonlinear (8)
            'sampen', 'apen', 'dfa_alpha1', 'dfa_alpha2',
            'sd1', 'sd2', 'sd1_sd2_ratio', 'corr_dim'
        ]