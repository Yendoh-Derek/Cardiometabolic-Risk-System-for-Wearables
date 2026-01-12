"""Wavelet denoising for PPG signals."""

import pywt
import numpy as np

class WaveletDenoiser:
    def __init__(self, wavelet='db4', level=5, threshold_method='soft'):
        self.wavelet = wavelet
        self.level = level
        self.threshold_method = threshold_method
    
    def denoise(self, signal: np.ndarray) -> np.ndarray:
        """Apply DWT denoising."""
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        # Noise estimation
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Threshold detail coefficients
        coeffs_thresh = [coeffs[0]]
        for detail in coeffs[1:]:
            thresh = pywt.threshold(detail, threshold, mode=self.threshold_method)
            coeffs_thresh.append(thresh)
        
        # Reconstruct
        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        
        # Match length
        if len(denoised) > len(signal):
            denoised = denoised[:len(signal)]
        elif len(denoised) < len(signal):
            denoised = np.pad(denoised, (0, len(signal) - len(denoised)), mode='edge')
        
        return denoised