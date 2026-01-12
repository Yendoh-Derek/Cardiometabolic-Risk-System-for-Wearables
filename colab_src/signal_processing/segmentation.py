"""Signal segmentation into fixed-length windows."""

import numpy as np
from typing import List, Tuple, Dict

class SignalSegmenter:
    def __init__(self, window_sec=600, overlap_sec=0, fs=125):
        self.window_samples = int(window_sec * fs)
        self.overlap_samples = int(overlap_sec * fs)
        self.step_samples = self.window_samples - self.overlap_samples
        self.fs = fs
    
    def segment(self, signal: np.ndarray) -> List[np.ndarray]:
        """Split signal into windows."""
        segments = []
        start_idx = 0
        
        while start_idx + self.window_samples <= len(signal):
            segment = signal[start_idx:start_idx + self.window_samples]
            segments.append(segment)
            start_idx += self.step_samples
        
        return segments
    
    def segment_with_sqi(self, signal: np.ndarray, sqi_engine, 
                         min_sqi=0.5) -> List[Tuple[np.ndarray, Dict]]:
        """Segment and compute SQI for each window."""
        segments = self.segment(signal)
        results = []
        
        for seg in segments:
            sqi_metrics = sqi_engine.compute_sqi(seg)
            if sqi_metrics['sqi_score'] >= min_sqi:
                results.append((seg, sqi_metrics))
        
        return results