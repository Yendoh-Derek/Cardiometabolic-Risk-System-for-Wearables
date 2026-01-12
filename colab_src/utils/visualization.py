"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np

def plot_signals(signals, metadata, n_samples=5, figsize=(15, 10)):
    """Plot random signal samples with SQI scores."""
    fig, axes = plt.subplots(n_samples, 1, figsize=figsize)
    
    indices = np.random.choice(len(signals), min(n_samples, len(signals)), replace=False)
    
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        signal = signals[idx]
        sqi = metadata.iloc[idx]['sqi_score']
        grade = metadata.iloc[idx]['quality_grade']
        
        time = np.arange(len(signal)) / 125
        ax.plot(time, signal, linewidth=0.5)
        ax.set_title(f"Segment {idx}: SQI={sqi:.3f} ({grade})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("PPG Amplitude")
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_sqi_distribution(metadata, save_path=None):
    """Plot SQI score distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(metadata['sqi_score'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(metadata['sqi_score'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {metadata['sqi_score'].mean():.3f}")
    axes[0].set_xlabel("SQI Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("SQI Score Distribution")
    axes[0].legend()
    
    # Quality grades
    grade_counts = metadata['quality_grade'].value_counts()
    axes[1].bar(grade_counts.index, grade_counts.values)
    axes[1].set_xlabel("Quality Grade")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Quality Grade Distribution")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig