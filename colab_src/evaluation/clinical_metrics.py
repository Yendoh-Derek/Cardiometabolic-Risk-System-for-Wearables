"""
Clinical evaluation metrics.
Focuses on AUPRC, sensitivity, specificity for imbalanced data.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, 
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt

class ClinicalMetricsCalculator:
    """Calculate clinical metrics for model evaluation."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_proba: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive metrics for binary classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            # Primary metrics (for imbalanced data)
            'auprc': average_precision_score(y_true, y_proba),
            'auroc': roc_auc_score(y_true, y_proba),
            
            # Clinical metrics
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            
            # Confusion matrix components
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            
            # F-scores
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'f2_score': 5 * tp / (5 * tp + 4 * fn + fp) if (5 * tp + 4 * fn + fp) > 0 else 0,
        }
        
        return metrics
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray,
                                    condition: str, save_path: str = None):
        """Plot precision-recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        auprc = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, 
                label=f'AUPRC = {auprc:.3f}')
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title(f'Precision-Recall Curve: {condition}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return plt.gcf()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                      condition: str, save_path: str = None):
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auroc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC Curve: {condition}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return plt.gcf()