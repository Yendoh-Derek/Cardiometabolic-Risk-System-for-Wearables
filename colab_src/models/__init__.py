"""Model training and evaluation modules."""

# Conditionally import optional modules to avoid import errors
try:
    from .xgboost_classifier import XGBoostMultiLabelClassifier
except ImportError:
    XGBoostMultiLabelClassifier = None

try:
    from .losses import FocalLoss
except ImportError:
    FocalLoss = None

__all__ = [
    'XGBoostMultiLabelClassifier',
    'FocalLoss'
]