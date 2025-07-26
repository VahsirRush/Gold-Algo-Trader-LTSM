"""
Gold trading algorithm framework.
"""

__version__ = "1.0.0"
__author__ = "Gold Trading Team"

from .strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    EnhancedNNStrategy,
    EnhancedDLRegressionStrategy
)

__all__ = [
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'EnhancedNNStrategy',
    'EnhancedDLRegressionStrategy'
]
