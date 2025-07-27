"""
Gold trading algorithm framework.
"""

__version__ = "1.0.0"
__author__ = "Gold Trading Team"

# Import strategies that exist
try:
    from .strategies import (
        UltraAggressiveStrategy,
        AggressiveTradingStrategy,
        Quantitative2PlusSharpeStrategy,
        RiskEnhancedStrategy
    )
    
    __all__ = [
        'UltraAggressiveStrategy',
        'AggressiveTradingStrategy',
        'Quantitative2PlusSharpeStrategy',
        'RiskEnhancedStrategy'
    ]
except ImportError:
    __all__ = []
