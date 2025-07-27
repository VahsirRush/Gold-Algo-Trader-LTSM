"""
Strategy module for the gold trading framework.
"""

from .base import BaseStrategy

# Import strategies that exist
try:
    from .ultra_aggressive_strategy import UltraAggressiveStrategy
except ImportError:
    pass

try:
    from .aggressive_trading_strategy import AggressiveTradingStrategy
except ImportError:
    pass

try:
    from .quantitative_2plus_sharpe_strategy import Quantitative2PlusSharpeStrategy
except ImportError:
    pass

try:
    from .risk_enhanced_strategy import RiskEnhancedStrategy
except ImportError:
    pass

__all__ = [
    'BaseStrategy',
    'UltraAggressiveStrategy',
    'AggressiveTradingStrategy', 
    'Quantitative2PlusSharpeStrategy',
    'RiskEnhancedStrategy'
]
