"""
Data pipeline module for the gold trading framework.
"""

from .base import BaseDataCollector
from .gold_spot import GoldSpotCollector
from .macro import MacroDataCollector
from .news_sentiment import NewsSentimentCollector
from .cot import CoTDataCollector
from .central_bank import CentralBankDataCollector

__all__ = [
    'BaseDataCollector',
    'GoldSpotCollector',
    'MacroDataCollector', 
    'NewsSentimentCollector',
    'CoTDataCollector',
    'CentralBankDataCollector'
] 