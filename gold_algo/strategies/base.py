"""
Base strategy class for the gold trading framework.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str):
        """
        Initialize base strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.logger = logging.getLogger(f"strategy.{name}")
        self.signals = None
        self.confidence = None
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Series with trading signals (-1, 0, 1)
        """
        pass
    
    @abstractmethod
    def calculate_confidence(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence score for each signal.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Series with confidence scores (0 to 1)
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        if data.empty:
            self.logger.error("Data is empty")
            return False
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.signals is None:
            return {}
        
        metrics = {
            'total_signals': len(self.signals[self.signals != 0]),
            'buy_signals': len(self.signals[self.signals == 1]),
            'sell_signals': len(self.signals[self.signals == -1]),
            'signal_frequency': len(self.signals[self.signals != 0]) / len(self.signals)
        }
        
        if self.confidence is not None:
            metrics['avg_confidence'] = self.confidence.mean()
            metrics['confidence_std'] = self.confidence.std()
        
        return metrics
