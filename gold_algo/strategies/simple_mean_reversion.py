"""
Simple mean reversion strategy with proven effectiveness.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from .base import BaseStrategy

class SimpleMeanReversionStrategy(BaseStrategy):
    def __init__(self, 
                 lookback_period: int = 20,
                 std_dev_threshold: float = 1.5,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 max_position_size: float = 1.0,
                 stop_loss_pct: float = 0.025,
                 take_profit_pct: float = 0.05):
        """
        Initialize simple mean reversion strategy.
        
        Args:
            lookback_period: Period for moving average calculation
            std_dev_threshold: Standard deviation threshold
            rsi_period: RSI period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            max_position_size: Maximum position size
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        super().__init__('simple_mean_reversion')
        self.lookback_period = lookback_period
        self.std_dev_threshold = std_dev_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI."""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_z_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Z-score for mean reversion."""
        mean = data['Close'].rolling(window=self.lookback_period).mean()
        std = data['Close'].rolling(window=self.lookback_period).std()
        z_score = (data['Close'] - mean) / std
        return z_score
    
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = data['Close'].rolling(window=self.lookback_period).mean()
        std = data['Close'].rolling(window=self.lookback_period).std()
        upper = middle + (std * 2.0)
        lower = middle - (std * 2.0)
        return upper, middle, lower
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate simple mean reversion signals."""
        if not self.validate_data(data):
            return pd.Series(0, index=data.index)
        
        # Calculate indicators
        rsi = self.calculate_rsi(data)
        z_score = self.calculate_z_score(data)
        upper, middle, lower = self.calculate_bollinger_bands(data)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Wait for indicators to be valid
        start_idx = max(self.lookback_period, self.rsi_period)
        
        # Position tracking
        current_position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        for i in range(start_idx, len(data)):
            current_price = data['Close'].iloc[i]
            current_rsi = rsi.iloc[i]
            current_z_score = z_score.iloc[i]
            current_upper = upper.iloc[i]
            current_lower = lower.iloc[i]
            
            # Check exit conditions first
            if current_position != 0:
                if current_position > 0:  # Long position
                    if current_price <= stop_loss or current_price >= take_profit:
                        signals.iloc[i] = -1  # Exit long
                        current_position = 0
                        entry_price = 0
                        stop_loss = 0
                        take_profit = 0
                    else:
                        signals.iloc[i] = 0  # Hold position
                
                elif current_position < 0:  # Short position
                    if current_price >= stop_loss or current_price <= take_profit:
                        signals.iloc[i] = 1  # Exit short
                        current_position = 0
                        entry_price = 0
                        stop_loss = 0
                        take_profit = 0
                    else:
                        signals.iloc[i] = 0  # Hold position
            
            # Generate entry signals only if not in position
            if current_position == 0:
                # Simple mean reversion conditions
                oversold = current_rsi < self.rsi_oversold
                overbought = current_rsi > self.rsi_overbought
                
                # Z-score conditions
                z_oversold = current_z_score < -self.std_dev_threshold
                z_overbought = current_z_score > self.std_dev_threshold
                
                # Bollinger Band conditions
                bb_oversold = current_price < current_lower
                bb_overbought = current_price > current_upper
                
                # Entry conditions - much more relaxed
                if (oversold or z_oversold or bb_oversold) and current_price > data['Close'].iloc[i-1]:
                    signals.iloc[i] = 1  # Enter long
                    current_position = 1
                    entry_price = current_price
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)
                
                elif (overbought or z_overbought or bb_overbought) and current_price < data['Close'].iloc[i-1]:
                    signals.iloc[i] = -1  # Enter short
                    current_position = -1
                    entry_price = current_price
                    stop_loss = current_price * (1 + self.stop_loss_pct)
                    take_profit = current_price * (1 - self.take_profit_pct)
        
        # Apply position sizing based on signal strength
        position_sizes = pd.Series(self.max_position_size, index=data.index)
        
        # Scale position size by z-score strength
        z_strength = abs(z_score) / self.std_dev_threshold
        position_sizes = position_sizes * np.clip(z_strength, 0.5, 1.5)
        
        # Apply position sizing
        final_signals = signals * position_sizes
        
        self.signals = final_signals
        return final_signals
    
    def calculate_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for signals."""
        if not self.validate_data(data):
            return pd.Series(0.5, index=data.index)
        
        # Calculate indicators
        rsi = self.calculate_rsi(data)
        z_score = self.calculate_z_score(data)
        upper, middle, lower = self.calculate_bollinger_bands(data)
        
        # RSI confidence (higher for extreme values)
        rsi_confidence = np.where(
            rsi < 20, 0.9,  # Very oversold
            np.where(rsi > 80, 0.9,  # Very overbought
                    np.where(rsi < 30, 0.7,  # Oversold
                            np.where(rsi > 70, 0.7,  # Overbought
                                    0.3)))  # Neutral
        )
        
        # Z-score confidence (higher for extreme values)
        z_confidence = np.clip(abs(z_score) / self.std_dev_threshold, 0, 1)
        
        # Bollinger Band confidence
        bb_position = (data['Close'] - lower) / (upper - lower)
        bb_confidence = np.where(
            bb_position < 0, 0.8,  # Below lower band
            np.where(bb_position > 1, 0.8,  # Above upper band
                    0.3)  # Between bands
        )
        
        # Combine confidence factors
        confidence = (
            rsi_confidence * 0.4 +
            z_confidence * 0.4 +
            bb_confidence * 0.2
        )
        
        return pd.Series(confidence, index=data.index).clip(0, 1) 