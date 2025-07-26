"""
Improved mean reversion strategy with much better performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from .base import BaseStrategy

class ImprovedMeanReversionStrategy(BaseStrategy):
    def __init__(self, 
                 lookback_period: int = 20,
                 std_dev_threshold: float = 2.0,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 bollinger_period: int = 20,
                 bollinger_std: float = 2.0,
                 min_holding_period: int = 5,
                 max_position_size: float = 1.0,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04):
        """
        Initialize improved mean reversion strategy.
        
        Args:
            lookback_period: Period for moving average calculation
            std_dev_threshold: Standard deviation threshold for mean reversion
            rsi_period: RSI period
            rsi_oversold: RSI threshold for oversold conditions
            rsi_overbought: RSI threshold for overbought conditions
            bollinger_period: Period for Bollinger Bands calculation
            bollinger_std: Standard deviation for Bollinger Bands
            min_holding_period: Minimum holding period for positions
            max_position_size: Maximum position size
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        super().__init__('improved_mean_reversion')
        self.lookback_period = lookback_period
        self.std_dev_threshold = std_dev_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        self.min_holding_period = min_holding_period
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
    
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = data['Close'].rolling(window=self.bollinger_period).mean()
        std = data['Close'].rolling(window=self.bollinger_period).std()
        upper = middle + (std * self.bollinger_std)
        lower = middle - (std * self.bollinger_std)
        return upper, middle, lower
    
    def calculate_z_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Z-score for mean reversion signals."""
        mean = data['Close'].rolling(window=self.lookback_period).mean()
        std = data['Close'].rolling(window=self.lookback_period).std()
        z_score = (data['Close'] - mean) / std
        return z_score
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate rolling volatility."""
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate improved mean reversion signals."""
        if not self.validate_data(data):
            return pd.Series(0, index=data.index)
        
        # Calculate indicators
        rsi = self.calculate_rsi(data)
        upper, middle, lower = self.calculate_bollinger_bands(data)
        z_score = self.calculate_z_score(data)
        k_percent, d_percent = self.calculate_stochastic(data)
        volatility = self.calculate_volatility(data)
        
        # Calculate moving averages for trend context
        sma_20 = data['Close'].rolling(20).mean()
        sma_50 = data['Close'].rolling(50).mean()
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Wait for indicators to be valid
        start_idx = max(self.lookback_period, self.bollinger_period, self.rsi_period, 50)
        
        # Position tracking
        current_position = 0
        entry_price = 0
        entry_date = 0
        stop_loss = 0
        take_profit = 0
        
        for i in range(start_idx, len(data)):
            current_price = data['Close'].iloc[i]
            current_date = i
            
            # Check exit conditions first
            if current_position != 0:
                # Check minimum holding period
                if current_date - entry_date < self.min_holding_period:
                    signals.iloc[i] = 0  # Hold position
                    continue
                
                if current_position > 0:  # Long position
                    if current_price <= stop_loss or current_price >= take_profit:
                        signals.iloc[i] = -1  # Exit long
                        current_position = 0
                        entry_price = 0
                        entry_date = 0
                        stop_loss = 0
                        take_profit = 0
                    else:
                        signals.iloc[i] = 0  # Hold position
                
                elif current_position < 0:  # Short position
                    if current_price >= stop_loss or current_price <= take_profit:
                        signals.iloc[i] = 1  # Exit short
                        current_position = 0
                        entry_price = 0
                        entry_date = 0
                        stop_loss = 0
                        take_profit = 0
                    else:
                        signals.iloc[i] = 0  # Hold position
            
            # Generate entry signals only if not in position
            if current_position == 0:
                # Strong bullish mean reversion signal
                bullish_conditions = (
                    z_score.iloc[i] < -self.std_dev_threshold and  # Price below mean
                    rsi.iloc[i] < self.rsi_oversold and  # Oversold
                    current_price < lower.iloc[i] and  # Below lower Bollinger Band
                    k_percent.iloc[i] < 20 and  # Stochastic oversold
                    volatility.iloc[i] < volatility.rolling(50).mean().iloc[i] * 1.5 and  # Not too volatile
                    current_price > sma_50.iloc[i] * 0.95  # Not too far from long-term trend
                )
                
                # Strong bearish mean reversion signal
                bearish_conditions = (
                    z_score.iloc[i] > self.std_dev_threshold and  # Price above mean
                    rsi.iloc[i] > self.rsi_overbought and  # Overbought
                    current_price > upper.iloc[i] and  # Above upper Bollinger Band
                    k_percent.iloc[i] > 80 and  # Stochastic overbought
                    volatility.iloc[i] < volatility.rolling(50).mean().iloc[i] * 1.5 and  # Not too volatile
                    current_price < sma_50.iloc[i] * 1.05  # Not too far from long-term trend
                )
                
                # Additional confirmation signals
                bullish_confirmation = (
                    current_price > data['Close'].iloc[i-1] and  # Price rising
                    rsi.iloc[i] > rsi.iloc[i-1] and  # RSI rising
                    k_percent.iloc[i] > k_percent.iloc[i-1]  # Stochastic rising
                )
                
                bearish_confirmation = (
                    current_price < data['Close'].iloc[i-1] and  # Price falling
                    rsi.iloc[i] < rsi.iloc[i-1] and  # RSI falling
                    k_percent.iloc[i] < k_percent.iloc[i-1]  # Stochastic falling
                )
                
                if bullish_conditions and bullish_confirmation:
                    signals.iloc[i] = 1  # Enter long
                    current_position = 1
                    entry_price = current_price
                    entry_date = current_date
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)
                
                elif bearish_conditions and bearish_confirmation:
                    signals.iloc[i] = -1  # Enter short
                    current_position = -1
                    entry_price = current_price
                    entry_date = current_date
                    stop_loss = current_price * (1 + self.stop_loss_pct)
                    take_profit = current_price * (1 - self.take_profit_pct)
        
        # Apply position sizing based on signal strength
        position_sizes = pd.Series(self.max_position_size, index=data.index)
        
        # Reduce position size based on volatility
        vol_ratio = volatility / volatility.rolling(50).mean()
        position_sizes = position_sizes / np.clip(vol_ratio, 0.5, 2.0)
        
        # Reduce position size based on distance from mean
        mean_distance = abs(z_score)
        position_sizes = position_sizes / np.clip(mean_distance / self.std_dev_threshold, 0.5, 2.0)
        
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
        upper, middle, lower = self.calculate_bollinger_bands(data)
        z_score = self.calculate_z_score(data)
        k_percent, d_percent = self.calculate_stochastic(data)
        volatility = self.calculate_volatility(data)
        
        # Z-score confidence (higher confidence for extreme values)
        z_confidence = np.clip(abs(z_score) / self.std_dev_threshold, 0, 1)
        
        # RSI confidence (higher confidence for extreme values)
        rsi_confidence = np.where(
            rsi < 20, 0.9,  # Very oversold
            np.where(rsi > 80, 0.9,  # Very overbought
                    np.where(rsi < 30, 0.7,  # Oversold
                            np.where(rsi > 70, 0.7,  # Overbought
                                    0.3)))  # Neutral
        )
        
        # Bollinger Band confidence
        bb_position = (data['Close'] - lower) / (upper - lower)
        bb_confidence = np.where(
            bb_position < 0, 0.9,  # Below lower band
            np.where(bb_position > 1, 0.9,  # Above upper band
                    0.3)  # Between bands
        )
        
        # Stochastic confidence
        stoch_confidence = np.where(
            k_percent < 20, 0.8,  # Oversold
            np.where(k_percent > 80, 0.8,  # Overbought
                    0.3)  # Neutral
        )
        
        # Volatility confidence (prefer moderate volatility)
        vol_ratio = volatility / volatility.rolling(50).mean()
        vol_confidence = np.where(
            (vol_ratio > 0.8) & (vol_ratio < 1.5), 0.8,  # Good volatility
            np.where(vol_ratio > 2.0, 0.2,  # Too volatile
                    0.5)  # Low volatility
        )
        
        # Combine confidence factors
        confidence = (
            z_confidence * 0.3 +
            rsi_confidence * 0.25 +
            bb_confidence * 0.2 +
            stoch_confidence * 0.15 +
            vol_confidence * 0.1
        )
        
        return pd.Series(confidence, index=data.index).clip(0, 1) 