"""
Improved trend-following strategy with much better performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from .base import BaseStrategy

class ImprovedTrendFollowingStrategy(BaseStrategy):
    def __init__(self, 
                 ema_fast: int = 12,
                 ema_slow: int = 26,
                 ema_signal: int = 9,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 atr_period: int = 14,
                 stop_loss_atr: float = 2.0,
                 take_profit_atr: float = 4.0,
                 min_trend_strength: float = 0.6,
                 max_position_size: float = 1.0):
        """
        Initialize improved trend-following strategy.
        
        Args:
            ema_fast: Fast EMA period
            ema_slow: Slow EMA period
            ema_signal: MACD signal line period
            rsi_period: RSI period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            atr_period: ATR period for volatility
            stop_loss_atr: Stop loss in ATR multiples
            take_profit_atr: Take profit in ATR multiples
            min_trend_strength: Minimum trend strength (0-1)
            max_position_size: Maximum position size
        """
        super().__init__('improved_trend_following')
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_signal = ema_signal
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.min_trend_strength = min_trend_strength
        self.max_position_size = max_position_size
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period).mean()
    
    def calculate_macd(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = self.calculate_ema(data['Close'], self.ema_fast)
        ema_slow = self.calculate_ema(data['Close'], self.ema_slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, self.ema_signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI."""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using multiple indicators."""
        # Price momentum
        momentum_5 = data['Close'].pct_change(5)
        momentum_10 = data['Close'].pct_change(10)
        momentum_20 = data['Close'].pct_change(20)
        
        # Moving average alignment
        ema_12 = self.calculate_ema(data['Close'], 12)
        ema_26 = self.calculate_ema(data['Close'], 26)
        ema_50 = self.calculate_ema(data['Close'], 50)
        
        # Trend alignment score
        trend_alignment = (
            (ema_12 > ema_26).astype(int) +
            (ema_26 > ema_50).astype(int) +
            (momentum_5 > 0).astype(int) +
            (momentum_10 > 0).astype(int) +
            (momentum_20 > 0).astype(int)
        ) / 5.0
        
        return trend_alignment
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate improved trend-following signals."""
        if not self.validate_data(data):
            return pd.Series(0, index=data.index)
        
        # Calculate indicators
        macd_line, signal_line, histogram = self.calculate_macd(data)
        rsi = self.calculate_rsi(data)
        atr = self.calculate_atr(data)
        trend_strength = self.calculate_trend_strength(data)
        
        # Calculate EMAs for trend direction
        ema_fast = self.calculate_ema(data['Close'], self.ema_fast)
        ema_slow = self.calculate_ema(data['Close'], self.ema_slow)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Wait for indicators to be valid
        start_idx = max(self.ema_slow, self.rsi_period, self.atr_period)
        
        # Position tracking
        current_position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        for i in range(start_idx, len(data)):
            current_price = data['Close'].iloc[i]
            current_atr = atr.iloc[i]
            
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
                        # Update trailing stop
                        new_stop = current_price - (current_atr * self.stop_loss_atr)
                        if new_stop > stop_loss:
                            stop_loss = new_stop
                        signals.iloc[i] = 0  # Hold position
                
                elif current_position < 0:  # Short position
                    if current_price >= stop_loss or current_price <= take_profit:
                        signals.iloc[i] = 1  # Exit short
                        current_position = 0
                        entry_price = 0
                        stop_loss = 0
                        take_profit = 0
                    else:
                        # Update trailing stop
                        new_stop = current_price + (current_atr * self.stop_loss_atr)
                        if new_stop < stop_loss or stop_loss == 0:
                            stop_loss = new_stop
                        signals.iloc[i] = 0  # Hold position
            
            # Generate entry signals only if not in position
            if current_position == 0:
                # Strong bullish signal
                bullish_conditions = (
                    ema_fast.iloc[i] > ema_slow.iloc[i] and  # EMA crossover
                    macd_line.iloc[i] > signal_line.iloc[i] and  # MACD bullish
                    histogram.iloc[i] > 0 and  # MACD histogram positive
                    rsi.iloc[i] > 40 and rsi.iloc[i] < 80 and  # RSI in good range
                    trend_strength.iloc[i] >= self.min_trend_strength and  # Strong trend
                    current_atr > 0  # Sufficient volatility
                )
                
                # Strong bearish signal
                bearish_conditions = (
                    ema_fast.iloc[i] < ema_slow.iloc[i] and  # EMA crossover
                    macd_line.iloc[i] < signal_line.iloc[i] and  # MACD bearish
                    histogram.iloc[i] < 0 and  # MACD histogram negative
                    rsi.iloc[i] < 60 and rsi.iloc[i] > 20 and  # RSI in good range
                    trend_strength.iloc[i] <= (1 - self.min_trend_strength) and  # Strong downtrend
                    current_atr > 0  # Sufficient volatility
                )
                
                if bullish_conditions:
                    signals.iloc[i] = 1  # Enter long
                    current_position = 1
                    entry_price = current_price
                    stop_loss = current_price - (current_atr * self.stop_loss_atr)
                    take_profit = current_price + (current_atr * self.take_profit_atr)
                
                elif bearish_conditions:
                    signals.iloc[i] = -1  # Enter short
                    current_position = -1
                    entry_price = current_price
                    stop_loss = current_price + (current_atr * self.stop_loss_atr)
                    take_profit = current_price - (current_atr * self.take_profit_atr)
        
        # Apply position sizing based on volatility
        position_sizes = pd.Series(self.max_position_size, index=data.index)
        
        # Reduce position size in high volatility
        volatility_ratio = atr / atr.rolling(20).mean()
        position_sizes = position_sizes / np.clip(volatility_ratio, 0.5, 2.0)
        
        # Apply position sizing
        final_signals = signals * position_sizes
        
        self.signals = final_signals
        return final_signals
    
    def calculate_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for signals."""
        if not self.validate_data(data):
            return pd.Series(0.5, index=data.index)
        
        # Calculate indicators
        macd_line, signal_line, histogram = self.calculate_macd(data)
        rsi = self.calculate_rsi(data)
        atr = self.calculate_atr(data)
        trend_strength = self.calculate_trend_strength(data)
        
        # Normalize indicators
        macd_strength = abs(histogram) / (atr + 1e-8)
        macd_strength = np.clip(macd_strength, 0, 1)
        
        rsi_confidence = 1 - abs(rsi - 50) / 50  # Higher confidence when RSI is near 50
        
        # Trend strength confidence
        trend_confidence = np.where(
            trend_strength > 0.7, 0.9,  # Strong uptrend
            np.where(trend_strength < 0.3, 0.9,  # Strong downtrend
                    trend_strength)  # Moderate trend
        )
        
        # Volatility confidence (prefer moderate volatility)
        vol_ratio = atr / atr.rolling(20).mean()
        vol_confidence = np.where(
            (vol_ratio > 0.8) & (vol_ratio < 1.5), 0.8,  # Good volatility
            np.where(vol_ratio > 2.0, 0.3,  # Too volatile
                    0.5)  # Low volatility
        )
        
        # Combine confidence factors
        confidence = (
            macd_strength * 0.3 +
            trend_confidence * 0.3 +
            vol_confidence * 0.2 +
            rsi_confidence * 0.2
        )
        
        return pd.Series(confidence, index=data.index).clip(0, 1) 