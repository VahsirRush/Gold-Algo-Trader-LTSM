"""
Mean reversion strategy implementation for the gold trading algorithm.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from .base import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, 
                 lookback_period: int = 20,
                 std_dev_threshold: float = 2.0,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 bollinger_period: int = 20,
                 bollinger_std: float = 2.0,
                 min_holding_period: int = 5):
        """
        Initialize mean reversion strategy.
        
        Args:
            lookback_period: Period for moving average calculation
            std_dev_threshold: Standard deviation threshold for mean reversion
            rsi_oversold: RSI threshold for oversold conditions
            rsi_overbought: RSI threshold for overbought conditions
            bollinger_period: Period for Bollinger Bands calculation
            bollinger_std: Standard deviation for Bollinger Bands
            min_holding_period: Minimum holding period for positions
        """
        super().__init__('mean_reversion')
        self.lookback_period = lookback_period
        self.std_dev_threshold = std_dev_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        self.min_holding_period = min_holding_period
        
    def calculate_rsi(self, data, period: int = 14) -> pd.Series:
        """
        Calculate RSI. Accepts either a Series or a DataFrame with 'Close'.
        """
        if isinstance(data, pd.DataFrame):
            prices = data['Close']
        else:
            prices = data
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = data['Close'].rolling(window=self.bollinger_period).mean()
        std = data['Close'].rolling(window=self.bollinger_period).std()
        upper = middle + (std * self.bollinger_std)
        lower = middle - (std * self.bollinger_std)
        return upper, middle, lower
    
    def calculate_z_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Z-score for mean reversion signals.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with Z-scores
        """
        mean = data['Close'].rolling(window=self.lookback_period).mean()
        std = data['Close'].rolling(window=self.lookback_period).std()
        z_score = (data['Close'] - mean) / std
        return z_score
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            data: DataFrame with price data
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D)
        """
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate enhanced mean reversion signals with much more flexible logic.
        """
        # Calculate Bollinger Bands
        bb_middle = data['Close'].rolling(window=self.bollinger_period).mean()
        bb_std = data['Close'].rolling(window=self.bollinger_period).std()
        bb_upper = bb_middle + (bb_std * self.bollinger_std)
        bb_lower = bb_middle - (bb_std * self.bollinger_std)
        
        # Calculate RSI
        rsi = self.calculate_rsi(data)
        
        # Calculate mean reversion indicators - Much more flexible
        z_score = (data['Close'] - bb_middle) / bb_std
        
        # Calculate momentum for confirmation
        momentum_3 = data['Close'].pct_change(3)
        momentum_5 = data['Close'].pct_change(5)
        momentum_10 = data['Close'].pct_change(10)
        
        # Calculate volatility
        volatility = data['Close'].pct_change().rolling(20).std()
        vol_ma = volatility.rolling(100).mean()
        
        # Calculate moving averages
        sma_20 = data['Close'].rolling(20).mean()
        sma_50 = data['Close'].rolling(50).mean()
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Very flexible bullish mean reversion conditions
        bullish_conditions_1 = (
            (z_score < -0.5) &  # Much more flexible
            (rsi < 45) &  # Much more flexible
            (momentum_3 > -0.005) &  # Very flexible
            (data['Close'] > data['Close'].shift(1))  # Price confirmation
        )
        
        bullish_conditions_2 = (
            (data['Close'] < bb_lower * 1.01) &  # Near lower band
            (rsi < 50) &  # Below neutral
            (momentum_5 > -0.01) &  # Not too negative
            (data['Close'] > sma_20 * 0.98)  # Near moving average
        )
        
        bullish_conditions_3 = (
            (data['Close'] < sma_20 * 0.97) &  # Below short MA
            (rsi < 40) &  # Oversold
            (momentum_10 > -0.02) &  # Not too negative long term
            (volatility < vol_ma * 2.5)  # Not extreme volatility
        )
        
        # Very flexible bearish mean reversion conditions
        bearish_conditions_1 = (
            (z_score > 0.5) &  # Much more flexible
            (rsi > 55) &  # Much more flexible
            (momentum_3 < 0.005) &  # Very flexible
            (data['Close'] < data['Close'].shift(1))  # Price confirmation
        )
        
        bearish_conditions_2 = (
            (data['Close'] > bb_upper * 0.99) &  # Near upper band
            (rsi > 50) &  # Above neutral
            (momentum_5 < 0.01) &  # Not too positive
            (data['Close'] < sma_20 * 1.02)  # Near moving average
        )
        
        bearish_conditions_3 = (
            (data['Close'] > sma_20 * 1.03) &  # Above short MA
            (rsi > 60) &  # Overbought
            (momentum_10 < 0.02) &  # Not too positive long term
            (volatility < vol_ma * 2.5)  # Not extreme volatility
        )
        
        # Generate signals with position sizing
        bullish_all = bullish_conditions_1 | bullish_conditions_2 | bullish_conditions_3
        bearish_all = bearish_conditions_1 | bearish_conditions_2 | bearish_conditions_3
        
        # Strong signals
        signals[bullish_all] = np.clip(-z_score[bullish_all] / 1.5, 0.4, 1.0)
        signals[bearish_all] = -np.clip(z_score[bearish_all] / 1.5, 0.4, 1.0)
        
        # Add breakout signals when price moves outside bands
        breakout_bullish = (
            (data['Close'] < bb_lower) &
            (rsi < 60) &
            (momentum_3 > -0.01) &
            (signals == 0)
        )
        
        breakout_bearish = (
            (data['Close'] > bb_upper) &
            (rsi > 40) &
            (momentum_3 < 0.01) &
            (signals == 0)
        )
        
        signals[breakout_bullish] = 0.6
        signals[breakout_bearish] = -0.6
        
        # Add momentum reversal signals
        momentum_reversal_bullish = (
            (rsi < 35) &
            (momentum_3 > momentum_10) &  # Short-term momentum improving
            (data['Close'] > data['Close'].shift(2)) &
            (signals == 0)
        )
        
        momentum_reversal_bearish = (
            (rsi > 65) &
            (momentum_3 < momentum_10) &  # Short-term momentum deteriorating
            (data['Close'] < data['Close'].shift(2)) &
            (signals == 0)
        )
        
        signals[momentum_reversal_bullish] = 0.4
        signals[momentum_reversal_bearish] = -0.4
        
        # Add mean reversion from moving averages
        ma_reversion_bullish = (
            (data['Close'] < sma_20 * 0.95) &
            (data['Close'] < sma_50 * 0.92) &
            (rsi < 45) &
            (momentum_5 > -0.015) &
            (signals == 0)
        )
        
        ma_reversion_bearish = (
            (data['Close'] > sma_20 * 1.05) &
            (data['Close'] > sma_50 * 1.08) &
            (rsi > 55) &
            (momentum_5 < 0.015) &
            (signals == 0)
        )
        
        signals[ma_reversion_bullish] = 0.5
        signals[ma_reversion_bearish] = -0.5
        
        # Risk management - much more flexible
        extreme_volatility = volatility > vol_ma * 4  # Much more flexible
        signals[extreme_volatility] = 0
        
        # Add signal smoothing
        signals = signals.rolling(window=2, min_periods=1).mean()
        
        # Ensure signals are within bounds
        signals = np.clip(signals, -1, 1)
        
        self.signals = signals
        return signals
    
    def _apply_holding_period(self, signals: pd.Series) -> pd.Series:
        """
        Apply minimum holding period to signals.
        
        Args:
            signals: Raw signals
            
        Returns:
            Signals with holding period applied
        """
        filtered_signals = signals.copy()
        
        for i in range(self.min_holding_period, len(signals)):
            # Check if we have a recent signal
            recent_signals = signals.iloc[i-self.min_holding_period:i]
            if recent_signals.sum() != 0:
                # Keep the most recent signal for the holding period
                filtered_signals.iloc[i] = 0
        
        return filtered_signals
    
    def calculate_confidence(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence score for each signal.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Series with confidence scores (0 to 1)
        """
        # Calculate indicators if not already present
        if 'RSI' not in data.columns:
            data['RSI'] = self.calculate_rsi(data)
        if 'Z_Score' not in data.columns:
            data['Z_Score'] = self.calculate_z_score(data)
        if 'BB_Upper' not in data.columns:
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data)
            data['BB_Upper'] = bb_upper
            data['BB_Middle'] = bb_middle
            data['BB_Lower'] = bb_lower
        
        # Calculate confidence factors
        # RSI confidence (closer to extremes = higher confidence)
        rsi_confidence = np.where(
            data['RSI'] < 30,
            (30 - data['RSI']) / 30,  # Oversold confidence
            np.where(
                data['RSI'] > 70,
                (data['RSI'] - 70) / 30,  # Overbought confidence
                0
            )
        )
        
        # Z-score confidence (higher absolute value = higher confidence)
        z_score_confidence = abs(data['Z_Score']) / self.std_dev_threshold
        z_score_confidence = np.clip(z_score_confidence, 0, 1)
        
        # Bollinger Band confidence
        bb_position = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        bb_confidence = np.where(
            bb_position < 0,
            abs(bb_position),  # Below lower band
            np.where(
                bb_position > 1,
                bb_position - 1,  # Above upper band
                0
            )
        )
        bb_confidence = np.clip(bb_confidence, 0, 1)
        
        # Combine confidence factors
        confidence = (
            rsi_confidence * 0.4 + 
            z_score_confidence * 0.4 + 
            bb_confidence * 0.2
        )
        
        return pd.Series(confidence, index=data.index) 