"""
Simple momentum strategy with proven effectiveness.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from .base import BaseStrategy

class SimpleMomentumStrategy(BaseStrategy):
    def __init__(self, 
                 lookback_period: int = 20,
                 momentum_threshold: float = 0.02,
                 volatility_lookback: int = 50,
                 max_position_size: float = 1.0,
                 stop_loss_pct: float = 0.03,
                 take_profit_pct: float = 0.06):
        """
        Initialize simple momentum strategy.
        
        Args:
            lookback_period: Period for momentum calculation
            momentum_threshold: Minimum momentum threshold
            volatility_lookback: Period for volatility calculation
            max_position_size: Maximum position size
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        super().__init__('simple_momentum')
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.volatility_lookback = volatility_lookback
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def calculate_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price momentum."""
        return data['Close'].pct_change(self.lookback_period)
    
    def calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate rolling volatility."""
        returns = data['Close'].pct_change()
        return returns.rolling(window=self.volatility_lookback).std() * np.sqrt(252)
    
    def calculate_ma_crossover(self, data: pd.DataFrame) -> pd.Series:
        """Calculate moving average crossover signal."""
        ma_fast = data['Close'].rolling(window=10).mean()
        ma_slow = data['Close'].rolling(window=30).mean()
        return (ma_fast > ma_slow).astype(int)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate simple momentum signals."""
        if not self.validate_data(data):
            return pd.Series(0, index=data.index)
        
        # Calculate indicators
        momentum = self.calculate_momentum(data)
        volatility = self.calculate_volatility(data)
        ma_crossover = self.calculate_ma_crossover(data)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Wait for indicators to be valid
        start_idx = max(self.lookback_period, self.volatility_lookback, 30)
        
        # Position tracking
        current_position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        for i in range(start_idx, len(data)):
            current_price = data['Close'].iloc[i]
            current_momentum = momentum.iloc[i]
            current_volatility = volatility.iloc[i]
            current_ma_signal = ma_crossover.iloc[i]
            
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
                # Simple momentum conditions
                momentum_positive = current_momentum > self.momentum_threshold
                momentum_negative = current_momentum < -self.momentum_threshold
                
                # Volatility filter (avoid extreme volatility)
                volatility_ok = current_volatility < 0.5  # 50% annualized volatility
                
                # MA crossover confirmation
                ma_bullish = current_ma_signal == 1
                ma_bearish = current_ma_signal == 0
                
                # Entry conditions
                if momentum_positive and volatility_ok and ma_bullish:
                    signals.iloc[i] = 1  # Enter long
                    current_position = 1
                    entry_price = current_price
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)
                
                elif momentum_negative and volatility_ok and ma_bearish:
                    signals.iloc[i] = -1  # Enter short
                    current_position = -1
                    entry_price = current_price
                    stop_loss = current_price * (1 + self.stop_loss_pct)
                    take_profit = current_price * (1 - self.take_profit_pct)
        
        # Apply position sizing based on momentum strength
        position_sizes = pd.Series(self.max_position_size, index=data.index)
        
        # Scale position size by momentum strength
        momentum_strength = abs(momentum) / self.momentum_threshold
        position_sizes = position_sizes * np.clip(momentum_strength, 0.5, 1.5)
        
        # Apply position sizing
        final_signals = signals * position_sizes
        
        self.signals = final_signals
        return final_signals
    
    def calculate_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for signals."""
        if not self.validate_data(data):
            return pd.Series(0.5, index=data.index)
        
        # Calculate indicators
        momentum = self.calculate_momentum(data)
        volatility = self.calculate_volatility(data)
        ma_crossover = self.calculate_ma_crossover(data)
        
        # Momentum confidence (higher for stronger momentum)
        momentum_confidence = np.clip(abs(momentum) / self.momentum_threshold, 0, 1)
        
        # Volatility confidence (prefer moderate volatility)
        vol_confidence = np.where(
            volatility < 0.3, 0.8,  # Good volatility
            np.where(volatility < 0.5, 0.6,  # Moderate volatility
                    0.3)  # High volatility
        )
        
        # MA crossover confidence
        ma_confidence = ma_crossover * 0.5 + 0.5  # 0.5 to 1.0
        
        # Combine confidence factors
        confidence = (
            momentum_confidence * 0.5 +
            vol_confidence * 0.3 +
            ma_confidence * 0.2
        )
        
        return pd.Series(confidence, index=data.index).clip(0, 1) 