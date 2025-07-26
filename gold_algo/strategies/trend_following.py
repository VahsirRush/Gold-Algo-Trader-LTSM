"""
Trend-following strategy implementation for the gold trading algorithm.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from .base import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, 
                 sma_short: int = 5,
                 sma_medium: int = 15,
                 sma_long: int = 50,
                 adx_period: int = 14,
                 adx_threshold: float = 15.0,
                 atr_period: int = 14,
                 macd_fast: int = 8,
                 macd_slow: int = 21,
                 macd_signal: int = 5,
                 rsi_period: int = 14,
                 rsi_oversold: float = 25.0,
                 rsi_overbought: float = 75.0,
                 volatility_threshold: float = 0.01,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04,
                 trailing_stop_pct: float = 0.015,
                 max_position_size: float = 1.0,
                 volatility_lookback: int = 20,
                 # Anti-overfitting parameters
                 min_data_points: int = 200,
                 signal_consistency_threshold: float = 0.7,
                 max_signal_frequency: float = 0.3,
                 regime_filter_enabled: bool = True,
                 regime_lookback: int = 60,
                 regime_threshold: float = 0.6,
                 ensemble_enabled: bool = True,
                 ensemble_size: int = 3,
                 robustness_check_enabled: bool = True,
                 robustness_window: int = 20,
                 min_trade_interval: int = 5,
                 max_consecutive_losses: int = 5,
                 adaptive_parameters: bool = True,
                 parameter_smoothing: float = 0.1):
        """
        Initialize trend-following strategy with aggressive parameters for positive returns.
        
        Args:
            sma_short: Short-term SMA period (very short for quick signals)
            sma_medium: Medium-term SMA period (short for quick signals)
            sma_long: Long-term SMA period (moderate for trend confirmation)
            adx_period: ADX calculation period
            adx_threshold: ADX threshold for trend strength (lower for more signals)
            atr_period: ATR calculation period
            macd_fast: MACD fast EMA period (faster for quicker signals)
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period (faster for quicker signals)
            rsi_period: RSI period for momentum confirmation
            rsi_oversold: RSI oversold threshold (more aggressive)
            rsi_overbought: RSI overbought threshold (more aggressive)
            volatility_threshold: Minimum volatility threshold for trading
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            trailing_stop_pct: Trailing stop percentage
            max_position_size: Maximum position size (1.0 = 100%)
            volatility_lookback: Lookback period for volatility calculation
        """
        super().__init__('trend_following')
        self.sma_short = sma_short
        self.sma_medium = sma_medium
        self.sma_long = sma_long
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volatility_threshold = volatility_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_position_size = max_position_size
        self.volatility_lookback = volatility_lookback
        
        # Anti-overfitting parameters
        self.min_data_points = min_data_points
        self.signal_consistency_threshold = signal_consistency_threshold
        self.max_signal_frequency = max_signal_frequency
        self.regime_filter_enabled = regime_filter_enabled
        self.regime_lookback = regime_lookback
        self.regime_threshold = regime_threshold
        self.ensemble_enabled = ensemble_enabled
        self.ensemble_size = ensemble_size
        self.robustness_check_enabled = robustness_check_enabled
        self.robustness_window = robustness_window
        self.min_trade_interval = min_trade_interval
        self.max_consecutive_losses = max_consecutive_losses
        self.adaptive_parameters = adaptive_parameters
        self.parameter_smoothing = parameter_smoothing
        
        # Performance tracking for overfitting detection
        self.trade_history = []
        self.consecutive_losses = 0
        self.last_trade_date = None
        self.parameter_history = []
        self.performance_metrics = {}
        
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with ATR values
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
        
    def calculate_adx(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.adx_period).mean()
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        # Use pandas operations instead of numpy
        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
        
        # Calculate +DI and -DI
        plus_di = 100 * plus_dm.rolling(window=self.adx_period).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=self.adx_period).mean() / atr
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(window=self.adx_period).mean()
        
        return adx, plus_di, minus_di
    
    def calculate_macd(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        ema_fast = data['Close'].ewm(span=self.macd_fast).mean()
        ema_slow = data['Close'].ewm(span=self.macd_slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with RSI values
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            data: DataFrame with price data
            window: Rolling window for volatility calculation
            
        Returns:
            Series with rolling volatility values
        """
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def detect_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect market regime (trending vs ranging) to avoid overfitting to specific conditions.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with regime indicators (1 = trending, 0 = ranging)
        """
        if not self.regime_filter_enabled:
            return pd.Series(1, index=data.index)  # Assume trending regime
        
        # Calculate trend strength using multiple methods
        returns = data['Close'].pct_change()
        
        # 1. Linear regression R-squared
        def rolling_r_squared(series, window):
            r_squared = pd.Series(index=series.index, dtype=float)
            for i in range(window, len(series)):
                y = series.iloc[i-window:i]
                x = np.arange(len(y))
                if len(y) > 1:
                    slope, intercept = np.polyfit(x, y, 1)
                    y_pred = slope * x + intercept
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r_squared.iloc[i] = 1 - (ss_res / (ss_tot + 1e-8))
            return r_squared
        
        r_squared = rolling_r_squared(data['Close'], self.regime_lookback)
        
        # 2. ADX trend strength
        adx, _, _ = self.calculate_adx(data)
        adx_normalized = adx / 100
        
        # 3. Volatility clustering
        volatility = self.calculate_volatility(data, self.regime_lookback)
        vol_ratio = volatility / volatility.rolling(self.regime_lookback).mean()
        
        # 4. Price momentum
        momentum = data['Close'].pct_change(self.regime_lookback).abs()
        
        # Combine regime indicators
        regime_score = (
            r_squared * 0.4 +
            adx_normalized * 0.3 +
            (1 / vol_ratio).clip(0, 2) * 0.2 +
            momentum.clip(0, 0.1) * 10 * 0.1
        )
        
        # Determine regime
        regime = (regime_score > self.regime_threshold).astype(int)
        
        return regime
    
    def check_signal_consistency(self, signals: pd.Series, window: int = 10) -> pd.Series:
        """
        Check signal consistency to avoid noise-based signals.
        
        Args:
            signals: Signal series
            window: Window for consistency check
            
        Returns:
            Series with consistency scores (0 to 1)
        """
        consistency = pd.Series(0.0, index=signals.index)
        
        for i in range(window, len(signals)):
            recent_signals = signals.iloc[i-window:i]
            if len(recent_signals) > 0:
                # Check if signals are consistent in direction
                positive_signals = (recent_signals > 0).sum()
                negative_signals = (recent_signals < 0).sum()
                total_signals = len(recent_signals)
                
                if total_signals > 0:
                    consistency.iloc[i] = max(positive_signals, negative_signals) / total_signals
        
        return consistency
    
    def check_signal_frequency(self, signals: pd.Series, window: int = 20) -> pd.Series:
        """
        Check signal frequency to avoid over-trading.
        
        Args:
            signals: Signal series
            window: Window for frequency check
            
        Returns:
            Series with frequency scores (0 to 1)
        """
        frequency = pd.Series(0.0, index=signals.index)
        
        for i in range(window, len(signals)):
            recent_signals = signals.iloc[i-window:i]
            if len(recent_signals) > 0:
                # Calculate percentage of non-zero signals
                non_zero_signals = (recent_signals != 0).sum()
                frequency.iloc[i] = non_zero_signals / len(recent_signals)
        
        return frequency
    
    def ensemble_signal_generation(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate ensemble signals using multiple parameter sets to reduce overfitting.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with ensemble signals
        """
        if not self.ensemble_enabled:
            return self._generate_base_signals(data)
        
        ensemble_signals = []
        
        # Create parameter variations
        base_params = {
            'sma_short': self.sma_short,
            'sma_medium': self.sma_medium,
            'sma_long': self.sma_long,
            'adx_period': self.adx_period,
            'adx_threshold': self.adx_threshold,
            'atr_period': self.atr_period,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'volatility_threshold': self.volatility_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trailing_stop_pct': self.trailing_stop_pct,
            'max_position_size': self.max_position_size,
            'volatility_lookback': self.volatility_lookback
        }
        
        # Generate parameter variations
        param_variations = []
        
        # Conservative variation
        conservative = base_params.copy()
        conservative['sma_short'] = int(base_params['sma_short'] * 1.5)
        conservative['sma_medium'] = int(base_params['sma_medium'] * 1.3)
        conservative['adx_threshold'] = base_params['adx_threshold'] * 1.2
        param_variations.append(conservative)
        
        # Aggressive variation
        aggressive = base_params.copy()
        aggressive['sma_short'] = int(base_params['sma_short'] * 0.8)
        aggressive['sma_medium'] = int(base_params['sma_medium'] * 0.9)
        aggressive['adx_threshold'] = base_params['adx_threshold'] * 0.8
        param_variations.append(aggressive)
        
        # Base variation
        param_variations.append(base_params)
        
        # Generate signals for each parameter set
        for params in param_variations[:self.ensemble_size]:
            try:
                # Create temporary strategy instance
                temp_strategy = TrendFollowingStrategy(**params)
                temp_strategy.ensemble_enabled = False  # Prevent recursion
                temp_strategy.regime_filter_enabled = False
                temp_strategy.robustness_check_enabled = False
                
                signals = temp_strategy._generate_base_signals(data.copy())
                ensemble_signals.append(signals)
                
            except Exception as e:
                print(f"Error in ensemble signal generation: {e}")
                continue
        
        if not ensemble_signals:
            return self._generate_base_signals(data)
        
        # Combine ensemble signals
        ensemble_df = pd.concat(ensemble_signals, axis=1)
        ensemble_df.columns = [f'signal_{i}' for i in range(len(ensemble_signals))]
        
        # Use median for robustness
        final_signals = ensemble_df.median(axis=1)
        
        return final_signals
    
    def robustness_check(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Perform robustness check to ensure signals are stable across different conditions.
        
        Args:
            data: DataFrame with price data
            signals: Signal series
            
        Returns:
            Series with robustness-adjusted signals
        """
        if not self.robustness_check_enabled:
            return signals
        
        robustness_scores = pd.Series(1.0, index=signals.index)
        
        for i in range(self.robustness_window, len(signals)):
            if signals.iloc[i] != 0:
                # Check signal stability in recent window
                recent_signals = signals.iloc[i-self.robustness_window:i]
                recent_prices = data['Close'].iloc[i-self.robustness_window:i]
                
                # Calculate signal consistency
                signal_consistency = self.check_signal_consistency(recent_signals, min(10, len(recent_signals)))
                
                # Calculate price stability
                price_volatility = recent_prices.pct_change().std()
                price_stability = 1 / (1 + price_volatility * 100)
                
                # Calculate robustness score
                robustness_scores.iloc[i] = (
                    signal_consistency.iloc[-1] * 0.6 +
                    price_stability * 0.4
                )
        
        # Apply robustness filter
        robust_signals = signals * robustness_scores
        
        return robust_signals
    
    def adaptive_parameter_adjustment(self, data: pd.DataFrame, performance_window: int = 50):
        """
        Adaptively adjust parameters based on recent performance to prevent overfitting.
        
        Args:
            data: DataFrame with price data
            performance_window: Window for performance calculation
        """
        if not self.adaptive_parameters or len(self.trade_history) < performance_window:
            return
        
        # Calculate recent performance
        recent_trades = self.trade_history[-performance_window:]
        if len(recent_trades) < 10:
            return
        
        # Calculate performance metrics
        returns = [trade['return'] for trade in recent_trades]
        avg_return = np.mean(returns)
        return_std = np.std(returns)
        sharpe_ratio = avg_return / (return_std + 1e-8)
        
        # Store performance metrics
        self.performance_metrics = {
            'avg_return': avg_return,
            'return_std': return_std,
            'sharpe_ratio': sharpe_ratio,
            'trade_count': len(recent_trades)
        }
        
        # Adjust parameters based on performance
        if sharpe_ratio < 0.5:  # Poor performance
            # Make strategy more conservative
            self.adx_threshold = min(self.adx_threshold * 1.1, 30.0)
            self.volatility_threshold = max(self.volatility_threshold * 0.9, 0.005)
            self.max_position_size = max(self.max_position_size * 0.9, 0.5)
            
        elif sharpe_ratio > 1.5:  # Good performance
            # Slightly increase aggressiveness
            self.adx_threshold = max(self.adx_threshold * 0.95, 10.0)
            self.volatility_threshold = min(self.volatility_threshold * 1.05, 0.02)
            self.max_position_size = min(self.max_position_size * 1.05, 1.0)
        
        # Store parameter history
        self.parameter_history.append({
            'date': data.index[-1] if len(data) > 0 else None,
            'adx_threshold': self.adx_threshold,
            'volatility_threshold': self.volatility_threshold,
            'max_position_size': self.max_position_size,
            'sharpe_ratio': sharpe_ratio
        })
    
    def check_overfitting_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Check for overfitting indicators and return risk scores.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary with overfitting risk scores
        """
        risk_scores = {}
        
        # 1. Check data sufficiency
        if len(data) < self.min_data_points:
            risk_scores['data_insufficiency'] = 1.0
        else:
            risk_scores['data_insufficiency'] = 0.0
        
        # 2. Check parameter stability
        if len(self.parameter_history) > 10:
            recent_params = self.parameter_history[-10:]
            adx_variation = np.std([p['adx_threshold'] for p in recent_params])
            vol_variation = np.std([p['volatility_threshold'] for p in recent_params])
            
            risk_scores['parameter_instability'] = min(adx_variation + vol_variation, 1.0)
        else:
            risk_scores['parameter_instability'] = 0.0
        
        # 3. Check performance consistency
        if len(self.trade_history) > 20:
            recent_returns = [t['return'] for t in self.trade_history[-20:]]
            return_consistency = 1 - np.std(recent_returns)
            risk_scores['performance_inconsistency'] = max(0, 1 - return_consistency)
        else:
            risk_scores['performance_inconsistency'] = 0.0
        
        # 4. Check consecutive losses
        risk_scores['consecutive_losses'] = min(self.consecutive_losses / 10, 1.0)
        
        # 5. Overall overfitting risk
        risk_scores['overall_risk'] = np.mean(list(risk_scores.values()))
        
        return risk_scores
    
    def calculate_position_size(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate position size based on volatility and signal strength.
        
        Args:
            data: DataFrame with price data
            signals: Signal series
            
        Returns:
            Series with position sizes (0 to max_position_size)
        """
        # Calculate volatility
        volatility = self.calculate_volatility(data, self.volatility_lookback)
        
        # Calculate ATR for risk adjustment
        atr = self.calculate_atr(data)
        
        # Calculate signal strength (absolute value of signals)
        signal_strength = abs(signals)
        
        # Base position size on volatility (inverse relationship)
        # Lower volatility = larger position size
        volatility_factor = 1 / (volatility + 1e-8)
        volatility_factor = np.clip(volatility_factor, 0.1, 2.0)  # Limit range
        
        # ATR factor (inverse relationship)
        atr_factor = 1 / (atr / data['Close'] + 1e-8)
        atr_factor = np.clip(atr_factor, 0.1, 2.0)
        
        # Combine factors
        position_size = signal_strength * volatility_factor * atr_factor * self.max_position_size
        
        # Normalize to max position size
        position_size = np.clip(position_size, 0, self.max_position_size)
        
        return position_size
    
    def apply_exit_logic(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply stop loss, take profit, and trailing stop logic.
        
        Args:
            data: DataFrame with price data
            signals: Raw signals
            
        Returns:
            Series with exit signals
        """
        exit_signals = pd.Series(0, index=data.index)
        
        # Track entry prices and positions
        entry_prices = pd.Series(0.0, index=data.index)
        positions = pd.Series(0, index=data.index)
        trailing_stops = pd.Series(0.0, index=data.index)
        
        current_position = 0
        current_entry_price = 0.0
        current_trailing_stop = 0.0
        
        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            
            # Check for new entry signals
            if current_position == 0 and signals.iloc[i] != 0:
                current_position = np.sign(signals.iloc[i])
                current_entry_price = current_price
                current_trailing_stop = current_price * (1 - self.trailing_stop_pct * current_position)
                positions.iloc[i] = current_position
                entry_prices.iloc[i] = current_entry_price
                trailing_stops.iloc[i] = current_trailing_stop
            
            # Check exit conditions if we have a position
            elif current_position != 0:
                # Stop loss
                if current_position > 0:  # Long position
                    stop_loss_price = current_entry_price * (1 - self.stop_loss_pct)
                    take_profit_price = current_entry_price * (1 + self.take_profit_pct)
                    
                    if current_price <= stop_loss_price or current_price >= take_profit_price:
                        exit_signals.iloc[i] = -1
                        current_position = 0
                        current_entry_price = 0.0
                        current_trailing_stop = 0.0
                    else:
                        # Update trailing stop
                        new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
                        if new_trailing_stop > current_trailing_stop:
                            current_trailing_stop = new_trailing_stop
                        
                        # Check trailing stop
                        if current_price <= current_trailing_stop:
                            exit_signals.iloc[i] = -1
                            current_position = 0
                            current_entry_price = 0.0
                            current_trailing_stop = 0.0
                        else:
                            positions.iloc[i] = current_position
                            trailing_stops.iloc[i] = current_trailing_stop
                
                elif current_position < 0:  # Short position
                    stop_loss_price = current_entry_price * (1 + self.stop_loss_pct)
                    take_profit_price = current_entry_price * (1 - self.take_profit_pct)
                    
                    if current_price >= stop_loss_price or current_price <= take_profit_price:
                        exit_signals.iloc[i] = 1
                        current_position = 0
                        current_entry_price = 0.0
                        current_trailing_stop = 0.0
                    else:
                        # Update trailing stop
                        new_trailing_stop = current_price * (1 + self.trailing_stop_pct)
                        if new_trailing_stop < current_trailing_stop or current_trailing_stop == 0:
                            current_trailing_stop = new_trailing_stop
                        
                        # Check trailing stop
                        if current_price >= current_trailing_stop:
                            exit_signals.iloc[i] = 1
                            current_position = 0
                            current_entry_price = 0.0
                            current_trailing_stop = 0.0
                        else:
                            positions.iloc[i] = current_position
                            trailing_stops.iloc[i] = current_trailing_stop
        
        return exit_signals
    
    def _generate_base_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate base trend-following signals (original implementation).
        """
        # Validate data first
        if not self.validate_data(data):
            return pd.Series(0, index=data.index)
        
        # Ensure we have enough data for calculations
        min_periods = max(self.sma_long, self.adx_period * 2, self.rsi_period * 2)
        if len(data) < min_periods:
            return pd.Series(0, index=data.index)
        
        # Calculate all indicators
        data['SMA_short'] = data['Close'].rolling(window=self.sma_short).mean()
        data['SMA_medium'] = data['Close'].rolling(window=self.sma_medium).mean()
        data['SMA_long'] = data['Close'].rolling(window=self.sma_long).mean()
        
        adx, plus_di, minus_di = self.calculate_adx(data)
        macd_line, signal_line, histogram = self.calculate_macd(data)
        atr = self.calculate_atr(data)
        rsi = self.calculate_rsi(data)
        volatility = self.calculate_volatility(data)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Wait for indicators to be valid
        start_idx = max(self.sma_long, self.adx_period * 2, self.rsi_period * 2)
        
        # Aggressive signal generation logic
        for i in range(start_idx, len(data)):
            signal_strength = 0.0
            
            # 1. Moving Average Crossover (40% weight)
            if data['SMA_short'].iloc[i] > data['SMA_medium'].iloc[i] > data['SMA_long'].iloc[i]:
                signal_strength += 0.4  # Strong uptrend
            elif data['SMA_short'].iloc[i] < data['SMA_medium'].iloc[i] < data['SMA_long'].iloc[i]:
                signal_strength -= 0.4  # Strong downtrend
            elif data['SMA_short'].iloc[i] > data['SMA_medium'].iloc[i]:
                signal_strength += 0.2  # Weak uptrend
            elif data['SMA_short'].iloc[i] < data['SMA_medium'].iloc[i]:
                signal_strength -= 0.2  # Weak downtrend
            
            # 2. MACD Confirmation (30% weight)
            if macd_line.iloc[i] > signal_line.iloc[i] and histogram.iloc[i] > 0:
                signal_strength += 0.3
            elif macd_line.iloc[i] < signal_line.iloc[i] and histogram.iloc[i] < 0:
                signal_strength -= 0.3
            
            # 3. RSI Momentum (20% weight)
            if rsi.iloc[i] < self.rsi_oversold:
                signal_strength += 0.2  # Oversold - potential buy
            elif rsi.iloc[i] > self.rsi_overbought:
                signal_strength -= 0.2  # Overbought - potential sell
            
            # 4. ADX Trend Strength (10% weight)
            if adx.iloc[i] > self.adx_threshold:
                if plus_di.iloc[i] > minus_di.iloc[i]:
                    signal_strength += 0.1
                else:
                    signal_strength -= 0.1
            
            # 5. Volatility Filter
            if volatility.iloc[i] < self.volatility_threshold:
                signal_strength *= 0.5  # Reduce signal strength in low volatility
            
            # Convert to final signal
            if signal_strength > 0.3:
                signals.iloc[i] = 1.0
            elif signal_strength < -0.3:
                signals.iloc[i] = -1.0
            elif signal_strength > 0.1:
                signals.iloc[i] = 0.5
            elif signal_strength < -0.1:
                signals.iloc[i] = -0.5
        
        # Apply exit logic
        exit_signals = self.apply_exit_logic(data, signals)
        
        # Combine entry and exit signals
        final_signals = signals + exit_signals
        
        # Ensure signals are within bounds
        final_signals = np.clip(final_signals, -1, 1)
        
        # Calculate position sizes
        position_sizes = self.calculate_position_size(data, final_signals)
        
        # Apply position sizing to signals
        final_signals = final_signals * position_sizes
        
        self.signals = final_signals
        return final_signals
    
    def calculate_confidence(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate improved confidence score for each signal.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Series with confidence scores (0 to 1)
        """
        if not self.validate_data(data):
            return pd.Series(0.5, index=data.index)
        
        adx, plus_di, minus_di = self.calculate_adx(data)
        macd_line, signal_line, histogram = self.calculate_macd(data)
        atr = self.calculate_atr(data)
        rsi = self.calculate_rsi(data)
        volatility = self.calculate_volatility(data)
        
        # Normalize indicators
        adx_normalized = adx / 100
        rsi_normalized = 1 - abs(rsi - 50) / 50  # Higher confidence when RSI is near 50
        volatility_normalized = volatility / volatility.rolling(100).mean()
        volatility_normalized = np.clip(volatility_normalized, 0.5, 2.0) / 2.0
        
        # Trend strength
        trend_strength = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        
        # Moving average alignment
        ma_alignment = (
            (data['SMA_short'] > data['SMA_medium']).astype(int) +
            (data['SMA_medium'] > data['SMA_long']).astype(int)
        ) / 2
        
        # MACD strength
        macd_strength = abs(histogram) / (atr + 1e-8)
        macd_strength = np.clip(macd_strength, 0, 1)
        
        # Combine confidence factors with weights
        confidence = (
            adx_normalized * 0.25 +
            trend_strength * 0.20 +
            ma_alignment * 0.20 +
            macd_strength * 0.15 +
            rsi_normalized * 0.10 +
            volatility_normalized * 0.10
        )
        
        return confidence.clip(0, 1)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trend-following signals with comprehensive anti-overfitting mechanisms.
        
        This enhanced method incorporates:
        - Market regime detection
        - Signal consistency checks
        - Ensemble signal generation
        - Robustness validation
        - Adaptive parameter adjustment
        - Overfitting risk monitoring
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with robust signals
        """
        # Check data sufficiency
        if len(data) < self.min_data_points:
            print(f"Warning: Insufficient data points ({len(data)} < {self.min_data_points})")
            return pd.Series(0, index=data.index)
        
        # Check overfitting indicators
        overfitting_risks = self.check_overfitting_indicators(data)
        if overfitting_risks['overall_risk'] > 0.7:
            print(f"Warning: High overfitting risk detected ({overfitting_risks['overall_risk']:.3f})")
            # Reduce position sizes when overfitting risk is high
            self.max_position_size *= 0.5
        
        # Detect market regime
        regime = self.detect_market_regime(data)
        
        # Generate base signals
        if self.ensemble_enabled:
            base_signals = self.ensemble_signal_generation(data)
        else:
            base_signals = self._generate_base_signals(data)
        
        # Apply regime filter
        if self.regime_filter_enabled:
            # Only trade in trending regimes
            base_signals = base_signals * regime
        
        # Check signal consistency
        consistency = self.check_signal_consistency(base_signals)
        base_signals = base_signals * consistency
        
        # Check signal frequency
        frequency = self.check_signal_frequency(base_signals)
        # Reduce signals if frequency is too high
        frequency_filter = np.where(frequency > self.max_signal_frequency, 
                                   self.max_signal_frequency / frequency, 1.0)
        base_signals = base_signals * frequency_filter
        
        # Apply robustness check
        robust_signals = self.robustness_check(data, base_signals)
        
        # Apply minimum trade interval
        if self.min_trade_interval > 1:
            filtered_signals = pd.Series(0, index=robust_signals.index)
            last_signal_idx = -self.min_trade_interval
            
            for i in range(len(robust_signals)):
                if i > last_signal_idx + self.min_trade_interval and robust_signals.iloc[i] != 0:
                    filtered_signals.iloc[i] = robust_signals.iloc[i]
                    last_signal_idx = i
            
            robust_signals = filtered_signals
        
        # Apply consecutive loss filter
        if self.consecutive_losses >= self.max_consecutive_losses:
            print(f"Warning: Maximum consecutive losses reached ({self.consecutive_losses})")
            robust_signals = robust_signals * 0.5  # Reduce position sizes
        
        # Calculate position sizes
        position_sizes = self.calculate_position_size(data, robust_signals)
        
        # Apply position sizing
        final_signals = robust_signals * position_sizes
        
        # Store signals for analysis
        self.signals = final_signals
        
        # Adaptive parameter adjustment
        self.adaptive_parameter_adjustment(data)
        
        # Log performance metrics
        if len(self.performance_metrics) > 0:
            print(f"Performance - Sharpe: {self.performance_metrics.get('sharpe_ratio', 0):.3f}, "
                  f"Avg Return: {self.performance_metrics.get('avg_return', 0):.4f}")
        
        return final_signals
    
    def record_trade(self, entry_date, exit_date, entry_price, exit_price, position_size, signal_strength):
        """
        Record trade for performance tracking and overfitting detection.
        
        Args:
            entry_date: Trade entry date
            exit_date: Trade exit date
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size
            signal_strength: Signal strength at entry
        """
        trade_return = (exit_price - entry_price) / entry_price * np.sign(position_size)
        
        trade_record = {
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'signal_strength': signal_strength,
            'return': trade_return,
            'duration': (exit_date - entry_date).days if hasattr(exit_date, 'days') else 0
        }
        
        self.trade_history.append(trade_record)
        
        # Update consecutive losses
        if trade_return < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update last trade date
        self.last_trade_date = exit_date
    
    def get_overfitting_report(self, data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive overfitting report.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary with overfitting analysis results
        """
        risk_scores = self.check_overfitting_indicators(data)
        
        report = {
            'risk_scores': risk_scores,
            'trade_count': len(self.trade_history),
            'consecutive_losses': self.consecutive_losses,
            'parameter_stability': len(self.parameter_history),
            'performance_metrics': self.performance_metrics,
            'recommendations': []
        }
        
        # Generate recommendations
        if risk_scores['overall_risk'] > 0.7:
            report['recommendations'].append("HIGH overfitting risk - consider reducing strategy complexity")
        elif risk_scores['overall_risk'] > 0.4:
            report['recommendations'].append("MEDIUM overfitting risk - monitor performance closely")
        
        if risk_scores['data_insufficiency'] > 0.5:
            report['recommendations'].append("Insufficient data - collect more historical data")
        
        if risk_scores['parameter_instability'] > 0.3:
            report['recommendations'].append("Parameter instability - consider fixing parameters")
        
        if risk_scores['consecutive_losses'] > 0.5:
            report['recommendations'].append("High consecutive losses - review risk management")
        
        return report 