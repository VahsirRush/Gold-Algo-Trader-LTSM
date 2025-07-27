#!/usr/bin/env python3
"""
Shared Utilities Module
Contains common functions used across multiple strategy files to eliminate redundancy.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Set, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Common technical indicators used across strategies."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        
        return macd - signal_line
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, lower_band

class FeatureScaler:
    """Standardized feature scaling functionality."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_scaler(self, features: pd.DataFrame):
        """Fit the scaler."""
        self.scaler.fit(features)
        self.is_fitted = True
    
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if not self.is_fitted:
            return features
        return pd.DataFrame(
            self.scaler.transform(features),
            index=features.index,
            columns=features.columns
        )

class ModelTrainer:
    """Common model training functionality."""
    
    @staticmethod
    def prepare_target(data: pd.DataFrame, lookback_days: int = 5) -> pd.Series:
        """Prepare target variable (future returns)."""
        future_returns = data['Close'].pct_change(lookback_days).shift(-lookback_days)
        return future_returns
    
    @staticmethod
    def train_models(models: Dict, features: pd.DataFrame, target: pd.Series):
        """Train all models."""
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]
        
        if len(X) < 100:
            print("❌ Not enough data for training")
            return
        
        # Train each model
        for name, model in models.items():
            model.fit(X, y)
        
        print(f"✅ Models trained on {len(X)} samples")

class RiskManager:
    """Common risk management functionality."""
    
    @staticmethod
    def apply_volatility_targeting(signals: pd.Series, data: pd.DataFrame, 
                                 target_volatility: float = 0.15, 
                                 max_position_size: float = 0.8) -> pd.Series:
        """Apply volatility targeting to position sizing."""
        returns = data['Close'].pct_change()
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        
        # Scale positions based on volatility
        vol_scale = target_volatility / rolling_vol
        vol_scale = vol_scale.clip(0, max_position_size)
        
        return signals * vol_scale
    
    @staticmethod
    def apply_stop_loss_take_profit(signals: pd.Series, data: pd.DataFrame,
                                   stop_loss_pct: float = 0.02,
                                   take_profit_pct: float = 0.03) -> pd.Series:
        """Apply stop-loss and take-profit logic."""
        # This is a simplified implementation
        # In practice, you'd track positions and apply SL/TP dynamically
        return signals

class PerformanceMetrics:
    """Comprehensive performance metrics calculation."""
    
    @staticmethod
    def calculate_metrics(returns: pd.Series, name: str = "Strategy", risk_free_rate: float = 0.02) -> Dict:
        """Calculate comprehensive performance metrics."""
        if len(returns) == 0:
            return {}
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        sortino_ratio = (returns.mean() - risk_free_rate/252) / downside_deviation * np.sqrt(252)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trading statistics
        winning_trades = (returns > 0).sum()
        losing_trades = (returns < 0).sum()
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,  # Changed to match expected key
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'equity_curve': cumulative_returns
        }
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """Print formatted performance metrics."""
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"   Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"   Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.3f}")
        print(f"   Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"   Profit Factor: {metrics.get('profit_factor', 0):.3f}")
        print(f"   Total Trades: {metrics.get('total_trades', 0)}")
        print(f"   VaR (95%): {metrics.get('var_95', 0):.3f}")
        print(f"   CVaR (95%): {metrics.get('cvar_95', 0):.3f}")

class DataFetcher:
    """Common data fetching functionality."""
    
    @staticmethod
    def fetch_data(symbol: str, period: str = '5y') -> pd.DataFrame:
        """Fetch data using yfinance."""
        try:
            import yfinance as yf
            data = yf.download(symbol, period=period)
            
            # Handle MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                # Get the first level (Open, High, Low, Close, Volume)
                data.columns = data.columns.get_level_values(0)
            
            return data
        except ImportError:
            print("❌ yfinance not available")
            return pd.DataFrame()

class FeatureEngineer:
    """Comprehensive feature engineering functionality."""
    
    @staticmethod
    def create_basic_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create basic technical features."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['price_change'] = data['Close'] - data['Close'].shift(1)
        
        # Moving averages
        for period in [3, 5, 8, 13, 21]:
            features[f'sma_{period}'] = data['Close'].rolling(period).mean()
            features[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
            features[f'price_sma_{period}_ratio'] = data['Close'] / features[f'sma_{period}']
            features[f'price_ema_{period}_ratio'] = data['Close'] / features[f'ema_{period}']
        
        # Momentum indicators
        for period in [1, 2, 3, 5, 8]:
            features[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
            features[f'roc_{period}'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
        
        # Volatility indicators
        for period in [3, 5, 8, 13, 21]:
            features[f'volatility_{period}'] = features['returns'].rolling(period).std()
        
        # Volume indicators
        if 'Volume' in data.columns:
            features['volume_ma_3'] = data['Volume'].rolling(3).mean()
            features['volume_ma_5'] = data['Volume'].rolling(5).mean()
            features['volume_ratio_3'] = data['Volume'] / features['volume_ma_3']
            features['volume_ratio_5'] = data['Volume'] / features['volume_ma_5']
        
        # Technical oscillators
        features['rsi_7'] = TechnicalIndicators.calculate_rsi(data['Close'], 7)
        features['rsi_14'] = TechnicalIndicators.calculate_rsi(data['Close'], 14)
        features['macd'] = TechnicalIndicators.calculate_macd(data['Close'])
        
        # Bollinger Bands
        for period in [5, 10, 20]:
            bb_sma = data['Close'].rolling(period).mean()
            bb_std = data['Close'].rolling(period).std()
            features[f'bb_upper_{period}'] = bb_sma + (bb_std * 2)
            features[f'bb_lower_{period}'] = bb_sma - (bb_std * 2)
            features[f'bb_position_{period}'] = (data['Close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        # Mean reversion indicators
        for period in [5, 10, 20]:
            bb_sma = data['Close'].rolling(period).mean()
            bb_std = data['Close'].rolling(period).std()
            features[f'mean_reversion_{period}'] = (data['Close'] - bb_sma) / bb_std
        
        # High-Low analysis
        if all(col in data.columns for col in ['High', 'Low']):
            features['hl_spread'] = (data['High'] - data['Low']) / data['Close']
            features['hl_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        # Additional signals
        features['price_acceleration'] = features['returns'].diff()
        if 'Volume' in data.columns:
            features['volume_acceleration'] = data['Volume'].pct_change()
        
        # Support and resistance
        if all(col in data.columns for col in ['High', 'Low']):
            features['support_level'] = data['Low'].rolling(5).min()
            features['resistance_level'] = data['High'].rolling(5).max()
            features['support_distance'] = (data['Close'] - features['support_level']) / data['Close']
            features['resistance_distance'] = (features['resistance_level'] - data['Close']) / data['Close']
        
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features

class SignalGenerator:
    """Standardized signal generation functionality."""
    
    @staticmethod
    def calculate_momentum_signal(features: pd.DataFrame) -> pd.Series:
        """Calculate momentum-based signal."""
        momentum_indicators = []
        
        # Add available momentum indicators
        for col in ['momentum_1', 'momentum_2', 'momentum_3', 'momentum_5']:
            if col in features.columns:
                momentum_indicators.append(features[col])
        
        for col in ['roc_1', 'roc_2', 'roc_3', 'roc_5']:
            if col in features.columns:
                momentum_indicators.append(features[col])
        
        for col in ['price_sma_3_ratio', 'price_ema_3_ratio']:
            if col in features.columns:
                momentum_indicators.append(features[col] - 1)
        
        if 'price_acceleration' in features.columns:
            momentum_indicators.append(features['price_acceleration'])
        
        # Combine indicators
        if momentum_indicators:
            momentum_signal = pd.concat(momentum_indicators, axis=1).mean(axis=1)
        else:
            momentum_signal = pd.Series(0, index=features.index)
        
        return momentum_signal
    
    @staticmethod
    def calculate_mean_reversion_signal(features: pd.DataFrame) -> pd.Series:
        """Calculate mean reversion signal."""
        mean_reversion_indicators = []
        
        # Add available mean reversion indicators
        for col in ['mean_reversion_5', 'mean_reversion_10']:
            if col in features.columns:
                mean_reversion_indicators.append(-features[col])
        
        for col in ['bb_position_5', 'bb_position_10']:
            if col in features.columns:
                mean_reversion_indicators.append(-features[col])
        
        for col in ['rsi_7', 'rsi_14']:
            if col in features.columns:
                mean_reversion_indicators.append(features[col] - 50)
        
        for col in ['price_sma_3_ratio', 'price_sma_5_ratio', 'price_sma_8_ratio']:
            if col in features.columns:
                mean_reversion_indicators.append(-features[col] + 1)
        
        for col in ['support_distance']:
            if col in features.columns:
                mean_reversion_indicators.append(features[col])
        
        for col in ['resistance_distance']:
            if col in features.columns:
                mean_reversion_indicators.append(-features[col])
        
        # Combine indicators
        if mean_reversion_indicators:
            mean_reversion_signal = pd.concat(mean_reversion_indicators, axis=1).mean(axis=1)
        else:
            mean_reversion_signal = pd.Series(0, index=features.index)
        
        return mean_reversion_signal
    
    @staticmethod
    def calculate_volume_signal(features: pd.DataFrame) -> pd.Series:
        """Calculate volume-based signal."""
        volume_indicators = []
        
        # Add available volume indicators
        for col in ['volume_ratio_3', 'volume_ratio_5']:
            if col in features.columns:
                volume_indicators.append(features[col] - 1)
        
        if 'volume_acceleration' in features.columns:
            volume_indicators.append(features['volume_acceleration'])
        
        if 'hl_spread' in features.columns:
            volume_indicators.append(features['hl_spread'] - features['hl_spread'].rolling(5).mean())
        
        if 'hl_position' in features.columns:
            volume_indicators.append(features['hl_position'] - 0.5)
        
        # Combine indicators
        if volume_indicators:
            volume_signal = pd.concat(volume_indicators, axis=1).mean(axis=1)
        else:
            volume_signal = pd.Series(0, index=features.index)
        
        return volume_signal
    
    @staticmethod
    def calculate_technical_signal(features: pd.DataFrame) -> pd.Series:
        """Calculate technical oscillator signal."""
        technical_indicators = []
        
        # Add available technical indicators
        if 'macd' in features.columns:
            technical_indicators.append(features['macd'])
        
        for col in ['rsi_7', 'rsi_14']:
            if col in features.columns:
                technical_indicators.append(features[col] - 50)
        
        for col in ['bb_position_5', 'bb_position_10']:
            if col in features.columns:
                technical_indicators.append(features[col] - 0.5)
        
        for col in ['volatility_3', 'volatility_5']:
            if col in features.columns:
                technical_indicators.append(features[col] - features[col].rolling(10).mean())
        
        # Combine indicators
        if technical_indicators:
            technical_signal = pd.concat(technical_indicators, axis=1).mean(axis=1)
        else:
            technical_signal = pd.Series(0, index=features.index)
        
        return technical_signal

class KellySizer:
    @staticmethod
    def calculate_kelly_fraction(win_rate, payoff_ratio):
        """
        Calculate the Kelly fraction given win rate and payoff ratio.
        Kelly fraction = win_rate - (1 - win_rate) / payoff_ratio
        """
        if payoff_ratio <= 0:
            return 0.0
        kelly = win_rate - (1 - win_rate) / payoff_ratio
        return max(0.0, min(kelly, 1.0))  # Clamp between 0 and 1

class BaseStrategy:
    """Base strategy class with common functionality."""
    
    def __init__(self, 
                 long_threshold: float = 0.05,
                 short_threshold: float = -0.05,
                 exit_threshold: float = 0.01,
                 risk_free_rate: float = 0.02):
        
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.exit_threshold = exit_threshold
        self.risk_free_rate = risk_free_rate
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest with common logic."""
        print("🔄 Running strategy backtest...")
        
        # Calculate features
        features = self._calculate_features(data)
        
        # Generate signals - pass both features and original data
        signals = self._generate_signals(features, data)
        
        # After calculating signals:
        # If Kelly sizing is enabled, adjust position sizes
        if hasattr(self, 'use_kelly') and getattr(self, 'use_kelly', False):
            # Calculate rolling win rate and payoff ratio
            trades = signals.diff().fillna(0) != 0
            returns = features['returns'].fillna(0)
            trade_returns = returns[trades]
            wins = trade_returns[trade_returns > 0].count()
            losses = trade_returns[trade_returns < 0].count()
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5
            avg_win = trade_returns[trade_returns > 0].mean() if wins > 0 else 0.0
            avg_loss = -trade_returns[trade_returns < 0].mean() if losses > 0 else 0.0
            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
            kelly_fraction = KellySizer.calculate_kelly_fraction(win_rate, payoff_ratio)
            # Scale position sizes by Kelly fraction
            signals = signals * kelly_fraction
        
        # Calculate returns
        close_col = 'Close' if 'Close' in data.columns else 'close'
        returns = data[close_col].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # Remove NaN values
        strategy_returns = strategy_returns.dropna()
        returns = returns[strategy_returns.index]
        
        # Calculate performance metrics
        metrics = PerformanceMetrics.calculate_metrics(strategy_returns, risk_free_rate=self.risk_free_rate)
        
        # Add signals to metrics
        metrics['signals'] = signals
        metrics['thresholds'] = {
            'long': self.long_threshold,
            'short': self.short_threshold,
            'exit': self.exit_threshold
        }
        
        return metrics
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features - to be overridden by subclasses."""
        return FeatureEngineer.create_basic_features(data)
    
    def _generate_signals(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Generate signals - to be overridden by subclasses."""
        # Default implementation
        signals = pd.Series(0, index=features.index)
        return signals 
        return features 

def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate comprehensive technical indicators for trading strategies.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Dictionary of technical indicators
    """
    try:
        indicators = {}
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            # Try capitalized versions
            data = data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # RSI
        indicators['rsi'] = TechnicalIndicators.calculate_rsi(close, period=14)
        
        # MACD
        macd = TechnicalIndicators.calculate_macd(close)
        indicators['macd'] = macd
        indicators['macd_signal'] = macd.ewm(span=9).mean()
        
        # Bollinger Bands
        bb_upper, bb_lower = TechnicalIndicators.calculate_bollinger_bands(close)
        indicators['bb_upper'] = bb_upper
        indicators['bb_lower'] = bb_lower
        indicators['bb_middle'] = close.rolling(20).mean()
        
        # Moving Averages
        indicators['sma_20'] = close.rolling(20).mean()
        indicators['sma_50'] = close.rolling(50).mean()
        indicators['ema_12'] = close.ewm(span=12).mean()
        indicators['ema_26'] = close.ewm(span=26).mean()
        
        # Stochastic
        low_min = low.rolling(14).min()
        high_max = high.rolling(14).max()
        indicators['stoch_k'] = 100 * (close - low_min) / (high_max - low_min)
        indicators['stoch_d'] = indicators['stoch_k'].rolling(3).mean()
        
        # Volume indicators
        indicators['volume_sma'] = volume.rolling(20).mean()
        indicators['obv'] = (volume * np.sign(close.diff())).cumsum()
        indicators['obv_sma'] = indicators['obv'].rolling(20).mean()
        
        # ADX (Average Directional Index)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / tr.rolling(14).mean()
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / tr.rolling(14).mean()
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        indicators['adx'] = dx.rolling(14).mean()
        
        # CCI (Commodity Channel Index)
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        indicators['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Williams %R
        highest_high = high.rolling(14).max()
        lowest_low = low.rolling(14).min()
        indicators['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        # ATR (Average True Range)
        indicators['atr'] = tr.rolling(14).mean()
        
        return indicators
        
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return {} 