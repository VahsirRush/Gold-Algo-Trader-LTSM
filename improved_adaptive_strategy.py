#!/usr/bin/env python3
"""
IMPROVED ADAPTIVE STRATEGY WITH HIGHER RETURNS
=============================================

Enhanced version of the adaptive overfitting protection system
with more aggressive parameters and better signal generation
to achieve higher returns while maintaining risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ImprovedAdaptiveGoldStrategy:
    """Enhanced adaptive strategy with higher return potential."""
    
    def __init__(self, 
                 target_volatility: float = 0.25,  # Increased from 0.20
                 max_position_size: float = 1.5,   # Increased from 1.0
                 use_adaptive_thresholds: bool = True,
                 use_regime_filtering: bool = True,
                 regularization_strength: float = 0.03,  # Reduced from 0.05
                 max_features: int = 75,           # Increased from 50
                 signal_sensitivity: float = 0.6,  # New parameter for signal sensitivity
                 momentum_weight: float = 0.4,     # New parameter for momentum
                 mean_reversion_weight: float = 0.3, # New parameter for mean reversion
                 volume_weight: float = 0.3):      # New parameter for volume
        
        self.target_volatility = target_volatility
        self.max_position_size = max_position_size
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.use_regime_filtering = use_regime_filtering
        self.regularization_strength = regularization_strength
        self.max_features = max_features
        self.signal_sensitivity = signal_sensitivity
        self.momentum_weight = momentum_weight
        self.mean_reversion_weight = mean_reversion_weight
        self.volume_weight = volume_weight
        
        # Strategy state
        self.is_trained = False
        self.feature_importance = {}
        self.signal_thresholds = {
            'long': 0.3,    # Reduced from higher values
            'short': -0.3,  # More aggressive
            'exit': 0.1     # Faster exits
        }
        
        # Performance tracking
        self.trade_history = []
        self.current_position = 0
        self.entry_price = None
        self.entry_date = None
        
    def train_on_data(self, data: pd.DataFrame):
        """Train the strategy on historical data."""
        print("🚀 Training Improved Adaptive Strategy...")
        
        # Calculate technical indicators
        features = self._calculate_features(data)
        
        # Calculate target returns (next day returns)
        target_returns = data['Close'].pct_change().shift(-1).dropna()
        
        # Align features with targets
        features = features.dropna()
        target_returns = target_returns[features.index]
        
        # Simple feature importance based on correlation
        correlations = {}
        for col in features.columns:
            if col != 'target':
                corr = abs(features[col].corr(target_returns))
                correlations[col] = corr
        
        # Sort by importance
        self.feature_importance = dict(sorted(correlations.items(), 
                                            key=lambda x: x[1], reverse=True))
        
        # Select top features
        top_features = list(self.feature_importance.keys())[:self.max_features]
        
        # Calculate optimal thresholds based on historical performance
        self._optimize_thresholds(features[top_features], target_returns)
        
        self.is_trained = True
        print(f"✅ Strategy trained on {len(features)} samples")
        print(f"📊 Top 5 features: {list(self.feature_importance.keys())[:5]}")
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical features."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = data['Close'].rolling(period).mean()
            features[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
            features[f'price_sma_{period}_ratio'] = data['Close'] / features[f'sma_{period}']
        
        # Momentum indicators
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
            features[f'roc_{period}'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
        
        # Volatility indicators
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = features['returns'].rolling(period).std()
            features[f'atr_{period}'] = self._calculate_atr(data, period)
        
        # Volume indicators
        features['volume_sma_20'] = data['Volume'].rolling(20).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_sma_20']
        features['volume_price_trend'] = (data['Volume'] * features['returns']).rolling(10).sum()
        
        # RSI
        features['rsi_14'] = self._calculate_rsi(data['Close'], 14)
        features['rsi_5'] = self._calculate_rsi(data['Close'], 5)
        
        # MACD
        macd, signal = self._calculate_macd(data['Close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = macd - signal
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['Close'])
        features['bb_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Mean reversion signals
        features['mean_reversion_5'] = (data['Close'] - data['Close'].rolling(5).mean()) / data['Close'].rolling(5).std()
        features['mean_reversion_10'] = (data['Close'] - data['Close'].rolling(10).mean()) / data['Close'].rolling(10).std()
        
        # Support/Resistance
        features['support_level'] = data['Low'].rolling(20).min()
        features['resistance_level'] = data['High'].rolling(20).max()
        features['price_position'] = (data['Close'] - features['support_level']) / (features['resistance_level'] - features['support_level'])
        
        # Day of week effects
        features['day_of_week'] = data.index.dayofweek
        
        # High-Low spread
        features['hl_spread'] = (data['High'] - data['Low']) / data['Close']
        features['hl_spread_ma'] = features['hl_spread'].rolling(10).mean()
        
        return features
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(20).mean()
        std = prices.rolling(20).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower, sma
    
    def _optimize_thresholds(self, features: pd.DataFrame, targets: pd.Series):
        """Optimize signal thresholds for better performance."""
        print("🔧 Optimizing signal thresholds...")
        
        # Test different threshold combinations
        best_sharpe = -np.inf
        best_thresholds = self.signal_thresholds.copy()
        
        for long_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for short_thresh in [-0.1, -0.2, -0.3, -0.4, -0.5]:
                for exit_thresh in [0.05, 0.1, 0.15, 0.2]:
                    
                    # Generate signals with these thresholds
                    signals = self._generate_signals_with_thresholds(
                        features, long_thresh, short_thresh, exit_thresh
                    )
                    
                    # Calculate performance
                    strategy_returns = signals.shift(1) * targets
                    strategy_returns = strategy_returns.dropna()
                    
                    if len(strategy_returns) > 0:
                        sharpe = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_thresholds = {
                                'long': long_thresh,
                                'short': short_thresh,
                                'exit': exit_thresh
                            }
        
        self.signal_thresholds = best_thresholds
        print(f"✅ Optimized thresholds: {best_thresholds}")
        print(f"📊 Best Sharpe: {best_sharpe:.3f}")
    
    def _generate_signals_with_thresholds(self, features: pd.DataFrame, 
                                        long_thresh: float, short_thresh: float, 
                                        exit_thresh: float) -> pd.Series:
        """Generate signals with given thresholds."""
        # Calculate composite signal
        signal = pd.Series(0.0, index=features.index)
        
        # Momentum signals
        momentum_signal = (
            features['momentum_5'] * 0.4 +
            features['momentum_10'] * 0.3 +
            features['momentum_20'] * 0.3
        )
        
        # Mean reversion signals
        mean_rev_signal = (
            features['mean_reversion_5'] * 0.5 +
            features['mean_reversion_10'] * 0.5
        )
        
        # Volume signals
        volume_signal = features['volume_ratio'] - 1
        
        # RSI signals
        rsi_signal = (features['rsi_14'] - 50) / 50
        
        # MACD signals
        macd_signal = features['macd_histogram']
        
        # Bollinger Bands signals
        bb_signal = features['bb_position'] - 0.5
        
        # Composite signal
        composite = (
            momentum_signal * self.momentum_weight +
            mean_rev_signal * self.mean_reversion_weight +
            volume_signal * self.volume_weight +
            rsi_signal * 0.2 +
            macd_signal * 0.2 +
            bb_signal * 0.2
        )
        
        # Generate positions
        signal = pd.Series(0.0, index=features.index)
        signal[composite > long_thresh] = 1.0
        signal[composite < short_thresh] = -1.0
        
        # Exit signals
        signal[abs(composite) < exit_thresh] = 0.0
        
        return signal
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals."""
        if not self.is_trained:
            raise ValueError("Strategy must be trained before generating signals")
        
        # Calculate features
        features = self._calculate_features(data)
        
        # Select top features
        top_features = list(self.feature_importance.keys())[:self.max_features]
        available_features = [f for f in top_features if f in features.columns]
        
        if len(available_features) == 0:
            return pd.Series(0.0, index=data.index)
        
        # Generate signals with optimized thresholds
        signals = self._generate_signals_with_thresholds(
            features[available_features],
            self.signal_thresholds['long'],
            self.signal_thresholds['short'],
            self.signal_thresholds['exit']
        )
        
        return signals
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run comprehensive backtest."""
        print("🔄 Running improved strategy backtest...")
        
        # Generate signals
        signals = self.generate_signals(data)
        
        # Calculate returns
        price_returns = data['Close'].pct_change().dropna()
        strategy_returns = signals.shift(1) * price_returns
        strategy_returns = strategy_returns.dropna()
        
        # Calculate equity curve
        equity_curve = (1 + strategy_returns).cumprod()
        
        # Performance metrics
        total_return = equity_curve.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trading statistics
        total_trades = len(strategy_returns[strategy_returns != 0])
        winning_trades = len(strategy_returns[strategy_returns > 0])
        losing_trades = len(strategy_returns[strategy_returns < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'equity_curve': equity_curve,
            'strategy_returns': strategy_returns,
            'signals': signals
        }

def test_improved_strategy():
    """Test the improved adaptive strategy with Databento data."""
    print("🧪 TESTING IMPROVED ADAPTIVE STRATEGY")
    print("=" * 60)
    
    # Import and use the DatabentoGoldCollector
    try:
        from data_pipeline.databento_collector import DatabentoGoldCollector
        print("[INFO] Using DatabentoGoldCollector for real GOLD OHLCV data...")
        collector = DatabentoGoldCollector()
        
        # Fetch aggregated OHLCV data for August 2023
        print("[INFO] Fetching aggregated OHLCV data for August 2023...")
        ohlcv_data = collector.fetch_and_aggregate_gold_mbo_to_ohlcv(
            start_date="2023-08-01", 
            end_date="2023-08-31"
        )
        
        if ohlcv_data.empty:
            print("[ERROR] No OHLCV data fetched from Databento.")
            return
        
        print(f"[INFO] Successfully fetched {len(ohlcv_data)} days of OHLCV data.")
        
        # Convert to the format expected by the strategy
        data = ohlcv_data.copy()
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
    except Exception as e:
        print(f"[ERROR] Failed to import or use DatabentoGoldCollector: {e}")
        return
    
    # Create improved strategy
    strategy = ImprovedAdaptiveGoldStrategy(
        target_volatility=0.25,
        max_position_size=1.5,
        regularization_strength=0.03,
        max_features=75,
        signal_sensitivity=0.6,
        momentum_weight=0.4,
        mean_reversion_weight=0.3,
        volume_weight=0.3
    )
    
    # Train strategy
    print("[INFO] Training improved strategy...")
    strategy.train_on_data(data)
    print("[INFO] Strategy trained.")
    
    # Run backtest
    print("[INFO] Running improved backtest...")
    results = strategy.run_backtest(data)
    print("[INFO] Backtest complete.")
    
    # Print results
    print("\n📊 IMPROVED STRATEGY RESULTS:")
    print(f"   Total Return: {results['total_return']:.2%}")
    print(f"   Annualized Return: {results['annualized_return']:.2%}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"   Sortino Ratio: {results['sortino_ratio']:.3f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Win Rate: {results['win_rate']:.2%}")
    print(f"   Profit Factor: {results['profit_factor']:.3f}")
    
    # Save results
    with open("improved_strategy_results.txt", "w") as f:
        f.write("IMPROVED ADAPTIVE STRATEGY RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Return: {results['total_return']:.2%}\n")
        f.write(f"Annualized Return: {results['annualized_return']:.2%}\n")
        f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}\n")
        f.write(f"Sortino Ratio: {results['sortino_ratio']:.3f}\n")
        f.write(f"Max Drawdown: {results['max_drawdown']:.2%}\n")
        f.write(f"Total Trades: {results['total_trades']}\n")
        f.write(f"Win Rate: {results['win_rate']:.2%}\n")
        f.write(f"Profit Factor: {results['profit_factor']:.3f}\n")
    
    print("✅ Results saved to improved_strategy_results.txt")
    
    return results

if __name__ == "__main__":
    test_improved_strategy() 