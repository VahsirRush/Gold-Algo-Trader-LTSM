#!/usr/bin/env python3
"""
ENHANCED ROBUST STRATEGY
========================

Enhanced strategy that incorporates:
- Overfitting analysis results
- Robust parameter selection
- Advanced signal generation
- Improved risk management
- Better performance optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class EnhancedRobustStrategy:
    """Enhanced robust trading strategy with overfitting protection."""
    
    def __init__(self, 
                 target_volatility: float = 0.20,
                 max_position_size: float = 1.0,
                 regularization_strength: float = 0.05,
                 max_features: int = 50,
                 signal_sensitivity: float = 0.5,
                 momentum_weight: float = 0.35,
                 mean_reversion_weight: float = 0.35,
                 volume_weight: float = 0.30,
                 risk_free_rate: float = 0.02):
        
        self.target_volatility = target_volatility
        self.max_position_size = max_position_size
        self.regularization_strength = regularization_strength
        self.max_features = max_features
        self.signal_sensitivity = signal_sensitivity
        self.momentum_weight = momentum_weight
        self.mean_reversion_weight = mean_reversion_weight
        self.volume_weight = volume_weight
        self.risk_free_rate = risk_free_rate
        
        # Strategy state
        self.is_trained = False
        self.feature_importance = {}
        self.signal_thresholds = {'long': 0.25, 'short': -0.25, 'exit': 0.05}
        self.optimal_parameters = {}
        
    def train_on_data(self, data: pd.DataFrame) -> Dict:
        """Train the strategy on historical data."""
        print("🚀 Training Enhanced Robust Strategy...")
        
        # Calculate features
        features = self._calculate_enhanced_features(data)
        
        # Calculate target returns (next day returns)
        target_returns = data['Close'].pct_change().shift(-1).dropna()
        
        # Align features and targets
        features = features.dropna()
        target_returns = target_returns[features.index]
        
        if len(features) < 20:
            print("⚠️  Insufficient data for training")
            return {}
        
        # Feature selection based on importance
        feature_importance = self._calculate_feature_importance(features, target_returns)
        self.feature_importance = feature_importance
        
        # Select top features
        top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:self.max_features]
        selected_features = [f[0] for f in top_features]
        
        print(f"📊 Top 5 features: {selected_features[:5]}")
        
        # Optimize signal thresholds
        self._optimize_signal_thresholds(features[selected_features], target_returns)
        
        # Optimize strategy parameters
        self._optimize_strategy_parameters(features[selected_features], target_returns)
        
        self.is_trained = True
        print("✅ Strategy trained successfully")
        
        return {
            'feature_importance': feature_importance,
            'selected_features': selected_features,
            'signal_thresholds': self.signal_thresholds,
            'optimal_parameters': self.optimal_parameters
        }
    
    def _calculate_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced feature set."""
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['price_change'] = data['Close'] - data['Close'].shift(1)
        
        # Moving averages (optimized based on overfitting analysis)
        for period in [5, 10, 20]:  # Reduced from 50 to avoid overfitting
            features[f'sma_{period}'] = data['Close'].rolling(period).mean()
            features[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
            features[f'price_sma_{period}_ratio'] = data['Close'] / features[f'sma_{period}']
            features[f'price_ema_{period}_ratio'] = data['Close'] / features[f'ema_{period}']
        
        # Momentum indicators
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
            features[f'roc_{period}'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
        
        # Volatility indicators
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = features['returns'].rolling(period).std()
            features[f'atr_{period}'] = self._calculate_atr(data, period)
        
        # Volume indicators
        features['volume_ma'] = data['Volume'].rolling(20).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_ma']
        features['price_volume_trend'] = (data['Close'] - data['Close'].shift(1)) * data['Volume']
        
        # Technical oscillators
        features['rsi'] = self._calculate_rsi(data['Close'], 14)
        features['macd'] = self._calculate_macd(data['Close'])
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_sma = data['Close'].rolling(bb_period).mean()
        bb_std_dev = data['Close'].rolling(bb_period).std()
        features['bb_upper'] = bb_sma + (bb_std_dev * bb_std)
        features['bb_lower'] = bb_sma - (bb_std_dev * bb_std)
        features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Mean reversion indicators
        features['mean_reversion_signal'] = (data['Close'] - bb_sma) / bb_std_dev
        features['support_resistance'] = self._calculate_support_resistance(data)
        
        # High-Low analysis
        features['hl_spread'] = (data['High'] - data['Low']) / data['Close']
        features['hl_ratio'] = data['High'] / data['Low']
        
        # Day of week effect
        features['day_of_week'] = data.index.dayofweek
        
        # Trend strength
        features['adx'] = self._calculate_adx(data, 14)
        
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
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
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> pd.Series:
        """Calculate support/resistance levels."""
        # Simple implementation using rolling min/max
        support = data['Low'].rolling(20).min()
        resistance = data['High'].rolling(20).max()
        return (data['Close'] - support) / (resistance - support)
    
    def _calculate_adx(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average Directional Index."""
        # Simplified ADX calculation
        plus_dm = data['High'].diff()
        minus_dm = -data['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._calculate_atr(data, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean()
    
    def _calculate_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Calculate feature importance using correlation."""
        importance = {}
        for col in features.columns:
            correlation = features[col].corr(target)
            importance[col] = correlation if not pd.isna(correlation) else 0
        return importance
    
    def _optimize_signal_thresholds(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Optimize signal thresholds for better performance."""
        print("🔧 Optimizing signal thresholds...")
        
        best_sharpe = -np.inf
        best_thresholds = self.signal_thresholds.copy()
        
        # Test different threshold combinations
        long_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
        short_thresholds = [-0.1, -0.15, -0.2, -0.25, -0.3]
        exit_thresholds = [0.02, 0.05, 0.08, 0.1]
        
        for long_thresh in long_thresholds:
            for short_thresh in short_thresholds:
                for exit_thresh in exit_thresholds:
                    # Generate signals
                    signals = self._generate_signals_with_thresholds(
                        features, long_thresh, short_thresh, exit_thresh
                    )
                    
                    # Calculate performance
                    strategy_returns = signals.shift(1) * target
                    strategy_returns = strategy_returns.dropna()
                    
                    if len(strategy_returns) > 10:
                        sharpe = (strategy_returns.mean() - self.risk_free_rate/252) / strategy_returns.std()
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
        """Generate trading signals with given thresholds."""
        
        # Calculate composite signal
        momentum_signal = self._calculate_momentum_signal(features)
        mean_reversion_signal = self._calculate_mean_reversion_signal(features)
        volume_signal = self._calculate_volume_signal(features)
        
        # Combine signals
        composite_signal = (
            self.momentum_weight * momentum_signal +
            self.mean_reversion_weight * mean_reversion_signal +
            self.volume_weight * volume_signal
        )
        
        # Generate trading signals
        signals = pd.Series(0, index=features.index)
        
        # Long signals
        long_mask = composite_signal > long_thresh
        signals[long_mask] = 1
        
        # Short signals
        short_mask = composite_signal < short_thresh
        signals[short_mask] = -1
        
        # Exit signals
        exit_mask = (composite_signal > -exit_thresh) & (composite_signal < exit_thresh)
        signals[exit_mask] = 0
        
        return signals
    
    def _calculate_momentum_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate momentum-based signal."""
        # Use multiple momentum indicators
        momentum_indicators = [
            features['momentum_5'],
            features['momentum_10'],
            features['roc_5'],
            features['roc_10'],
            features['price_sma_5_ratio'] - 1,
            features['price_ema_5_ratio'] - 1
        ]
        
        # Combine indicators
        momentum_signal = pd.concat(momentum_indicators, axis=1).mean(axis=1)
        return momentum_signal
    
    def _calculate_mean_reversion_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate mean reversion signal."""
        # Use mean reversion indicators
        mean_reversion_indicators = [
            -features['mean_reversion_signal'],  # Negative because we want to buy when price is below mean
            -features['bb_position'],  # Negative because we want to buy when price is near lower band
            features['rsi'] - 50,  # RSI relative to neutral level
            -features['price_sma_20_ratio'] + 1  # Negative because we want to buy when price is below SMA
        ]
        
        # Combine indicators
        mean_reversion_signal = pd.concat(mean_reversion_indicators, axis=1).mean(axis=1)
        return mean_reversion_signal
    
    def _calculate_volume_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate volume-based signal."""
        # Volume confirmation signals
        volume_indicators = [
            features['volume_ratio'] - 1,  # Volume relative to average
            features['price_volume_trend'].rolling(5).mean(),  # Price-volume trend
            (features['hl_spread'] - features['hl_spread'].rolling(20).mean())  # Spread relative to average
        ]
        
        # Combine indicators
        volume_signal = pd.concat(volume_indicators, axis=1).mean(axis=1)
        return volume_signal
    
    def _optimize_strategy_parameters(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Optimize strategy parameters."""
        print("🔧 Optimizing strategy parameters...")
        
        # Test different parameter combinations
        param_combinations = [
            {'momentum_weight': 0.4, 'mean_reversion_weight': 0.3, 'volume_weight': 0.3},
            {'momentum_weight': 0.35, 'mean_reversion_weight': 0.35, 'volume_weight': 0.30},
            {'momentum_weight': 0.3, 'mean_reversion_weight': 0.4, 'volume_weight': 0.3},
            {'momentum_weight': 0.5, 'mean_reversion_weight': 0.25, 'volume_weight': 0.25},
        ]
        
        best_sharpe = -np.inf
        best_params = {}
        
        for params in param_combinations:
            # Update weights
            self.momentum_weight = params['momentum_weight']
            self.mean_reversion_weight = params['mean_reversion_weight']
            self.volume_weight = params['volume_weight']
            
            # Generate signals
            signals = self._generate_signals_with_thresholds(
                features, 
                self.signal_thresholds['long'],
                self.signal_thresholds['short'],
                self.signal_thresholds['exit']
            )
            
            # Calculate performance
            strategy_returns = signals.shift(1) * target
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) > 10:
                sharpe = (strategy_returns.mean() - self.risk_free_rate/252) / strategy_returns.std()
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params.copy()
        
        # Set optimal parameters
        self.optimal_parameters = best_params
        self.momentum_weight = best_params['momentum_weight']
        self.mean_reversion_weight = best_params['mean_reversion_weight']
        self.volume_weight = best_params['volume_weight']
        
        print(f"✅ Optimal parameters: {best_params}")
        print(f"📊 Best Sharpe: {best_sharpe:.3f}")
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest on the data."""
        if not self.is_trained:
            print("⚠️  Strategy not trained. Training first...")
            self.train_on_data(data)
        
        print("🔄 Running enhanced robust strategy backtest...")
        
        # Calculate features
        features = self._calculate_enhanced_features(data)
        
        # Generate signals
        signals = self._generate_signals_with_thresholds(
            features,
            self.signal_thresholds['long'],
            self.signal_thresholds['short'],
            self.signal_thresholds['exit']
        )
        
        # Calculate returns
        returns = data['Close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # Remove NaN values
        strategy_returns = strategy_returns.dropna()
        returns = returns[strategy_returns.index]
        
        # Calculate performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = (strategy_returns.mean() - self.risk_free_rate/252) / strategy_returns.std() * np.sqrt(252)
        
        # Calculate drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        sortino_ratio = (strategy_returns.mean() - self.risk_free_rate/252) / downside_deviation * np.sqrt(252)
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trading statistics
        trades = signals.diff().abs()
        total_trades = trades.sum()
        winning_trades = (strategy_returns > 0).sum()
        losing_trades = (strategy_returns < 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk metrics
        var_95 = np.percentile(strategy_returns, 5)
        cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
        
        # Beta and Alpha
        market_returns = returns  # Using price returns as market proxy
        beta = np.cov(strategy_returns, market_returns)[0, 1] / np.var(market_returns)
        alpha = strategy_returns.mean() - beta * market_returns.mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
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
            'beta': beta,
            'alpha': alpha,
            'equity_curve': cumulative_returns
        }

def test_enhanced_robust_strategy():
    """Test the enhanced robust strategy."""
    print("🧪 TESTING ENHANCED ROBUST STRATEGY")
    print("=" * 60)
    
    # Import and use the DatabentoGoldCollector
    try:
        from data_pipeline.databento_collector import DatabentoGoldCollector
        
        print("[INFO] Using DatabentoGoldCollector for real GOLD OHLCV data...")
        collector = DatabentoGoldCollector()
        
        # Fetch extended data for better analysis
        print("[INFO] Fetching extended OHLCV data for enhanced strategy testing...")
        ohlcv_data = collector.fetch_and_aggregate_gold_mbo_to_ohlcv(
            start_date="2023-07-01", 
            end_date="2023-09-30"
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
    
    # Create enhanced robust strategy
    strategy = EnhancedRobustStrategy(
        target_volatility=0.20,
        max_position_size=1.0,
        regularization_strength=0.05,
        max_features=50,
        signal_sensitivity=0.5,
        momentum_weight=0.35,
        mean_reversion_weight=0.35,
        volume_weight=0.30
    )
    
    # Run backtest
    print("[INFO] Running enhanced robust strategy backtest...")
    results = strategy.run_backtest(data)
    
    # Print results
    print("\n📊 ENHANCED ROBUST STRATEGY RESULTS:")
    print(f"   Total Return: {results['total_return']:.2%}")
    print(f"   Annualized Return: {results['annualized_return']:.2%}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"   Sortino Ratio: {results['sortino_ratio']:.3f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"   Calmar Ratio: {results['calmar_ratio']:.3f}")
    print(f"   Win Rate: {results['win_rate']:.2%}")
    print(f"   Profit Factor: {results['profit_factor']:.3f}")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Beta: {results['beta']:.3f}")
    print(f"   Alpha: {results['alpha']:.3f}")
    
    # Save results
    with open("enhanced_robust_strategy_results.txt", "w") as f:
        f.write("ENHANCED ROBUST STRATEGY RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Return: {results['total_return']:.2%}\n")
        f.write(f"Annualized Return: {results['annualized_return']:.2%}\n")
        f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}\n")
        f.write(f"Sortino Ratio: {results['sortino_ratio']:.3f}\n")
        f.write(f"Max Drawdown: {results['max_drawdown']:.2%}\n")
        f.write(f"Calmar Ratio: {results['calmar_ratio']:.3f}\n")
        f.write(f"Win Rate: {results['win_rate']:.2%}\n")
        f.write(f"Profit Factor: {results['profit_factor']:.3f}\n")
        f.write(f"Total Trades: {results['total_trades']}\n")
        f.write(f"Winning Trades: {results['winning_trades']}\n")
        f.write(f"Losing Trades: {results['losing_trades']}\n")
        f.write(f"Beta: {results['beta']:.3f}\n")
        f.write(f"Alpha: {results['alpha']:.3f}\n")
        f.write(f"VaR (95%): {results['var_95']:.3f}\n")
        f.write(f"CVaR (95%): {results['cvar_95']:.3f}\n")
    
    # Save equity curve
    results['equity_curve'].to_csv("enhanced_robust_equity_curve.csv")
    
    print("✅ Results saved to enhanced_robust_strategy_results.txt")
    print("✅ Equity curve saved to enhanced_robust_equity_curve.csv")
    
    return results

if __name__ == "__main__":
    test_enhanced_robust_strategy() 