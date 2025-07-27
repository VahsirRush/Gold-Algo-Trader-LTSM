#!/usr/bin/env python3
"""
AGGRESSIVE TRADING STRATEGY
===========================

Aggressive strategy designed to generate more trades by:
- Using very sensitive signal thresholds
- Implementing multiple signal sources
- Avoiding complex index alignment issues
- Using simple but effective indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from shared_utilities import BaseStrategy, FeatureEngineer, SignalGenerator, PerformanceMetrics

class AggressiveTradingStrategy(BaseStrategy):
    """Aggressive trading strategy for maximum trade generation."""
    
    def __init__(self, 
                 long_threshold: float = 0.05,  # Very sensitive
                 short_threshold: float = -0.05,  # Very sensitive
                 exit_threshold: float = 0.01,  # Quick exits
                 momentum_weight: float = 0.4,
                 mean_reversion_weight: float = 0.3,
                 volume_weight: float = 0.3,
                 risk_free_rate: float = 0.02):
        
        super().__init__(long_threshold, short_threshold, exit_threshold, risk_free_rate)
        self.momentum_weight = momentum_weight
        self.mean_reversion_weight = mean_reversion_weight
        self.volume_weight = volume_weight
        
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest on the data."""
        print("🔄 Running aggressive trading strategy backtest...")
        
        # Use parent class backtest logic
        results = super().run_backtest(data)
        
        # Print results
        PerformanceMetrics.print_metrics(results)
        
        return results
        
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for aggressive strategy."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['price_change'] = data['Close'] - data['Close'].shift(1)
        
        # Moving averages
        for period in [3, 5, 8, 13, 21]:
            sma = data['Close'].rolling(period).mean()
            ema = data['Close'].ewm(span=period).mean()
            features[f'sma_{period}'] = sma
            features[f'ema_{period}'] = ema
            features[f'price_sma_{period}_ratio'] = data['Close'] / sma
            features[f'price_ema_{period}_ratio'] = data['Close'] / ema
        
        # Momentum indicators
        for period in [1, 2, 3, 5, 8]:
            features[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
            features[f'roc_{period}'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
        
        # Volatility indicators
        for period in [3, 5, 8, 13, 21]:
            features[f'volatility_{period}'] = features['returns'].rolling(period).std()
        
        # Volume indicators
        if 'Volume' in data.columns:
            volume_ma_3 = data['Volume'].rolling(3).mean()
            volume_ma_5 = data['Volume'].rolling(5).mean()
            features['volume_ma_3'] = volume_ma_3
            features['volume_ma_5'] = volume_ma_5
            features['volume_ratio_3'] = data['Volume'] / volume_ma_3
            features['volume_ratio_5'] = data['Volume'] / volume_ma_5
        
        # Technical oscillators
        from shared_utilities import TechnicalIndicators
        features['rsi_7'] = TechnicalIndicators.calculate_rsi(data['Close'], 7)
        features['rsi_14'] = TechnicalIndicators.calculate_rsi(data['Close'], 14)
        features['macd'] = TechnicalIndicators.calculate_macd(data['Close'])
        
        # Bollinger Bands
        for period in [5, 10, 20]:
            bb_sma = data['Close'].rolling(period).mean()
            bb_std = data['Close'].rolling(period).std()
            bb_upper = bb_sma + (bb_std * 2)
            bb_lower = bb_sma - (bb_std * 2)
            features[f'bb_upper_{period}'] = bb_upper
            features[f'bb_lower_{period}'] = bb_lower
            features[f'bb_position_{period}'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        
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
            support_level = data['Low'].rolling(5).min()
            resistance_level = data['High'].rolling(5).max()
            features['support_level'] = support_level
            features['resistance_level'] = resistance_level
            features['support_distance'] = (data['Close'] - support_level) / data['Close']
            features['resistance_distance'] = (resistance_level - data['Close']) / data['Close']
        
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate aggressive trading signals."""
        # Calculate different signal types using shared utilities
        momentum_signal = SignalGenerator.calculate_momentum_signal(features)
        mean_reversion_signal = SignalGenerator.calculate_mean_reversion_signal(features)
        volume_signal = SignalGenerator.calculate_volume_signal(features)
        technical_signal = SignalGenerator.calculate_technical_signal(features)
        
        # Combine signals with weighted approach for aggressive strategy
        combined_signal = (
            momentum_signal * self.momentum_weight +
            mean_reversion_signal * self.mean_reversion_weight +
            volume_signal * self.volume_weight +
            technical_signal * (1 - self.momentum_weight - self.mean_reversion_weight - self.volume_weight)
        )
        
        # Generate final signals based on thresholds
        signals = pd.Series(0, index=features.index)
        
        # Long signals
        long_condition = combined_signal > self.long_threshold
        signals[long_condition] = 1
        
        # Short signals
        short_condition = combined_signal < self.short_threshold
        signals[short_condition] = -1
        
        # Exit signals
        exit_condition = (combined_signal.abs() < self.exit_threshold) & (signals != 0)
        signals[exit_condition] = 0
        
        return signals

def test_aggressive_strategy():
    """Test the aggressive strategy."""
    print("🧪 Testing Aggressive Strategy")
    
    # Fetch test data
    from shared_utilities import DataFetcher
    data = DataFetcher.fetch_data('GLD', '3mo')  # Original period
    
    if data.empty:
        print("❌ No data available for testing")
        return
    
    # Create and test strategy
    strategy = AggressiveTradingStrategy()
    results = strategy.run_backtest(data)
    
    # Save results
    if 'equity_curve' in results:
        results['equity_curve'].to_csv('aggressive_equity_curve.csv')
    
    if 'signals' in results:
        results['signals'].to_csv('aggressive_signals.csv')
    
    # Save summary
    with open('aggressive_strategy_results.txt', 'w') as f:
        f.write("AGGRESSIVE STRATEGY RESULTS\n")
        f.write("==========================\n\n")
        f.write(f"Total Return: {results.get('total_return', 0):.2%}\n")
        f.write(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}\n")
        f.write(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}\n")
        f.write(f"Total Trades: {results.get('total_trades', 0)}\n")
        f.write(f"Win Rate: {results.get('win_rate', 0):.2%}\n")
    
    print("✅ Aggressive strategy test completed!")

if __name__ == "__main__":
    test_aggressive_strategy() 
    def _calculate_momentum_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate momentum-based signal."""
        momentum_indicators = [
            features['momentum_1'],
            features['momentum_2'],
            features['momentum_3'],
            features['momentum_5'],
            features['roc_1'],
            features['roc_2'],
            features['roc_3'],
            features['price_sma_3_ratio'] - 1,
            features['price_ema_3_ratio'] - 1,
            features['price_acceleration']
        ]
        
        # Combine indicators
        momentum_signal = pd.concat(momentum_indicators, axis=1).mean(axis=1)
        return momentum_signal
    
    def _calculate_mean_reversion_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate mean reversion signal."""
        mean_reversion_indicators = [
            -features['mean_reversion_5'],
            -features['mean_reversion_10'],
            -features['bb_position_5'],
            -features['bb_position_10'],
            features['rsi_7'] - 50,
            features['rsi_14'] - 50,
            -features['price_sma_5_ratio'] + 1,
            -features['price_sma_10_ratio'] + 1,
            features['support_distance'],
            -features['resistance_distance']
        ]
        
        # Combine indicators
        mean_reversion_signal = pd.concat(mean_reversion_indicators, axis=1).mean(axis=1)
        return mean_reversion_signal
    
    def _calculate_volume_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate volume-based signal."""
        volume_indicators = [
            features['volume_ratio_3'] - 1,
            features['volume_ratio_5'] - 1,
            features['volume_acceleration'],
            (features['hl_spread'] - features['hl_spread'].rolling(5).mean()),
            features['hl_position'] - 0.5
        ]
        
        # Combine indicators
        volume_signal = pd.concat(volume_indicators, axis=1).mean(axis=1)
        return volume_signal

def test_aggressive_strategy():
    """Test the aggressive trading strategy."""
    print("🧪 TESTING AGGRESSIVE TRADING STRATEGY")
    print("=" * 60)
    
    # Import and use the DatabentoGoldCollector
    try:
        from data_pipeline.databento_collector import DatabentoGoldCollector
        
        print("[INFO] Using DatabentoGoldCollector for real GOLD OHLCV data...")
        collector = DatabentoGoldCollector()
        
        # Fetch extended data for better analysis
        print("[INFO] Fetching extended OHLCV data for aggressive strategy testing...")
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
    
    # Test multiple threshold combinations for maximum trades
    threshold_combinations = [
        {'long': 0.03, 'short': -0.03, 'exit': 0.005},  # Very aggressive
        {'long': 0.05, 'short': -0.05, 'exit': 0.01},   # Aggressive
        {'long': 0.08, 'short': -0.08, 'exit': 0.02},   # Moderate
        {'long': 0.1, 'short': -0.1, 'exit': 0.03},     # Conservative
    ]
    
    best_results = None
    best_trade_count = 0
    best_sharpe = -np.inf
    
    for i, thresholds in enumerate(threshold_combinations):
        print(f"\n🔧 Testing threshold combination {i+1}: {thresholds}")
        
        # Create strategy with current thresholds
        strategy = AggressiveTradingStrategy(
            long_threshold=thresholds['long'],
            short_threshold=thresholds['short'],
            exit_threshold=thresholds['exit'],
            momentum_weight=0.4,
            mean_reversion_weight=0.3,
            volume_weight=0.3
        )
        
        # Run backtest
        results = strategy.run_backtest(data)
        
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Total Return: {results['total_return']:.2%}")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"   Win Rate: {results['win_rate']:.2%}")
        
        # Prefer strategies with more trades and good Sharpe
        if (results['total_trades'] >= 5 and 
            results['sharpe_ratio'] > best_sharpe and 
            results['total_trades'] > best_trade_count):
            best_results = results
            best_trade_count = results['total_trades']
            best_sharpe = results['sharpe_ratio']
    
    if best_results is None:
        print("⚠️  No strategy met the minimum criteria. Using the most aggressive one.")
        strategy = AggressiveTradingStrategy(
            long_threshold=0.02,  # Extremely aggressive
            short_threshold=-0.02,
            exit_threshold=0.005
        )
        best_results = strategy.run_backtest(data)
    
    # Print final results
    print("\n📊 AGGRESSIVE TRADING STRATEGY FINAL RESULTS:")
    print(f"   Total Return: {best_results['total_return']:.2%}")
    print(f"   Annualized Return: {best_results['annualized_return']:.2%}")
    print(f"   Sharpe Ratio: {best_results['sharpe_ratio']:.3f}")
    print(f"   Sortino Ratio: {best_results['sortino_ratio']:.3f}")
    print(f"   Max Drawdown: {best_results['max_drawdown']:.2%}")
    print(f"   Calmar Ratio: {best_results['calmar_ratio']:.3f}")
    print(f"   Win Rate: {best_results['win_rate']:.2%}")
    print(f"   Profit Factor: {best_results['profit_factor']:.3f}")
    print(f"   Total Trades: {best_results['total_trades']}")
    print(f"   Winning Trades: {best_results['winning_trades']}")
    print(f"   Losing Trades: {best_results['losing_trades']}")
    print(f"   Beta: {best_results['beta']:.3f}")
    print(f"   Alpha: {best_results['alpha']:.3f}")
    print(f"   Thresholds: {best_results['thresholds']}")
    
    # Save results
    with open("aggressive_strategy_results.txt", "w") as f:
        f.write("AGGRESSIVE TRADING STRATEGY RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Return: {best_results['total_return']:.2%}\n")
        f.write(f"Annualized Return: {best_results['annualized_return']:.2%}\n")
        f.write(f"Sharpe Ratio: {best_results['sharpe_ratio']:.3f}\n")
        f.write(f"Sortino Ratio: {best_results['sortino_ratio']:.3f}\n")
        f.write(f"Max Drawdown: {best_results['max_drawdown']:.2%}\n")
        f.write(f"Calmar Ratio: {best_results['calmar_ratio']:.3f}\n")
        f.write(f"Win Rate: {best_results['win_rate']:.2%}\n")
        f.write(f"Profit Factor: {best_results['profit_factor']:.3f}\n")
        f.write(f"Total Trades: {best_results['total_trades']}\n")
        f.write(f"Winning Trades: {best_results['winning_trades']}\n")
        f.write(f"Losing Trades: {best_results['losing_trades']}\n")
        f.write(f"Beta: {best_results['beta']:.3f}\n")
        f.write(f"Alpha: {best_results['alpha']:.3f}\n")
        f.write(f"Thresholds: {best_results['thresholds']}\n")
        f.write(f"VaR (95%): {best_results['var_95']:.3f}\n")
        f.write(f"CVaR (95%): {best_results['cvar_95']:.3f}\n")
    
    # Save equity curve
    best_results['equity_curve'].to_csv("aggressive_equity_curve.csv")
    
    # Save signals for analysis
    signals_df = pd.DataFrame({
        'Date': best_results['signals'].index,
        'Signal': best_results['signals'].values
    })
    signals_df.to_csv("aggressive_signals.csv", index=False)
    
    print("✅ Results saved to aggressive_strategy_results.txt")
    print("✅ Equity curve saved to aggressive_equity_curve.csv")
    print("✅ Signals saved to aggressive_signals.csv")
    
    return best_results

if __name__ == "__main__":
    test_aggressive_strategy() 