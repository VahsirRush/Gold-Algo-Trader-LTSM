#!/usr/bin/env python3
"""
ULTRA AGGRESSIVE TRADING STRATEGY
=================================

Ultra-aggressive strategy designed to generate maximum trades by:
- Using extremely sensitive signal thresholds
- Implementing simple but effective indicators
- Avoiding complex feature dependencies
- Using multiple signal sources
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from gold_algo.shared_utilities import BaseStrategy, FeatureEngineer, SignalGenerator, PerformanceMetrics

class UltraAggressiveStrategy(BaseStrategy):
    """Ultra-aggressive trading strategy for maximum trade generation."""
    
    def __init__(self, 
                 long_threshold: float = 0.01,  # Extremely sensitive - original working value
                 short_threshold: float = -0.01,  # Extremely sensitive - original working value
                 exit_threshold: float = 0.002,  # Very quick exits - original working value
                 risk_free_rate: float = 0.02):
        
        super().__init__(long_threshold, short_threshold, exit_threshold, risk_free_rate)
        
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest on the data."""
        print("🔄 Running ultra-aggressive trading strategy backtest...")
        
        # Use parent class backtest logic
        results = super().run_backtest(data)
        
        # Print results
        PerformanceMetrics.print_metrics(results)
        
        return results
        
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple trading features for ultra-aggressive strategy."""
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        close_col = 'Close' if 'Close' in data.columns else 'close'
        features['returns'] = data[close_col].pct_change()
        features['price_change'] = data[close_col] - data[close_col].shift(1)
        
        # Simple moving averages
        for period in [3, 5, 8, 13]:
            sma = data[close_col].rolling(period).mean()
            ema = data[close_col].ewm(span=period).mean()
            features[f'sma_{period}'] = sma
            features[f'ema_{period}'] = ema
            features[f'price_sma_{period}_ratio'] = data[close_col] / sma
            features[f'price_ema_{period}_ratio'] = data[close_col] / ema
        
        # Momentum indicators
        for period in [1, 2, 3, 5]:
            features[f'momentum_{period}'] = data[close_col] / data[close_col].shift(period) - 1
            features[f'roc_{period}'] = (data[close_col] - data[close_col].shift(period)) / data[close_col].shift(period)
        
        # Volatility indicators
        for period in [3, 5, 8]:
            features[f'volatility_{period}'] = features['returns'].rolling(period).std()
        
        # Volume indicators
        volume_col = 'Volume' if 'Volume' in data.columns else 'volume'
        if volume_col in data.columns:
            volume_ma_3 = data[volume_col].rolling(3).mean()
            volume_ma_5 = data[volume_col].rolling(5).mean()
            features['volume_ma_3'] = volume_ma_3
            features['volume_ma_5'] = volume_ma_5
            features['volume_ratio_3'] = data[volume_col] / volume_ma_3
            features['volume_ratio_5'] = data[volume_col] / volume_ma_5
        
        # Technical oscillators
        from gold_algo.shared_utilities import TechnicalIndicators
        features['rsi_7'] = TechnicalIndicators.calculate_rsi(data[close_col], 7)
        features['rsi_14'] = TechnicalIndicators.calculate_rsi(data[close_col], 14)
        features['macd'] = TechnicalIndicators.calculate_macd(data[close_col])
        
        # Bollinger Bands
        for period in [5, 10]:
            bb_sma = data[close_col].rolling(period).mean()
            bb_std = data[close_col].rolling(period).std()
            bb_upper = bb_sma + (bb_std * 2)
            bb_lower = bb_sma - (bb_std * 2)
            features[f'bb_upper_{period}'] = bb_upper
            features[f'bb_lower_{period}'] = bb_lower
            features[f'bb_position_{period}'] = (data[close_col] - bb_lower) / (bb_upper - bb_lower)
        
        # Mean reversion indicators
        for period in [5, 10]:
            bb_sma = data[close_col].rolling(period).mean()
            bb_std = data[close_col].rolling(period).std()
            features[f'mean_reversion_{period}'] = (data[close_col] - bb_sma) / bb_std
        
        # High-Low analysis
        high_col = 'High' if 'High' in data.columns else 'high'
        low_col = 'Low' if 'Low' in data.columns else 'low'
        if all(col in data.columns for col in [high_col, low_col]):
            features['hl_spread'] = (data[high_col] - data[low_col]) / data[close_col]
            features['hl_position'] = (data[close_col] - data[low_col]) / (data[high_col] - data[low_col])
        
        # Additional signals
        features['price_acceleration'] = features['returns'].diff()
        if volume_col in data.columns:
            features['volume_acceleration'] = data[volume_col].pct_change()
        
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _generate_signals(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Generate ultra-aggressive trading signals."""
        # Calculate multiple signal components using original logic
        momentum_signal = self._calculate_momentum_signal(features)
        mean_reversion_signal = self._calculate_mean_reversion_signal(features)
        volume_signal = self._calculate_volume_signal(features)
        technical_signal = self._calculate_technical_signal(features)
        
        # Combine all signals with equal weights
        composite_signal = (
            0.25 * momentum_signal +
            0.25 * mean_reversion_signal +
            0.25 * volume_signal +
            0.25 * technical_signal
        )
        
        # Generate trading signals
        signals = pd.Series(0, index=features.index)
        
        # Long signals
        long_mask = composite_signal > self.long_threshold
        signals[long_mask] = 1
        
        # Short signals
        short_mask = composite_signal < self.short_threshold
        signals[short_mask] = -1
        
        # Exit signals
        exit_mask = (composite_signal > -self.exit_threshold) & (composite_signal < self.exit_threshold)
        signals[exit_mask] = 0
        
        return signals
    
    def _calculate_momentum_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate momentum-based signal."""
        momentum_indicators = []
        
        # Add available momentum indicators
        for col in ['momentum_1', 'momentum_2', 'momentum_3', 'momentum_5']:
            if col in features.columns:
                momentum_indicators.append(features[col])
        
        for col in ['roc_1', 'roc_2', 'roc_3', 'roc_5']:
            if col in features.columns:
                momentum_indicators.append(features[col])
        
        for col in ['price_sma_3_ratio', 'price_sma_5_ratio', 'price_sma_8_ratio']:
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
    
    def _calculate_mean_reversion_signal(self, features: pd.DataFrame) -> pd.Series:
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
    
    def _calculate_volume_signal(self, features: pd.DataFrame) -> pd.Series:
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
    
    def _calculate_technical_signal(self, features: pd.DataFrame) -> pd.Series:
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

def test_ultra_aggressive_strategy():
    """Test the ultra-aggressive strategy."""
    print("🧪 Testing Ultra-Aggressive Strategy")
    
    # Fetch test data
    from shared_utilities import DataFetcher
    data = DataFetcher.fetch_data('GLD', '3mo')  # Original period that achieved 228% returns
    
    if data.empty:
        print("❌ No data available for testing")
        return
    
    # Create and test strategy
    strategy = UltraAggressiveStrategy()
    results = strategy.run_backtest(data)
    
    # Save results
    if 'equity_curve' in results:
        results['equity_curve'].to_csv('ultra_aggressive_equity_curve.csv')
    
    if 'signals' in results:
        results['signals'].to_csv('ultra_aggressive_signals.csv')
    
    # Save summary
    with open('ultra_aggressive_strategy_results.txt', 'w') as f:
        f.write("ULTRA AGGRESSIVE STRATEGY RESULTS\n")
        f.write("==================================\n\n")
        f.write(f"Total Return: {results.get('total_return', 0):.2%}\n")
        f.write(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}\n")
        f.write(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}\n")
        f.write(f"Total Trades: {results.get('total_trades', 0)}\n")
        f.write(f"Win Rate: {results.get('win_rate', 0):.2%}\n")
    
    print("✅ Ultra-aggressive strategy test completed!")

if __name__ == "__main__":
    test_ultra_aggressive_strategy() 