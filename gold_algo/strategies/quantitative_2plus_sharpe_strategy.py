#!/usr/bin/env python3
"""
Quantitative 2+ Sharpe Strategy
Uses proven statistical techniques and sophisticated risk management
"""

import pandas as pd
import numpy as np
from shared_utilities import BaseStrategy, PerformanceMetrics

class Quantitative2PlusSharpeStrategy(BaseStrategy):
    """
    Quantitative strategy for 2+ Sharpe ratio with statistical rigor
    """
    
    def __init__(self, 
                 long_threshold: float = 0.015,  # Statistical threshold
                 short_threshold: float = -0.015,  # Statistical threshold
                 exit_threshold: float = 0.008,  # Tight exit for risk management
                 risk_free_rate: float = 0.02):
        
        super().__init__(long_threshold, short_threshold, exit_threshold, risk_free_rate)
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate quantitative features with statistical rigor."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = data['Close'].rolling(period).mean()
            features[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
            features[f'price_sma_{period}_ratio'] = data['Close'] / features[f'sma_{period}']
            features[f'price_ema_{period}_ratio'] = data['Close'] / features[f'ema_{period}']
        
        # Momentum indicators
        for period in [1, 3, 5, 10, 20]:
            features[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
            features[f'roc_{period}'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
        
        # Price acceleration
        features['price_acceleration'] = features['returns'].diff()
        
        # RSI
        for period in [7, 14, 21]:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Volume indicators
        for period in [5, 10, 20]:
            features[f'volume_ma_{period}'] = data['Volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = data['Volume'] / features[f'volume_ma_{period}']
        
        # Volatility
        for period in [10, 20, 50]:
            features[f'volatility_{period}'] = data['Close'].rolling(period).std()
        
        # Statistical indicators
        features['z_score_20'] = (data['Close'] - features['sma_20']) / features['volatility_20']
        features['z_score_50'] = (data['Close'] - features['sma_50']) / features['volatility_50']
        
        # Trend indicators
        features['trend_strength'] = abs(features['price_sma_20_ratio'] - 1)
        features['trend_direction'] = np.where(features['price_sma_20_ratio'] > 1, 1, -1)
        
        # Momentum strength
        features['momentum_strength'] = abs(features['momentum_10'])
        features['momentum_consistency'] = features['momentum_5'].rolling(5).sum() / 5
        
        # Statistical filters
        features['volatility_regime'] = features['volatility_20'] / features['volatility_20'].rolling(50).mean()
        features['volume_regime'] = features['volume_ratio_20'] / features['volume_ratio_20'].rolling(20).mean()
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate quantitative signals with statistical rigor."""
        # Calculate signal components
        statistical_signal = self._calculate_statistical_signal(features)
        momentum_signal = self._calculate_momentum_signal(features)
        trend_signal = self._calculate_trend_signal(features)
        regime_signal = self._calculate_regime_signal(features)
        
        # Combine signals with statistical weights
        composite_signal = (
            0.4 * statistical_signal +   # Statistical signal (most reliable)
            0.3 * momentum_signal +      # Momentum signal
            0.2 * trend_signal +         # Trend signal
            0.1 * regime_signal          # Regime signal
        )
        
        # Apply statistical filters
        quality_filter = self._calculate_statistical_filter(features)
        composite_signal = composite_signal * quality_filter
        
        # Generate trading signals with statistical criteria
        signals = pd.Series(0, index=features.index)
        
        # Long signals with statistical confirmation
        long_mask = (
            (composite_signal > self.long_threshold) & 
            (quality_filter > 0.7) &  # High quality requirement
            (features['z_score_20'] > 0.5) &  # Statistical confirmation
            (features['trend_direction'] > 0)  # Trend confirmation
        )
        signals[long_mask] = 1
        
        # Short signals with statistical confirmation
        short_mask = (
            (composite_signal < self.short_threshold) & 
            (quality_filter > 0.7) &  # High quality requirement
            (features['z_score_20'] < -0.5) &  # Statistical confirmation
            (features['trend_direction'] < 0)  # Trend confirmation
        )
        signals[short_mask] = -1
        
        # Exit signals
        exit_mask = (composite_signal > -self.exit_threshold) & (composite_signal < self.exit_threshold)
        signals[exit_mask] = 0
        
        return signals
    
    def _calculate_statistical_filter(self, features: pd.DataFrame) -> pd.Series:
        """Calculate statistical filter for signal quality."""
        filter_indicators = []
        
        # Volatility regime (avoid extreme volatility)
        if 'volatility_regime' in features.columns:
            vol_filter = pd.Series(np.where(features['volatility_regime'] > 2, 0.3, 
                                           np.where(features['volatility_regime'] < 0.5, 0.3, 1.0)), 
                                 index=features.index)
            filter_indicators.append(vol_filter)
        
        # Volume regime (prefer normal volume)
        if 'volume_regime' in features.columns:
            vol_regime = pd.Series(np.where(features['volume_regime'] > 2, 0.5, 
                                           np.where(features['volume_regime'] < 0.5, 0.5, 1.0)), 
                                 index=features.index)
            filter_indicators.append(vol_regime)
        
        # Trend strength
        if 'trend_strength' in features.columns:
            filter_indicators.append(features['trend_strength'])
        
        # Momentum consistency
        if 'momentum_consistency' in features.columns:
            filter_indicators.append(abs(features['momentum_consistency']))
        
        # Combine indicators
        if filter_indicators:
            quality_filter = pd.concat(filter_indicators, axis=1).mean(axis=1)
            # Normalize to 0-1 range
            quality_filter = (quality_filter - quality_filter.min()) / (quality_filter.max() - quality_filter.min())
            quality_filter = quality_filter.fillna(0.5)
        else:
            quality_filter = pd.Series(0.5, index=features.index)
        
        return quality_filter
    
    def _calculate_statistical_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate statistical signal using z-scores and statistical measures."""
        statistical_indicators = []
        
        # Z-scores (statistical significance)
        for col in ['z_score_20', 'z_score_50']:
            if col in features.columns:
                statistical_indicators.append(features[col])
        
        # RSI statistical extremes
        for col in ['rsi_7', 'rsi_14']:
            if col in features.columns:
                # Normalize RSI to -1 to 1 range
                rsi_norm = (features[col] - 50) / 50
                statistical_indicators.append(rsi_norm)
        
        # MACD statistical signal
        if 'macd_histogram' in features.columns:
            # Normalize MACD histogram
            macd_norm = features['macd_histogram'] / features['macd_histogram'].rolling(20).std()
            statistical_indicators.append(macd_norm)
        
        # Combine indicators
        if statistical_indicators:
            statistical_signal = pd.concat(statistical_indicators, axis=1).mean(axis=1)
        else:
            statistical_signal = pd.Series(0, index=features.index)
        
        return statistical_signal
    
    def _calculate_momentum_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate momentum signal."""
        momentum_indicators = []
        
        # Short-term momentum
        for col in ['momentum_1', 'momentum_3', 'momentum_5']:
            if col in features.columns:
                momentum_indicators.append(features[col])
        
        # Medium-term momentum
        for col in ['momentum_10', 'momentum_20']:
            if col in features.columns:
                momentum_indicators.append(features[col])
        
        # Rate of change
        for col in ['roc_5', 'roc_10', 'roc_20']:
            if col in features.columns:
                momentum_indicators.append(features[col])
        
        # Price acceleration
        if 'price_acceleration' in features.columns:
            momentum_indicators.append(features['price_acceleration'])
        
        # Combine indicators
        if momentum_indicators:
            momentum_signal = pd.concat(momentum_indicators, axis=1).mean(axis=1)
        else:
            momentum_signal = pd.Series(0, index=features.index)
        
        return momentum_signal
    
    def _calculate_trend_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate trend signal."""
        trend_indicators = []
        
        # Price vs moving averages
        for col in ['price_sma_10_ratio', 'price_sma_20_ratio', 'price_sma_50_ratio']:
            if col in features.columns:
                trend_indicators.append(features[col] - 1)
        
        # EMA ratios
        for col in ['price_ema_10_ratio', 'price_ema_20_ratio']:
            if col in features.columns:
                trend_indicators.append(features[col] - 1)
        
        # MACD trend
        if 'macd_histogram' in features.columns:
            trend_indicators.append(features['macd_histogram'])
        
        # Trend strength
        if 'trend_strength' in features.columns:
            trend_indicators.append(features['trend_strength'])
        
        # Combine indicators
        if trend_indicators:
            trend_signal = pd.concat(trend_indicators, axis=1).mean(axis=1)
        else:
            trend_signal = pd.Series(0, index=features.index)
        
        return trend_signal
    
    def _calculate_regime_signal(self, features: pd.DataFrame) -> pd.Series:
        """Calculate regime signal."""
        regime_indicators = []
        
        # Volatility regime
        if 'volatility_regime' in features.columns:
            regime_indicators.append(features['volatility_regime'] - 1)
        
        # Volume regime
        if 'volume_regime' in features.columns:
            regime_indicators.append(features['volume_regime'] - 1)
        
        # Combine indicators
        if regime_indicators:
            regime_signal = pd.concat(regime_indicators, axis=1).mean(axis=1)
        else:
            regime_signal = pd.Series(0, index=features.index)
        
        return regime_signal

def test_quantitative_2plus_sharpe_strategy():
    """Test the quantitative 2+ Sharpe strategy."""
    print("🧪 Testing Quantitative 2+ Sharpe Strategy")
    
    # Fetch test data
    from shared_utilities import DataFetcher
    data = DataFetcher.fetch_data('GLD', '2y')  # Use 2 years for better testing
    
    if data.empty:
        print("❌ Failed to fetch data")
        return
    
    print(f"📊 Data loaded: {len(data)} days")
    
    # Create and run strategy
    strategy = Quantitative2PlusSharpeStrategy()
    print("🔄 Running quantitative 2+ Sharpe strategy backtest...")
    
    results = strategy.run_backtest(data)
    
    # Print results
    print("\n📊 PERFORMANCE METRICS:")
    PerformanceMetrics.print_metrics(results)
    
    print("✅ Quantitative 2+ Sharpe strategy test completed!")

if __name__ == "__main__":
    test_quantitative_2plus_sharpe_strategy() 
 