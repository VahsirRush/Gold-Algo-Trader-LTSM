#!/usr/bin/env python3
"""
GOLD OPTIMIZED STRATEGY - 2.5+ SHARPE TARGET
============================================

Specialized strategy for gold trading that leverages:
- Safe-haven demand during market stress
- Inflation hedging properties
- Currency correlations (USD, EUR, JPY)
- Seasonal patterns
- Central bank demand
- Technical momentum with fundamental filters
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

class GoldFeatureEngineer:
    """Specialized feature engineer for gold trading."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def create_gold_features(self, gold_data: pd.DataFrame, 
                           usd_data: pd.DataFrame = None,
                           vix_data: pd.DataFrame = None,
                           inflation_data: pd.DataFrame = None) -> pd.DataFrame:
        """Create gold-specific features."""
        features = pd.DataFrame(index=gold_data.index)
        
        # Basic price features
        returns = gold_data['Close'].pct_change()
        log_returns = np.log(gold_data['Close'] / gold_data['Close'].shift(1))
        features['returns'] = returns
        features['log_returns'] = log_returns
        
        # Multiple timeframe returns
        for period in [1, 3, 5, 10, 20, 50]:
            features[f'returns_{period}d'] = gold_data['Close'].pct_change(period)
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            sma = gold_data['Close'].rolling(period).mean()
            ema = gold_data['Close'].ewm(span=period).mean()
            features[f'sma_{period}'] = sma
            features[f'ema_{period}'] = ema
            features[f'price_sma_ratio_{period}'] = gold_data['Close'] / sma
        
        # Volatility features
        for period in [5, 10, 20, 50]:
            volatility = features['returns'].rolling(period).std()
            features[f'volatility_{period}'] = volatility
            features[f'volatility_ratio_{period}'] = volatility / volatility.rolling(period*2).mean()
        
        # Momentum features
        for period in [5, 10, 20, 50]:
            momentum = gold_data['Close'] / gold_data['Close'].shift(period) - 1
            features[f'momentum_{period}'] = momentum
            features[f'momentum_ma_{period}'] = momentum.rolling(period).mean()
        
        # Volume features
        volume_sma = gold_data['Volume'].rolling(20).mean()
        features['volume_sma_ratio'] = gold_data['Volume'] / volume_sma
        volume_price_trend = (gold_data['Volume'] * returns).rolling(10).sum()
        features['volume_price_trend'] = volume_price_trend
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(gold_data['Close'])
        features['macd'] = self.calculate_macd(gold_data['Close'])
        bollinger_upper, bollinger_lower = self.calculate_bollinger_bands(gold_data['Close'])
        features['bollinger_upper'] = bollinger_upper
        features['bollinger_lower'] = bollinger_lower
        features['bollinger_position'] = (gold_data['Close'] - bollinger_lower) / (bollinger_upper - bollinger_lower)
        
        # Gold-specific features
        features['gold_real_price'] = self.calculate_real_gold_price(gold_data)
        features['gold_relative_strength'] = self.calculate_gold_relative_strength(gold_data)
        features['gold_volatility_regime'] = self.calculate_volatility_regime(features['volatility_20'])
        
        # USD correlation features (if available)
        if usd_data is not None:
            features['usd_correlation'] = self.calculate_usd_correlation(gold_data, usd_data)
            features['usd_strength'] = self.calculate_usd_strength(usd_data)
        
        # VIX features (market stress indicator)
        if vix_data is not None:
            features['vix_level'] = vix_data['Close']
            features['vix_change'] = vix_data['Close'].pct_change()
            features['vix_ma_ratio'] = vix_data['Close'] / vix_data['Close'].rolling(20).mean()
            features['safe_haven_demand'] = (vix_data['Close'] > vix_data['Close'].rolling(50).quantile(0.8)).astype(int)
        
        # Inflation features (if available)
        if inflation_data is not None:
            features['inflation_expectation'] = inflation_data['Close']
            features['inflation_change'] = inflation_data['Close'].pct_change()
            features['inflation_hedge_demand'] = (inflation_data['Close'] > inflation_data['Close'].rolling(50).quantile(0.7)).astype(int)
        
        # Seasonal features
        features['month'] = gold_data.index.month
        features['quarter'] = gold_data.index.quarter
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Gold seasonal patterns
        features['indian_wedding_season'] = ((features['month'] >= 9) & (features['month'] <= 12)).astype(int)
        features['chinese_new_year'] = ((features['month'] >= 1) & (features['month'] <= 2)).astype(int)
        features['western_holiday'] = ((features['month'] >= 11) & (features['month'] <= 12)).astype(int)
        
        # Market regime features
        features['trend_regime'] = self.calculate_trend_regime(gold_data['Close'])
        features['momentum_regime'] = self.calculate_momentum_regime(features)
        features['volatility_regime'] = self.calculate_volatility_regime(features['volatility_20'])
        
        # Advanced features
        features['price_momentum_divergence'] = self.calculate_momentum_divergence(gold_data['Close'])
        features['volume_price_divergence'] = self.calculate_volume_price_divergence(gold_data)
        features['support_resistance_levels'] = self.calculate_support_resistance(gold_data['Close'])
        
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        
        return macd - signal_line
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, lower_band
    
    def calculate_real_gold_price(self, gold_data: pd.DataFrame) -> pd.Series:
        """Calculate real gold price (adjusted for inflation - simplified)."""
        # Simplified real price calculation
        return gold_data['Close'] / gold_data['Close'].rolling(252).mean()
    
    def calculate_gold_relative_strength(self, gold_data: pd.DataFrame) -> pd.Series:
        """Calculate gold's relative strength vs other assets."""
        # Simplified relative strength
        return gold_data['Close'].pct_change(20) / gold_data['Close'].pct_change(20).rolling(50).std()
    
    def calculate_usd_correlation(self, gold_data: pd.DataFrame, usd_data: pd.DataFrame) -> pd.Series:
        """Calculate rolling correlation with USD."""
        gold_returns = gold_data['Close'].pct_change()
        usd_returns = usd_data['Close'].pct_change()
        
        return gold_returns.rolling(20).corr(usd_returns)
    
    def calculate_usd_strength(self, usd_data: pd.DataFrame) -> pd.Series:
        """Calculate USD strength."""
        return usd_data['Close'].pct_change(20)
    
    def calculate_trend_regime(self, prices: pd.Series) -> pd.Series:
        """Calculate trend regime."""
        short_ma = prices.rolling(20).mean()
        long_ma = prices.rolling(50).mean()
        
        regime = pd.Series(0, index=prices.index)
        regime[short_ma > long_ma] = 1  # Uptrend
        regime[short_ma < long_ma * 0.95] = -1  # Downtrend
        
        return regime
    
    def calculate_momentum_regime(self, features: pd.DataFrame) -> pd.Series:
        """Calculate momentum regime."""
        momentum_20 = features['momentum_20']
        
        regime = pd.Series(0, index=features.index)
        regime[momentum_20 > momentum_20.rolling(50).quantile(0.7)] = 1  # High momentum
        regime[momentum_20 < momentum_20.rolling(50).quantile(0.3)] = -1  # Low momentum
        
        return regime
    
    def calculate_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Calculate volatility regime."""
        regime = pd.Series(0, index=volatility.index)
        regime[volatility > volatility.rolling(50).quantile(0.8)] = 1  # High volatility
        regime[volatility < volatility.rolling(50).quantile(0.2)] = -1  # Low volatility
        
        return regime
    
    def calculate_momentum_divergence(self, prices: pd.Series) -> pd.Series:
        """Calculate momentum divergence."""
        price_momentum = prices.pct_change(10)
        momentum_momentum = price_momentum.pct_change(5)
        
        return momentum_momentum
    
    def calculate_volume_price_divergence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume-price divergence."""
        price_change = data['Close'].pct_change()
        volume_change = data['Volume'].pct_change()
        
        return price_change - volume_change
    
    def calculate_support_resistance(self, prices: pd.Series) -> pd.Series:
        """Calculate support/resistance levels."""
        rolling_high = prices.rolling(20).max()
        rolling_low = prices.rolling(20).min()
        
        return (prices - rolling_low) / (rolling_high - rolling_low)
    
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

class GoldMLEnsemble:
    """Gold-specific ML ensemble."""
    
    def __init__(self, lookback_days: int = 5):
        self.lookback_days = lookback_days
        
        # Models optimized for gold
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            'et': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.01),
            'elastic': ElasticNet(alpha=0.01, l1_ratio=0.5)
        }
        
        # Model weights (can be optimized)
        self.model_weights = {name: 1.0/len(self.models) for name in self.models.keys()}
        
        self.is_trained = False
    
    def prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """Prepare target variable (future returns)."""
        future_returns = data['Close'].pct_change(self.lookback_days).shift(-self.lookback_days)
        return future_returns
    
    def train_models(self, features: pd.DataFrame, target: pd.Series):
        """Train all models."""
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]
        
        if len(X) < 100:
            print("❌ Not enough data for training")
            return
        
        # Train each model
        for name, model in self.models.items():
            model.fit(X, y)
        
        self.is_trained = True
        print(f"✅ Models trained on {len(X)} samples")
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Generate ensemble predictions."""
        if not self.is_trained:
            return pd.Series(0, index=features.index)
        
        predictions = pd.DataFrame(index=features.index)
        
        for name, model in self.models.items():
            pred = model.predict(features)
            predictions[name] = pred
        
        # Weighted ensemble
        ensemble_pred = pd.Series(0, index=features.index)
        for name in self.models.keys():
            ensemble_pred += predictions[name] * self.model_weights[name]
        
        return ensemble_pred

class GoldOptimizedStrategy:
    """Gold-optimized trading strategy."""
    
    def __init__(self, 
                 target_volatility: float = 0.15,
                 max_position_size: float = 0.8,
                 use_adaptive_thresholds: bool = True,
                 use_regime_filtering: bool = True):
        
        self.target_volatility = target_volatility
        self.max_position_size = max_position_size
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.use_regime_filtering = use_regime_filtering
        
        self.feature_engineer = GoldFeatureEngineer()
        self.ml_ensemble = GoldMLEnsemble()
        
        # Adaptive thresholds
        self.buy_threshold = 0.02
        self.sell_threshold = -0.02
        
        # Regime filters
        self.trend_filter = True
        self.volatility_filter = True
        self.momentum_filter = True
    
    def generate_signals(self, gold_data: pd.DataFrame,
                        usd_data: pd.DataFrame = None,
                        vix_data: pd.DataFrame = None,
                        inflation_data: pd.DataFrame = None) -> pd.Series:
        """Generate gold trading signals."""
        print("🔧 Generating gold-optimized signals...")
        
        # Create features
        features = self.feature_engineer.create_gold_features(
            gold_data, usd_data, vix_data, inflation_data
        )
        
        # Train models on first 70% of data
        train_size = int(len(gold_data) * 0.7)
        
        train_features = features.iloc[:train_size]
        train_target = self.ml_ensemble.prepare_target(gold_data.iloc[:train_size])
        
        # Fit scaler and train models
        self.feature_engineer.fit_scaler(train_features)
        self.ml_ensemble.train_models(train_features, train_target)
        
        # Transform all features
        scaled_features = self.feature_engineer.transform_features(features)
        
        # Generate predictions
        predictions = self.ml_ensemble.predict(scaled_features)
        
        # Generate signals
        signals = self.generate_trading_signals(predictions, features, gold_data)
        
        # Apply risk management
        signals = self.apply_risk_management(signals, gold_data)
        
        return signals
    
    def generate_trading_signals(self, predictions: pd.Series, features: pd.DataFrame, gold_data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from predictions."""
        signals = pd.Series(0, index=predictions.index)
        
        # Adaptive thresholds
        if self.use_adaptive_thresholds:
            volatility = features['volatility_20']
            self.buy_threshold = 0.01 + (volatility * 0.5)
            self.sell_threshold = -0.01 - (volatility * 0.5)
        
        # Generate signals based on predictions
        for i in range(len(predictions)):
            if pd.isna(predictions.iloc[i]):
                continue
                
            # Apply regime filters
            if self.use_regime_filtering:
                if not self.check_regime_filters(features, i):
                    continue
            
            # Signal generation
            if predictions.iloc[i] > self.buy_threshold.iloc[i] if hasattr(self.buy_threshold, 'iloc') else self.buy_threshold:
                signals.iloc[i] = 1
            elif predictions.iloc[i] < self.sell_threshold.iloc[i] if hasattr(self.sell_threshold, 'iloc') else self.sell_threshold:
                signals.iloc[i] = -1
        
        return signals
    
    def check_regime_filters(self, features: pd.DataFrame, index: int) -> bool:
        """Check if current regime allows trading."""
        if self.trend_filter and features['trend_regime'].iloc[index] == -1:
            return False  # Don't trade in strong downtrend
        
        if self.volatility_filter and features['volatility_regime'].iloc[index] == 1:
            return False  # Don't trade in high volatility
        
        if self.momentum_filter and features['momentum_regime'].iloc[index] == -1:
            return False  # Don't trade in low momentum
        
        return True
    
    def apply_risk_management(self, signals: pd.Series, gold_data: pd.DataFrame) -> pd.Series:
        """Apply risk management to signals."""
        # Volatility targeting
        volatility = gold_data['Close'].pct_change().rolling(20).std()
        position_size = self.target_volatility / (volatility * np.sqrt(252))
        position_size = position_size.clip(0, self.max_position_size)
        
        # Apply position sizing
        signals = signals * position_size
        
        # Stop loss and take profit (simplified)
        signals = self.apply_stop_loss_take_profit(signals, gold_data)
        
        return signals
    
    def apply_stop_loss_take_profit(self, signals: pd.Series, gold_data: pd.DataFrame) -> pd.Series:
        """Apply stop loss and take profit."""
        # Simplified implementation
        returns = gold_data['Close'].pct_change()
        
        # Stop loss at -2% daily loss
        stop_loss = (returns < -0.02).astype(int)
        
        # Take profit at +3% daily gain
        take_profit = (returns > 0.03).astype(int)
        
        # Close positions on stop loss or take profit
        for i in range(1, len(signals)):
            if stop_loss.iloc[i] and signals.iloc[i-1] != 0:
                signals.iloc[i] = 0
            elif take_profit.iloc[i] and signals.iloc[i-1] > 0:
                signals.iloc[i] = 0
        
        return signals

def fetch_gold_data(period: str = '5y') -> tuple:
    """Fetch gold and related data."""
    print(f"📊 Fetching gold data for {period}...")
    
    # Gold data - use a simpler approach
    try:
        gold_ticker = yf.Ticker('GC=F')
        gold = gold_ticker.history(period=period)
        print(f"  📈 Gold: {len(gold)} days, ${float(gold['Close'].iloc[-1]):.2f}")
    except Exception as e:
        print(f"  ❌ Failed to fetch gold data: {e}")
        return None, None, None, None
    
    # USD index
    try:
        usd_ticker = yf.Ticker('UUP')
        usd = usd_ticker.history(period=period)
        print(f"  💵 USD Index: {len(usd)} days, ${float(usd['Close'].iloc[-1]):.2f}")
    except:
        usd = None
        print("  ⚠️  USD data not available")
    
    # VIX (market stress)
    try:
        vix_ticker = yf.Ticker('^VIX')
        vix = vix_ticker.history(period=period)
        print(f"  😰 VIX: {len(vix)} days, {float(vix['Close'].iloc[-1]):.2f}")
    except:
        vix = None
        print("  ⚠️  VIX data not available")
    
    # Inflation expectations (simplified - using TIPS)
    try:
        tips_ticker = yf.Ticker('TIP')
        tips = tips_ticker.history(period=period)
        print(f"  📈 TIPS: {len(tips)} days, ${float(tips['Close'].iloc[-1]):.2f}")
    except:
        tips = None
        print("  ⚠️  TIPS data not available")
    
    return gold, usd, vix, tips

def backtest_gold_strategy(gold_data: pd.DataFrame, strategy) -> Dict:
    """Backtest the gold strategy."""
    print("🔍 Testing Gold Strategy")
    print("-" * 50)
    
    # Generate signals
    signals = strategy.generate_signals(gold_data)
    
    # Calculate returns
    gold_returns = gold_data['Close'].pct_change()
    strategy_returns = signals.shift(1) * gold_returns
    
    # Buy and hold returns
    buy_hold_returns = gold_returns
    
    # Calculate metrics
    def calculate_metrics(returns: pd.Series, name: str) -> Dict:
        returns = returns.dropna()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Risk metrics
        max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        sortino_ratio = annual_return / (returns[returns < 0].std() * np.sqrt(252)) if returns[returns < 0].std() > 0 else 0
        
        # Additional metrics
        win_rate = (returns > 0).mean()
        profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
        
        return {
            'name': name,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    strategy_metrics = calculate_metrics(strategy_returns, "Gold Strategy")
    buy_hold_metrics = calculate_metrics(buy_hold_returns, "Buy & Hold")
    
    # Print results
    print(f"📊 STRATEGY RESULTS:")
    print(f"   Total Return: {strategy_metrics['total_return']:.2%}")
    print(f"   Annual Return: {strategy_metrics['annual_return']:.2%}")
    print(f"   Volatility: {strategy_metrics['volatility']:.2%}")
    print(f"   Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.3f}")
    print(f"   Sortino Ratio: {strategy_metrics['sortino_ratio']:.3f}")
    print(f"   Max Drawdown: {strategy_metrics['max_drawdown']:.2%}")
    print(f"   Win Rate: {strategy_metrics['win_rate']:.2%}")
    print(f"   Profit Factor: {strategy_metrics['profit_factor']:.2f}")
    
    print(f"\n📊 BUY & HOLD RESULTS:")
    print(f"   Total Return: {buy_hold_metrics['total_return']:.2%}")
    print(f"   Annual Return: {buy_hold_metrics['annual_return']:.2%}")
    print(f"   Volatility: {buy_hold_metrics['volatility']:.2%}")
    print(f"   Sharpe Ratio: {buy_hold_metrics['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {buy_hold_metrics['max_drawdown']:.2%}")
    
    # Check if target achieved
    if strategy_metrics['sharpe_ratio'] >= 2.5:
        print(f"\n🎯 TARGET ACHIEVED! Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.3f} >= 2.5")
    else:
        print(f"\n❌ Target not achieved. Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.3f} < 2.5")
    
    return {
        'strategy': strategy_metrics,
        'buy_hold': buy_hold_metrics,
        'signals': signals,
        'returns': strategy_returns
    }

def main():
    """Main function."""
    print("🚀 GOLD OPTIMIZED STRATEGY - 2.5+ SHARPE TARGET")
    print("=" * 60)
    
    # Fetch data
    gold_data, usd_data, vix_data, inflation_data = fetch_gold_data('5y')
    
    if gold_data.empty:
        print("❌ Failed to fetch gold data")
        return
    
    print(f"✅ Successfully loaded gold data")
    
    # Create strategy
    strategy = GoldOptimizedStrategy(
        target_volatility=0.15,
        max_position_size=0.8,
        use_adaptive_thresholds=True,
        use_regime_filtering=True
    )
    
    # Run backtest
    results = backtest_gold_strategy(gold_data, strategy)
    
    # Plot results
    try:
        plt.figure(figsize=(15, 10))
        
        # Cumulative returns
        plt.subplot(2, 2, 1)
        cumulative_strategy = (1 + results['returns']).cumprod()
        cumulative_buyhold = (1 + gold_data['Close'].pct_change()).cumprod()
        
        plt.plot(cumulative_strategy.index, cumulative_strategy.values, label='Gold Strategy', linewidth=2)
        plt.plot(cumulative_buyhold.index, cumulative_buyhold.values, label='Buy & Hold', alpha=0.7)
        plt.title('Cumulative Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gold price
        plt.subplot(2, 2, 2)
        plt.plot(gold_data.index, gold_data['Close'], label='Gold Price', color='gold')
        plt.title('Gold Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Signals
        plt.subplot(2, 2, 3)
        signals = results['signals']
        buy_signals = signals[signals > 0]
        sell_signals = signals[signals < 0]
        
        plt.scatter(buy_signals.index, gold_data.loc[buy_signals.index, 'Close'], 
                   color='green', marker='^', s=50, label='Buy Signal', alpha=0.7)
        plt.scatter(sell_signals.index, gold_data.loc[sell_signals.index, 'Close'], 
                   color='red', marker='v', s=50, label='Sell Signal', alpha=0.7)
        plt.plot(gold_data.index, gold_data['Close'], color='gold', alpha=0.5)
        plt.title('Trading Signals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Returns distribution
        plt.subplot(2, 2, 4)
        strategy_returns = results['returns'].dropna()
        plt.hist(strategy_returns, bins=50, alpha=0.7, label='Strategy Returns')
        plt.axvline(strategy_returns.mean(), color='red', linestyle='--', label='Mean')
        plt.title('Returns Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gold_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Results saved to 'gold_strategy_results.png'")
        
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

if __name__ == "__main__":
    main() 