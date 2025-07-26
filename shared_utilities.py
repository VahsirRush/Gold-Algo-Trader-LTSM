#!/usr/bin/env python3
"""
Shared Utilities Module
Contains common functions used across multiple strategy files to eliminate redundancy.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Set, Tuple
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
        """Apply volatility targeting to signals."""
        volatility = data['Close'].pct_change().rolling(20).std()
        position_size = target_volatility / (volatility * np.sqrt(252))
        position_size = position_size.clip(0, max_position_size)
        
        return signals * position_size
    
    @staticmethod
    def apply_stop_loss_take_profit(signals: pd.Series, data: pd.DataFrame,
                                   stop_loss_pct: float = 0.02,
                                   take_profit_pct: float = 0.03) -> pd.Series:
        """Apply stop loss and take profit."""
        returns = data['Close'].pct_change()
        
        # Stop loss
        stop_loss = (returns < -stop_loss_pct).astype(int)
        
        # Take profit
        take_profit = (returns > take_profit_pct).astype(int)
        
        # Close positions on stop loss or take profit
        for i in range(1, len(signals)):
            if stop_loss.iloc[i] and signals.iloc[i-1] != 0:
                signals.iloc[i] = 0
            elif take_profit.iloc[i] and signals.iloc[i-1] > 0:
                signals.iloc[i] = 0
        
        return signals

class PerformanceMetrics:
    """Common performance calculation functions."""
    
    @staticmethod
    def calculate_metrics(returns: pd.Series, name: str = "Strategy") -> Dict:
        """Calculate comprehensive performance metrics."""
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
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """Print formatted performance metrics."""
        print(f"📊 {metrics['name'].upper()} RESULTS:")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annual Return: {metrics['annual_return']:.2%}")
        print(f"   Volatility: {metrics['volatility']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"   Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")

class DataFetcher:
    """Common data fetching functionality."""
    
    @staticmethod
    def fetch_data(symbol: str, period: str = '5y') -> pd.DataFrame:
        """Fetch data with error handling."""
        try:
            import yfinance as yf
            data = yf.download(symbol, period=period, progress=False)
            
            # Handle MultiIndex if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(1)
            
            print(f"  📈 {symbol}: {len(data)} days, ${float(data['Close'].iloc[-1]):.2f}")
            return data
        except Exception as e:
            print(f"  ❌ Failed to fetch {symbol}: {e}")
            return pd.DataFrame()

class FeatureEngineer:
    """Common feature engineering functionality."""
    
    @staticmethod
    def create_basic_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create basic price and volume features."""
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Multiple timeframe returns
        for period in [1, 3, 5, 10, 20]:
            features[f'returns_{period}d'] = data['Close'].pct_change(period)
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            sma = data['Close'].rolling(period).mean()
            ema = data['Close'].ewm(span=period).mean()
            features[f'sma_{period}'] = sma
            features[f'ema_{period}'] = ema
            features[f'price_sma_ratio_{period}'] = data['Close'] / sma
        
        # Volatility features
        for period in [5, 10, 20]:
            volatility = features['returns'].rolling(period).std()
            features[f'volatility_{period}'] = volatility
            features[f'volatility_ratio_{period}'] = volatility / volatility.rolling(period*2).mean()
        
        # Volume features
        if 'Volume' in data.columns:
            volume_sma = data['Volume'].rolling(20).mean()
            features['volume_sma_ratio'] = data['Volume'] / volume_sma
            volume_price_trend = (data['Volume'] * features['returns']).rolling(10).sum()
            features['volume_price_trend'] = volume_price_trend
        
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features 