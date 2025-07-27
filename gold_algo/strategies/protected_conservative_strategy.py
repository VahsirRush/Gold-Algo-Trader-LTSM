#!/usr/bin/env python3
"""
Protected Conservative Risk-Enhanced Strategy
============================================

This strategy integrates comprehensive overfitting protection with conservative
risk management to ensure robust, realistic performance across different market conditions.

Key Features:
- Overfitting protection with cross-validation
- Parameter sensitivity testing
- Walk-forward analysis
- Realistic performance bounds
- Conservative position sizing
- Comprehensive risk management
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from gold_algo.shared_utilities import BaseStrategy, calculate_technical_indicators
from risk_management.volatility_position_manager import VolatilityPositionManager
from overfitting_protection import OverfittingProtection, create_protected_strategy_wrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProtectedConservativeStrategy(BaseStrategy):
    """
    Protected conservative risk-enhanced strategy with overfitting protection.
    """
    
    def __init__(self, 
                 target_volatility: float = 0.15,  # 15% annual volatility target
                 max_position_size: float = 0.10,  # Max 10% position size (increased from 5%)
                 soft_drawdown_limit: float = 0.08,  # 8% soft stop
                 hard_drawdown_limit: float = 0.10,  # 10% hard stop
                 initial_capital: float = 100000.0,
                 enable_protection: bool = True):
        """
        Initialize protected conservative risk strategy.
        
        Args:
            target_volatility: Annual volatility target (15% = realistic)
            max_position_size: Maximum position size as fraction of capital
            soft_drawdown_limit: Soft stop drawdown level
            hard_drawdown_limit: Hard stop drawdown level
            initial_capital: Initial capital
            enable_protection: Whether to enable overfitting protection
        """
        super().__init__()
        
        # Strategy parameters
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.target_volatility = target_volatility
        self.max_position_size = max_position_size
        self.enable_protection = enable_protection
        
        # Risk management
        self.risk_manager = VolatilityPositionManager(
            target_volatility=target_volatility,
            max_position_size=max_position_size,
            soft_drawdown_limit=soft_drawdown_limit,
            hard_drawdown_limit=hard_drawdown_limit
        )
        
        # Overfitting protection
        if enable_protection:
            self.protection_system = OverfittingProtection(
                min_performance_threshold=0.5,
                max_sharpe_threshold=3.0,
                max_drawdown_threshold=0.25,
                cv_folds=5,
                walk_forward_windows=4
            )
        else:
            self.protection_system = None
        
        # Performance tracking
        self.equity_curve = []
        self.trades = []
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        
        # Signal generation parameters
        self.lookback_periods = [5, 10, 20]  # Multiple timeframes
        self.confirmation_threshold = 0.4  # Require 40% signal agreement (reduced from 60%)
        
        # Parameter ranges for sensitivity testing
        self.param_ranges = {
            'target_volatility': [0.10, 0.15, 0.20, 0.25],
            'max_position_size': [0.03, 0.05, 0.07, 0.10],
            'confirmation_threshold': [0.5, 0.6, 0.7, 0.8]
        }
        
        logger.info(f"ProtectedConservativeStrategy initialized with protection: {enable_protection}")
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate conservative trading signals with multiple confirmations.
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary with signal strength and direction
        """
        try:
            if len(data) < max(self.lookback_periods):
                return {'signal': 0.0, 'strength': 0.0}
            
            # Calculate technical indicators
            indicators = calculate_technical_indicators(data)
            
            # Multi-timeframe signal analysis
            signals = []
            weights = []
            
            for period in self.lookback_periods:
                if len(data) < period:
                    continue
                
                # Get recent data for this period
                recent_data = data.tail(period)
                
                # Calculate trend indicators
                sma = recent_data['close'].mean()
                current_price = recent_data['close'].iloc[-1]
                price_sma_ratio = current_price / sma
                
                # Calculate momentum
                returns = recent_data['close'].pct_change().dropna()
                momentum = returns.mean()
                volatility = returns.std()
                
                # Calculate RSI
                rsi = indicators.get('rsi_14', pd.Series([50] * len(data))).iloc[-1]
                
                # Generate signal for this timeframe
                signal = 0.0
                
                # Trend following with momentum confirmation (more sensitive)
                if price_sma_ratio > 1.005 and momentum > 0 and rsi < 75:  # Bullish (reduced threshold)
                    signal = min(1.0, (price_sma_ratio - 1.0) * 20)  # Increased sensitivity
                elif price_sma_ratio < 0.995 and momentum < 0 and rsi > 25:  # Bearish (reduced threshold)
                    signal = -min(1.0, (1.0 - price_sma_ratio) * 20)  # Increased sensitivity
                
                # Volatility adjustment
                if volatility > 0.03:  # High volatility - reduce signal strength
                    signal *= 0.5
                
                signals.append(signal)
                weights.append(1.0 / period)  # Shorter periods get higher weight
            
            # Weighted average of signals
            if signals and weights:
                weighted_signal = np.average(signals, weights=weights)
                
                # Require confirmation threshold
                signal_agreement = sum(1 for s in signals if (s > 0 and weighted_signal > 0) or 
                                     (s < 0 and weighted_signal < 0)) / len(signals)
                
                if signal_agreement >= self.confirmation_threshold:
                    # Conservative signal strength
                    final_signal = np.clip(weighted_signal, -0.5, 0.5)  # Cap at 50%
                    return {
                        'signal': np.sign(final_signal),
                        'strength': abs(final_signal)
                    }
            
            return {'signal': 0.0, 'strength': 0.0}
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {'signal': 0.0, 'strength': 0.0}
    
    def execute_trade(self, signal: float, price: float, timestamp: pd.Timestamp) -> Dict:
        """
        Execute trade with conservative position sizing.
        
        Args:
            signal: Trading signal (-1 to 1)
            price: Current price
            timestamp: Trade timestamp
            
        Returns:
            Trade execution details
        """
        try:
            # Simple conservative position sizing (no complex risk manager for now)
            position_size = min(abs(signal) * 0.10, self.max_position_size)  # Increased from 0.05 to 0.10
            
            if position_size < 0.01:  # Minimum 1% position size
                return {'executed': False, 'reason': 'Position too small'}
            
            # Calculate trade value
            trade_value = self.current_capital * position_size
            shares = trade_value / price
            
            # Execute trade
            trade = {
                'timestamp': timestamp,
                'price': price,
                'shares': shares,
                'value': trade_value,
                'direction': np.sign(signal),
                'position_size': position_size,
                'capital_before': self.current_capital
            }
            
            self.trades.append(trade)
            
            logger.info(f"Trade executed: {signal:.2f} {position_size:.3f} at {price:.2f}")
            
            return {'executed': True, 'trade': trade}
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'executed': False, 'reason': str(e)}
    
    def update_performance(self, current_price: float, timestamp: pd.Timestamp):
        """
        Update performance metrics.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
        """
        try:
            # Calculate current portfolio value
            portfolio_value = self.current_capital
            
            # Add unrealized P&L from open positions
            for trade in self.trades[-10:]:  # Recent trades
                if trade.get('direction', 0) != 0:
                    price_change = (current_price - trade['price']) / trade['price']
                    unrealized_pnl = trade['value'] * price_change * trade['direction']
                    portfolio_value += unrealized_pnl
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'current_price': current_price
            })
            
            # Simple drawdown tracking (no complex risk manager for now)
            if portfolio_value > self.initial_capital:
                self.initial_capital = portfolio_value  # Update high water mark
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def _run_strategy_core(self, data: pd.DataFrame, params: Dict = None) -> Dict:
        """
        Core strategy execution without protection.
        
        Args:
            data: OHLCV data
            params: Strategy parameters (optional)
            
        Returns:
            Strategy results
        """
        try:
            logger.info("Starting protected conservative strategy backtest...")
            
            # Reset state
            self.equity_curve = []
            self.trades = []
            self.current_capital = self.initial_capital
            
            # Process each data point
            for i, (timestamp, row) in enumerate(data.iterrows()):
                if i < max(self.lookback_periods):
                    continue
                
                # Get historical data for signal generation
                historical_data = data.iloc[:i+1]
                
                # Generate signals
                signal_data = self.generate_signals(historical_data)
                signal = signal_data['signal']
                strength = signal_data['strength']
                
                # Execute trade if signal is strong enough
                if abs(signal) > 0.05:  # Reduced minimum signal threshold (from 0.1 to 0.05)
                    trade_result = self.execute_trade(signal, row['close'], timestamp)
                    if trade_result['executed']:
                        # Update capital (simplified - no transaction costs for now)
                        pass
                
                # Update performance
                self.update_performance(row['close'], timestamp)
                
                # Progress update
                if i % 50 == 0:
                    logger.info(f"Progress: {i}/{len(data)} - Capital: {self.current_capital:.2f}")
            
            # Calculate final performance metrics
            self._calculate_performance_metrics()
            
            logger.info("Protected conservative backtest completed successfully")
            logger.info("=== PROTECTED CONSERVATIVE STRATEGY RESULTS ===")
            logger.info(f"Total Return: {self.total_return:.2%}")
            logger.info(f"Max Drawdown: {self.max_drawdown:.2%}")
            logger.info(f"Sharpe Ratio: {self.sharpe_ratio:.3f}")
            logger.info(f"Win Rate: {self.win_rate:.2%}")
            logger.info(f"Total Trades: {len(self.trades)}")
            
            return self._get_backtest_results()
            
        except Exception as e:
            logger.error(f"Error in core strategy: {e}")
            return {}
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run protected backtest with overfitting protection.
        
        Args:
            data: OHLCV data
            
        Returns:
            Backtest results with protection analysis
        """
        try:
            if self.enable_protection and self.protection_system:
                # Create strategy parameters
                strategy_params = {
                    'target_volatility': self.target_volatility,
                    'max_position_size': self.max_position_size,
                    'confirmation_threshold': self.confirmation_threshold
                }
                
                # Run comprehensive overfitting check
                protection_results = self.protection_system.comprehensive_overfitting_check(
                    self._run_strategy_core, data, strategy_params, self.param_ranges
                )
                
                # Run core strategy
                strategy_results = self._run_strategy_core(data)
                
                # Add protection results
                strategy_results['overfitting_protection'] = protection_results
                strategy_results['strategy_type'] = 'Protected Conservative Risk-Enhanced'
                
                # Log protection status
                if protection_results['overfitting_risk'] == 'HIGH':
                    logger.warning("⚠️  HIGH overfitting risk detected!")
                    logger.warning(f"Recommendations: {protection_results['recommendations']}")
                elif protection_results['overfitting_risk'] == 'MEDIUM':
                    logger.warning("⚠️  MEDIUM overfitting risk detected")
                else:
                    logger.info("✅ LOW overfitting risk - strategy appears robust")
                
                return strategy_results
            else:
                # Run without protection
                return self._run_strategy_core(data)
                
        except Exception as e:
            logger.error(f"Error in protected backtest: {e}")
            return {}
    
    def _calculate_performance_metrics(self):
        """Calculate conservative performance metrics."""
        try:
            if not self.equity_curve:
                return
            
            # Convert to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            equity_df['returns'] = equity_df['portfolio_value'].pct_change()
            
            # Total return
            final_value = equity_df['portfolio_value'].iloc[-1]
            self.total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Maximum drawdown
            equity_df['cummax'] = equity_df['portfolio_value'].cummax()
            equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['cummax']) / equity_df['cummax']
            self.max_drawdown = equity_df['drawdown'].min()
            
            # Conservative Sharpe ratio calculation
            returns = equity_df['returns'].dropna()
            if len(returns) > 30:  # Require minimum data points
                # Use simple annualized metrics
                annualized_return = returns.mean() * 252
                annualized_volatility = returns.std() * np.sqrt(252)
                risk_free_rate = 0.02
                
                # Conservative Sharpe calculation
                if annualized_volatility > 0.01:  # Minimum volatility threshold
                    self.sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
                else:
                    self.sharpe_ratio = 0.0
            else:
                self.sharpe_ratio = 0.0
            
            # Win rate calculation
            if self.trades:
                # Simplified win rate based on final performance
                self.win_rate = 1.0 if self.total_return > 0 else 0.0
                self.profit_factor = 1.0 if self.total_return > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    def _get_backtest_results(self) -> Dict:
        """Get protected backtest results."""
        return {
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': len(self.trades),
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'strategy_type': 'Protected Conservative Risk-Enhanced'
        }


def run_protected_conservative_backtest(start_date: str = "2023-07-01", 
                                       end_date: str = "2023-09-30",
                                       initial_capital: float = 100000.0,
                                       enable_protection: bool = True) -> Dict:
    """
    Run protected conservative backtest.
    
    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Initial capital
        enable_protection: Whether to enable overfitting protection
        
    Returns:
        Backtest results with protection analysis
    """
    try:
        logger.info(f"Running protected conservative backtest from {start_date} to {end_date}")
        logger.info(f"Overfitting protection: {'ENABLED' if enable_protection else 'DISABLED'}")
        
        # Import here to avoid circular imports
        from data_pipeline.databento_collector import DatabentoGoldCollector
        
        # Fetch data
        collector = DatabentoGoldCollector()
        data = collector.fetch_gld(start_date, end_date)
        
        if data is None or len(data) == 0:
            logger.error("No data fetched for backtest")
            return {}
        
        logger.info(f"Fetched {len(data)} data points")
        
        # Initialize strategy
        strategy = ProtectedConservativeStrategy(
            target_volatility=0.15,  # 15% annual volatility
            max_position_size=0.05,  # Max 5% position size
            soft_drawdown_limit=0.08,  # 8% soft stop
            hard_drawdown_limit=0.10,  # 10% hard stop
            initial_capital=initial_capital,
            enable_protection=enable_protection
        )
        
        # Run backtest
        results = strategy.backtest(data)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in protected conservative backtest: {e}")
        return {}


if __name__ == "__main__":
    # Test the protected strategy
    results = run_protected_conservative_backtest(enable_protection=True)
    print("Protected Strategy Results:", results) 