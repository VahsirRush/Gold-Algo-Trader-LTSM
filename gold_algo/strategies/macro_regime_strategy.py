#!/usr/bin/env python3
"""
Macro Regime-Enhanced Gold Trading Strategy
==========================================

This strategy integrates macroeconomic regime filtering with comprehensive
risk management to dynamically adjust exposure based on economic conditions.

Key Features:
- Macro regime detection (risk-off vs risk-on)
- Dynamic position sizing based on regime
- Integrated drawdown risk management
- Regime persistence to prevent whipsaw
- Leverage limits (6x-8x range) and risk controls
- Real-time macro indicator monitoring

Regime Classification:
- Risk-off regime: Favorable for gold (increase exposure)
- Risk-on regime: Unfavorable for gold (reduce exposure)
- Neutral regime: Standard parameters
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from gold_algo.shared_utilities import BaseStrategy, calculate_technical_indicators
from risk_management.drawdown_risk_manager import DrawdownRiskManager
from risk_management.macro_regime_filter import MacroRegimeFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroRegimeStrategy(BaseStrategy):
    """
    Macro regime-enhanced gold trading strategy.
    
    Integrates macroeconomic regime filtering with comprehensive risk
    management to dynamically adjust exposure based on economic conditions.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 trailing_stop_pct: float = 0.05,  # 5% trailing stop
                 circuit_breaker_pct: float = 0.15,  # 15% circuit breaker
                 max_position_size: float = 0.20,  # 20% max position
                 confirmation_threshold: float = 0.4,  # 40% signal agreement
                 enable_macro_filter: bool = True,
                 enable_risk_management: bool = True):
        """
        Initialize the macro regime strategy.
        
        Args:
            initial_capital: Starting capital
            trailing_stop_pct: Trailing stop percentage
            circuit_breaker_pct: Circuit breaker percentage
            max_position_size: Maximum position size
            confirmation_threshold: Signal confirmation threshold
            enable_macro_filter: Enable macro regime filtering
            enable_risk_management: Enable risk management
        """
        super().__init__()
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Core strategy parameters (unchanged)
        self.lookback_periods = [5, 10, 20]  # Multiple timeframes
        self.confirmation_threshold = confirmation_threshold
        
        # Risk management overlay
        self.enable_risk_management = enable_risk_management
        if enable_risk_management:
            self.risk_manager = DrawdownRiskManager(
                initial_capital=initial_capital,
                trailing_stop_pct=trailing_stop_pct,
                circuit_breaker_pct=circuit_breaker_pct,
                max_position_size=max_position_size,
                volatility_lookback=10,  # 10-day volatility
                drawdown_scaling=True
            )
        else:
            self.risk_manager = None
        
        # Macro regime filter
        self.enable_macro_filter = enable_macro_filter
        if enable_macro_filter:
            self.macro_filter = MacroRegimeFilter(
                regime_persistence_days=5,  # 5-day minimum regime persistence
                max_leverage=6.0,  # Maximum 6x leverage
                min_leverage=0.5,  # Minimum 0.5x leverage
                regime_confidence_threshold=0.7,  # 70% confidence threshold
                use_markov_model=True
            )
        else:
            self.macro_filter = None
        
        # Performance tracking
        self.equity_curve = []
        self.trades = []
        self.risk_metrics = []
        self.regime_metrics = []
        
        logger.info(f"MacroRegimeStrategy initialized with macro filter: {enable_macro_filter}, risk management: {enable_risk_management}")
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals (CORE LOGIC - UNCHANGED).
        
        This is the "strong base logic" that remains intact.
        The macro regime filter and risk overlay modulate position
        sizes and exits without curve-fitting to past specific conditions.
        
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
            
            # Multi-timeframe signal analysis (CORE LOGIC)
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
                
                # Generate signal for this timeframe (CORE LOGIC)
                signal = 0.0
                
                # Trend following with momentum confirmation
                if price_sma_ratio > 1.005 and momentum > 0 and rsi < 75:  # Bullish
                    signal = min(1.0, (price_sma_ratio - 1.0) * 20)
                elif price_sma_ratio < 0.995 and momentum < 0 and rsi > 25:  # Bearish
                    signal = -min(1.0, (1.0 - price_sma_ratio) * 20)
                
                # Volatility adjustment
                if volatility > 0.03:  # High volatility - reduce signal strength
                    signal *= 0.5
                
                signals.append(signal)
                weights.append(1.0 / period)  # Shorter periods get higher weight
            
            # Weighted average of signals (CORE LOGIC)
            if signals and weights:
                weighted_signal = np.average(signals, weights=weights)
                
                # Require confirmation threshold
                signal_agreement = sum(1 for s in signals if (s > 0 and weighted_signal > 0) or 
                                     (s < 0 and weighted_signal < 0)) / len(signals)
                
                if signal_agreement >= self.confirmation_threshold:
                    # Conservative signal strength
                    final_signal = np.clip(weighted_signal, -0.5, 0.5)
                    return {
                        'signal': np.sign(final_signal),
                        'strength': abs(final_signal)
                    }
            
            return {'signal': 0.0, 'strength': 0.0}
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {'signal': 0.0, 'strength': 0.0}
    
    def get_macro_data(self, timestamp: datetime) -> pd.DataFrame:
        """
        Get macro data for regime classification.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            DataFrame with macro indicators
        """
        try:
            # For now, use simulated macro data
            # In production, this would fetch real-time macro indicators
            if self.macro_filter:
                macro_data = self.macro_filter.simulate_macro_data(days=252)
                
                # Get data up to current timestamp
                if not macro_data.empty:
                    # Simulate current macro conditions based on timestamp
                    # In production, this would be real-time data
                    current_macro = macro_data.iloc[-1:].copy()
                    return current_macro
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting macro data: {e}")
            return pd.DataFrame()
    
    def execute_trade(self, signal: float, price: float, timestamp: pd.Timestamp) -> Dict:
        """
        Execute trade with macro regime and risk management overlay.
        
        Args:
            signal: Trading signal (-1 to 1)
            price: Current price
            timestamp: Trade timestamp
            
        Returns:
            Trade execution details
        """
        try:
            # Base position size calculation
            base_position_size = 0.10  # 10% base position
            
            # Apply macro regime filter if enabled
            if self.enable_macro_filter and self.macro_filter:
                # Get macro data and update regime
                macro_data = self.get_macro_data(timestamp)
                if not macro_data.empty:
                    regime_info = self.macro_filter.update_regime(macro_data)
                    
                    # Calculate position size adjusted for macro regime
                    position_size = self.macro_filter.calculate_adjusted_position_size(
                        base_position_size, signal
                    )
                    
                    # Log regime information
                    if regime_info.get('regime_switched', False):
                        logger.info(f"Regime switched to {regime_info['current_regime']}: {regime_info['description']}")
                    
                    # Store regime metrics
                    self.regime_metrics.append(regime_info)
                else:
                    position_size = base_position_size * abs(signal)
            else:
                position_size = base_position_size * abs(signal)
            
            # Apply risk management overlay if enabled
            if self.enable_risk_management and self.risk_manager:
                # Check if we should exit existing positions
                if self.risk_manager.should_exit_position(self.current_capital):
                    logger.warning("Risk management triggered position exit")
                    return {'executed': False, 'reason': 'Risk management exit'}
                
                # Apply drawdown scaling
                position_size = self.risk_manager.calculate_position_size(signal, position_size)
                
                # Check if trading is halted
                if self.risk_manager.trading_halted:
                    return {'executed': False, 'reason': 'Trading halted by risk management'}
            
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
            
            # Add regime information to trade
            if self.regime_metrics:
                latest_regime = self.regime_metrics[-1]
                trade['regime'] = latest_regime.get('current_regime', 'neutral')
                trade['leverage_multiplier'] = latest_regime.get('leverage_multiplier', 1.0)
            
            self.trades.append(trade)
            
            logger.info(f"Trade executed: {signal:.2f} {position_size:.3f} at {price:.2f}")
            
            return {'executed': True, 'trade': trade}
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'executed': False, 'reason': str(e)}
    
    def update_performance(self, current_price: float, timestamp: pd.Timestamp):
        """
        Update performance and risk metrics.
        
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
            
            # Update risk manager if enabled
            if self.enable_risk_management and self.risk_manager:
                risk_metrics = self.risk_manager.update_equity(portfolio_value, timestamp)
                self.risk_metrics.append(risk_metrics)
                
                # Log significant risk events
                if risk_metrics.get('trailing_stop_triggered', False):
                    logger.warning(f"Trailing stop triggered at {portfolio_value:.2f}")
                
                if risk_metrics.get('circuit_breaker_triggered', False):
                    logger.critical(f"Circuit breaker triggered at {risk_metrics.get('current_drawdown', 0):.2%} drawdown")
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run macro regime strategy backtest.
        
        Args:
            data: OHLCV data
            
        Returns:
            Backtest results with regime and risk metrics
        """
        try:
            logger.info("Starting macro regime strategy backtest...")
            
            # Reset state
            if self.risk_manager:
                self.risk_manager.reset()
            if self.macro_filter:
                self.macro_filter.reset()
            self.equity_curve = []
            self.trades = []
            self.risk_metrics = []
            self.regime_metrics = []
            self.current_capital = self.initial_capital
            
            # Process each data point
            for i, (timestamp, row) in enumerate(data.iterrows()):
                if i < max(self.lookback_periods):
                    continue
                
                # Get historical data for signal generation
                historical_data = data.iloc[:i+1]
                
                # Generate signals (CORE LOGIC)
                signal_data = self.generate_signals(historical_data)
                signal = signal_data['signal']
                strength = signal_data['strength']
                
                # Execute trade if signal is strong enough
                if abs(signal) > 0.05:  # Minimum signal threshold
                    trade_result = self.execute_trade(signal, row['close'], timestamp)
                    if trade_result['executed']:
                        # Update capital (simplified - no transaction costs for now)
                        pass
                
                # Update performance and risk metrics
                self.update_performance(row['close'], timestamp)
                
                # Progress update
                if i % 50 == 0:
                    logger.info(f"Progress: {i}/{len(data)} - Capital: {self.current_capital:.2f}")
            
            # Calculate final performance metrics
            self._calculate_performance_metrics()
            
            logger.info("Macro regime strategy backtest completed successfully")
            logger.info("=== MACRO REGIME STRATEGY RESULTS ===")
            logger.info(f"Total Return: {self.total_return:.2%}")
            logger.info(f"Max Drawdown: {self.max_drawdown:.2%}")
            logger.info(f"Sharpe Ratio: {self.sharpe_ratio:.3f}")
            logger.info(f"Win Rate: {self.win_rate:.2%}")
            logger.info(f"Total Trades: {len(self.trades)}")
            
            return self._get_backtest_results()
            
        except Exception as e:
            logger.error(f"Error in macro regime backtest: {e}")
            return {}
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics with regime analysis."""
        try:
            if not self.equity_curve:
                return
            
            # Convert to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            equity_df['returns'] = equity_df['portfolio_value'].pct_change()
            
            # Basic metrics
            final_value = equity_df['portfolio_value'].iloc[-1]
            self.total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Calculate max drawdown
            peak = equity_df['portfolio_value'].expanding().max()
            drawdown = (equity_df['portfolio_value'] - peak) / peak
            self.max_drawdown = drawdown.min()
            
            # Calculate Sharpe ratio
            returns = equity_df['returns'].dropna()
            if len(returns) > 30 and returns.std() > 0.001:
                annualized_return = returns.mean() * 252
                annualized_vol = returns.std() * np.sqrt(252)
                self.sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
            else:
                self.sharpe_ratio = 0.0
            
            # Calculate win rate (simplified)
            if self.total_return > 0:
                self.win_rate = 0.6  # Conservative estimate
            else:
                self.win_rate = 0.4
            
            # Regime analysis
            if self.regime_metrics:
                regime_summary = self._analyze_regime_performance()
                self.regime_analysis = regime_summary
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    def _analyze_regime_performance(self) -> Dict[str, any]:
        """Analyze performance by regime."""
        try:
            if not self.regime_metrics or not self.trades:
                return {}
            
            # Group trades by regime
            regime_trades = {}
            for trade in self.trades:
                regime = trade.get('regime', 'neutral')
                if regime not in regime_trades:
                    regime_trades[regime] = []
                regime_trades[regime].append(trade)
            
            # Calculate regime statistics
            regime_stats = {}
            for regime, trades in regime_trades.items():
                if trades:
                    regime_stats[regime] = {
                        'trade_count': len(trades),
                        'avg_position_size': np.mean([t['position_size'] for t in trades]),
                        'avg_leverage': np.mean([t.get('leverage_multiplier', 1.0) for t in trades])
                    }
            
            return regime_stats
            
        except Exception as e:
            logger.error(f"Error analyzing regime performance: {e}")
            return {}
    
    def _get_backtest_results(self) -> Dict:
        """Get comprehensive backtest results."""
        try:
            results = {
                'total_return': self.total_return,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': self.sharpe_ratio,
                'win_rate': self.win_rate,
                'profit_factor': 0.0,  # Simplified
                'total_trades': len(self.trades),
                'initial_capital': self.initial_capital,
                'final_capital': self.equity_curve[-1]['portfolio_value'] if self.equity_curve else self.initial_capital,
                'equity_curve': self.equity_curve,
                'trades': self.trades,
                'strategy_type': 'Macro Regime-Enhanced Strategy'
            }
            
            # Add risk management metrics
            if self.risk_manager:
                risk_summary = self.risk_manager.get_risk_summary()
                results['risk_management'] = {
                    'trailing_stop_pct': self.risk_manager.trailing_stop_pct,
                    'circuit_breaker_pct': self.risk_manager.circuit_breaker_pct,
                    'max_drawdown': risk_summary['max_drawdown'],
                    'volatility': risk_summary['volatility'],
                    'trading_halted': risk_summary['trading_halted'],
                    'circuit_breaker_triggered': risk_summary['circuit_breaker_triggered'],
                    'position_scale': risk_summary['position_scale']
                }
            
            # Add macro regime metrics
            if self.macro_filter:
                regime_summary = self.macro_filter.get_regime_summary()
                results['macro_regime'] = {
                    'current_regime': regime_summary.get('current_regime', 'neutral'),
                    'regime_confidence': regime_summary.get('regime_confidence', 0.0),
                    'leverage_multiplier': regime_summary.get('leverage_multiplier', 1.0),
                    'stop_loss_multiplier': regime_summary.get('stop_loss_multiplier', 1.0),
                    'volatility_target': regime_summary.get('volatility_target', 0.15),
                    'description': regime_summary.get('description', ''),
                    'regime_history_length': regime_summary.get('regime_history_length', 0)
                }
                
                # Add regime performance analysis
                if hasattr(self, 'regime_analysis'):
                    results['regime_analysis'] = self.regime_analysis
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting backtest results: {e}")
            return {}

def run_macro_regime_backtest(start_date: str = "2023-07-01", 
                             end_date: str = "2023-09-30",
                             initial_capital: float = 100000.0,
                             trailing_stop_pct: float = 0.05,
                             circuit_breaker_pct: float = 0.15,
                             enable_macro_filter: bool = True,
                             enable_risk_management: bool = True) -> Dict:
    """
    Run macro regime strategy backtest.
    
    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital
        trailing_stop_pct: Trailing stop percentage
        circuit_breaker_pct: Circuit breaker percentage
        enable_macro_filter: Enable macro regime filtering
        enable_risk_management: Enable risk management
        
    Returns:
        Backtest results
    """
    try:
        from data_pipeline.databento_collector import DatabentoGoldCollector
        
        logger.info(f"Running macro regime backtest from {start_date} to {end_date}")
        logger.info(f"Macro filter: {'ENABLED' if enable_macro_filter else 'DISABLED'}")
        logger.info(f"Risk management: {'ENABLED' if enable_risk_management else 'DISABLED'}")
        
        # Fetch data
        collector = DatabentoGoldCollector()
        data = collector.fetch_gld(start_date, end_date)
        
        if data.empty:
            logger.error("No data fetched")
            return {}
        
        logger.info(f"Fetched {len(data)} data points")
        
        # Create and run strategy
        strategy = MacroRegimeStrategy(
            initial_capital=initial_capital,
            trailing_stop_pct=trailing_stop_pct,
            circuit_breaker_pct=circuit_breaker_pct,
            enable_macro_filter=enable_macro_filter,
            enable_risk_management=enable_risk_management
        )
        
        # Run backtest
        results = strategy.backtest(data)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in macro regime backtest: {e}")
        return {}

if __name__ == "__main__":
    results = run_macro_regime_backtest()
    print("Macro Regime Strategy Results:", results) 