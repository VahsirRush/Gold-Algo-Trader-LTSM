#!/usr/bin/env python3
"""
Optimized Performance Strategy
=============================

This strategy addresses the performance degradation issues identified in the analysis:
1. Excessive risk management constraints
2. Over-complicated macro regime filtering
3. Strategy complexity creep
4. Signal quality degradation

Key optimizations:
- Simplified risk management parameters
- Reduced macro regime complexity
- Increased position sizes and leverage
- Streamlined signal generation
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

class OptimizedPerformanceStrategy(BaseStrategy):
    """
    Optimized performance strategy with simplified parameters and reduced complexity.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 # OPTIMIZED RISK MANAGEMENT PARAMETERS
                 trailing_stop_pct: float = 0.08,  # Increased from 0.05
                 circuit_breaker_pct: float = 0.25,  # Increased from 0.15
                 max_position_size: float = 0.15,  # Increased from 0.10
                 confirmation_threshold: float = 0.3,  # Reduced from 0.4
                 # OPTIMIZED MACRO REGIME PARAMETERS
                 regime_persistence_days: int = 3,  # Reduced from 5
                 regime_confidence_threshold: float = 0.6,  # Reduced from 0.7
                 min_leverage: float = 0.5,  # Increased from 0.3
                 max_leverage: float = 8.0,  # Increased from 6.0
                 max_leverage_multiplier: float = 2.0,  # Increased from 1.5
                 enable_macro_filter: bool = True,
                 enable_risk_management: bool = True):
        
        super().__init__()
        
        # Core strategy parameters
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.confirmation_threshold = confirmation_threshold
        self.max_position_size = max_position_size
        
        # Risk management
        self.enable_risk_management = enable_risk_management
        if enable_risk_management:
            self.risk_manager = DrawdownRiskManager(
                initial_capital=initial_capital,
                trailing_stop_pct=trailing_stop_pct,
                circuit_breaker_pct=circuit_breaker_pct,
                max_position_size=max_position_size,
                min_position_size=0.01,
                drawdown_scaling=True
            )
        
        # Macro regime filter
        self.enable_macro_filter = enable_macro_filter
        if enable_macro_filter:
            self.macro_filter = MacroRegimeFilter(
                regime_persistence_days=regime_persistence_days,
                max_leverage=max_leverage,
                min_leverage=min_leverage,
                regime_confidence_threshold=regime_confidence_threshold,
                use_markov_model=True
            )
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        
        # Strategy state
        self.current_position = 0.0
        self.current_signal = 0.0
        self.trade_count = 0
        
        logger.info(f"OptimizedPerformanceStrategy initialized with:")
        logger.info(f"  - Trailing stop: {trailing_stop_pct:.1%}")
        logger.info(f"  - Circuit breaker: {circuit_breaker_pct:.1%}")
        logger.info(f"  - Max position size: {max_position_size:.1%}")
        logger.info(f"  - Max leverage: {max_leverage:.1f}x")
        logger.info(f"  - Macro filter: {'ENABLED' if enable_macro_filter else 'DISABLED'}")
        logger.info(f"  - Risk management: {'ENABLED' if enable_risk_management else 'DISABLED'}")
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals with optimized parameters.
        """
        if data.empty:
            return {'signal': 0.0, 'strength': 0.0}
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(data)
        
        # Core signal components (simplified and optimized)
        signals = []
        
        # 1. Price momentum (weighted more heavily)
        if 'sma_20' in indicators and 'sma_50' in indicators:
            price_sma_ratio = data['close'].iloc[-1] / indicators['sma_20'].iloc[-1]
            if price_sma_ratio > 1.005:  # Reduced threshold from 1.02
                signals.append(1.0 * 20)  # Increased weight from 10
            elif price_sma_ratio < 0.995:  # Reduced threshold from 0.98
                signals.append(-1.0 * 20)  # Increased weight from 10
        
        # 2. RSI momentum (simplified)
        if 'rsi' in indicators:
            rsi = indicators['rsi'].iloc[-1]
            if rsi > 70:
                signals.append(-0.5)  # Bearish signal
            elif rsi < 30:
                signals.append(0.5)   # Bullish signal
        
        # 3. MACD momentum (simplified)
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            if macd > macd_signal:
                signals.append(0.3)
            else:
                signals.append(-0.3)
        
        # 4. Volume confirmation (simplified)
        if 'volume_sma' in indicators:
            volume_ratio = data['volume'].iloc[-1] / indicators['volume_sma'].iloc[-1]
            if volume_ratio > 1.2:
                signals.append(0.2)
            elif volume_ratio < 0.8:
                signals.append(-0.2)
        
        # Calculate composite signal
        if signals:
            composite_signal = np.mean(signals)
            signal_strength = abs(composite_signal)
            
            # Apply confirmation threshold (reduced for more activity)
            if signal_strength >= self.confirmation_threshold:
                return {
                    'signal': np.sign(composite_signal),
                    'strength': signal_strength
                }
        
        return {'signal': 0.0, 'strength': 0.0}
    
    def get_macro_data(self, timestamp) -> pd.DataFrame:
        """
        Simulate macro data for regime detection.
        """
        # Simplified macro data simulation
        if hasattr(timestamp, 'timestamp'):
            seed = int(timestamp.timestamp())
        else:
            seed = hash(str(timestamp)) % 1000000
        
        np.random.seed(seed)
        
        macro_data = pd.DataFrame({
            'treasury_yields': np.random.normal(2.5, 0.5, 1),
            'inflation_breakevens': np.random.normal(2.0, 0.3, 1),
            'usd_index': np.random.normal(100, 5, 1),
            'vix': np.random.normal(20, 5, 1),
            'credit_spreads': np.random.normal(100, 20, 1),
            'equity_returns': np.random.normal(0.001, 0.02, 1)
        })
        
        return macro_data
    
    def execute_trade(self, signal: float, price: float, timestamp: pd.Timestamp) -> Dict:
        """
        Execute trade with optimized position sizing.
        """
        if abs(signal) < 0.05:  # Reduced threshold from 0.1
            return {'action': 'hold', 'position': self.current_position}
        
        # Get macro regime adjustment if enabled
        leverage_multiplier = 1.0
        if self.enable_macro_filter:
            macro_data = self.get_macro_data(timestamp)
            regime_info = self.macro_filter.update_regime(macro_data)
            leverage_multiplier = regime_info.get('leverage_multiplier', 1.0)
        
        # Calculate base position size (increased)
        base_position_size = self.max_position_size * abs(signal)
        
        # Apply leverage multiplier
        adjusted_position_size = base_position_size * leverage_multiplier
        
        # Apply risk management if enabled
        if self.enable_risk_management:
            adjusted_position_size = self.risk_manager.calculate_position_size(
                signal, adjusted_position_size
            )
        
        # Cap position size
        adjusted_position_size = min(adjusted_position_size, self.max_position_size)
        
        # Execute trade
        if signal > 0 and self.current_position <= 0:
            # Buy signal
            self.current_position = adjusted_position_size
            action = 'buy'
        elif signal < 0 and self.current_position >= 0:
            # Sell signal
            self.current_position = -adjusted_position_size
            action = 'sell'
        else:
            action = 'hold'
        
        # Record trade
        if action != 'hold':
            self.trade_count += 1
            self.trades.append({
                'timestamp': timestamp,
                'action': action,
                'price': price,
                'position': self.current_position,
                'signal': signal,
                'leverage_multiplier': leverage_multiplier
            })
            
            logger.info(f"Trade executed: {signal:.2f} {adjusted_position_size:.3f} at {price:.2f}")
        
        return {
            'action': action,
            'position': self.current_position,
            'leverage_multiplier': leverage_multiplier
        }
    
    def update_performance(self, current_price: float, timestamp: pd.Timestamp):
        """
        Update performance metrics.
        """
        # Calculate P&L
        if self.current_position != 0:
            price_change = (current_price - self.last_price) / self.last_price
            pnl = self.current_position * price_change * self.current_capital
            self.current_capital += pnl
        
        self.last_price = current_price
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.current_capital,
            'position': self.current_position
        })
        
        # Update peak equity and drawdown
        if self.current_capital > self.peak_equity:
            self.peak_equity = self.current_capital
        
        self.current_drawdown = (self.peak_equity - self.current_capital) / self.peak_equity
        
        # Update risk manager if enabled
        if self.enable_risk_management:
            risk_update = self.risk_manager.update_equity(self.current_capital, timestamp)
            
            # Check for risk management actions
            if risk_update.get('trailing_stop_triggered', False):
                logger.warning(f"Trailing stop triggered at {self.current_capital:.2f}")
                self.current_position = 0.0  # Close position
            
            if risk_update.get('circuit_breaker_triggered', False):
                logger.critical(f"Circuit breaker triggered at {self.current_drawdown:.2%} drawdown")
                self.current_position = 0.0  # Close position
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run optimized backtest.
        """
        logger.info("Starting optimized performance strategy backtest...")
        
        # Reset state
        self.current_capital = self.initial_capital
        self.peak_equity = self.initial_capital
        self.current_drawdown = 0.0
        self.current_position = 0.0
        self.trades = []
        self.equity_curve = []
        self.trade_count = 0
        
        if self.enable_risk_management:
            self.risk_manager.reset()
        
        if self.enable_macro_filter:
            self.macro_filter.reset()
        
        # Initialize
        self.last_price = data['close'].iloc[0]
        
        # Main backtest loop
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']
            
            # Generate signals
            signal_data = self.generate_signals(data.iloc[:i+1])
            signal = signal_data['signal']
            strength = signal_data['strength']
            
            # Execute trade
            trade_result = self.execute_trade(signal, current_price, timestamp)
            
            # Update performance
            self.update_performance(current_price, timestamp)
            
            # Progress logging
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i+1}/{len(data)} - Capital: {self.current_capital:.2f}")
        
        # Calculate final performance metrics
        self._calculate_performance_metrics()
        
        logger.info("Optimized performance strategy backtest completed successfully")
        logger.info("=== OPTIMIZED PERFORMANCE STRATEGY RESULTS ===")
        logger.info(f"Total Return: {self.total_return:.2%}")
        logger.info(f"Max Drawdown: {self.max_drawdown:.2%}")
        logger.info(f"Sharpe Ratio: {self.sharpe_ratio:.3f}")
        logger.info(f"Win Rate: {self.win_rate:.1%}")
        logger.info(f"Total Trades: {self.trade_count}")
        
        return self._get_backtest_results()
    
    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics.
        """
        if not self.equity_curve:
            self.total_return = 0.0
            self.max_drawdown = 0.0
            self.sharpe_ratio = 0.0
            self.win_rate = 0.0
            return
        
        # Calculate returns
        equity_series = pd.DataFrame(self.equity_curve)
        equity_series.set_index('timestamp', inplace=True)
        
        returns = equity_series['equity'].pct_change().dropna()
        
        # Total return
        self.total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Max drawdown
        peak = equity_series['equity'].expanding().max()
        drawdown = (equity_series['equity'] - peak) / peak
        self.max_drawdown = drawdown.min()
        
        # Sharpe ratio
        if returns.std() > 0:
            self.sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        else:
            self.sharpe_ratio = 0.0
        
        # Win rate
        if self.trades:
            winning_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
            self.win_rate = winning_trades / len(self.trades)
        else:
            self.win_rate = 0.0
    
    def _get_backtest_results(self) -> Dict:
        """
        Get comprehensive backtest results.
        """
        return {
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'total_trades': self.trade_count,
            'final_capital': self.current_capital,
            'peak_equity': self.peak_equity,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }

def run_optimized_performance_backtest(start_date: str = "2023-01-01", 
                                      end_date: str = "2023-12-31",
                                      initial_capital: float = 100000.0,
                                      enable_macro_filter: bool = True,
                                      enable_risk_management: bool = True) -> Dict:
    """
    Run optimized performance strategy backtest.
    """
    try:
        from data_pipeline.databento_collector import DatabentoGoldCollector
        
        # Fetch data
        collector = DatabentoGoldCollector()
        data = collector.fetch_gld(start_date, end_date)
        
        if data.empty:
            logger.error("No data fetched")
            return {}
        
        logger.info(f"Fetched {len(data)} data points")
        
        # Create and run strategy
        strategy = OptimizedPerformanceStrategy(
            initial_capital=initial_capital,
            enable_macro_filter=enable_macro_filter,
            enable_risk_management=enable_risk_management
        )
        
        results = strategy.backtest(data)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in optimized performance backtest: {e}")
        return {}

if __name__ == "__main__":
    # Test the optimized strategy
    results = run_optimized_performance_backtest()
    print(f"Optimized Strategy Results: {results}") 