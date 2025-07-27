"""
Risk-Enhanced Gold Trading Strategy
==================================

This strategy integrates volatility-targeted position sizing and drawdown protection
with the ultra-aggressive signal generation to achieve better risk-adjusted returns.

Key Features:
- Ultra-aggressive signal generation (multiple technical indicators)
- Volatility-targeted position sizing using ATR and realized volatility
- Drawdown protection with soft (-10%) and hard (-15%) stops
- Dynamic position scaling based on market conditions
- Comprehensive risk monitoring and reporting
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from gold_algo.shared_utilities import calculate_technical_indicators
from risk_management.volatility_position_manager import RiskOverlay
from data_pipeline.databento_collector import DatabentoGoldCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskEnhancedStrategy:
    """
    Risk-enhanced gold trading strategy with volatility targeting and drawdown protection.
    
    This strategy combines:
    1. Ultra-aggressive signal generation from multiple sources
    2. Volatility-targeted position sizing
    3. Comprehensive drawdown protection
    4. Dynamic risk management
    """
    
    def __init__(self, 
                 target_volatility: float = 0.02,
                 atr_period: int = 15,
                 vol_window: int = 20,
                 soft_drawdown_limit: float = 0.10,
                 hard_drawdown_limit: float = 0.05,  # Reduced to 5% for maximum protection
                 initial_capital: float = 100000.0):
        
        # Risk management parameters
        self.target_volatility = target_volatility
        self.atr_period = atr_period
        self.vol_window = vol_window
        self.soft_drawdown_limit = soft_drawdown_limit
        self.hard_drawdown_limit = hard_drawdown_limit
        
        # Capital and performance tracking
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
        # Risk overlay
        self.risk_overlay = RiskOverlay(
            target_volatility=target_volatility,
            atr_period=atr_period,
            vol_window=vol_window,
            soft_drawdown_limit=soft_drawdown_limit,
            hard_drawdown_limit=hard_drawdown_limit
        )
        
        # Signal generation parameters (ultra-aggressive)
        self.long_threshold = 0.01
        self.short_threshold = -0.01
        self.exit_threshold = 0.002
        
        # Performance metrics
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        
        logger.info(f"RiskEnhancedStrategy initialized with target volatility: {target_volatility}")
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate ultra-aggressive trading signals from multiple sources.
        
        Args:
            data: OHLCV price data
            
        Returns:
            Dictionary with signal components and combined signal
        """
        try:
            # Calculate technical indicators
            indicators = calculate_technical_indicators(data)
            
            # Momentum signals (25% weight)
            momentum_signals = []
            if 'rsi' in indicators:
                rsi = indicators['rsi'].iloc[-1]
                momentum_signals.append(1.0 if rsi < 30 else -1.0 if rsi > 70 else 0)
            
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'].iloc[-1]
                macd_signal = indicators['macd_signal'].iloc[-1]
                momentum_signals.append(1.0 if macd > macd_signal else -1.0)
            
            if 'stoch_k' in indicators:
                stoch_k = indicators['stoch_k'].iloc[-1]
                momentum_signals.append(1.0 if stoch_k < 20 else -1.0 if stoch_k > 80 else 0)
            
            momentum_signal = np.mean(momentum_signals) if momentum_signals else 0.0
            
            # Mean reversion signals (25% weight)
            mean_reversion_signals = []
            if 'bb_upper' in indicators and 'bb_lower' in indicators and 'close' in data:
                close = data['close'].iloc[-1]
                bb_upper = indicators['bb_upper'].iloc[-1]
                bb_lower = indicators['bb_lower'].iloc[-1]
                
                if close < bb_lower:
                    mean_reversion_signals.append(1.0)  # Oversold
                elif close > bb_upper:
                    mean_reversion_signals.append(-1.0)  # Overbought
                else:
                    mean_reversion_signals.append(0.0)
            
            if 'sma_20' in indicators and 'close' in data:
                close = data['close'].iloc[-1]
                sma_20 = indicators['sma_20'].iloc[-1]
                mean_reversion_signals.append(1.0 if close < sma_20 * 0.98 else -1.0 if close > sma_20 * 1.02 else 0.0)
            
            mean_reversion_signal = np.mean(mean_reversion_signals) if mean_reversion_signals else 0.0
            
            # Volume signals (25% weight)
            volume_signals = []
            if 'volume' in data and 'volume_sma' in indicators:
                volume = data['volume'].iloc[-1]
                volume_sma = indicators['volume_sma'].iloc[-1]
                volume_signals.append(1.0 if volume > volume_sma * 1.5 else -1.0 if volume < volume_sma * 0.5 else 0.0)
            
            if 'obv' in indicators:
                obv = indicators['obv'].iloc[-1]
                obv_sma = indicators['obv_sma'].iloc[-1] if 'obv_sma' in indicators else obv
                volume_signals.append(1.0 if obv > obv_sma else -1.0)
            
            volume_signal = np.mean(volume_signals) if volume_signals else 0.0
            
            # Technical signals (25% weight)
            technical_signals = []
            if 'adx' in indicators:
                adx = indicators['adx'].iloc[-1]
                technical_signals.append(1.0 if adx > 25 else 0.0)  # Strong trend
            
            if 'cci' in indicators:
                cci = indicators['cci'].iloc[-1]
                technical_signals.append(1.0 if cci < -100 else -1.0 if cci > 100 else 0.0)
            
            if 'williams_r' in indicators:
                williams_r = indicators['williams_r'].iloc[-1]
                technical_signals.append(1.0 if williams_r < -80 else -1.0 if williams_r > -20 else 0.0)
            
            technical_signal = np.mean(technical_signals) if technical_signals else 0.0
            
            # Combine signals with equal weights
            combined_signal = (
                0.25 * momentum_signal +
                0.25 * mean_reversion_signal +
                0.25 * volume_signal +
                0.25 * technical_signal
            )
            
            # Apply ultra-aggressive thresholds
            if combined_signal > self.long_threshold:
                final_signal = 1.0
            elif combined_signal < self.short_threshold:
                final_signal = -1.0
            else:
                final_signal = 0.0
            
            return {
                'momentum_signal': momentum_signal,
                'mean_reversion_signal': mean_reversion_signal,
                'volume_signal': volume_signal,
                'technical_signal': technical_signal,
                'combined_signal': combined_signal,
                'final_signal': final_signal
            }
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {
                'momentum_signal': 0.0,
                'mean_reversion_signal': 0.0,
                'volume_signal': 0.0,
                'technical_signal': 0.0,
                'combined_signal': 0.0,
                'final_signal': 0.0
            }
    
    def execute_trade(self, signal: float, price: float, timestamp: pd.Timestamp) -> Dict:
        """
        Execute trade with risk management overlay.
        
        Args:
            signal: Trading signal (-1, 0, 1)
            price: Current price
            timestamp: Trade timestamp
            
        Returns:
            Trade execution details
        """
        try:
            # Apply risk controls
            risk_adjusted_position = self.risk_overlay.apply_risk_controls(
                signal=signal,
                price_data=self.current_data,
                current_nav=self.current_capital
            )
            
            position_size = risk_adjusted_position['position_size']
            position_direction = risk_adjusted_position['position_direction']
            
            # Execute trade if position size > 0
            if position_size > 0 and position_direction != 0:
                # Calculate trade details
                position_value = position_size * self.current_capital
                shares = position_value / price
                
                # Record trade
                trade = {
                    'timestamp': timestamp,
                    'signal': signal,
                    'price': price,
                    'position_size': position_size,
                    'position_direction': position_direction,
                    'position_value': position_value,
                    'shares': shares,
                    'capital': self.current_capital,
                    'drawdown': risk_adjusted_position['current_drawdown'],
                    'volatility_measure': risk_adjusted_position['volatility_measure'],
                    'atr': risk_adjusted_position['atr'],
                    'realized_volatility': risk_adjusted_position['realized_volatility'],
                    'vol_target_scale': risk_adjusted_position['vol_target_scale'],
                    'expected_daily_risk': risk_adjusted_position['expected_daily_risk'],
                    'soft_stop_triggered': risk_adjusted_position['soft_stop_triggered'],
                    'hard_stop_triggered': risk_adjusted_position['hard_stop_triggered'],
                    'recovery_mode': risk_adjusted_position['recovery_mode']
                }
                
                self.trades.append(trade)
                
                # Update position
                self.positions.append({
                    'timestamp': timestamp,
                    'price': price,
                    'shares': shares,
                    'direction': position_direction,
                    'value': position_value
                })
                
                logger.info(f"Trade executed: {position_direction} {position_size:.3f} at {price:.2f}")
                
                return trade
            else:
                logger.info(f"No trade executed: signal={signal}, position_size={position_size}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def update_performance(self, current_price: float, timestamp: pd.Timestamp):
        """
        Update performance metrics and equity curve.
        
        Args:
            current_price: Current asset price
            timestamp: Current timestamp
        """
        try:
            # Calculate current portfolio value
            portfolio_value = self.current_capital
            
            # Add unrealized P&L from open positions
            for position in self.positions:
                if position['direction'] == 1:  # Long position
                    unrealized_pnl = (current_price - position['price']) * position['shares']
                else:  # Short position
                    unrealized_pnl = (position['price'] - current_price) * position['shares']
                
                portfolio_value += unrealized_pnl
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'capital': self.current_capital,
                'unrealized_pnl': portfolio_value - self.current_capital,
                'drawdown': self.risk_overlay.get_risk_metrics()['current_drawdown']
            })
            
            # Update current capital
            self.current_capital = portfolio_value
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run backtest with risk management overlay.
        
        Args:
            data: OHLCV price data
            
        Returns:
            Backtest results with performance metrics
        """
        try:
            logger.info("Starting risk-enhanced strategy backtest...")
            
            # Reset state
            self.current_capital = self.initial_capital
            self.positions = []
            self.trades = []
            self.equity_curve = []
            self.risk_overlay.reset()
            
            # Initialize performance tracking
            self.total_return = 0.0
            self.max_drawdown = 0.0
            self.sharpe_ratio = 0.0
            self.win_rate = 0.0
            self.profit_factor = 0.0
            
            # Run backtest
            for i in range(len(data)):
                current_data = data.iloc[:i+1]
                current_price = data['close'].iloc[i]
                timestamp = data.index[i]
                
                # Store current data for risk overlay
                self.current_data = current_data
                
                # Generate signals
                signals = self.generate_signals(current_data)
                signal = signals['final_signal']
                
                # Execute trade
                trade = self.execute_trade(signal, current_price, timestamp)
                
                # Update performance
                self.update_performance(current_price, timestamp)
                
                # Log progress
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(data)} - Capital: {self.current_capital:.2f}")
            
            # Calculate final performance metrics
            self._calculate_performance_metrics()
            
            logger.info("Backtest completed successfully")
            return self._get_backtest_results()
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {}
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
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
            
            # Sharpe ratio (proper calculation)
            returns = equity_df['returns'].dropna()
            if len(returns) > 0:
                # Calculate annualized return and volatility
                total_return = (1 + returns).prod() - 1
                annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
                volatility = returns.std() * np.sqrt(252)
                risk_free_rate = 0.02  # 2% annual risk-free rate
                
                # Sharpe ratio = (Annualized Return - Risk Free Rate) / Volatility
                self.sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Win rate and profit factor
            if self.trades:
                profitable_trades = [t for t in self.trades if t.get('unrealized_pnl', 0) > 0]
                self.win_rate = len(profitable_trades) / len(self.trades) if self.trades else 0
                
                # Calculate profit factor
                gross_profit = sum([t.get('unrealized_pnl', 0) for t in profitable_trades])
                gross_loss = sum([abs(t.get('unrealized_pnl', 0)) for t in self.trades if t.get('unrealized_pnl', 0) < 0])
                self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    def _get_backtest_results(self) -> Dict:
        """Get comprehensive backtest results."""
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
            'risk_metrics': self.risk_overlay.get_risk_metrics()
        }
    
    def get_risk_summary(self) -> Dict:
        """Get current risk management summary."""
        return self.risk_overlay.get_risk_metrics()


def run_risk_enhanced_backtest(start_date: str = "2023-07-01", 
                              end_date: str = "2023-09-30",
                              initial_capital: float = 100000.0) -> Dict:
    """
    Run comprehensive backtest of the risk-enhanced strategy.
    
    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Initial capital
        
    Returns:
        Backtest results
    """
    try:
        logger.info(f"Running risk-enhanced backtest from {start_date} to {end_date}")
        
        # Fetch data
        collector = DatabentoGoldCollector()
        data = collector.fetch_gld(start_date, end_date)
        
        if data is None or len(data) == 0:
            logger.error("No data fetched for backtest")
            return {}
        
        logger.info(f"Fetched {len(data)} data points")
        
        # Initialize strategy
        strategy = RiskEnhancedStrategy(
            target_volatility=0.02,
            atr_period=15,
            vol_window=20,
            soft_drawdown_limit=0.10,
            hard_drawdown_limit=0.05,  # Reduced from 15% to 12%
            initial_capital=initial_capital
        )
        
        # Run backtest
        results = strategy.backtest(data)
        
        # Log results
        logger.info("=== RISK-ENHANCED STRATEGY RESULTS ===")
        logger.info(f"Total Return: {results.get('total_return', 0):.2%}")
        logger.info(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
        logger.info(f"Win Rate: {results.get('win_rate', 0):.2%}")
        logger.info(f"Profit Factor: {results.get('profit_factor', 0):.3f}")
        logger.info(f"Total Trades: {results.get('total_trades', 0)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in risk-enhanced backtest: {e}")
        return {}


if __name__ == "__main__":
    # Run backtest
    results = run_risk_enhanced_backtest()
    
    if results:
        print("\n=== RISK-ENHANCED STRATEGY BACKTEST RESULTS ===")
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Profit Factor: {results.get('profit_factor', 0):.3f}")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Initial Capital: ${results.get('initial_capital', 0):,.2f}")
        print(f"Final Capital: ${results.get('final_capital', 0):,.2f}")
        
        # Risk metrics
        risk_metrics = results.get('risk_metrics', {})
        print(f"\n=== RISK METRICS ===")
        print(f"Current Drawdown: {risk_metrics.get('current_drawdown', 0):.2%}")
        print(f"High Water Mark: {risk_metrics.get('high_water_mark', 1.0):.3f}")
        print(f"Position Scale: {risk_metrics.get('position_scale', 1.0):.3f}")
        print(f"Soft Stop Triggered: {risk_metrics.get('soft_stop_triggered', False)}")
        print(f"Hard Stop Triggered: {risk_metrics.get('hard_stop_triggered', False)}")
        print(f"Recovery Mode: {risk_metrics.get('recovery_mode', False)}")
        print(f"Average ATR: {risk_metrics.get('avg_atr', 0):.4f}")
        print(f"Average Volatility: {risk_metrics.get('avg_volatility', 0):.2%}") 