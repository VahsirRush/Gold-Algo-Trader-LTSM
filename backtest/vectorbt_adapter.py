"""
VectorBT backtesting adapter for gold trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

try:
    import vectorbt as vbt
except ImportError:
    vbt = None

class VectorBTBacktester:
    """VectorBT-based backtesting engine."""
    
    def __init__(self):
        """Initialize VectorBT backtester."""
        self.logger = logging.getLogger(__name__)
        
        if vbt is None:
            self.logger.warning("VectorBT not available, using simplified backtesting")
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series, 
                    confidence: pd.Series, initial_capital: float = 100000,
                    commission: float = 0.001, slippage: float = 0.0005) -> Dict[str, Any]:
        """
        Run backtest using VectorBT.
        
        Args:
            data: Price data with OHLCV
            signals: Trading signals (-1, 0, 1)
            confidence: Signal confidence scores
            initial_capital: Initial capital
            commission: Commission rate
            slippage: Slippage rate
            
        Returns:
            Dictionary with backtest results
        """
        if vbt is None:
            return self._run_simplified_backtest(data, signals, confidence, initial_capital, commission, slippage)
        
        try:
            return self._run_vectorbt_backtest(data, signals, confidence, initial_capital, commission, slippage)
        except Exception as e:
            self.logger.error(f"VectorBT backtest failed: {e}")
            return self._run_simplified_backtest(data, signals, confidence, initial_capital, commission, slippage)
    
    def _run_vectorbt_backtest(self, data: pd.DataFrame, signals: pd.Series, 
                              confidence: pd.Series, initial_capital: float,
                              commission: float, slippage: float) -> Dict[str, Any]:
        """Run backtest using VectorBT."""
        
        # Prepare data
        close_prices = data['Close']
        
        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=close_prices,
            entries=signals == 1,
            exits=signals == -1,
            init_cash=initial_capital,
            fees=commission,
            slippage=slippage,
            freq='1D'
        )
        
        # Calculate metrics
        total_return = portfolio.total_return()
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        total_trades = portfolio.count_trades()
        
        # Get equity curve
        equity_curve = portfolio.value()
        
        # Calculate additional metrics
        daily_returns = portfolio.returns()
        volatility = daily_returns.std() * np.sqrt(252)
        annual_return = total_return * (252 / len(data))
        
        # Calculate win rate
        trades = portfolio.trades
        if len(trades) > 0:
            winning_trades = trades[trades['PnL'] > 0]
            win_rate = len(winning_trades) / len(trades)
            avg_win = winning_trades['PnL'].mean() if len(winning_trades) > 0 else 0
            losing_trades = trades[trades['PnL'] < 0]
            avg_loss = losing_trades['PnL'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'num_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': equity_curve,
            'daily_returns': daily_returns,
            'final_value': equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
        }
    
    def _run_simplified_backtest(self, data: pd.DataFrame, signals: pd.Series, 
                                confidence: pd.Series, initial_capital: float,
                                commission: float, slippage: float) -> Dict[str, Any]:
        """Run simplified backtest without VectorBT."""
        
        position = 0
        capital = initial_capital
        trades = []
        equity_curve = []
        
        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            current_date = data.index[i]
            
            # Check for signal changes
            if signals.iloc[i] != signals.iloc[i-1]:
                if signals.iloc[i] == 1:  # Buy signal
                    if position == 0:
                        # Position size based on confidence
                        conf = confidence.iloc[i] if not pd.isna(confidence.iloc[i]) else 0.5
                        position_size = capital * min(conf, 1.0)
                        position = position_size / current_price
                        capital -= position_size
                        trades.append({
                            'date': current_date,
                            'action': 'BUY',
                            'price': current_price,
                            'position': position,
                            'confidence': conf
                        })
                elif signals.iloc[i] == -1:  # Sell signal
                    if position > 0:
                        # Apply slippage and commission
                        sell_price = current_price * (1 - slippage)
                        capital = position * sell_price * (1 - commission)
                        position = 0
                        trades.append({
                            'date': current_date,
                            'action': 'SELL',
                            'price': sell_price,
                            'capital': capital
                        })
            
            # Calculate current equity
            current_equity = capital + (position * current_price)
            equity_curve.append(current_equity)
        
        # Final position
        if position > 0:
            final_price = data['Close'].iloc[-1] * (1 - slippage)
            capital = position * final_price * (1 - commission)
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital
        annual_return = total_return * (252 / len(data))
        
        # Calculate Sharpe ratio
        equity_series = pd.Series(equity_curve)
        daily_returns = equity_series.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        if len(trades) > 0:
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                trade_returns = []
                for i in range(min(len(buy_trades), len(sell_trades))):
                    buy_price = buy_trades[i]['price']
                    sell_price = sell_trades[i]['price']
                    trade_return = (sell_price - buy_price) / buy_price
                    trade_returns.append(trade_return)
                
                winning_trades = [r for r in trade_returns if r > 0]
                win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
                avg_win = np.mean(winning_trades) if winning_trades else 0
                losing_trades = [r for r in trade_returns if r < 0]
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': equity_curve,
            'daily_returns': daily_returns.tolist(),
            'final_value': capital
        }
    
    def run_walk_forward_analysis(self, data: pd.DataFrame, signals: pd.Series,
                                 confidence: pd.Series, initial_capital: float = 100000,
                                 train_period: int = 252, test_period: int = 63,
                                 step_size: int = 21) -> Dict[str, Any]:
        """
        Run walk-forward analysis.
        
        Args:
            data: Price data
            signals: Trading signals
            confidence: Signal confidence
            initial_capital: Initial capital
            train_period: Training period in days
            test_period: Test period in days
            step_size: Step size for forward walk
            
        Returns:
            Dictionary with walk-forward results
        """
        results = []
        
        for i in range(train_period, len(data) - test_period, step_size):
            # Training period
            train_start = i - train_period
            train_end = i
            train_data = data.iloc[train_start:train_end]
            train_signals = signals.iloc[train_start:train_end]
            train_confidence = confidence.iloc[train_start:train_end]
            
            # Test period
            test_start = i
            test_end = min(i + test_period, len(data))
            test_data = data.iloc[test_start:test_end]
            test_signals = signals.iloc[test_start:test_end]
            test_confidence = confidence.iloc[test_start:test_end]
            
            # Run backtest on test period
            test_result = self.run_backtest(test_data, test_signals, test_confidence, initial_capital)
            
            results.append({
                'period_start': test_data.index[0],
                'period_end': test_data.index[-1],
                'total_return': test_result['total_return'],
                'sharpe_ratio': test_result['sharpe_ratio'],
                'max_drawdown': test_result['max_drawdown'],
                'num_trades': test_result['num_trades']
            })
        
        # Aggregate results
        if results:
            returns = [r['total_return'] for r in results]
            sharpe_ratios = [r['sharpe_ratio'] for r in results]
            max_drawdowns = [r['max_drawdown'] for r in results]
            
            return {
                'num_periods': len(results),
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'avg_sharpe': np.mean(sharpe_ratios),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'positive_periods': sum(1 for r in returns if r > 0),
                'period_results': results
            }
        else:
            return {
                'num_periods': 0,
                'avg_return': 0,
                'std_return': 0,
                'avg_sharpe': 0,
                'avg_max_drawdown': 0,
                'positive_periods': 0,
                'period_results': []
            }
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary of risk metrics
        """
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        mean_return = returns.mean()
        std_return = returns.std()
        annualized_return = mean_return * 252
        annualized_volatility = std_return * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information ratio
        benchmark_returns = pd.Series(0.0001, index=returns.index)  # 0.01% daily benchmark
        excess_returns = returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        return {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        } 