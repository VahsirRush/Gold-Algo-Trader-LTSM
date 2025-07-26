"""
Trade visualization module for gold trading strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TradeVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize trade visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_strategy_performance(self, data: pd.DataFrame, signals: pd.Series, 
                                strategy_name: str = "Strategy") -> None:
        """Plot strategy performance with signals and returns."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Price and signals
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=1)
        
        # Plot signals
        buy_signals = signals > 0
        sell_signals = signals < 0
        
        ax1.scatter(data.index[buy_signals], data['Close'][buy_signals], 
                   color='green', marker='^', s=50, label='Buy Signal')
        ax1.scatter(data.index[sell_signals], data['Close'][sell_signals], 
                   color='red', marker='v', s=50, label='Sell Signal')
        
        ax1.set_title(f'{strategy_name} - Price and Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative returns
        price_returns = data['Close'].pct_change()
        strategy_returns = signals.shift(1) * price_returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        ax2.plot(data.index, cumulative_returns, label='Strategy Returns', 
                color='blue', linewidth=2)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Cumulative Returns')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        self._print_performance_metrics(strategy_returns)
    
    def plot_strategy_comparison(self, data: pd.DataFrame, 
                               strategies: Dict[str, pd.Series]) -> None:
        """Plot comparison of multiple strategies."""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        price_returns = data['Close'].pct_change()
        
        for name, signals in strategies.items():
            strategy_returns = signals.shift(1) * price_returns
            cumulative_returns = (1 + strategy_returns).cumprod()
            ax.plot(data.index, cumulative_returns, label=name, linewidth=2)
        
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Strategy Comparison - Cumulative Returns')
        ax.set_ylabel('Cumulative Returns')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def _print_performance_metrics(self, returns: pd.Series) -> None:
        """Print performance metrics."""
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            print("No returns data available")
            return
        
        total_return = (1 + returns_clean).prod() - 1
        volatility = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = returns_clean.mean() / returns_clean.std() * np.sqrt(252) if returns_clean.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        positive_returns = returns_clean[returns_clean > 0]
        win_rate = len(positive_returns) / len(returns_clean) if len(returns_clean) > 0 else 0
        
        print(f"\nPerformance Metrics:")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Win Rate: {win_rate:.2%}")
        
        if sharpe_ratio >= 2.0:
            print(f"🎉 EXCELLENT! Sharpe ratio >= 2.0 achieved!")
        elif sharpe_ratio >= 1.0:
            print(f"✅ GOOD! Sharpe ratio >= 1.0 achieved!")
        else:
            print(f"📈 Room for improvement - target Sharpe ratio >= 2.0")
    
    def plot_strategy_signals(self, 
                            data: pd.DataFrame, 
                            signals: pd.Series,
                            strategy_name: str = "Strategy",
                            show_indicators: bool = True) -> None:
        """
        Plot strategy signals on price chart.
        
        Args:
            data: DataFrame with OHLCV data
            signals: Series with trading signals
            strategy_name: Name of the strategy
            show_indicators: Whether to show technical indicators
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price data
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=1, alpha=0.8)
        
        # Add moving averages
        if show_indicators:
            sma_20 = data['Close'].rolling(20).mean()
            sma_50 = data['Close'].rolling(50).mean()
            ax1.plot(data.index, sma_20, label='SMA 20', alpha=0.7, linewidth=1)
            ax1.plot(data.index, sma_50, label='SMA 50', alpha=0.7, linewidth=1)
        
        # Plot signals
        buy_signals = signals > 0
        sell_signals = signals < 0
        
        ax1.scatter(data.index[buy_signals], data['Close'][buy_signals], 
                   color='green', marker='^', s=50, label='Buy Signal', alpha=0.8)
        ax1.scatter(data.index[sell_signals], data['Close'][sell_signals], 
                   color='red', marker='v', s=50, label='Sell Signal', alpha=0.8)
        
        # Plot signal strength
        ax2.fill_between(data.index, signals, 0, 
                        where=signals > 0, color='green', alpha=0.3, label='Long')
        ax2.fill_between(data.index, signals, 0, 
                        where=signals < 0, color='red', alpha=0.3, label='Short')
        ax2.plot(data.index, signals, color='blue', linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Signal Strength')
        ax2.set_ylim(-1.1, 1.1)
        
        # Formatting
        ax1.set_title(f'{strategy_name} - Price and Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_trades_with_pnl(self, 
                           data: pd.DataFrame,
                           signals: pd.Series,
                           strategy_name: str = "Strategy") -> None:
        """
        Plot trades with P&L visualization.
        
        Args:
            data: DataFrame with OHLCV data
            signals: Series with trading signals
            strategy_name: Name of the strategy
        """
        # Calculate returns
        price_returns = data['Close'].pct_change()
        strategy_returns = signals.shift(1) * price_returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Identify trades
        trades = self._identify_trades(signals, data['Close'])
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, 
                                           gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot 1: Price and trades
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=1, alpha=0.8)
        
        # Plot trade entries and exits
        for trade in trades:
            if trade['type'] == 'long':
                ax1.scatter(trade['entry_date'], trade['entry_price'], 
                           color='green', marker='^', s=100, alpha=0.8)
                if trade['exit_date'] is not None:
                    ax1.scatter(trade['exit_date'], trade['exit_price'], 
                               color='red', marker='v', s=100, alpha=0.8)
                    ax1.plot([trade['entry_date'], trade['exit_date']], 
                            [trade['entry_price'], trade['exit_price']], 
                            'g-', alpha=0.5, linewidth=2)
            else:  # short
                ax1.scatter(trade['entry_date'], trade['entry_price'], 
                           color='red', marker='v', s=100, alpha=0.8)
                if trade['exit_date'] is not None:
                    ax1.scatter(trade['exit_date'], trade['exit_price'], 
                               color='green', marker='^', s=100, alpha=0.8)
                    ax1.plot([trade['entry_date'], trade['exit_date']], 
                            [trade['entry_price'], trade['exit_price']], 
                            'r-', alpha=0.5, linewidth=2)
        
        ax1.set_title(f'{strategy_name} - Trades')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative returns
        ax2.plot(data.index, cumulative_returns, label='Strategy Returns', 
                color='blue', linewidth=2)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Cumulative Returns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        drawdown = self._calculate_drawdown(cumulative_returns)
        ax3.fill_between(data.index, drawdown, 0, color='red', alpha=0.3)
        ax3.plot(data.index, drawdown, color='red', linewidth=1)
        ax3.set_ylabel('Drawdown')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print trade summary
        self._print_trade_summary(trades)
    
    def plot_performance_metrics(self, 
                               data: pd.DataFrame,
                               signals: pd.Series,
                               strategy_name: str = "Strategy") -> None:
        """
        Plot comprehensive performance metrics.
        
        Args:
            data: DataFrame with OHLCV data
            signals: Series with trading signals
            strategy_name: Name of the strategy
        """
        # Calculate metrics
        price_returns = data['Close'].pct_change()
        strategy_returns = signals.shift(1) * price_returns
        
        # Calculate rolling metrics
        rolling_sharpe = self._calculate_rolling_sharpe(strategy_returns, window=63)
        rolling_vol = strategy_returns.rolling(63).std() * np.sqrt(252)
        rolling_returns = strategy_returns.rolling(63).mean() * 252
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Rolling Sharpe Ratio
        ax1.plot(data.index, rolling_sharpe, color='blue', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        ax1.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Sharpe = 2')
        ax1.set_title('Rolling Sharpe Ratio (63-day window)')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling Volatility
        ax2.plot(data.index, rolling_vol, color='red', linewidth=2)
        ax2.set_title('Rolling Volatility (63-day window)')
        ax2.set_ylabel('Annualized Volatility')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rolling Returns
        ax3.plot(data.index, rolling_returns, color='green', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Rolling Returns (63-day window)')
        ax3.set_ylabel('Annualized Returns')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Returns Distribution
        ax4.hist(strategy_returns.dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('Returns Distribution')
        ax4.set_xlabel('Returns')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(f'{strategy_name} - Performance Metrics', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def _identify_trades(self, signals: pd.Series, prices: pd.Series) -> List[Dict]:
        """Identify individual trades from signals."""
        trades = []
        current_trade = None
        
        for i, (date, signal) in enumerate(signals.items()):
            if current_trade is None:
                if signal > 0:  # Long entry
                    current_trade = {
                        'type': 'long',
                        'entry_date': date,
                        'entry_price': prices.iloc[i],
                        'exit_date': None,
                        'exit_price': None,
                        'pnl': None
                    }
                elif signal < 0:  # Short entry
                    current_trade = {
                        'type': 'short',
                        'entry_date': date,
                        'entry_price': prices.iloc[i],
                        'exit_date': None,
                        'exit_price': None,
                        'pnl': None
                    }
            else:
                # Check for exit
                if (current_trade['type'] == 'long' and signal < 0) or \
                   (current_trade['type'] == 'short' and signal > 0):
                    current_trade['exit_date'] = date
                    current_trade['exit_price'] = prices.iloc[i]
                    
                    # Calculate P&L
                    if current_trade['type'] == 'long':
                        current_trade['pnl'] = (current_trade['exit_price'] - current_trade['entry_price']) / current_trade['entry_price']
                    else:
                        current_trade['pnl'] = (current_trade['entry_price'] - current_trade['exit_price']) / current_trade['entry_price']
                    
                    trades.append(current_trade)
                    current_trade = None
        
        return trades
    
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 63) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        sharpe = rolling_mean / rolling_std
        return sharpe
    
    def _print_trade_summary(self, trades: List[Dict]) -> None:
        """Print summary of trades."""
        if not trades:
            print("No trades identified")
            return
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] and t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] and t['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        if winning_trades:
            avg_win = np.mean([t['pnl'] for t in winning_trades])
            max_win = max([t['pnl'] for t in winning_trades])
        else:
            avg_win = max_win = 0
        
        if losing_trades:
            avg_loss = np.mean([t['pnl'] for t in losing_trades])
            max_loss = min([t['pnl'] for t in losing_trades])
        else:
            avg_loss = max_loss = 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        print(f"\nTrade Summary:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Win: {avg_win:.2%}")
        print(f"Average Loss: {avg_loss:.2%}")
        print(f"Max Win: {max_win:.2%}")
        print(f"Max Loss: {max_loss:.2%}")
        print(f"Profit Factor: {profit_factor:.2f}") 