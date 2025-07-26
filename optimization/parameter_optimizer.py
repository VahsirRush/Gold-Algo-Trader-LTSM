"""
Parameter optimization for trading strategies using grid search and random search.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import warnings
warnings.filterwarnings('ignore')

from gold_algo.strategies import TrendFollowingStrategy, MeanReversionStrategy, EnhancedNNStrategy

class ParameterOptimizer:
    """Parameter optimization engine for trading strategies."""
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000, metric: str = 'sharpe_ratio'):
        """
        Initialize parameter optimizer.
        
        Args:
            data: Historical price data
            initial_capital: Initial capital for backtesting
            metric: Optimization metric ('sharpe_ratio', 'total_return', 'calmar_ratio')
        """
        self.data = data
        self.initial_capital = initial_capital
        self.metric = metric
        self.logger = logging.getLogger(__name__)
        self.results = []
        
        # Strategy parameter spaces
        self.parameter_spaces = {
            'trend_following': {
                'sma_short': [10, 15, 20, 25, 30],
                'sma_medium': [40, 50, 60, 70],
                'sma_long': [150, 200, 250],
                'adx_threshold': [10.0, 15.0, 20.0, 25.0, 30.0],
                'atr_period': [10, 14, 20],
                'macd_fast': [8, 12, 16],
                'macd_slow': [20, 26, 32],
                'macd_signal': [7, 9, 11]
            },
            'mean_reversion': {
                'lookback_period': [15, 20, 25, 30],
                'std_dev_threshold': [1.5, 2.0, 2.5, 3.0],
                'rsi_oversold': [20, 25, 30, 35],
                'rsi_overbought': [65, 70, 75, 80],
                'bollinger_period': [15, 20, 25],
                'bollinger_std': [1.5, 2.0, 2.5],
                'min_holding_period': [3, 5, 7, 10]
            },
            'ml_strategy': {
                'lookback_period': [15, 20, 25],
                'prediction_horizon': [3, 5, 7, 10],
                'confidence_threshold': [0.5, 0.6, 0.7, 0.8],
                'retrain_frequency': [20, 30, 45],
                'model_type': ['random_forest', 'gradient_boosting']
            }
        }
    
    def generate_parameter_combinations(self, strategy_name: str, max_combinations: int = 1000) -> List[Dict]:
        """
        Generate parameter combinations for optimization.
        
        Args:
            strategy_name: Name of the strategy
            max_combinations: Maximum number of combinations to test
            
        Returns:
            List of parameter dictionaries
        """
        if strategy_name not in self.parameter_spaces:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        param_space = self.parameter_spaces[strategy_name]
        
        # Generate all combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        combinations = list(itertools.product(*param_values))
        
        # Convert to dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_combinations.append(param_dict)
        
        # Limit combinations if too many
        if len(param_combinations) > max_combinations:
            # Use random sampling for large parameter spaces
            np.random.seed(42)
            indices = np.random.choice(len(param_combinations), max_combinations, replace=False)
            param_combinations = [param_combinations[i] for i in indices]
        
        return param_combinations
    
    def backtest_strategy(self, strategy_name: str, params: Dict) -> Dict[str, Any]:
        """
        Backtest a strategy with given parameters.
        
        Args:
            strategy_name: Name of the strategy
            params: Strategy parameters
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Initialize strategy
            if strategy_name == 'trend_following':
                strategy = TrendFollowingStrategy(**params)
            elif strategy_name == 'mean_reversion':
                strategy = MeanReversionStrategy(**params)
            elif strategy_name == 'enhanced_nn':
                strategy = EnhancedNNStrategy(**params)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            # Generate signals
            signals = strategy.generate_signals(self.data)
            confidence = strategy.calculate_confidence(self.data)
            
            # Run backtest
            results = self._run_backtest(signals, confidence)
            
            # Add strategy info
            results.update({
                'strategy': strategy_name,
                'parameters': params,
                'num_signals': len(signals[signals != 0]),
                'buy_signals': len(signals[signals == 1]),
                'sell_signals': len(signals[signals == -1]),
                'avg_confidence': confidence.mean()
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed for {strategy_name} with params {params}: {e}")
            return {
                'strategy': strategy_name,
                'parameters': params,
                'total_return': -1.0,
                'sharpe_ratio': -1.0,
                'max_drawdown': -1.0,
                'num_trades': 0,
                'error': str(e)
            }
    
    def _run_backtest(self, signals: pd.Series, confidence: pd.Series) -> Dict[str, Any]:
        """
        Run backtest with signals and confidence.
        
        Args:
            signals: Trading signals
            confidence: Signal confidence scores
            
        Returns:
            Dictionary with backtest metrics
        """
        # Validate inputs
        if signals is None or signals.empty:
            return {
                'total_return': -1.0,
                'annual_return': -1.0,
                'sharpe_ratio': -999,
                'max_drawdown': -1.0,
                'volatility': 0.0,
                'num_trades': 0,
                'final_capital': self.initial_capital,
                'error': 'No signals provided'
            }
        
        if len(signals) != len(self.data):
            return {
                'total_return': -1.0,
                'annual_return': -1.0,
                'sharpe_ratio': -999,
                'max_drawdown': -1.0,
                'volatility': 0.0,
                'num_trades': 0,
                'final_capital': self.initial_capital,
                'error': 'Signal length mismatch'
            }
        
        position = 0
        capital = self.initial_capital
        trades = []
        equity_curve = [self.initial_capital]  # Start with initial capital
        
        # Calculate daily returns for the underlying asset
        price_returns = self.data['Close'].pct_change().fillna(0)
        
        for i in range(1, len(self.data)):
            current_price = self.data['Close'].iloc[i]
            current_date = self.data.index[i]
            current_signal = signals.iloc[i-1]  # Use previous day's signal
            
            # Calculate current equity including unrealized P&L
            if position > 0:
                unrealized_pnl = position * (current_price - self.data['Close'].iloc[i-1])
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
            
            # Check for signal changes
            if current_signal != 0:
                if current_signal > 0 and position == 0:  # Buy signal
                    # Position size based on confidence and signal strength
                    conf = confidence.iloc[i-1] if not pd.isna(confidence.iloc[i-1]) else 0.5
                    signal_strength = abs(current_signal)
                    position_size = capital * min(conf * signal_strength, 0.95)  # Max 95% of capital
                    position = position_size / current_price
                    capital -= position_size
                    trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'price': current_price,
                        'position': position,
                        'confidence': conf,
                        'signal_strength': signal_strength
                    })
                elif current_signal < 0 and position > 0:  # Sell signal
                    capital = position * current_price * 0.999  # 0.1% commission
                    position = 0
                    trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'price': current_price,
                        'capital': capital
                    })
        
        # Final position
        if position > 0:
            capital = position * self.data['Close'].iloc[-1] * 0.999
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        # Calculate daily strategy returns
        strategy_returns = pd.Series(0.0, index=self.data.index)
        for i in range(1, len(self.data)):
            if i < len(equity_curve):
                strategy_returns.iloc[i] = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
        
        # Remove any infinite or NaN values
        strategy_returns = strategy_returns.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Calculate annualized metrics
        if len(strategy_returns) > 1:
            annual_return = strategy_returns.mean() * 252
            volatility = strategy_returns.std() * np.sqrt(252)
            
            # FIXED: Proper Sharpe ratio calculation with risk-free rate
            risk_free_rate = 0.02  # 2% annual risk-free rate
            excess_return = annual_return - risk_free_rate
            
            if volatility > 0:
                sharpe_ratio = excess_return / volatility
            else:
                sharpe_ratio = 0 if excess_return == 0 else -999
        else:
            annual_return = 0
            volatility = 0
            sharpe_ratio = -999
        
        # Calculate maximum drawdown
        if len(equity_curve) > 1:
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'num_trades': len(trades),
            'final_capital': capital,
            'strategy_returns': strategy_returns
        }
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        # Validate input
        if returns is None or returns.empty:
            return {
                'sharpe_ratio': -999,
                'total_return': -999,
                'max_drawdown': -999,
                'calmar_ratio': -999,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        # Clean returns
        returns_clean = returns.replace([np.inf, -np.inf], 0).dropna()
        
        if len(returns_clean) == 0:
            return {
                'sharpe_ratio': -999,
                'total_return': -999,
                'max_drawdown': -999,
                'calmar_ratio': -999,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        annualized_return = returns_clean.mean() * 252
        volatility = returns_clean.std() * np.sqrt(252)
        
        # FIXED: Proper Sharpe ratio calculation
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_return = annualized_return - risk_free_rate
        
        if volatility > 0:
            sharpe_ratio = excess_return / volatility
        else:
            sharpe_ratio = 0 if excess_return == 0 else -999
        
        # Maximum drawdown
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else -999
        
        # Win rate and profit factor
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]
        
        win_rate = len(positive_returns) / len(returns_clean) if len(returns_clean) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def evaluate_strategy(self, strategy_class, params: Dict) -> Dict[str, Any]:
        """Evaluate a single parameter combination."""
        try:
            # Initialize strategy
            strategy = strategy_class(**params)
            
            # Generate signals
            signals = strategy.generate_signals(self.data.copy())
            
            # Calculate returns
            price_returns = self.data['Close'].pct_change()
            strategy_returns = signals.shift(1) * price_returns
            
            # Calculate metrics
            metrics = self.calculate_performance_metrics(strategy_returns)
            
            # Count signals
            signal_counts = signals.value_counts()
            total_signals = signal_counts.sum() - signal_counts.get(0, 0)
            
            return {
                'params': params,
                'metrics': metrics,
                'total_signals': total_signals,
                'success': True
            }
            
        except Exception as e:
            return {
                'params': params,
                'metrics': {self.metric: -999},
                'total_signals': 0,
                'success': False,
                'error': str(e)
            }
    
    def grid_search(self, strategy_class, param_grid: Dict[str, List], 
                   max_workers: int = 4) -> pd.DataFrame:
        """
        Perform grid search optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dictionary of parameter names and their possible values
            max_workers: Number of parallel workers
            
        Returns:
            DataFrame with optimization results
        """
        print(f"Starting grid search for {strategy_class.__name__}...")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        results = []
        
        # Use parallel processing for faster optimization
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for combination in param_combinations:
                params = dict(zip(param_names, combination))
                future = executor.submit(self.evaluate_strategy, strategy_class, params)
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                
                if (i + 1) % 50 == 0:
                    print(f"  Completed {i + 1}/{len(param_combinations)} combinations")
        
        # Convert to DataFrame
        df_results = []
        for result in results:
            if result['success']:
                row = {
                    'params': result['params'],
                    'total_signals': result['total_signals'],
                    **result['metrics']
                }
                df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        if len(df) > 0:
            # Sort by optimization metric
            df = df.sort_values(self.metric, ascending=False)
            
            print(f"\nBest {self.metric}: {df[self.metric].iloc[0]:.4f}")
            print(f"Best parameters: {df['params'].iloc[0]}")
            print(f"Total signals: {df['total_signals'].iloc[0]}")
        
        return df
    
    def random_search(self, strategy_class, param_ranges: Dict[str, Tuple], 
                     n_trials: int = 100, max_workers: int = 4) -> pd.DataFrame:
        """
        Perform random search optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            param_ranges: Dictionary of parameter names and their (min, max) ranges
            n_trials: Number of random trials
            max_workers: Number of parallel workers
            
        Returns:
            DataFrame with optimization results
        """
        print(f"Starting random search for {strategy_class.__name__}...")
        print(f"Testing {n_trials} random parameter combinations...")
        
        results = []
        
        # Use parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for _ in range(n_trials):
                # Generate random parameters
                params = {}
                for param_name, (min_val, max_val) in param_ranges.items():
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        params[param_name] = np.random.uniform(min_val, max_val)
                
                future = executor.submit(self.evaluate_strategy, strategy_class, params)
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                
                if (i + 1) % 20 == 0:
                    print(f"  Completed {i + 1}/{n_trials} trials")
        
        # Convert to DataFrame
        df_results = []
        for result in results:
            if result['success']:
                row = {
                    'params': result['params'],
                    'total_signals': result['total_signals'],
                    **result['metrics']
                }
                df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        if len(df) > 0:
            df = df.sort_values(self.metric, ascending=False)
            
            print(f"\nBest {self.metric}: {df[self.metric].iloc[0]:.4f}")
            print(f"Best parameters: {df['params'].iloc[0]}")
            print(f"Total signals: {df['total_signals'].iloc[0]}")
        
        return df
    
    def optimize_strategy(self, strategy_name: str, 
                         optimization_metric: str = 'sharpe_ratio',
                         max_combinations: int = 500,
                         n_jobs: int = -1) -> Tuple[Dict, List[Dict]]:
        """
        Optimize strategy parameters.
        
        Args:
            strategy_name: Name of the strategy to optimize
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'calmar_ratio')
            max_combinations: Maximum parameter combinations to test
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            Tuple of (best_params, all_results)
        """
        self.logger.info(f"Starting optimization for {strategy_name}")
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(strategy_name, max_combinations)
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Run backtests
        results = []
        
        if n_jobs == 1:
            # Sequential processing
            for i, params in enumerate(param_combinations):
                if i % 50 == 0:
                    self.logger.info(f"Progress: {i}/{len(param_combinations)}")
                result = self.backtest_strategy(strategy_name, params)
                results.append(result)
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
                futures = [executor.submit(self.backtest_strategy, strategy_name, params) 
                          for params in param_combinations]
                
                for i, future in enumerate(as_completed(futures)):
                    if i % 50 == 0:
                        self.logger.info(f"Progress: {i}/{len(param_combinations)}")
                    result = future.result()
                    results.append(result)
        
        # Filter out failed backtests
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            raise ValueError("No valid backtest results found")
        
        # Sort by optimization metric
        if optimization_metric == 'calmar_ratio':
            # Calculate Calmar ratio (annual return / max drawdown)
            for result in valid_results:
                if result['max_drawdown'] != 0:
                    result['calmar_ratio'] = result['annual_return'] / abs(result['max_drawdown'])
                else:
                    result['calmar_ratio'] = 0
        
        # Sort results
        valid_results.sort(key=lambda x: x.get(optimization_metric, 0), reverse=True)
        
        # Get best parameters
        best_result = valid_results[0]
        best_params = best_result['parameters']
        
        self.logger.info(f"Best {optimization_metric}: {best_result.get(optimization_metric, 0):.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return best_params, valid_results
    
    def save_optimization_results(self, results: List[Dict], filename: str):
        """Save optimization results to file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for saving
        clean_results = []
        for result in results:
            clean_result = {}
            for key, value in result.items():
                if key == 'parameters':
                    clean_result[key] = {k: convert_numpy(v) for k, v in value.items()}
                else:
                    clean_result[key] = convert_numpy(value)
            clean_results.append(clean_result)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        self.logger.info(f"Optimization results saved to {filename}")
    
    def load_optimization_results(self, filename: str) -> List[Dict]:
        """Load optimization results from file."""
        with open(filename, 'r') as f:
            results = json.load(f)
        return results
    
    def plot_optimization_results(self, results: List[Dict], 
                                metric1: str = 'sharpe_ratio',
                                metric2: str = 'total_return',
                                top_n: int = 50):
        """Plot optimization results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get top results
            top_results = results[:top_n]
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            
            x_values = [r.get(metric1, 0) for r in top_results]
            y_values = [r.get(metric2, 0) for r in top_results]
            
            plt.scatter(x_values, y_values, alpha=0.6)
            plt.xlabel(metric1.replace('_', ' ').title())
            plt.ylabel(metric2.replace('_', ' ').title())
            plt.title(f'Optimization Results: {metric1} vs {metric2}')
            
            # Highlight best result
            best_result = top_results[0]
            plt.scatter(best_result.get(metric1, 0), best_result.get(metric2, 0), 
                       color='red', s=100, marker='*', label='Best')
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_filename = f'optimization_results_{metric1}_{metric2}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"Optimization plot saved to {plot_filename}")
            
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Plotting failed: {e}")

    def optimize_trend_following(self) -> pd.DataFrame:
        """Optimize trend following strategy parameters."""
        from gold_algo.strategies.trend_following import TrendFollowingStrategy
        
        param_grid = {
            'sma_short': [5, 10, 15, 20],
            'sma_medium': [20, 30, 50, 100],
            'sma_long': [50, 100, 200],
            'adx_threshold': [15.0, 20.0, 25.0, 30.0],
            'rsi_period': [10, 14, 20],
            'volatility_threshold': [0.01, 0.015, 0.02, 0.025]
        }
        
        return self.grid_search(TrendFollowingStrategy, param_grid)
    
    def optimize_mean_reversion(self) -> pd.DataFrame:
        """Optimize mean reversion strategy parameters."""
        from gold_algo.strategies.mean_reversion import MeanReversionStrategy
        
        param_grid = {
            'lookback_period': [10, 15, 20, 30],
            'std_dev_threshold': [1.5, 2.0, 2.5, 3.0],
            'rsi_oversold': [20.0, 25.0, 30.0, 35.0],
            'rsi_overbought': [65.0, 70.0, 75.0, 80.0],
            'bollinger_period': [15, 20, 25],
            'bollinger_std': [1.5, 2.0, 2.5],
            'min_holding_period': [3, 5, 7, 10]
        }
        
        return self.grid_search(MeanReversionStrategy, param_grid)
    
    def optimize_neural_net(self) -> pd.DataFrame:
        """Optimize neural net strategy parameters."""
        from gold_algo.strategies.dl_regression import DLRegressionStrategy
        
        param_ranges = {
            'lookback_period': (10, 50),
            'prediction_horizon': (1, 10),
            'epochs': (10, 50),
            'batch_size': (16, 64),
            'threshold': (0.001, 0.005)
        }
        
        return self.random_search(DLRegressionStrategy, param_ranges, n_trials=50)
    
    def get_best_parameters(self, df: pd.DataFrame) -> Dict:
        """Get the best parameters from optimization results."""
        if len(df) == 0:
            return {}
        
        return df['params'].iloc[0]
    
    def plot_optimization_results(self, df: pd.DataFrame, param_name: str = None):
        """Plot optimization results."""
        if len(df) == 0:
            print("No results to plot")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Top 10 results
        top_10 = df.head(10)
        
        # Plot 1: Metric distribution
        axes[0, 0].hist(df[self.metric], bins=20, alpha=0.7)
        axes[0, 0].axvline(df[self.metric].iloc[0], color='red', linestyle='--', label='Best')
        axes[0, 0].set_title(f'{self.metric} Distribution')
        axes[0, 0].legend()
        
        # Plot 2: Top 10 results
        axes[0, 1].bar(range(len(top_10)), top_10[self.metric])
        axes[0, 1].set_title('Top 10 Results')
        axes[0, 1].set_ylabel(self.metric)
        
        # Plot 3: Signals vs Performance
        axes[1, 0].scatter(df['total_signals'], df[self.metric], alpha=0.6)
        axes[1, 0].set_xlabel('Total Signals')
        axes[1, 0].set_ylabel(self.metric)
        axes[1, 0].set_title('Signals vs Performance')
        
        # Plot 4: Win Rate vs Performance
        if 'win_rate' in df.columns:
            axes[1, 1].scatter(df['win_rate'], df[self.metric], alpha=0.6)
            axes[1, 1].set_xlabel('Win Rate')
            axes[1, 1].set_ylabel(self.metric)
            axes[1, 1].set_title('Win Rate vs Performance')
        
        plt.tight_layout()
        plt.show() 