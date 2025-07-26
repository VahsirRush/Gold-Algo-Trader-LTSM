"""
Overfitting detection and reporting for algorithmic trading strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class OverfittingDetector:
    def __init__(self, train_ratio: float = 0.6, validation_ratio: float = 0.2, test_ratio: float = 0.2):
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios sum to 1
        total = train_ratio + validation_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        n = len(data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.validation_ratio))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        return train_data, val_data, test_data
    
    def calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        if len(returns) == 0 or returns.isna().all():
            return {'sharpe_ratio': 0.0, 'total_return': 0.0, 'max_drawdown': 0.0, 'volatility': 0.0}
        
        # Remove NaN values
        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return {'sharpe_ratio': 0.0, 'total_return': 0.0, 'max_drawdown': 0.0, 'volatility': 0.0}
        
        total_return = (1 + returns_clean).prod() - 1
        volatility = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = returns_clean.mean() / returns_clean.std() * np.sqrt(252) if returns_clean.std() > 0 else 0
        
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility
        }
    
    def detect_overfitting(self, strategy_name: str, data: pd.DataFrame, 
                          strategy_class, strategy_params: Dict) -> Dict:
        """Detect overfitting by comparing in-sample vs out-of-sample performance."""
        print(f"Analyzing overfitting for {strategy_name}...")
        
        # Ensure we have enough data
        if len(data) < 200:
            print(f"  Warning: Only {len(data)} days of data available. Need at least 200 for reliable analysis.")
            return {
                'in_sample': {'sharpe_ratio': 0.0, 'total_return': 0.0, 'max_drawdown': 0.0},
                'out_of_sample': {'sharpe_ratio': 0.0, 'total_return': 0.0, 'max_drawdown': 0.0},
                'overfitting_score': 0.0,
                'overfitting_status': 'INSUFFICIENT_DATA',
                'train_returns': pd.Series(),
                'test_returns': pd.Series()
            }
        
        train_data, val_data, test_data = self.split_data(data)
        
        try:
            strategy = strategy_class(**strategy_params)
            
            # In-sample performance (train + validation)
            train_val_data = pd.concat([train_data, val_data])
            train_signals = strategy.generate_signals(train_val_data.copy())
            train_returns = self.calculate_strategy_returns(train_val_data, train_signals)
            in_sample_metrics = self.calculate_metrics(train_returns)
            
            # Out-of-sample performance (test)
            test_signals = strategy.generate_signals(test_data.copy())
            test_returns = self.calculate_strategy_returns(test_data, test_signals)
            out_sample_metrics = self.calculate_metrics(test_returns)
            
            # Overfitting analysis
            is_sharpe = in_sample_metrics['sharpe_ratio']
            oos_sharpe = out_sample_metrics['sharpe_ratio']
            overfitting_score = oos_sharpe - is_sharpe
            
            if overfitting_score < -0.5:
                status = "HIGH"
            elif overfitting_score < -0.2:
                status = "MEDIUM"
            elif overfitting_score < 0:
                status = "LOW"
            else:
                status = "NONE"
            
            return {
                'in_sample': in_sample_metrics,
                'out_of_sample': out_sample_metrics,
                'overfitting_score': overfitting_score,
                'overfitting_status': status,
                'train_returns': train_returns,
                'test_returns': test_returns
            }
            
        except Exception as e:
            print(f"  Error analyzing {strategy_name}: {str(e)}")
            return {
                'in_sample': {'sharpe_ratio': 0.0, 'total_return': 0.0, 'max_drawdown': 0.0},
                'out_of_sample': {'sharpe_ratio': 0.0, 'total_return': 0.0, 'max_drawdown': 0.0},
                'overfitting_score': 0.0,
                'overfitting_status': 'ERROR',
                'error': str(e),
                'train_returns': pd.Series(),
                'test_returns': pd.Series()
            }
    
    def calculate_strategy_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns based on signals."""
        if len(data) == 0 or len(signals) == 0:
            return pd.Series()
        
        # Ensure signals align with data
        if len(signals) != len(data):
            print(f"  Warning: Signal length ({len(signals)}) doesn't match data length ({len(data)})")
            return pd.Series()
        
        # Calculate price returns
        price_returns = data['Close'].pct_change()
        
        # Calculate strategy returns (shift signals by 1 to avoid look-ahead bias)
        strategy_returns = signals.shift(1) * price_returns
        
        return strategy_returns
    
    def generate_report(self, results: Dict, strategy_name: str) -> str:
        """Generate overfitting report."""
        if 'error' in results:
            return f"""
OVERFITTING ANALYSIS: {strategy_name}
=====================================
ERROR: {results['error']}
"""
        
        report = f"""
OVERFITTING ANALYSIS: {strategy_name}
=====================================

In-Sample Performance:
- Sharpe Ratio: {results['in_sample']['sharpe_ratio']:.4f}
- Total Return: {results['in_sample']['total_return']:.4f}
- Max Drawdown: {results['in_sample']['max_drawdown']:.4f}

Out-of-Sample Performance:
- Sharpe Ratio: {results['out_of_sample']['sharpe_ratio']:.4f}
- Total Return: {results['out_of_sample']['total_return']:.4f}
- Max Drawdown: {results['out_of_sample']['max_drawdown']:.4f}

Overfitting Analysis:
- Overfitting Score: {results['overfitting_score']:.4f}
- Status: {results['overfitting_status']}

Recommendations:
"""
        
        if results['overfitting_status'] == "HIGH":
            report += "- HIGH overfitting detected - reduce complexity\n"
        elif results['overfitting_status'] == "MEDIUM":
            report += "- MEDIUM overfitting - monitor closely\n"
        elif results['overfitting_status'] == "LOW":
            report += "- LOW overfitting - proceed with caution\n"
        elif results['overfitting_status'] == "INSUFFICIENT_DATA":
            report += "- Insufficient data for reliable analysis\n"
        else:
            report += "- Strategy appears robust\n"
        
        return report
    
    def plot_analysis(self, results: Dict, strategy_name: str):
        """Create overfitting analysis plots."""
        if 'error' in results or len(results['train_returns']) == 0:
            print(f"  Skipping plots for {strategy_name} due to errors or insufficient data")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cumulative returns
        train_cum = (1 + results['train_returns']).cumprod()
        test_cum = (1 + results['test_returns']).cumprod()
        
        ax1.plot(train_cum.index, train_cum.values, label='In-Sample', linewidth=2)
        ax1.plot(test_cum.index, test_cum.values, label='Out-of-Sample', linewidth=2)
        ax1.set_title(f'Cumulative Returns: {strategy_name}')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance metrics comparison
        metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
        is_metrics = [results['in_sample'][m] for m in metrics]
        oos_metrics = [results['out_of_sample'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax2.bar(x - width/2, is_metrics, width, label='In-Sample', alpha=0.8)
        ax2.bar(x + width/2, oos_metrics, width, label='Out-of-Sample', alpha=0.8)
        ax2.set_title('Performance Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_strategies(self, data: pd.DataFrame, strategies: Dict) -> Dict:
        """Analyze multiple strategies for overfitting."""
        all_results = {}
        
        for strategy_name, (strategy_class, params) in strategies.items():
            try:
                results = self.detect_overfitting(strategy_name, data, strategy_class, params)
                all_results[strategy_name] = results
                
                report = self.generate_report(results, strategy_name)
                print(report)
                
            except Exception as e:
                print(f"Error analyzing {strategy_name}: {str(e)}")
                all_results[strategy_name] = {'error': str(e)}
        
        return all_results
    
    def walk_forward_analysis(self, data: pd.DataFrame, strategy_class, strategy_params: Dict, 
                            n_periods: int = 5, min_test_size: int = 50) -> List[Dict]:
        """Perform walk-forward analysis for more robust overfitting detection."""
        print("Performing walk-forward analysis...")
        
        if len(data) < n_periods * min_test_size:
            print(f"  Insufficient data for walk-forward analysis. Need at least {n_periods * min_test_size} days.")
            return []
        
        period_length = len(data) // n_periods
        walk_forward_results = []
        
        for i in range(n_periods - 1):
            start_idx = i * period_length
            train_end = (i + 2) * period_length  # Use 2 periods for training
            test_end = (i + 3) * period_length   # Use 1 period for testing
            
            if test_end > len(data):
                break
                
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]
            
            if len(test_data) < min_test_size:
                continue
            
            try:
                strategy = strategy_class(**strategy_params)
                
                train_signals = strategy.generate_signals(train_data.copy())
                test_signals = strategy.generate_signals(test_data.copy())
                
                train_returns = self.calculate_strategy_returns(train_data, train_signals)
                test_returns = self.calculate_strategy_returns(test_data, test_signals)
                
                train_metrics = self.calculate_metrics(train_returns)
                test_metrics = self.calculate_metrics(test_returns)
                
                walk_forward_results.append({
                    'period': i + 1,
                    'train_sharpe': train_metrics['sharpe_ratio'],
                    'test_sharpe': test_metrics['sharpe_ratio'],
                    'overfitting_score': test_metrics['sharpe_ratio'] - train_metrics['sharpe_ratio'],
                    'train_return': train_metrics['total_return'],
                    'test_return': test_metrics['total_return']
                })
                
            except Exception as e:
                print(f"  Error in period {i+1}: {str(e)}")
                continue
        
        return walk_forward_results


def main():
    """Example usage of the overfitting detector."""
    # This would be used in practice with real data and strategies
    print("Overfitting detection system ready for use.")
    print("Use OverfittingDetector.analyze_strategies() to analyze strategies.")


if __name__ == "__main__":
    main() 