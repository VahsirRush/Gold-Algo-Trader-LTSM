#!/usr/bin/env python3
"""
COMPREHENSIVE OVERFITTING DETECTION SYSTEM
=========================================

Advanced overfitting detection and prevention system that:
- Tests across multiple time periods
- Validates on out-of-sample data
- Uses statistical methods to detect overfitting
- Implements walk-forward analysis
- Provides detailed overfitting reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

class OverfittingDetectionSystem:
    """Comprehensive overfitting detection and analysis system."""
    
    def __init__(self, 
                 strategy,
                 data: pd.DataFrame,
                 test_periods: int = 5,
                 min_train_size: int = 30,
                 max_train_size: int = 100):
        
        self.strategy = strategy
        self.data = data
        self.test_periods = test_periods
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        
        # Results storage
        self.walk_forward_results = []
        self.overfitting_metrics = {}
        self.stability_analysis = {}
        
    def run_comprehensive_overfitting_analysis(self) -> Dict:
        """Run comprehensive overfitting analysis."""
        print("🔍 COMPREHENSIVE OVERFITTING DETECTION ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        # 1. Walk-forward analysis
        print("📊 Running walk-forward analysis...")
        results['walk_forward'] = self._walk_forward_analysis()
        
        # 2. Out-of-sample testing
        print("📈 Running out-of-sample testing...")
        results['out_of_sample'] = self._out_of_sample_testing()
        
        # 3. Parameter stability analysis
        print("🔧 Running parameter stability analysis...")
        results['parameter_stability'] = self._parameter_stability_analysis()
        
        # 4. Performance degradation analysis
        print("📉 Running performance degradation analysis...")
        results['performance_degradation'] = self._performance_degradation_analysis()
        
        # 5. Statistical overfitting tests
        print("📊 Running statistical overfitting tests...")
        results['statistical_tests'] = self._statistical_overfitting_tests()
        
        # 6. Market regime analysis
        print("🌍 Running market regime analysis...")
        results['market_regime'] = self._market_regime_analysis()
        
        # 7. Generate comprehensive report
        print("📋 Generating comprehensive report...")
        results['comprehensive_report'] = self._generate_comprehensive_report(results)
        
        return results
    
    def _walk_forward_analysis(self) -> Dict:
        """Perform walk-forward analysis to detect overfitting."""
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=self.test_periods)
        
        train_metrics = []
        test_metrics = []
        parameter_changes = []
        
        for train_idx, test_idx in tscv.split(self.data):
            if len(train_idx) < self.min_train_size:
                continue
                
            # Split data
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            # Train strategy
            self.strategy.train_on_data(train_data)
            
            # Get initial parameters
            initial_params = self._get_strategy_parameters()
            
            # Test on training data
            train_results = self.strategy.run_backtest(train_data)
            train_metrics.append({
                'sharpe': train_results['sharpe_ratio'],
                'return': train_results['total_return'],
                'volatility': train_results['volatility'],
                'max_drawdown': train_results['max_drawdown'],
                'win_rate': train_results['win_rate']
            })
            
            # Test on out-of-sample data
            test_results = self.strategy.run_backtest(test_data)
            test_metrics.append({
                'sharpe': test_results['sharpe_ratio'],
                'return': test_results['total_return'],
                'volatility': test_results['volatility'],
                'max_drawdown': test_results['max_drawdown'],
                'win_rate': test_results['win_rate']
            })
            
            # Get final parameters
            final_params = self._get_strategy_parameters()
            parameter_changes.append(self._calculate_parameter_changes(initial_params, final_params))
        
        # Calculate overfitting indicators
        overfitting_indicators = self._calculate_overfitting_indicators(train_metrics, test_metrics)
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'parameter_changes': parameter_changes,
            'overfitting_indicators': overfitting_indicators
        }
    
    def _out_of_sample_testing(self) -> Dict:
        """Test strategy on completely out-of-sample data."""
        
        # Use first 70% for training, last 30% for testing
        split_point = int(len(self.data) * 0.7)
        train_data = self.data.iloc[:split_point]
        test_data = self.data.iloc[split_point:]
        
        # Train on training data
        self.strategy.train_on_data(train_data)
        
        # Test on training data
        train_results = self.strategy.run_backtest(train_data)
        
        # Test on out-of-sample data
        test_results = self.strategy.run_backtest(test_data)
        
        # Calculate degradation metrics
        degradation = {
            'sharpe_degradation': (train_results['sharpe_ratio'] - test_results['sharpe_ratio']) / train_results['sharpe_ratio'] if train_results['sharpe_ratio'] != 0 else 0,
            'return_degradation': (train_results['total_return'] - test_results['total_return']) / abs(train_results['total_return']) if train_results['total_return'] != 0 else 0,
            'volatility_increase': (test_results['volatility'] - train_results['volatility']) / train_results['volatility'] if train_results['volatility'] != 0 else 0,
            'drawdown_increase': (abs(test_results['max_drawdown']) - abs(train_results['max_drawdown'])) / abs(train_results['max_drawdown']) if train_results['max_drawdown'] != 0 else 0,
            'win_rate_degradation': (train_results['win_rate'] - test_results['win_rate']) / train_results['win_rate'] if train_results['win_rate'] != 0 else 0
        }
        
        return {
            'train_results': train_results,
            'test_results': test_results,
            'degradation': degradation,
            'is_overfitted': any(abs(v) > 0.3 for v in degradation.values())
        }
    
    def _parameter_stability_analysis(self) -> Dict:
        """Analyze parameter stability across different time periods."""
        
        parameter_history = []
        
        # Test parameters across different time windows
        for window_size in [20, 30, 40, 50]:
            if window_size >= len(self.data):
                continue
                
            # Use sliding window
            for i in range(0, len(self.data) - window_size, 10):
                window_data = self.data.iloc[i:i+window_size]
                
                # Train strategy
                self.strategy.train_on_data(window_data)
                
                # Get parameters
                params = self._get_strategy_parameters()
                parameter_history.append({
                    'window_start': i,
                    'window_size': window_size,
                    'parameters': params
                })
        
        # Calculate parameter stability
        stability_metrics = self._calculate_parameter_stability(parameter_history)
        
        return {
            'parameter_history': parameter_history,
            'stability_metrics': stability_metrics
        }
    
    def _performance_degradation_analysis(self) -> Dict:
        """Analyze performance degradation over time."""
        
        performance_windows = []
        
        # Test performance across different time windows
        for window_size in [10, 15, 20, 25]:
            if window_size >= len(self.data):
                continue
                
            window_performances = []
            
            for i in range(0, len(self.data) - window_size, 5):
                window_data = self.data.iloc[i:i+window_size]
                
                # Train and test
                self.strategy.train_on_data(window_data)
                results = self.strategy.run_backtest(window_data)
                
                window_performances.append({
                    'start_idx': i,
                    'sharpe': results['sharpe_ratio'],
                    'return': results['total_return'],
                    'volatility': results['volatility']
                })
            
            performance_windows.append({
                'window_size': window_size,
                'performances': window_performances
            })
        
        # Calculate degradation trends
        degradation_trends = self._calculate_degradation_trends(performance_windows)
        
        return {
            'performance_windows': performance_windows,
            'degradation_trends': degradation_trends
        }
    
    def _statistical_overfitting_tests(self) -> Dict:
        """Run statistical tests to detect overfitting."""
        
        # 1. Performance consistency test
        consistency_test = self._performance_consistency_test()
        
        # 2. Parameter sensitivity test
        sensitivity_test = self._parameter_sensitivity_test()
        
        # 3. Randomization test
        randomization_test = self._randomization_test()
        
        # 4. Cross-validation stability test
        cv_stability_test = self._cross_validation_stability_test()
        
        return {
            'consistency_test': consistency_test,
            'sensitivity_test': sensitivity_test,
            'randomization_test': randomization_test,
            'cv_stability_test': cv_stability_test
        }
    
    def _market_regime_analysis(self) -> Dict:
        """Analyze performance across different market regimes."""
        
        # Identify market regimes
        regimes = self._identify_market_regimes()
        
        regime_performances = {}
        
        for regime_name, regime_data in regimes.items():
            if len(regime_data) < 10:  # Need minimum data points
                continue
                
            # Train on other regimes
            other_data = pd.concat([data for name, data in regimes.items() if name != regime_name])
            
            if len(other_data) < 20:  # Need minimum training data
                continue
            
            # Train strategy
            self.strategy.train_on_data(other_data)
            
            # Test on this regime
            regime_results = self.strategy.run_backtest(regime_data)
            
            regime_performances[regime_name] = regime_results
        
        return {
            'regimes': regimes,
            'regime_performances': regime_performances
        }
    
    def _identify_market_regimes(self) -> Dict:
        """Identify different market regimes in the data."""
        
        # Calculate volatility regime
        returns = self.data['Close'].pct_change().dropna()
        volatility = returns.rolling(10).std()
        
        # High volatility regime
        high_vol_threshold = volatility.quantile(0.75)
        high_vol_mask = volatility > high_vol_threshold
        
        # Low volatility regime
        low_vol_threshold = volatility.quantile(0.25)
        low_vol_mask = volatility < low_vol_threshold
        
        # Trend regime
        sma_20 = self.data['Close'].rolling(20).mean()
        trend_mask = self.data['Close'] > sma_20
        
        # Mean reversion regime
        mean_rev_mask = self.data['Close'] < sma_20
        
        regimes = {
            'high_volatility': self.data[high_vol_mask],
            'low_volatility': self.data[low_vol_mask],
            'trending': self.data[trend_mask],
            'mean_reverting': self.data[mean_rev_mask]
        }
        
        return regimes
    
    def _get_strategy_parameters(self) -> Dict:
        """Get current strategy parameters."""
        return {
            'signal_thresholds': getattr(self.strategy, 'signal_thresholds', {}),
            'max_features': getattr(self.strategy, 'max_features', 0),
            'regularization_strength': getattr(self.strategy, 'regularization_strength', 0),
            'target_volatility': getattr(self.strategy, 'target_volatility', 0),
            'max_position_size': getattr(self.strategy, 'max_position_size', 0)
        }
    
    def _calculate_parameter_changes(self, initial: Dict, final: Dict) -> Dict:
        """Calculate parameter changes between initial and final states."""
        changes = {}
        for key in initial.keys():
            if key in final:
                if isinstance(initial[key], dict):
                    changes[key] = self._calculate_parameter_changes(initial[key], final[key])
                else:
                    changes[key] = (final[key] - initial[key]) / initial[key] if initial[key] != 0 else 0
        return changes
    
    def _calculate_overfitting_indicators(self, train_metrics: List, test_metrics: List) -> Dict:
        """Calculate overfitting indicators from walk-forward analysis."""
        
        if not train_metrics or not test_metrics:
            return {}
        
        # Calculate average metrics
        avg_train_sharpe = np.mean([m['sharpe'] for m in train_metrics])
        avg_test_sharpe = np.mean([m['sharpe'] for m in test_metrics])
        avg_train_return = np.mean([m['return'] for m in train_metrics])
        avg_test_return = np.mean([m['return'] for m in test_metrics])
        
        # Calculate degradation
        sharpe_degradation = (avg_train_sharpe - avg_test_sharpe) / abs(avg_train_sharpe) if avg_train_sharpe != 0 else 0
        return_degradation = (avg_train_return - avg_test_return) / abs(avg_train_return) if avg_train_return != 0 else 0
        
        # Calculate consistency
        train_sharpe_std = np.std([m['sharpe'] for m in train_metrics])
        test_sharpe_std = np.std([m['sharpe'] for m in test_metrics])
        
        return {
            'sharpe_degradation': sharpe_degradation,
            'return_degradation': return_degradation,
            'train_sharpe_std': train_sharpe_std,
            'test_sharpe_std': test_sharpe_std,
            'is_overfitted': sharpe_degradation > 0.3 or return_degradation > 0.3
        }
    
    def _calculate_parameter_stability(self, parameter_history: List) -> Dict:
        """Calculate parameter stability metrics."""
        
        if not parameter_history:
            return {}
        
        # Extract parameter values
        param_values = {}
        for entry in parameter_history:
            for param_name, param_value in entry['parameters'].items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(param_value)
        
        # Calculate stability metrics
        stability_metrics = {}
        for param_name, values in param_values.items():
            if len(values) > 1:
                stability_metrics[param_name] = {
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                    'range': max(values) - min(values)
                }
        
        return stability_metrics
    
    def _calculate_degradation_trends(self, performance_windows: List) -> Dict:
        """Calculate performance degradation trends."""
        
        trends = {}
        
        for window_data in performance_windows:
            performances = window_data['performances']
            if len(performances) < 2:
                continue
            
            # Calculate trend in Sharpe ratio
            sharpes = [p['sharpe'] for p in performances]
            sharpe_trend = np.polyfit(range(len(sharpes)), sharpes, 1)[0]
            
            # Calculate trend in returns
            returns = [p['return'] for p in performances]
            return_trend = np.polyfit(range(len(returns)), returns, 1)[0]
            
            trends[f"window_{window_data['window_size']}"] = {
                'sharpe_trend': sharpe_trend,
                'return_trend': return_trend,
                'degrading': sharpe_trend < 0 or return_trend < 0
            }
        
        return trends
    
    def _performance_consistency_test(self) -> Dict:
        """Test performance consistency across different time periods."""
        
        # Split data into chunks and test consistency
        chunk_size = len(self.data) // 4
        chunk_performances = []
        
        for i in range(0, len(self.data), chunk_size):
            chunk_data = self.data.iloc[i:i+chunk_size]
            if len(chunk_data) < 10:
                continue
            
            # Train on other chunks
            other_chunks = []
            for j in range(0, len(self.data), chunk_size):
                if j != i:
                    other_chunks.append(self.data.iloc[j:j+chunk_size])
            
            if not other_chunks:
                continue
            
            other_data = pd.concat(other_chunks)
            self.strategy.train_on_data(other_data)
            
            # Test on this chunk
            chunk_results = self.strategy.run_backtest(chunk_data)
            chunk_performances.append(chunk_results['sharpe_ratio'])
        
        # Calculate consistency
        consistency = {
            'performances': chunk_performances,
            'std': np.std(chunk_performances) if chunk_performances else 0,
            'cv': np.std(chunk_performances) / np.mean(chunk_performances) if chunk_performances and np.mean(chunk_performances) != 0 else 0,
            'is_consistent': np.std(chunk_performances) < 0.5 if chunk_performances else True
        }
        
        return consistency
    
    def _parameter_sensitivity_test(self) -> Dict:
        """Test parameter sensitivity to detect overfitting."""
        
        # Test with slightly different parameters
        original_params = self._get_strategy_parameters()
        
        # Train with original parameters
        self.strategy.train_on_data(self.data)
        original_results = self.strategy.run_backtest(self.data)
        
        # Test with perturbed parameters
        perturbations = []
        
        # Perturb signal thresholds
        if 'signal_thresholds' in original_params:
            for threshold_name in ['long', 'short', 'exit']:
                if threshold_name in original_params['signal_thresholds']:
                    original_value = original_params['signal_thresholds'][threshold_name]
                    
                    # Test with 10% perturbation
                    perturbed_value = original_value * 1.1
                    original_params['signal_thresholds'][threshold_name] = perturbed_value
                    
                    # Update strategy parameters
                    self._update_strategy_parameters(original_params)
                    
                    # Retrain and test
                    self.strategy.train_on_data(self.data)
                    perturbed_results = self.strategy.run_backtest(self.data)
                    
                    perturbations.append({
                        'parameter': f'signal_thresholds.{threshold_name}',
                        'original_value': original_value,
                        'perturbed_value': perturbed_value,
                        'original_sharpe': original_results['sharpe_ratio'],
                        'perturbed_sharpe': perturbed_results['sharpe_ratio'],
                        'sensitivity': abs(perturbed_results['sharpe_ratio'] - original_results['sharpe_ratio']) / abs(original_results['sharpe_ratio']) if original_results['sharpe_ratio'] != 0 else 0
                    })
                    
                    # Restore original value
                    original_params['signal_thresholds'][threshold_name] = original_value
        
        # Calculate overall sensitivity
        sensitivities = [p['sensitivity'] for p in perturbations]
        avg_sensitivity = np.mean(sensitivities) if sensitivities else 0
        
        return {
            'perturbations': perturbations,
            'avg_sensitivity': avg_sensitivity,
            'is_sensitive': avg_sensitivity > 0.2
        }
    
    def _randomization_test(self) -> Dict:
        """Randomization test to detect overfitting."""
        
        # Original performance
        self.strategy.train_on_data(self.data)
        original_results = self.strategy.run_backtest(self.data)
        original_sharpe = original_results['sharpe_ratio']
        
        # Test with randomized data
        randomized_performances = []
        
        for _ in range(10):
            # Randomize the data
            randomized_data = self.data.copy()
            randomized_data['Close'] = np.random.permutation(randomized_data['Close'].values)
            
            # Train and test on randomized data
            self.strategy.train_on_data(randomized_data)
            randomized_results = self.strategy.run_backtest(randomized_data)
            randomized_performances.append(randomized_results['sharpe_ratio'])
        
        # Calculate p-value (how often random performance beats original)
        better_than_random = sum(1 for p in randomized_performances if p >= original_sharpe)
        p_value = better_than_random / len(randomized_performances)
        
        return {
            'original_sharpe': original_sharpe,
            'randomized_performances': randomized_performances,
            'p_value': p_value,
            'is_significant': p_value < 0.1
        }
    
    def _cross_validation_stability_test(self) -> Dict:
        """Test cross-validation stability."""
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(self.data):
            if len(train_idx) < 20 or len(test_idx) < 10:
                continue
            
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            # Train and test
            self.strategy.train_on_data(train_data)
            results = self.strategy.run_backtest(test_data)
            cv_scores.append(results['sharpe_ratio'])
        
        # Calculate stability
        stability = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores) if cv_scores else 0,
            'std_score': np.std(cv_scores) if cv_scores else 0,
            'cv_std': np.std(cv_scores) / np.mean(cv_scores) if cv_scores and np.mean(cv_scores) != 0 else 0,
            'is_stable': np.std(cv_scores) < 0.5 if cv_scores else True
        }
        
        return stability
    
    def _update_strategy_parameters(self, params: Dict):
        """Update strategy parameters."""
        for key, value in params.items():
            if hasattr(self.strategy, key):
                setattr(self.strategy, key, value)
    
    def _generate_comprehensive_report(self, results: Dict) -> Dict:
        """Generate comprehensive overfitting report."""
        
        # Calculate overall overfitting score
        overfitting_score = 0
        overfitting_indicators = []
        
        # Walk-forward analysis
        if 'walk_forward' in results:
            indicators = results['walk_forward']['overfitting_indicators']
            if indicators.get('is_overfitted', False):
                overfitting_score += 25
                overfitting_indicators.append("Walk-forward analysis shows significant degradation")
        
        # Out-of-sample testing
        if 'out_of_sample' in results:
            if results['out_of_sample']['is_overfitted']:
                overfitting_score += 25
                overfitting_indicators.append("Out-of-sample performance significantly degraded")
        
        # Parameter stability
        if 'parameter_stability' in results:
            stability_metrics = results['parameter_stability']['stability_metrics']
            unstable_params = [param for param, metrics in stability_metrics.items() 
                             if metrics.get('cv', 0) > 0.5]
            if unstable_params:
                overfitting_score += 15
                overfitting_indicators.append(f"Unstable parameters: {unstable_params}")
        
        # Statistical tests
        if 'statistical_tests' in results:
            tests = results['statistical_tests']
            
            if tests.get('sensitivity_test', {}).get('is_sensitive', False):
                overfitting_score += 15
                overfitting_indicators.append("High parameter sensitivity detected")
            
            if not tests.get('randomization_test', {}).get('is_significant', False):
                overfitting_score += 10
                overfitting_indicators.append("Performance not significantly better than random")
            
            if not tests.get('cv_stability_test', {}).get('is_stable', False):
                overfitting_score += 10
                overfitting_indicators.append("Cross-validation shows instability")
        
        # Determine overfitting level
        if overfitting_score >= 70:
            overfitting_level = "HIGH"
            recommendation = "Significant overfitting detected. Reduce model complexity and increase regularization."
        elif overfitting_score >= 40:
            overfitting_level = "MEDIUM"
            recommendation = "Moderate overfitting detected. Consider parameter tuning and additional validation."
        else:
            overfitting_level = "LOW"
            recommendation = "Low overfitting risk. Strategy appears robust."
        
        return {
            'overfitting_score': overfitting_score,
            'overfitting_level': overfitting_level,
            'overfitting_indicators': overfitting_indicators,
            'recommendation': recommendation,
            'detailed_results': results
        }

def test_overfitting_detection():
    """Test the overfitting detection system."""
    print("🧪 TESTING OVERFITTING DETECTION SYSTEM")
    print("=" * 60)
    
    # Import and use the DatabentoGoldCollector
    try:
        from data_pipeline.databento_collector import DatabentoGoldCollector
        from improved_adaptive_strategy import ImprovedAdaptiveGoldStrategy
        
        print("[INFO] Using DatabentoGoldCollector for real GOLD OHLCV data...")
        collector = DatabentoGoldCollector()
        
        # Fetch more data for better overfitting analysis
        print("[INFO] Fetching extended OHLCV data for overfitting analysis...")
        ohlcv_data = collector.fetch_and_aggregate_gold_mbo_to_ohlcv(
            start_date="2023-07-01", 
            end_date="2023-09-30"
        )
        
        if ohlcv_data.empty:
            print("[ERROR] No OHLCV data fetched from Databento.")
            return
        
        print(f"[INFO] Successfully fetched {len(ohlcv_data)} days of OHLCV data.")
        
        # Convert to the format expected by the strategy
        data = ohlcv_data.copy()
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
    except Exception as e:
        print(f"[ERROR] Failed to import or use DatabentoGoldCollector: {e}")
        return
    
    # Create strategy
    strategy = ImprovedAdaptiveGoldStrategy(
        target_volatility=0.25,
        max_position_size=1.5,
        regularization_strength=0.03,
        max_features=75,
        signal_sensitivity=0.6,
        momentum_weight=0.4,
        mean_reversion_weight=0.3,
        volume_weight=0.3
    )
    
    # Create overfitting detection system
    overfitting_system = OverfittingDetectionSystem(
        strategy=strategy,
        data=data,
        test_periods=5,
        min_train_size=30,
        max_train_size=100
    )
    
    # Run comprehensive analysis
    print("[INFO] Running comprehensive overfitting analysis...")
    results = overfitting_system.run_comprehensive_overfitting_analysis()
    
    # Print results
    report = results['comprehensive_report']
    
    print("\n📊 OVERFITTING DETECTION RESULTS:")
    print(f"   Overfitting Score: {report['overfitting_score']}/100")
    print(f"   Overfitting Level: {report['overfitting_level']}")
    print(f"   Recommendation: {report['recommendation']}")
    
    if report['overfitting_indicators']:
        print("\n⚠️  OVERFITTING INDICATORS:")
        for indicator in report['overfitting_indicators']:
            print(f"   • {indicator}")
    
    # Save detailed results
    with open("overfitting_detection_results.txt", "w") as f:
        f.write("OVERFITTING DETECTION RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Overfitting Score: {report['overfitting_score']}/100\n")
        f.write(f"Overfitting Level: {report['overfitting_level']}\n")
        f.write(f"Recommendation: {report['recommendation']}\n\n")
        
        if report['overfitting_indicators']:
            f.write("OVERFITTING INDICATORS:\n")
            for indicator in report['overfitting_indicators']:
                f.write(f"• {indicator}\n")
    
    print("✅ Results saved to overfitting_detection_results.txt")
    
    return results

if __name__ == "__main__":
    test_overfitting_detection() 