#!/usr/bin/env python3
"""
ROBUST OVERFITTING DETECTION SYSTEM
==================================

Enhanced overfitting detection system with:
- Robust data handling
- Multiple validation methods
- Comprehensive statistical analysis
- Performance stability testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RobustOverfittingDetection:
    """Robust overfitting detection and analysis system."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.results = {}
        
    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive overfitting analysis."""
        print("🔍 ROBUST OVERFITTING DETECTION ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        # 1. Data quality analysis
        print("📊 Analyzing data quality...")
        results['data_quality'] = self._analyze_data_quality()
        
        # 2. Performance consistency analysis
        print("📈 Analyzing performance consistency...")
        results['performance_consistency'] = self._analyze_performance_consistency()
        
        # 3. Parameter sensitivity analysis
        print("🔧 Analyzing parameter sensitivity...")
        results['parameter_sensitivity'] = self._analyze_parameter_sensitivity()
        
        # 4. Time period analysis
        print("⏰ Analyzing time period stability...")
        results['time_period_analysis'] = self._analyze_time_periods()
        
        # 5. Statistical robustness tests
        print("📊 Running statistical robustness tests...")
        results['statistical_tests'] = self._run_statistical_tests()
        
        # 6. Market regime analysis
        print("🌍 Analyzing market regimes...")
        results['market_regime_analysis'] = self._analyze_market_regimes()
        
        # 7. Generate comprehensive report
        print("📋 Generating comprehensive report...")
        results['comprehensive_report'] = self._generate_comprehensive_report(results)
        
        return results
    
    def _analyze_data_quality(self) -> Dict:
        """Analyze data quality and characteristics."""
        
        # Basic data info
        data_info = {
            'total_days': len(self.data),
            'date_range': f"{self.data.index.min()} to {self.data.index.max()}",
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }
        
        # Price analysis
        price_analysis = {
            'price_range': {
                'min': self.data['Close'].min(),
                'max': self.data['Close'].max(),
                'mean': self.data['Close'].mean(),
                'std': self.data['Close'].std()
            },
            'returns_analysis': {
                'mean_return': self.data['Close'].pct_change().mean(),
                'std_return': self.data['Close'].pct_change().std(),
                'skewness': self.data['Close'].pct_change().skew(),
                'kurtosis': self.data['Close'].pct_change().kurtosis()
            }
        }
        
        # Volume analysis
        volume_analysis = {
            'volume_stats': {
                'mean': self.data['Volume'].mean(),
                'std': self.data['Volume'].std(),
                'min': self.data['Volume'].min(),
                'max': self.data['Volume'].max()
            }
        }
        
        return {
            'data_info': data_info,
            'price_analysis': price_analysis,
            'volume_analysis': volume_analysis
        }
    
    def _analyze_performance_consistency(self) -> Dict:
        """Analyze performance consistency across different time periods."""
        
        # Calculate daily returns
        returns = self.data['Close'].pct_change().dropna()
        
        # Split data into quarters
        quarter_performances = []
        for i in range(0, len(returns), len(returns)//4):
            quarter_returns = returns.iloc[i:i+len(returns)//4]
            if len(quarter_returns) > 5:  # Minimum data points
                quarter_performances.append({
                    'quarter': i//(len(returns)//4) + 1,
                    'mean_return': quarter_returns.mean(),
                    'std_return': quarter_returns.std(),
                    'sharpe': quarter_returns.mean() / quarter_returns.std() if quarter_returns.std() > 0 else 0,
                    'total_return': (1 + quarter_returns).prod() - 1
                })
        
        # Calculate consistency metrics
        if quarter_performances:
            sharpe_ratios = [q['sharpe'] for q in quarter_performances]
            returns_list = [q['total_return'] for q in quarter_performances]
            
            consistency_metrics = {
                'sharpe_std': np.std(sharpe_ratios),
                'return_std': np.std(returns_list),
                'sharpe_cv': np.std(sharpe_ratios) / np.mean(sharpe_ratios) if np.mean(sharpe_ratios) != 0 else 0,
                'return_cv': np.std(returns_list) / np.mean(returns_list) if np.mean(returns_list) != 0 else 0,
                'is_consistent': np.std(sharpe_ratios) < 0.5 and np.std(returns_list) < 0.1
            }
        else:
            consistency_metrics = {'is_consistent': True}
        
        return {
            'quarter_performances': quarter_performances,
            'consistency_metrics': consistency_metrics
        }
    
    def _analyze_parameter_sensitivity(self) -> Dict:
        """Analyze sensitivity to parameter changes."""
        
        # Test different moving average periods
        ma_periods = [5, 10, 20, 50]
        ma_sensitivities = []
        
        for period in ma_periods:
            if period < len(self.data):
                ma = self.data['Close'].rolling(period).mean()
                ma_signals = (self.data['Close'] > ma).astype(int)
                ma_returns = ma_signals.shift(1) * self.data['Close'].pct_change()
                ma_sharpe = ma_returns.mean() / ma_returns.std() if ma_returns.std() > 0 else 0
                
                ma_sensitivities.append({
                    'period': period,
                    'sharpe': ma_sharpe,
                    'total_return': (1 + ma_returns).prod() - 1
                })
        
        # Test different volatility thresholds
        vol_thresholds = [0.01, 0.02, 0.03, 0.05]
        vol_sensitivities = []
        
        returns = self.data['Close'].pct_change()
        for threshold in vol_thresholds:
            vol_filter = returns.rolling(20).std() < threshold
            vol_returns = vol_filter.shift(1) * returns
            vol_sharpe = vol_returns.mean() / vol_returns.std() if vol_returns.std() > 0 else 0
            
            vol_sensitivities.append({
                'threshold': threshold,
                'sharpe': vol_sharpe,
                'total_return': (1 + vol_returns).prod() - 1
            })
        
        # Calculate overall sensitivity
        if ma_sensitivities:
            ma_sharpes = [m['sharpe'] for m in ma_sensitivities]
            ma_sensitivity = np.std(ma_sharpes) / np.mean(ma_sharpes) if np.mean(ma_sharpes) != 0 else 0
        else:
            ma_sensitivity = 0
        
        if vol_sensitivities:
            vol_sharpes = [v['sharpe'] for v in vol_sensitivities]
            vol_sensitivity = np.std(vol_sharpes) / np.mean(vol_sharpes) if np.mean(vol_sharpes) != 0 else 0
        else:
            vol_sensitivity = 0
        
        return {
            'ma_sensitivities': ma_sensitivities,
            'vol_sensitivities': vol_sensitivities,
            'ma_sensitivity': ma_sensitivity,
            'vol_sensitivity': vol_sensitivity,
            'overall_sensitivity': (ma_sensitivity + vol_sensitivity) / 2,
            'is_sensitive': (ma_sensitivity + vol_sensitivity) / 2 > 0.3
        }
    
    def _analyze_time_periods(self) -> Dict:
        """Analyze performance across different time periods."""
        
        # Split data into different time periods
        periods = {
            'first_half': self.data.iloc[:len(self.data)//2],
            'second_half': self.data.iloc[len(self.data)//2:],
            'first_third': self.data.iloc[:len(self.data)//3],
            'middle_third': self.data.iloc[len(self.data)//3:2*len(self.data)//3],
            'last_third': self.data.iloc[2*len(self.data)//3:]
        }
        
        period_performances = {}
        
        for period_name, period_data in periods.items():
            if len(period_data) < 10:  # Minimum data points
                continue
            
            returns = period_data['Close'].pct_change().dropna()
            
            period_performances[period_name] = {
                'mean_return': returns.mean(),
                'std_return': returns.std(),
                'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'total_return': (1 + returns).prod() - 1,
                'max_drawdown': self._calculate_max_drawdown(period_data['Close'])
            }
        
        # Calculate stability across periods
        if len(period_performances) > 1:
            sharpes = [p['sharpe'] for p in period_performances.values()]
            returns_list = [p['total_return'] for p in period_performances.values()]
            
            stability_metrics = {
                'sharpe_std': np.std(sharpes),
                'return_std': np.std(returns_list),
                'sharpe_cv': np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) != 0 else 0,
                'return_cv': np.std(returns_list) / np.mean(returns_list) if np.mean(returns_list) != 0 else 0,
                'is_stable': np.std(sharpes) < 0.5 and np.std(returns_list) < 0.1
            }
        else:
            stability_metrics = {'is_stable': True}
        
        return {
            'period_performances': period_performances,
            'stability_metrics': stability_metrics
        }
    
    def _run_statistical_tests(self) -> Dict:
        """Run statistical tests for robustness."""
        
        returns = self.data['Close'].pct_change().dropna()
        
        # 1. Normality test (simplified)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        is_normal = abs(skewness) < 1 and abs(kurtosis) < 3
        
        # 2. Stationarity test (simplified)
        # Split data and compare means
        first_half = returns.iloc[:len(returns)//2]
        second_half = returns.iloc[len(returns)//2:]
        
        mean_diff = abs(first_half.mean() - second_half.mean())
        is_stationary = mean_diff < 0.001
        
        # 3. Autocorrelation test
        autocorr = returns.autocorr()
        has_autocorr = abs(autocorr) > 0.1
        
        # 4. Volatility clustering test
        vol_clustering = returns.rolling(5).std().autocorr()
        has_vol_clustering = abs(vol_clustering) > 0.1
        
        return {
            'normality_test': {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': is_normal
            },
            'stationarity_test': {
                'mean_difference': mean_diff,
                'is_stationary': is_stationary
            },
            'autocorrelation_test': {
                'autocorrelation': autocorr,
                'has_autocorrelation': has_autocorr
            },
            'volatility_clustering_test': {
                'vol_clustering': vol_clustering,
                'has_vol_clustering': has_vol_clustering
            }
        }
    
    def _analyze_market_regimes(self) -> Dict:
        """Analyze different market regimes."""
        
        # Identify market regimes
        returns = self.data['Close'].pct_change().dropna()
        
        # Volatility regime
        volatility = returns.rolling(10).std()
        high_vol_threshold = volatility.quantile(0.75)
        low_vol_threshold = volatility.quantile(0.25)
        
        # Trend regime
        sma_20 = self.data['Close'].rolling(20).mean()
        trend_mask = self.data['Close'] > sma_20
        
        # Volume regime
        volume_ma = self.data['Volume'].rolling(20).mean()
        high_volume_mask = self.data['Volume'] > volume_ma * 1.5
        low_volume_mask = self.data['Volume'] < volume_ma * 0.5
        
        # Create regime masks
        regimes = {
            'high_volatility': volatility > high_vol_threshold,
            'low_volatility': volatility < low_vol_threshold,
            'trending': trend_mask,
            'mean_reverting': ~trend_mask,
            'high_volume': high_volume_mask,
            'low_volume': low_volume_mask
        }
        
        # Analyze performance in each regime
        regime_performances = {}
        
        for regime_name, regime_mask in regimes.items():
            regime_returns = returns[regime_mask.iloc[1:]]  # Align indices
            if len(regime_returns) > 5:  # Minimum data points
                regime_performances[regime_name] = {
                    'count': len(regime_returns),
                    'mean_return': regime_returns.mean(),
                    'std_return': regime_returns.std(),
                    'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'total_return': (1 + regime_returns).prod() - 1
                }
        
        return {
            'regimes': regimes,
            'regime_performances': regime_performances
        }
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def _generate_comprehensive_report(self, results: Dict) -> Dict:
        """Generate comprehensive overfitting report."""
        
        # Calculate overall overfitting score
        overfitting_score = 0
        overfitting_indicators = []
        
        # Performance consistency
        if 'performance_consistency' in results:
            consistency = results['performance_consistency']['consistency_metrics']
            if not consistency.get('is_consistent', True):
                overfitting_score += 25
                overfitting_indicators.append("Performance inconsistent across quarters")
        
        # Parameter sensitivity
        if 'parameter_sensitivity' in results:
            sensitivity = results['parameter_sensitivity']
            if sensitivity.get('is_sensitive', False):
                overfitting_score += 25
                overfitting_indicators.append("High parameter sensitivity detected")
        
        # Time period stability
        if 'time_period_analysis' in results:
            stability = results['time_period_analysis']['stability_metrics']
            if not stability.get('is_stable', True):
                overfitting_score += 20
                overfitting_indicators.append("Performance unstable across time periods")
        
        # Statistical tests
        if 'statistical_tests' in results:
            tests = results['statistical_tests']
            
            if not tests.get('normality_test', {}).get('is_normal', True):
                overfitting_score += 10
                overfitting_indicators.append("Returns not normally distributed")
            
            if not tests.get('stationarity_test', {}).get('is_stationary', True):
                overfitting_score += 10
                overfitting_indicators.append("Data not stationary")
        
        # Market regime analysis
        if 'market_regime_analysis' in results:
            regime_performances = results['market_regime_analysis']['regime_performances']
            if len(regime_performances) > 1:
                regime_sharpes = [p['sharpe'] for p in regime_performances.values()]
                regime_std = np.std(regime_sharpes)
                if regime_std > 0.5:
                    overfitting_score += 10
                    overfitting_indicators.append("High performance variance across market regimes")
        
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

def test_robust_overfitting_detection():
    """Test the robust overfitting detection system."""
    print("🧪 TESTING ROBUST OVERFITTING DETECTION SYSTEM")
    print("=" * 60)
    
    # Import and use the DatabentoGoldCollector
    try:
        from data_pipeline.databento_collector import DatabentoGoldCollector
        
        print("[INFO] Using DatabentoGoldCollector for real GOLD OHLCV data...")
        collector = DatabentoGoldCollector()
        
        # Fetch extended data for better analysis
        print("[INFO] Fetching extended OHLCV data for robust overfitting analysis...")
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
    
    # Create robust overfitting detection system
    overfitting_system = RobustOverfittingDetection(data=data)
    
    # Run comprehensive analysis
    print("[INFO] Running robust overfitting analysis...")
    results = overfitting_system.run_comprehensive_analysis()
    
    # Print results
    report = results['comprehensive_report']
    
    print("\n📊 ROBUST OVERFITTING DETECTION RESULTS:")
    print(f"   Overfitting Score: {report['overfitting_score']}/100")
    print(f"   Overfitting Level: {report['overfitting_level']}")
    print(f"   Recommendation: {report['recommendation']}")
    
    if report['overfitting_indicators']:
        print("\n⚠️  OVERFITTING INDICATORS:")
        for indicator in report['overfitting_indicators']:
            print(f"   • {indicator}")
    
    # Print detailed analysis
    print("\n📈 DETAILED ANALYSIS:")
    
    # Data quality
    data_quality = results['data_quality']
    print(f"   Data Points: {data_quality['data_info']['total_days']}")
    print(f"   Date Range: {data_quality['data_info']['date_range']}")
    print(f"   Price Range: ${data_quality['price_analysis']['price_range']['min']:.2f} - ${data_quality['price_analysis']['price_range']['max']:.2f}")
    
    # Performance consistency
    consistency = results['performance_consistency']['consistency_metrics']
    print(f"   Performance Consistency: {'✅ Stable' if consistency.get('is_consistent', True) else '❌ Unstable'}")
    
    # Parameter sensitivity
    sensitivity = results['parameter_sensitivity']
    print(f"   Parameter Sensitivity: {'❌ High' if sensitivity.get('is_sensitive', False) else '✅ Low'}")
    
    # Time period stability
    stability = results['time_period_analysis']['stability_metrics']
    print(f"   Time Period Stability: {'✅ Stable' if stability.get('is_stable', True) else '❌ Unstable'}")
    
    # Save detailed results
    with open("robust_overfitting_results.txt", "w") as f:
        f.write("ROBUST OVERFITTING DETECTION RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Overfitting Score: {report['overfitting_score']}/100\n")
        f.write(f"Overfitting Level: {report['overfitting_level']}\n")
        f.write(f"Recommendation: {report['recommendation']}\n\n")
        
        if report['overfitting_indicators']:
            f.write("OVERFITTING INDICATORS:\n")
            for indicator in report['overfitting_indicators']:
                f.write(f"• {indicator}\n")
        
        f.write(f"\nDATA QUALITY:\n")
        f.write(f"Total Days: {data_quality['data_info']['total_days']}\n")
        f.write(f"Date Range: {data_quality['data_info']['date_range']}\n")
        f.write(f"Price Range: ${data_quality['price_analysis']['price_range']['min']:.2f} - ${data_quality['price_analysis']['price_range']['max']:.2f}\n")
    
    print("✅ Results saved to robust_overfitting_results.txt")
    
    return results

if __name__ == "__main__":
    test_robust_overfitting_detection() 