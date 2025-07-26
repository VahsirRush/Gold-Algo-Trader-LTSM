#!/usr/bin/env python3
"""
SIMPLE OVERFITTING DETECTION SYSTEM
==================================

Simple but effective overfitting detection system that:
- Avoids complex index alignment issues
- Provides clear, actionable results
- Tests strategy robustness
- Identifies potential overfitting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class SimpleOverfittingDetection:
    """Simple overfitting detection and analysis system."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.results = {}
        
    def run_analysis(self) -> Dict:
        """Run comprehensive overfitting analysis."""
        print("🔍 SIMPLE OVERFITTING DETECTION ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        # 1. Data analysis
        print("📊 Analyzing data characteristics...")
        results['data_analysis'] = self._analyze_data()
        
        # 2. Performance stability test
        print("📈 Testing performance stability...")
        results['performance_stability'] = self._test_performance_stability()
        
        # 3. Parameter sensitivity test
        print("🔧 Testing parameter sensitivity...")
        results['parameter_sensitivity'] = self._test_parameter_sensitivity()
        
        # 4. Time period consistency test
        print("⏰ Testing time period consistency...")
        results['time_consistency'] = self._test_time_consistency()
        
        # 5. Generate report
        print("📋 Generating comprehensive report...")
        results['report'] = self._generate_report(results)
        
        return results
    
    def _analyze_data(self) -> Dict:
        """Analyze data characteristics."""
        
        # Basic info
        data_info = {
            'total_days': len(self.data),
            'date_range': f"{self.data.index.min()} to {self.data.index.max()}",
            'missing_values': self.data.isnull().sum().sum()
        }
        
        # Price analysis
        returns = self.data['Close'].pct_change().dropna()
        price_analysis = {
            'price_range': {
                'min': self.data['Close'].min(),
                'max': self.data['Close'].max(),
                'mean': self.data['Close'].mean()
            },
            'returns_stats': {
                'mean': returns.mean(),
                'std': returns.std(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            }
        }
        
        return {
            'data_info': data_info,
            'price_analysis': price_analysis
        }
    
    def _test_performance_stability(self) -> Dict:
        """Test performance stability across different periods."""
        
        returns = self.data['Close'].pct_change().dropna()
        
        # Split into quarters
        quarter_size = len(returns) // 4
        quarter_performances = []
        
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = start_idx + quarter_size if i < 3 else len(returns)
            
            quarter_returns = returns.iloc[start_idx:end_idx]
            if len(quarter_returns) > 5:
                quarter_performances.append({
                    'quarter': i + 1,
                    'mean_return': quarter_returns.mean(),
                    'std_return': quarter_returns.std(),
                    'sharpe': quarter_returns.mean() / quarter_returns.std() if quarter_returns.std() > 0 else 0,
                    'total_return': (1 + quarter_returns).prod() - 1
                })
        
        # Calculate stability metrics
        if quarter_performances:
            sharpes = [q['sharpe'] for q in quarter_performances]
            returns_list = [q['total_return'] for q in quarter_performances]
            
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
            'quarter_performances': quarter_performances,
            'stability_metrics': stability_metrics
        }
    
    def _test_parameter_sensitivity(self) -> Dict:
        """Test sensitivity to parameter changes."""
        
        # Test different moving average periods
        ma_periods = [5, 10, 20, 50]
        ma_results = []
        
        for period in ma_periods:
            if period < len(self.data):
                ma = self.data['Close'].rolling(period).mean()
                ma_signals = (self.data['Close'] > ma).astype(int)
                ma_returns = ma_signals.shift(1) * self.data['Close'].pct_change()
                ma_returns = ma_returns.dropna()
                
                if len(ma_returns) > 0:
                    ma_results.append({
                        'period': period,
                        'sharpe': ma_returns.mean() / ma_returns.std() if ma_returns.std() > 0 else 0,
                        'total_return': (1 + ma_returns).prod() - 1
                    })
        
        # Test different volatility thresholds
        vol_thresholds = [0.01, 0.02, 0.03, 0.05]
        vol_results = []
        
        returns = self.data['Close'].pct_change().dropna()
        for threshold in vol_thresholds:
            vol_filter = returns.rolling(20).std() < threshold
            vol_returns = vol_filter.shift(1) * returns
            vol_returns = vol_returns.dropna()
            
            if len(vol_returns) > 0:
                vol_results.append({
                    'threshold': threshold,
                    'sharpe': vol_returns.mean() / vol_returns.std() if vol_returns.std() > 0 else 0,
                    'total_return': (1 + vol_returns).prod() - 1
                })
        
        # Calculate sensitivity
        if ma_results:
            ma_sharpes = [m['sharpe'] for m in ma_results]
            ma_sensitivity = np.std(ma_sharpes) / np.mean(ma_sharpes) if np.mean(ma_sharpes) != 0 else 0
        else:
            ma_sensitivity = 0
        
        if vol_results:
            vol_sharpes = [v['sharpe'] for v in vol_results]
            vol_sensitivity = np.std(vol_sharpes) / np.mean(vol_sharpes) if np.mean(vol_sharpes) != 0 else 0
        else:
            vol_sensitivity = 0
        
        return {
            'ma_results': ma_results,
            'vol_results': vol_results,
            'ma_sensitivity': ma_sensitivity,
            'vol_sensitivity': vol_sensitivity,
            'overall_sensitivity': (ma_sensitivity + vol_sensitivity) / 2,
            'is_sensitive': (ma_sensitivity + vol_sensitivity) / 2 > 0.3
        }
    
    def _test_time_consistency(self) -> Dict:
        """Test consistency across different time periods."""
        
        # Split data into different periods
        periods = {
            'first_half': self.data.iloc[:len(self.data)//2],
            'second_half': self.data.iloc[len(self.data)//2:],
            'first_third': self.data.iloc[:len(self.data)//3],
            'middle_third': self.data.iloc[len(self.data)//3:2*len(self.data)//3],
            'last_third': self.data.iloc[2*len(self.data)//3:]
        }
        
        period_results = {}
        
        for period_name, period_data in periods.items():
            if len(period_data) < 10:
                continue
            
            returns = period_data['Close'].pct_change().dropna()
            
            if len(returns) > 0:
                period_results[period_name] = {
                    'mean_return': returns.mean(),
                    'std_return': returns.std(),
                    'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                    'total_return': (1 + returns).prod() - 1
                }
        
        # Calculate consistency
        if len(period_results) > 1:
            sharpes = [p['sharpe'] for p in period_results.values()]
            returns_list = [p['total_return'] for p in period_results.values()]
            
            consistency_metrics = {
                'sharpe_std': np.std(sharpes),
                'return_std': np.std(returns_list),
                'sharpe_cv': np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) != 0 else 0,
                'return_cv': np.std(returns_list) / np.mean(returns_list) if np.mean(returns_list) != 0 else 0,
                'is_consistent': np.std(sharpes) < 0.5 and np.std(returns_list) < 0.1
            }
        else:
            consistency_metrics = {'is_consistent': True}
        
        return {
            'period_results': period_results,
            'consistency_metrics': consistency_metrics
        }
    
    def _generate_report(self, results: Dict) -> Dict:
        """Generate comprehensive overfitting report."""
        
        # Calculate overfitting score
        overfitting_score = 0
        overfitting_indicators = []
        
        # Performance stability
        if 'performance_stability' in results:
            stability = results['performance_stability']['stability_metrics']
            if not stability.get('is_stable', True):
                overfitting_score += 30
                overfitting_indicators.append("Performance unstable across quarters")
        
        # Parameter sensitivity
        if 'parameter_sensitivity' in results:
            sensitivity = results['parameter_sensitivity']
            if sensitivity.get('is_sensitive', False):
                overfitting_score += 30
                overfitting_indicators.append("High parameter sensitivity detected")
        
        # Time consistency
        if 'time_consistency' in results:
            consistency = results['time_consistency']['consistency_metrics']
            if not consistency.get('is_consistent', True):
                overfitting_score += 40
                overfitting_indicators.append("Performance inconsistent across time periods")
        
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

def test_simple_overfitting_detection():
    """Test the simple overfitting detection system."""
    print("🧪 TESTING SIMPLE OVERFITTING DETECTION SYSTEM")
    print("=" * 60)
    
    # Import and use the DatabentoGoldCollector
    try:
        from data_pipeline.databento_collector import DatabentoGoldCollector
        
        print("[INFO] Using DatabentoGoldCollector for real GOLD OHLCV data...")
        collector = DatabentoGoldCollector()
        
        # Fetch extended data for better analysis
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
    
    # Create simple overfitting detection system
    overfitting_system = SimpleOverfittingDetection(data=data)
    
    # Run analysis
    print("[INFO] Running simple overfitting analysis...")
    results = overfitting_system.run_analysis()
    
    # Print results
    report = results['report']
    
    print("\n📊 SIMPLE OVERFITTING DETECTION RESULTS:")
    print(f"   Overfitting Score: {report['overfitting_score']}/100")
    print(f"   Overfitting Level: {report['overfitting_level']}")
    print(f"   Recommendation: {report['recommendation']}")
    
    if report['overfitting_indicators']:
        print("\n⚠️  OVERFITTING INDICATORS:")
        for indicator in report['overfitting_indicators']:
            print(f"   • {indicator}")
    
    # Print detailed analysis
    print("\n📈 DETAILED ANALYSIS:")
    
    # Data analysis
    data_analysis = results['data_analysis']
    print(f"   Data Points: {data_analysis['data_info']['total_days']}")
    print(f"   Date Range: {data_analysis['data_info']['date_range']}")
    print(f"   Price Range: ${data_analysis['price_analysis']['price_range']['min']:.2f} - ${data_analysis['price_analysis']['price_range']['max']:.2f}")
    
    # Performance stability
    stability = results['performance_stability']['stability_metrics']
    print(f"   Performance Stability: {'✅ Stable' if stability.get('is_stable', True) else '❌ Unstable'}")
    
    # Parameter sensitivity
    sensitivity = results['parameter_sensitivity']
    print(f"   Parameter Sensitivity: {'❌ High' if sensitivity.get('is_sensitive', False) else '✅ Low'}")
    
    # Time consistency
    consistency = results['time_consistency']['consistency_metrics']
    print(f"   Time Consistency: {'✅ Consistent' if consistency.get('is_consistent', True) else '❌ Inconsistent'}")
    
    # Print parameter sensitivity details
    print("\n🔧 PARAMETER SENSITIVITY DETAILS:")
    ma_results = results['parameter_sensitivity']['ma_results']
    if ma_results:
        print("   Moving Average Sensitivity:")
        for result in ma_results:
            print(f"     MA{result['period']}: Sharpe={result['sharpe']:.3f}, Return={result['total_return']:.3f}")
    
    vol_results = results['parameter_sensitivity']['vol_results']
    if vol_results:
        print("   Volatility Threshold Sensitivity:")
        for result in vol_results:
            print(f"     Vol{result['threshold']}: Sharpe={result['sharpe']:.3f}, Return={result['total_return']:.3f}")
    
    # Save detailed results
    with open("simple_overfitting_results.txt", "w") as f:
        f.write("SIMPLE OVERFITTING DETECTION RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Overfitting Score: {report['overfitting_score']}/100\n")
        f.write(f"Overfitting Level: {report['overfitting_level']}\n")
        f.write(f"Recommendation: {report['recommendation']}\n\n")
        
        if report['overfitting_indicators']:
            f.write("OVERFITTING INDICATORS:\n")
            for indicator in report['overfitting_indicators']:
                f.write(f"• {indicator}\n")
        
        f.write(f"\nDATA ANALYSIS:\n")
        f.write(f"Total Days: {data_analysis['data_info']['total_days']}\n")
        f.write(f"Date Range: {data_analysis['data_info']['date_range']}\n")
        f.write(f"Price Range: ${data_analysis['price_analysis']['price_range']['min']:.2f} - ${data_analysis['price_analysis']['price_range']['max']:.2f}\n")
        
        f.write(f"\nPERFORMANCE STABILITY:\n")
        f.write(f"Stable: {stability.get('is_stable', True)}\n")
        f.write(f"Sharpe CV: {stability.get('sharpe_cv', 0):.3f}\n")
        f.write(f"Return CV: {stability.get('return_cv', 0):.3f}\n")
        
        f.write(f"\nPARAMETER SENSITIVITY:\n")
        f.write(f"Sensitive: {sensitivity.get('is_sensitive', False)}\n")
        f.write(f"MA Sensitivity: {sensitivity.get('ma_sensitivity', 0):.3f}\n")
        f.write(f"Vol Sensitivity: {sensitivity.get('vol_sensitivity', 0):.3f}\n")
        
        f.write(f"\nTIME CONSISTENCY:\n")
        f.write(f"Consistent: {consistency.get('is_consistent', True)}\n")
        f.write(f"Sharpe CV: {consistency.get('sharpe_cv', 0):.3f}\n")
        f.write(f"Return CV: {consistency.get('return_cv', 0):.3f}\n")
    
    print("✅ Results saved to simple_overfitting_results.txt")
    
    return results

if __name__ == "__main__":
    test_simple_overfitting_detection() 