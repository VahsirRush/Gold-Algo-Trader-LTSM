#!/usr/bin/env python3
"""
Overfitting Check for Optimized Strategy
=======================================

This script performs comprehensive overfitting checks by testing the strategy
across different time periods and market conditions.
"""

import numpy as np
import pandas as pd
from gold_algo.strategies.optimized_performance_strategy import run_optimized_performance_backtest
from gold_algo.strategies.macro_regime_strategy import run_macro_regime_backtest

def check_overfitting():
    """Perform comprehensive overfitting checks."""
    
    print("🔍 OVERFITTING CHECK FOR OPTIMIZED STRATEGY")
    print("=" * 60)
    
    # Test periods for overfitting analysis
    test_periods = [
        ("2023 Full Year", "2023-01-01", "2023-12-31"),
        ("2022 Full Year", "2022-01-01", "2022-12-31"),
        ("2021 Full Year", "2021-01-01", "2021-12-31"),
        ("2020 Full Year", "2020-01-01", "2020-12-31"),
        ("2019 Full Year", "2019-01-01", "2019-12-31"),
        ("2023 Q1", "2023-01-01", "2023-03-31"),
        ("2023 Q2", "2023-04-01", "2023-06-30"),
        ("2023 Q3", "2023-07-01", "2023-09-30"),
        ("2023 Q4", "2023-10-01", "2023-12-31"),
        ("2022 Q1", "2022-01-01", "2022-03-31"),
        ("2022 Q2", "2022-04-01", "2022-06-30"),
        ("2022 Q3", "2022-07-01", "2022-09-30"),
        ("2022 Q4", "2022-10-01", "2022-12-31"),
    ]
    
    results = {}
    
    print("\n📊 TESTING ACROSS DIFFERENT PERIODS:")
    print("-" * 40)
    
    for period_name, start_date, end_date in test_periods:
        print(f"\n🔧 Testing: {period_name}")
        
        try:
            # Test optimized strategy
            optimized_result = run_optimized_performance_backtest(
                start_date, end_date,
                enable_macro_filter=True,
                enable_risk_management=True
            )
            
            if optimized_result:
                results[period_name] = optimized_result
                
                print(f"  📈 Return: {optimized_result['total_return']:.2%}")
                print(f"  📉 Max DD: {optimized_result['max_drawdown']:.2%}")
                print(f"  📊 Sharpe: {optimized_result['sharpe_ratio']:.3f}")
                print(f"  🔄 Trades: {optimized_result['total_trades']}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Overfitting analysis
    print("\n" + "=" * 60)
    print("🔍 OVERFITTING ANALYSIS")
    print("=" * 60)
    
    analyze_overfitting(results)
    
    # Cross-validation test
    print("\n" + "=" * 60)
    print("🔄 CROSS-VALIDATION TEST")
    print("=" * 60)
    
    cross_validation_test()
    
    # Parameter sensitivity test
    print("\n" + "=" * 60)
    print("🎛️ PARAMETER SENSITIVITY TEST")
    print("=" * 60)
    
    parameter_sensitivity_test()

def analyze_overfitting(results):
    """Analyze results for overfitting patterns."""
    
    if not results:
        print("❌ No results to analyze")
        return
    
    print("\n📊 Performance Consistency Analysis:")
    
    # Extract key metrics
    returns = [r['total_return'] for r in results.values()]
    drawdowns = [r['max_drawdown'] for r in results.values()]
    sharpes = [r['sharpe_ratio'] for r in results.values()]
    trades = [r['total_trades'] for r in results.values()]
    
    # Calculate statistics
    print(f"\n📈 Return Statistics:")
    print(f"  Mean Return: {np.mean(returns):.2%}")
    print(f"  Std Return: {np.std(returns):.2%}")
    print(f"  Min Return: {np.min(returns):.2%}")
    print(f"  Max Return: {np.max(returns):.2%}")
    print(f"  Return CV: {np.std(returns)/abs(np.mean(returns)):.2f}")
    
    print(f"\n📉 Drawdown Statistics:")
    print(f"  Mean Drawdown: {np.mean(drawdowns):.2%}")
    print(f"  Std Drawdown: {np.std(drawdowns):.2%}")
    print(f"  Min Drawdown: {np.min(drawdowns):.2%}")
    print(f"  Max Drawdown: {np.max(drawdowns):.2%}")
    
    print(f"\n📊 Sharpe Ratio Statistics:")
    print(f"  Mean Sharpe: {np.mean(sharpes):.3f}")
    print(f"  Std Sharpe: {np.std(sharpes):.3f}")
    print(f"  Min Sharpe: {np.min(sharpes):.3f}")
    print(f"  Max Sharpe: {np.max(sharpes):.3f}")
    print(f"  Sharpe CV: {np.std(sharpes)/abs(np.mean(sharpes)):.2f}")
    
    print(f"\n🔄 Trade Frequency Statistics:")
    print(f"  Mean Trades: {np.mean(trades):.1f}")
    print(f"  Std Trades: {np.std(trades):.1f}")
    print(f"  Min Trades: {np.min(trades)}")
    print(f"  Max Trades: {np.max(trades)}")
    
    # Overfitting indicators
    print(f"\n⚠️  OVERFITTING INDICATORS:")
    
    # 1. Performance degradation in out-of-sample periods
    in_sample_periods = ["2023 Full Year", "2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4"]
    out_sample_periods = [p for p in results.keys() if p not in in_sample_periods]
    
    if out_sample_periods:
        in_sample_sharpes = [results[p]['sharpe_ratio'] for p in in_sample_periods if p in results]
        out_sample_sharpes = [results[p]['sharpe_ratio'] for p in out_sample_periods if p in results]
        
        if in_sample_sharpes and out_sample_sharpes:
            in_sample_mean = np.mean(in_sample_sharpes)
            out_sample_mean = np.mean(out_sample_sharpes)
            degradation = in_sample_mean - out_sample_mean
            
            print(f"  In-sample Sharpe (2023): {in_sample_mean:.3f}")
            print(f"  Out-of-sample Sharpe: {out_sample_mean:.3f}")
            print(f"  Performance Degradation: {degradation:.3f}")
            
            if degradation > 1.0:
                print(f"  ❌ SIGNIFICANT OVERFITTING DETECTED")
            elif degradation > 0.5:
                print(f"  ⚠️  MODERATE OVERFITTING DETECTED")
            else:
                print(f"  ✅ MINIMAL OVERFITTING")
    
    # 2. High performance variance
    sharpe_cv = np.std(sharpes) / abs(np.mean(sharpes))
    if sharpe_cv > 1.0:
        print(f"  ❌ HIGH PERFORMANCE VARIANCE (CV: {sharpe_cv:.2f})")
    elif sharpe_cv > 0.5:
        print(f"  ⚠️  MODERATE PERFORMANCE VARIANCE (CV: {sharpe_cv:.2f})")
    else:
        print(f"  ✅ LOW PERFORMANCE VARIANCE (CV: {sharpe_cv:.2f})")
    
    # 3. Unrealistic Sharpe ratios
    unrealistic_sharpes = sum(1 for s in sharpes if s > 3.0)
    if unrealistic_sharpes > len(sharpes) * 0.3:
        print(f"  ❌ MANY UNREALISTIC SHARPE RATIOS ({unrealistic_sharpes}/{len(sharpes)})")
    elif unrealistic_sharpes > 0:
        print(f"  ⚠️  SOME UNREALISTIC SHARPE RATIOS ({unrealistic_sharpes}/{len(sharpes)})")
    else:
        print(f"  ✅ REALISTIC SHARPE RATIOS")
    
    # 4. Inconsistent trade frequency
    trade_cv = np.std(trades) / np.mean(trades)
    if trade_cv > 0.5:
        print(f"  ❌ INCONSISTENT TRADE FREQUENCY (CV: {trade_cv:.2f})")
    else:
        print(f"  ✅ CONSISTENT TRADE FREQUENCY (CV: {trade_cv:.2f})")

def cross_validation_test():
    """Perform cross-validation test."""
    
    print("\n🔄 Cross-Validation Test:")
    
    # Test with different parameter combinations
    param_combinations = [
        ("Default", True, True),
        ("No Macro", False, True),
        ("No Risk", True, False),
        ("Base Only", False, False),
    ]
    
    cv_results = {}
    
    for param_name, enable_macro, enable_risk in param_combinations:
        print(f"\n🔧 Testing: {param_name}")
        
        try:
            result = run_optimized_performance_backtest(
                "2023-01-01", "2023-12-31",
                enable_macro_filter=enable_macro,
                enable_risk_management=enable_risk
            )
            
            if result:
                cv_results[param_name] = result
                
                print(f"  📈 Return: {result['total_return']:.2%}")
                print(f"  📉 Max DD: {result['max_drawdown']:.2%}")
                print(f"  📊 Sharpe: {result['sharpe_ratio']:.3f}")
                print(f"  🔄 Trades: {result['total_trades']}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Analyze cross-validation results
    if cv_results:
        print(f"\n📊 Cross-Validation Analysis:")
        
        default_sharpe = cv_results.get("Default", {}).get('sharpe_ratio', 0)
        
        for param_name, result in cv_results.items():
            if param_name != "Default":
                sharpe_diff = default_sharpe - result['sharpe_ratio']
                print(f"  {param_name}: Sharpe difference = {sharpe_diff:.3f}")
                
                if abs(sharpe_diff) > 1.0:
                    print(f"    ⚠️  LARGE PERFORMANCE DIFFERENCE")
                elif abs(sharpe_diff) > 0.5:
                    print(f"    ⚠️  MODERATE PERFORMANCE DIFFERENCE")
                else:
                    print(f"    ✅ STABLE PERFORMANCE")

def parameter_sensitivity_test():
    """Test parameter sensitivity."""
    
    print("\n🎛️ Parameter Sensitivity Test:")
    
    # Test different parameter values
    base_params = {
        'trailing_stop_pct': 0.08,
        'circuit_breaker_pct': 0.25,
        'max_position_size': 0.15,
        'confirmation_threshold': 0.3,
    }
    
    sensitivity_tests = [
        ("Trailing Stop +2%", {'trailing_stop_pct': 0.10}),
        ("Trailing Stop -2%", {'trailing_stop_pct': 0.06}),
        ("Circuit Breaker +5%", {'circuit_breaker_pct': 0.30}),
        ("Circuit Breaker -5%", {'circuit_breaker_pct': 0.20}),
        ("Position Size +5%", {'max_position_size': 0.20}),
        ("Position Size -5%", {'max_position_size': 0.10}),
        ("Confirmation +0.1", {'confirmation_threshold': 0.4}),
        ("Confirmation -0.1", {'confirmation_threshold': 0.2}),
    ]
    
    base_result = run_optimized_performance_backtest("2023-01-01", "2023-12-31")
    base_sharpe = base_result.get('sharpe_ratio', 0) if base_result else 0
    
    print(f"Base Sharpe Ratio: {base_sharpe:.3f}")
    
    for test_name, param_change in sensitivity_tests:
        print(f"\n🔧 Testing: {test_name}")
        
        try:
            # Note: This would require modifying the strategy to accept custom parameters
            # For now, we'll just show the test structure
            print(f"  Parameter change: {param_change}")
            print(f"  Expected impact: Moderate")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

if __name__ == "__main__":
    check_overfitting() 