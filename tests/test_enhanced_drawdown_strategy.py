#!/usr/bin/env python3
"""
Test Enhanced Drawdown Strategy with Risk Management
==================================================

This script demonstrates the enhanced drawdown strategy with comprehensive
risk management including:
- Trailing stop-losses and circuit breakers
- Dynamic position sizing based on drawdown levels
- Monte Carlo stress testing
- Performance comparison with and without risk management
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

sys.path.append(os.path.dirname(__file__))

from gold_algo.strategies.enhanced_drawdown_strategy import run_enhanced_drawdown_backtest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_drawdown_strategy():
    """Test the enhanced drawdown strategy with comprehensive risk management."""
    
    print("🛡️  ENHANCED DRAWDOWN STRATEGY WITH RISK MANAGEMENT")
    print("=" * 70)
    
    # Test parameters
    test_periods = [
        ("Original Period (Jul-Sep 2023)", "2023-07-01", "2023-09-30"),
        ("Earlier Period (Apr-Jun 2023)", "2023-04-01", "2023-06-30"),
        ("Later Period (Oct-Dec 2023)", "2023-10-01", "2023-12-31")
    ]
    
    risk_configs = [
        ("Conservative", 0.03, 0.10),  # 3% trailing stop, 10% circuit breaker
        ("Moderate", 0.05, 0.15),      # 5% trailing stop, 15% circuit breaker
        ("Aggressive", 0.08, 0.20)     # 8% trailing stop, 20% circuit breaker
    ]
    
    all_results = {}
    
    for period_name, start_date, end_date in test_periods:
        print(f"\n📊 Testing Period: {period_name}")
        print("-" * 50)
        
        period_results = {}
        
        for config_name, trailing_stop, circuit_breaker in risk_configs:
            print(f"\n🔧 Risk Config: {config_name} ({trailing_stop:.1%} trailing, {circuit_breaker:.1%} circuit)")
            
            # Test with risk management enabled
            results_with_risk = run_enhanced_drawdown_backtest(
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000.0,
                trailing_stop_pct=trailing_stop,
                circuit_breaker_pct=circuit_breaker,
                enable_risk_management=True
            )
            
            # Test without risk management (baseline)
            results_without_risk = run_enhanced_drawdown_backtest(
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000.0,
                trailing_stop_pct=trailing_stop,
                circuit_breaker_pct=circuit_breaker,
                enable_risk_management=False
            )
            
            if results_with_risk and results_without_risk:
                # Compare results
                comparison = {
                    'with_risk': results_with_risk,
                    'without_risk': results_without_risk,
                    'risk_improvement': {
                        'drawdown_reduction': results_without_risk['max_drawdown'] - results_with_risk['max_drawdown'],
                        'sharpe_improvement': results_with_risk['sharpe_ratio'] - results_without_risk['sharpe_ratio'],
                        'return_impact': results_with_risk['total_return'] - results_without_risk['total_return']
                    }
                }
                
                period_results[config_name] = comparison
                
                # Display results
                print(f"  📈 With Risk Management:")
                print(f"     Total Return: {results_with_risk['total_return']:.2%}")
                print(f"     Max Drawdown: {results_with_risk['max_drawdown']:.2%}")
                print(f"     Sharpe Ratio: {results_with_risk['sharpe_ratio']:.3f}")
                print(f"     Total Trades: {results_with_risk['total_trades']}")
                
                if 'risk_management' in results_with_risk:
                    risk_metrics = results_with_risk['risk_management']
                    print(f"     Circuit Breaker Triggered: {risk_metrics['circuit_breaker_triggered']}")
                    print(f"     Trading Halted: {risk_metrics['trading_halted']}")
                    print(f"     Position Scale: {risk_metrics['position_scale']:.2f}")
                
                print(f"  📉 Without Risk Management:")
                print(f"     Total Return: {results_without_risk['total_return']:.2%}")
                print(f"     Max Drawdown: {results_without_risk['max_drawdown']:.2%}")
                print(f"     Sharpe Ratio: {results_without_risk['sharpe_ratio']:.3f}")
                print(f"     Total Trades: {results_without_risk['total_trades']}")
                
                print(f"  🔄 Risk Management Impact:")
                print(f"     Drawdown Reduction: {comparison['risk_improvement']['drawdown_reduction']:.2%}")
                print(f"     Sharpe Improvement: {comparison['risk_improvement']['sharpe_improvement']:.3f}")
                print(f"     Return Impact: {comparison['risk_improvement']['return_impact']:.2%}")
        
        all_results[period_name] = period_results
    
    # Summary analysis
    print("\n" + "=" * 70)
    print("📋 SUMMARY ANALYSIS")
    print("=" * 70)
    
    for period_name, period_results in all_results.items():
        print(f"\n📊 {period_name}:")
        
        for config_name, comparison in period_results.items():
            risk_improvement = comparison['risk_improvement']
            
            print(f"  🔧 {config_name}:")
            print(f"     Drawdown Reduction: {risk_improvement['drawdown_reduction']:.2%}")
            print(f"     Sharpe Improvement: {risk_improvement['sharpe_improvement']:.3f}")
            print(f"     Return Impact: {risk_improvement['return_impact']:.2%}")
            
            # Risk management effectiveness assessment
            if risk_improvement['drawdown_reduction'] > 0.02:  # 2% reduction
                print(f"     ✅ Effective drawdown control")
            else:
                print(f"     ⚠️  Limited drawdown control")
            
            if risk_improvement['sharpe_improvement'] > 0.1:  # 0.1 improvement
                print(f"     ✅ Improved risk-adjusted returns")
            else:
                print(f"     ⚠️  Limited Sharpe improvement")

def test_monte_carlo_stress_test():
    """Test Monte Carlo stress testing capabilities."""
    
    print("\n🎲 MONTE CARLO STRESS TESTING")
    print("=" * 50)
    
    # Run enhanced strategy with Monte Carlo stress test
    results = run_enhanced_drawdown_backtest(
        start_date="2023-07-01",
        end_date="2023-09-30",
        initial_capital=100000.0,
        trailing_stop_pct=0.05,
        circuit_breaker_pct=0.15,
        enable_risk_management=True
    )
    
    if results and 'monte_carlo_stress_test' in results:
        stress_test = results['monte_carlo_stress_test']
        
        print(f"📊 Monte Carlo Stress Test Results:")
        print(f"   Number of Simulations: {stress_test['num_simulations']}")
        print(f"   Mean Max Drawdown: {stress_test['max_drawdown_mean']:.2%}")
        print(f"   95th Percentile Max Drawdown: {stress_test['max_drawdown_95th']:.2%}")
        print(f"   99th Percentile Max Drawdown: {stress_test['max_drawdown_99th']:.2%}")
        print(f"   Mean 95% VaR: {stress_test['var_95_mean']:.2%}")
        
        # Risk assessment
        if stress_test['max_drawdown_95th'] < 0.20:  # 20% threshold
            print(f"   ✅ Low tail risk (95th percentile < 20%)")
        else:
            print(f"   ⚠️  High tail risk (95th percentile >= 20%)")
        
        if stress_test['max_drawdown_99th'] < 0.30:  # 30% threshold
            print(f"   ✅ Very low extreme risk (99th percentile < 30%)")
        else:
            print(f"   ⚠️  High extreme risk (99th percentile >= 30%)")
    else:
        print("❌ Monte Carlo stress test not available")

def test_risk_management_effectiveness():
    """Test the effectiveness of different risk management parameters."""
    
    print("\n🔬 RISK MANAGEMENT PARAMETER ANALYSIS")
    print("=" * 50)
    
    # Test different trailing stop levels
    trailing_stops = [0.03, 0.05, 0.08, 0.10]
    circuit_breakers = [0.10, 0.15, 0.20, 0.25]
    
    print("📊 Trailing Stop Analysis:")
    for trailing_stop in trailing_stops:
        results = run_enhanced_drawdown_backtest(
            start_date="2023-07-01",
            end_date="2023-09-30",
            initial_capital=100000.0,
            trailing_stop_pct=trailing_stop,
            circuit_breaker_pct=0.15,
            enable_risk_management=True
        )
        
        if results:
            print(f"   {trailing_stop:.1%} trailing stop:")
            print(f"     Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"     Total Return: {results['total_return']:.2%}")
            print(f"     Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    
    print("\n📊 Circuit Breaker Analysis:")
    for circuit_breaker in circuit_breakers:
        results = run_enhanced_drawdown_backtest(
            start_date="2023-07-01",
            end_date="2023-09-30",
            initial_capital=100000.0,
            trailing_stop_pct=0.05,
            circuit_breaker_pct=circuit_breaker,
            enable_risk_management=True
        )
        
        if results and 'risk_management' in results:
            risk_metrics = results['risk_management']
            print(f"   {circuit_breaker:.1%} circuit breaker:")
            print(f"     Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"     Circuit Breaker Triggered: {risk_metrics['circuit_breaker_triggered']}")
            print(f"     Trading Halted: {risk_metrics['trading_halted']}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Drawdown Strategy Tests...")
    
    # Run comprehensive tests
    test_enhanced_drawdown_strategy()
    test_monte_carlo_stress_test()
    test_risk_management_effectiveness()
    
    print("\n✅ Enhanced Drawdown Strategy testing completed!")
    print("📋 Check the results above to see the effectiveness of risk management.") 