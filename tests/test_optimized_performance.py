#!/usr/bin/env python3
"""
Test Optimized Performance Strategy
==================================

This script compares the optimized performance strategy against previous versions
to demonstrate the improvements in performance and risk management.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

sys.path.append(os.path.dirname(__file__))

from gold_algo.strategies.optimized_performance_strategy import run_optimized_performance_backtest
from gold_algo.strategies.macro_regime_strategy import run_macro_regime_backtest
from gold_algo.strategies.enhanced_drawdown_strategy import run_enhanced_drawdown_backtest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_optimized_performance():
    """Test the optimized performance strategy against previous versions."""
    
    print("🚀 TESTING OPTIMIZED PERFORMANCE STRATEGY")
    print("=" * 60)
    
    # Test periods
    test_periods = [
        ("Full Year 2023", "2023-01-01", "2023-12-31"),
        ("Q1 2023", "2023-01-01", "2023-03-31"),
        ("Q2 2023", "2023-04-01", "2023-06-30"),
        ("Q3 2023", "2023-07-01", "2023-09-30"),
        ("Q4 2023", "2023-10-01", "2023-12-31")
    ]
    
    # Strategy configurations
    strategies = [
        ("Optimized Performance (Full)", "optimized_full"),
        ("Optimized Performance (No Risk)", "optimized_no_risk"),
        ("Optimized Performance (No Macro)", "optimized_no_macro"),
        ("Optimized Performance (Base)", "optimized_base"),
        ("Previous Macro Regime (Full)", "previous_macro_full"),
        ("Previous Enhanced Drawdown", "previous_enhanced_drawdown")
    ]
    
    all_results = {}
    
    for period_name, start_date, end_date in test_periods:
        print(f"\n📅 Testing Period: {period_name}")
        print("-" * 40)
        
        period_results = {}
        
        for strategy_name, strategy_type in strategies:
            print(f"\n🔧 Testing: {strategy_name}")
            
            try:
                if strategy_type == "optimized_full":
                    results = run_optimized_performance_backtest(
                        start_date, end_date, 
                        enable_macro_filter=True, 
                        enable_risk_management=True
                    )
                elif strategy_type == "optimized_no_risk":
                    results = run_optimized_performance_backtest(
                        start_date, end_date, 
                        enable_macro_filter=True, 
                        enable_risk_management=False
                    )
                elif strategy_type == "optimized_no_macro":
                    results = run_optimized_performance_backtest(
                        start_date, end_date, 
                        enable_macro_filter=False, 
                        enable_risk_management=True
                    )
                elif strategy_type == "optimized_base":
                    results = run_optimized_performance_backtest(
                        start_date, end_date, 
                        enable_macro_filter=False, 
                        enable_risk_management=False
                    )
                elif strategy_type == "previous_macro_full":
                    results = run_macro_regime_backtest(
                        start_date, end_date, 
                        enable_macro_filter=True, 
                        enable_risk_management=True
                    )
                elif strategy_type == "previous_enhanced_drawdown":
                    results = run_enhanced_drawdown_backtest(start_date, end_date)
                
                if results:
                    period_results[strategy_name] = results
                    
                    # Display key metrics
                    print(f"  📈 Return: {results['total_return']:.2%}")
                    print(f"  📉 Max DD: {results['max_drawdown']:.2%}")
                    print(f"  📊 Sharpe: {results['sharpe_ratio']:.3f}")
                    print(f"  🔄 Trades: {results['total_trades']}")
                    
            except Exception as e:
                logger.error(f"Error testing {strategy_name}: {e}")
                continue
        
        all_results[period_name] = period_results
    
    # Performance comparison analysis
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON ANALYSIS")
    print("=" * 60)
    
    analyze_performance_comparison(all_results)
    
    # Improvement summary
    print("\n" + "=" * 60)
    print("🎯 IMPROVEMENT SUMMARY")
    print("=" * 60)
    
    summarize_improvements(all_results)

def analyze_performance_comparison(all_results):
    """Analyze performance comparison between strategies."""
    
    print("\n🔍 Performance Comparison Analysis:")
    
    # Compare optimized vs previous strategies
    for period_name, period_results in all_results.items():
        if "Optimized Performance (Full)" in period_results and "Previous Macro Regime (Full)" in period_results:
            optimized = period_results["Optimized Performance (Full)"]
            previous = period_results["Previous Macro Regime (Full)"]
            
            return_improvement = optimized['total_return'] - previous['total_return']
            drawdown_improvement = previous['max_drawdown'] - optimized['max_drawdown']
            sharpe_improvement = optimized['sharpe_ratio'] - previous['sharpe_ratio']
            trade_improvement = optimized['total_trades'] - previous['total_trades']
            
            print(f"\n📊 {period_name}:")
            print(f"   Return Improvement: {return_improvement:.2%}")
            print(f"   Drawdown Improvement: {drawdown_improvement:.2%}")
            print(f"   Sharpe Improvement: {sharpe_improvement:.3f}")
            print(f"   Trade Frequency Change: {trade_improvement:+d}")
            
            # Overall assessment
            if sharpe_improvement > 0.1 and drawdown_improvement > 0.02:
                print(f"   ✅ SIGNIFICANT IMPROVEMENT")
            elif sharpe_improvement > 0.05:
                print(f"   ✅ MODERATE IMPROVEMENT")
            elif sharpe_improvement > 0:
                print(f"   ✅ SLIGHT IMPROVEMENT")
            else:
                print(f"   ⚠️  NEEDS FURTHER OPTIMIZATION")
    
    # Component analysis
    print("\n🔧 Component Analysis:")
    for period_name, period_results in all_results.items():
        if all(key in period_results for key in ["Optimized Performance (Full)", "Optimized Performance (No Risk)", "Optimized Performance (No Macro)"]):
            full = period_results["Optimized Performance (Full)"]
            no_risk = period_results["Optimized Performance (No Risk)"]
            no_macro = period_results["Optimized Performance (No Macro)"]
            
            risk_impact = full['total_return'] - no_risk['total_return']
            macro_impact = full['total_return'] - no_macro['total_return']
            
            print(f"\n📊 {period_name} Component Impact:")
            print(f"   Risk Management Impact: {risk_impact:.2%}")
            print(f"   Macro Filter Impact: {macro_impact:.2%}")

def summarize_improvements(all_results):
    """Summarize the improvements achieved."""
    
    print("\n🎯 Key Improvements Achieved:")
    
    # Overall performance improvements
    improvements = []
    for period_name, period_results in all_results.items():
        if "Optimized Performance (Full)" in period_results and "Previous Macro Regime (Full)" in period_results:
            optimized = period_results["Optimized Performance (Full)"]
            previous = period_results["Previous Macro Regime (Full)"]
            
            improvements.append({
                'period': period_name,
                'return_improvement': optimized['total_return'] - previous['total_return'],
                'drawdown_improvement': previous['max_drawdown'] - optimized['max_drawdown'],
                'sharpe_improvement': optimized['sharpe_ratio'] - previous['sharpe_ratio'],
                'trade_improvement': optimized['total_trades'] - previous['total_trades']
            })
    
    if improvements:
        avg_return_improvement = np.mean([imp['return_improvement'] for imp in improvements])
        avg_drawdown_improvement = np.mean([imp['drawdown_improvement'] for imp in improvements])
        avg_sharpe_improvement = np.mean([imp['sharpe_improvement'] for imp in improvements])
        avg_trade_improvement = np.mean([imp['trade_improvement'] for imp in improvements])
        
        print(f"\n📈 Average Improvements Across All Periods:")
        print(f"   Return: {avg_return_improvement:.2%}")
        print(f"   Drawdown: {avg_drawdown_improvement:.2%}")
        print(f"   Sharpe: {avg_sharpe_improvement:.3f}")
        print(f"   Trade Frequency: {avg_trade_improvement:+.0f}")
        
        if avg_sharpe_improvement > 0.1:
            print(f"\n✅ OPTIMIZATION SUCCESSFUL - Significant improvements achieved!")
        elif avg_sharpe_improvement > 0.05:
            print(f"\n✅ OPTIMIZATION MODERATELY SUCCESSFUL - Good improvements achieved!")
        else:
            print(f"\n⚠️  OPTIMIZATION NEEDS REFINEMENT - Further improvements needed")
    
    # Parameter optimization summary
    print(f"\n🔧 Parameter Optimization Summary:")
    print(f"   ✅ Trailing stop: 5% → 8% (less restrictive)")
    print(f"   ✅ Circuit breaker: 15% → 25% (allows recovery)")
    print(f"   ✅ Max position size: 10% → 15% (more aggressive)")
    print(f"   ✅ Confirmation threshold: 0.4 → 0.3 (more active)")
    print(f"   ✅ Max leverage: 6.0x → 8.0x (higher capacity)")
    print(f"   ✅ Regime persistence: 5 → 3 days (more responsive)")
    print(f"   ✅ Confidence threshold: 70% → 60% (less restrictive)")

def test_risk_management_effectiveness():
    """Test the effectiveness of the optimized risk management."""
    
    print("\n🛡️ TESTING RISK MANAGEMENT EFFECTIVENESS")
    print("=" * 60)
    
    # Test different risk management configurations
    risk_configs = [
        ("No Risk Management", False, False),
        ("Risk Management Only", False, True),
        ("Macro Filter Only", True, False),
        ("Full Risk + Macro", True, True)
    ]
    
    results = {}
    
    for config_name, enable_macro, enable_risk in risk_configs:
        print(f"\n🔧 Testing: {config_name}")
        
        try:
            result = run_optimized_performance_backtest(
                "2023-01-01", "2023-12-31",
                enable_macro_filter=enable_macro,
                enable_risk_management=enable_risk
            )
            
            if result:
                results[config_name] = result
                
                print(f"  📈 Return: {result['total_return']:.2%}")
                print(f"  📉 Max DD: {result['max_drawdown']:.2%}")
                print(f"  📊 Sharpe: {result['sharpe_ratio']:.3f}")
                print(f"  🔄 Trades: {result['total_trades']}")
                
        except Exception as e:
            logger.error(f"Error testing {config_name}: {e}")
            continue
    
    # Risk management analysis
    if len(results) >= 2:
        print(f"\n📊 Risk Management Analysis:")
        
        if "No Risk Management" in results and "Full Risk + Macro" in results:
            no_risk = results["No Risk Management"]
            full_risk = results["Full Risk + Macro"]
            
            risk_return_impact = full_risk['total_return'] - no_risk['total_return']
            risk_drawdown_improvement = no_risk['max_drawdown'] - full_risk['max_drawdown']
            
            print(f"   Return Impact: {risk_return_impact:.2%}")
            print(f"   Drawdown Improvement: {risk_drawdown_improvement:.2%}")
            
            if risk_drawdown_improvement > 0.05 and risk_return_impact > -0.02:
                print(f"   ✅ RISK MANAGEMENT EFFECTIVE")
            elif risk_drawdown_improvement > 0.02:
                print(f"   ✅ RISK MANAGEMENT MODERATELY EFFECTIVE")
            else:
                print(f"   ⚠️  RISK MANAGEMENT NEEDS ADJUSTMENT")

if __name__ == "__main__":
    print("🚀 Starting Optimized Performance Strategy Tests...")
    
    # Test optimized performance
    test_optimized_performance()
    
    # Test risk management effectiveness
    test_risk_management_effectiveness()
    
    print("\n✅ Optimized performance testing completed!")
    print("📋 Check the results above to see the improvements achieved.") 