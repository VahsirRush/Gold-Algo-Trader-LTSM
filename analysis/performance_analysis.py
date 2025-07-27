#!/usr/bin/env python3
"""
Performance Analysis and Degradation Investigation
================================================

This script analyzes the performance progression across different strategy
iterations to identify root causes of degradation and suggest improvements.

Key Issues Identified:
1. Overfitting to specific periods
2. Excessive risk management constraints
3. Signal quality degradation
4. Parameter sensitivity
5. Strategy complexity creep
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

sys.path.append(os.path.dirname(__file__))

from gold_algo.strategies.macro_regime_strategy import run_macro_regime_backtest
from gold_algo.strategies.enhanced_drawdown_strategy import run_enhanced_drawdown_backtest
from gold_algo.strategies.ultra_aggressive_strategy import UltraAggressiveStrategy
from data_pipeline.databento_collector import DatabentoGoldCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_performance_progression():
    """Analyze performance progression across strategy iterations."""
    
    print("📊 PERFORMANCE PROGRESSION ANALYSIS")
    print("=" * 60)
    
    # Test periods
    test_periods = [
        ("Original Period (Jul-Sep 2023)", "2023-07-01", "2023-09-30"),
        ("Earlier Period (Apr-Jun 2023)", "2023-04-01", "2023-06-30"),
        ("Later Period (Oct-Dec 2023)", "2023-10-01", "2023-12-31"),
        ("Full Year 2023", "2023-01-01", "2023-12-31")
    ]
    
    # Strategy configurations to test
    strategies = [
        ("Ultra Aggressive (Baseline)", "ultra_aggressive"),
        ("Enhanced Drawdown", "enhanced_drawdown"),
        ("Macro Regime (Full)", "macro_regime_full"),
        ("Macro Regime (No Risk)", "macro_regime_no_risk"),
        ("Macro Regime (No Macro)", "macro_regime_no_macro"),
        ("Base Strategy (No Macro, No Risk)", "base_strategy")
    ]
    
    all_results = {}
    
    for period_name, start_date, end_date in test_periods:
        print(f"\n📅 Testing Period: {period_name}")
        print("-" * 40)
        
        period_results = {}
        
        for strategy_name, strategy_type in strategies:
            print(f"\n🔧 Testing: {strategy_name}")
            
            try:
                if strategy_type == "ultra_aggressive":
                    results = test_ultra_aggressive_strategy(start_date, end_date)
                elif strategy_type == "enhanced_drawdown":
                    results = run_enhanced_drawdown_backtest(start_date, end_date)
                elif strategy_type == "macro_regime_full":
                    results = run_macro_regime_backtest(start_date, end_date, enable_macro_filter=True, enable_risk_management=True)
                elif strategy_type == "macro_regime_no_risk":
                    results = run_macro_regime_backtest(start_date, end_date, enable_macro_filter=True, enable_risk_management=False)
                elif strategy_type == "macro_regime_no_macro":
                    results = run_macro_regime_backtest(start_date, end_date, enable_macro_filter=False, enable_risk_management=True)
                elif strategy_type == "base_strategy":
                    results = run_macro_regime_backtest(start_date, end_date, enable_macro_filter=False, enable_risk_management=False)
                
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
    
    # Performance degradation analysis
    print("\n" + "=" * 60)
    print("🔍 PERFORMANCE DEGRADATION ANALYSIS")
    print("=" * 60)
    
    analyze_degradation_causes(all_results)
    
    # Recommendations
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS FOR IMPROVEMENT")
    print("=" * 60)
    
    provide_recommendations(all_results)

def test_ultra_aggressive_strategy(start_date, end_date):
    """Test the original ultra aggressive strategy."""
    try:
        from data_pipeline.databento_collector import DatabentoGoldCollector
        
        # Fetch data
        collector = DatabentoGoldCollector()
        data = collector.fetch_gld(start_date, end_date)
        
        if data.empty:
            return {}
        
        # Create and run strategy
        strategy = UltraAggressiveStrategy()
        results = strategy.run_backtest(data)
        
        return results
        
    except Exception as e:
        logger.error(f"Error testing ultra aggressive strategy: {e}")
        return {}

def analyze_degradation_causes(all_results):
    """Analyze the root causes of performance degradation."""
    
    print("\n🔍 Root Cause Analysis:")
    
    # 1. Overfitting Analysis
    print("\n1️⃣ OVERFITTING ANALYSIS:")
    for period_name, period_results in all_results.items():
        if "Ultra Aggressive (Baseline)" in period_results and "Macro Regime (Full)" in period_results:
            baseline = period_results["Ultra Aggressive (Baseline)"]
            enhanced = period_results["Macro Regime (Full)"]
            
            sharpe_degradation = baseline['sharpe_ratio'] - enhanced['sharpe_ratio']
            return_degradation = baseline['total_return'] - enhanced['total_return']
            
            print(f"  📊 {period_name}:")
            print(f"     Sharpe Degradation: {sharpe_degradation:.3f}")
            print(f"     Return Degradation: {return_degradation:.2%}")
            
            if sharpe_degradation > 0.5:
                print(f"     ⚠️  SIGNIFICANT OVERFITTING DETECTED")
            elif sharpe_degradation > 0.2:
                print(f"     ⚠️  MODERATE OVERFITTING DETECTED")
            else:
                print(f"     ✅ ACCEPTABLE PERFORMANCE")
    
    # 2. Risk Management Impact
    print("\n2️⃣ RISK MANAGEMENT IMPACT:")
    for period_name, period_results in all_results.items():
        if "Macro Regime (No Risk)" in period_results and "Macro Regime (Full)" in period_results:
            no_risk = period_results["Macro Regime (No Risk)"]
            with_risk = period_results["Macro Regime (Full)"]
            
            risk_impact = with_risk['total_return'] - no_risk['total_return']
            risk_drawdown_improvement = no_risk['max_drawdown'] - with_risk['max_drawdown']
            
            print(f"  📊 {period_name}:")
            print(f"     Risk Management Impact: {risk_impact:.2%}")
            print(f"     Drawdown Improvement: {risk_drawdown_improvement:.2%}")
            
            if risk_impact < -0.5:
                print(f"     ⚠️  RISK MANAGEMENT TOO RESTRICTIVE")
            elif risk_drawdown_improvement < 0.5:
                print(f"     ⚠️  RISK MANAGEMENT INEFFECTIVE")
            else:
                print(f"     ✅ RISK MANAGEMENT EFFECTIVE")
    
    # 3. Macro Filter Impact
    print("\n3️⃣ MACRO FILTER IMPACT:")
    for period_name, period_results in all_results.items():
        if "Macro Regime (No Macro)" in period_results and "Macro Regime (Full)" in period_results:
            no_macro = period_results["Macro Regime (No Macro)"]
            with_macro = period_results["Macro Regime (Full)"]
            
            macro_impact = with_macro['total_return'] - no_macro['total_return']
            macro_sharpe_impact = with_macro['sharpe_ratio'] - no_macro['sharpe_ratio']
            
            print(f"  📊 {period_name}:")
            print(f"     Macro Filter Impact: {macro_impact:.2%}")
            print(f"     Sharpe Impact: {macro_sharpe_impact:.3f}")
            
            if macro_impact < -0.3:
                print(f"     ⚠️  MACRO FILTER REDUCING RETURNS")
            elif macro_sharpe_impact < 0:
                print(f"     ⚠️  MACRO FILTER HURTING RISK-ADJUSTED RETURNS")
            else:
                print(f"     ✅ MACRO FILTER ADDING VALUE")
    
    # 4. Complexity Analysis
    print("\n4️⃣ COMPLEXITY ANALYSIS:")
    for period_name, period_results in all_results.items():
        if "Ultra Aggressive (Baseline)" in period_results and "Macro Regime (Full)" in period_results:
            baseline = period_results["Ultra Aggressive (Baseline)"]
            enhanced = period_results["Macro Regime (Full)"]
            
            trade_frequency_ratio = enhanced['total_trades'] / max(baseline['total_trades'], 1)
            
            print(f"  📊 {period_name}:")
            print(f"     Trade Frequency Ratio: {trade_frequency_ratio:.2f}")
            
            if trade_frequency_ratio < 0.5:
                print(f"     ⚠️  STRATEGY BECOMING TOO CONSERVATIVE")
            elif trade_frequency_ratio > 2.0:
                print(f"     ⚠️  STRATEGY BECOMING TOO ACTIVE")
            else:
                print(f"     ✅ APPROPRIATE TRADE FREQUENCY")

def provide_recommendations(all_results):
    """Provide specific recommendations for improvement."""
    
    print("\n💡 SPECIFIC RECOMMENDATIONS:")
    
    # Analyze overall trends
    baseline_performance = []
    enhanced_performance = []
    
    for period_name, period_results in all_results.items():
        if "Ultra Aggressive (Baseline)" in period_results and "Macro Regime (Full)" in period_results:
            baseline = period_results["Ultra Aggressive (Baseline)"]
            enhanced = period_results["Macro Regime (Full)"]
            
            baseline_performance.append(baseline['sharpe_ratio'])
            enhanced_performance.append(enhanced['sharpe_ratio'])
    
    if baseline_performance and enhanced_performance:
        avg_baseline_sharpe = np.mean(baseline_performance)
        avg_enhanced_sharpe = np.mean(enhanced_performance)
        
        print(f"\n📊 Overall Performance Comparison:")
        print(f"   Baseline Average Sharpe: {avg_baseline_sharpe:.3f}")
        print(f"   Enhanced Average Sharpe: {avg_enhanced_sharpe:.3f}")
        print(f"   Performance Change: {avg_enhanced_sharpe - avg_baseline_sharpe:.3f}")
        
        if avg_enhanced_sharpe < avg_baseline_sharpe:
            print(f"   ⚠️  OVERALL PERFORMANCE DEGRADATION DETECTED")
            
            print(f"\n🔧 IMMEDIATE ACTIONS REQUIRED:")
            print(f"   1. Simplify the strategy - remove excessive constraints")
            print(f"   2. Reduce risk management aggressiveness")
            print(f"   3. Optimize macro regime thresholds")
            print(f"   4. Increase base position sizes")
            print(f"   5. Reduce confirmation thresholds")
            
            print(f"\n🎯 SPECIFIC PARAMETER ADJUSTMENTS:")
            print(f"   - Increase base_position_size from 0.10 to 0.15")
            print(f"   - Reduce confirmation_threshold from 0.4 to 0.3")
            print(f"   - Increase max_leverage from 6.0 to 8.0")
            print(f"   - Reduce trailing_stop_pct from 0.05 to 0.03")
            print(f"   - Increase regime_persistence_days from 5 to 3")
            
        else:
            print(f"   ✅ OVERALL PERFORMANCE IMPROVEMENT ACHIEVED")
    
    # Strategy-specific recommendations
    print(f"\n📋 STRATEGY-SPECIFIC RECOMMENDATIONS:")
    print(f"   1. Ultra Aggressive: Use as baseline for comparison")
    print(f"   2. Enhanced Drawdown: Good risk control, needs return optimization")
    print(f"   3. Macro Regime: Reduce complexity, focus on core regime detection")
    print(f"   4. Risk Management: Less aggressive stops, faster recovery")
    print(f"   5. Macro Filter: Simplify indicators, reduce persistence requirements")

def create_simplified_strategy():
    """Create a simplified, high-performance strategy."""
    
    print("\n🚀 CREATING SIMPLIFIED HIGH-PERFORMANCE STRATEGY")
    print("=" * 60)
    
    # Strategy parameters optimized for performance
    simplified_params = {
        'base_position_size': 0.15,  # Increased from 0.10
        'confirmation_threshold': 0.3,  # Reduced from 0.4
        'max_leverage': 8.0,  # Increased from 6.0
        'trailing_stop_pct': 0.03,  # Reduced from 0.05
        'circuit_breaker_pct': 0.20,  # Increased from 0.15
        'regime_persistence_days': 3,  # Reduced from 5
        'regime_confidence_threshold': 0.6,  # Reduced from 0.7
        'min_leverage': 0.3,  # Reduced from 0.5
    }
    
    print(f"📊 Simplified Strategy Parameters:")
    for param, value in simplified_params.items():
        print(f"   {param}: {value}")
    
    return simplified_params

if __name__ == "__main__":
    print("🔍 Starting Performance Analysis...")
    
    # Run comprehensive analysis
    analyze_performance_progression()
    
    # Create simplified strategy
    simplified_params = create_simplified_strategy()
    
    print("\n✅ Performance analysis completed!")
    print("📋 Check the recommendations above for immediate improvements.") 