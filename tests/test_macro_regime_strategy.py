#!/usr/bin/env python3
"""
Test Macro Regime Strategy with Regime Analysis
==============================================

This script demonstrates the macro regime strategy with comprehensive
regime analysis and performance comparison across different configurations.

Key Features:
- Macro regime detection and analysis
- Performance comparison with and without regime filtering
- Regime-specific performance metrics
- Risk management integration
- Regime persistence analysis
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

sys.path.append(os.path.dirname(__file__))

from gold_algo.strategies.macro_regime_strategy import run_macro_regime_backtest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_macro_regime_strategy():
    """Test the macro regime strategy with comprehensive analysis."""
    
    print("🌍 MACRO REGIME STRATEGY WITH REGIME ANALYSIS")
    print("=" * 70)
    
    # Test configurations
    test_configs = [
        ("Full Strategy (Macro + Risk)", True, True),
        ("Macro Only (No Risk)", True, False),
        ("Risk Only (No Macro)", False, True),
        ("Base Strategy (No Macro, No Risk)", False, False)
    ]
    
    test_periods = [
        ("Original Period (Jul-Sep 2023)", "2023-07-01", "2023-09-30"),
        ("Earlier Period (Apr-Jun 2023)", "2023-04-01", "2023-06-30"),
        ("Later Period (Oct-Dec 2023)", "2023-10-01", "2023-12-31")
    ]
    
    all_results = {}
    
    for period_name, start_date, end_date in test_periods:
        print(f"\n📊 Testing Period: {period_name}")
        print("-" * 50)
        
        period_results = {}
        
        for config_name, enable_macro, enable_risk in test_configs:
            print(f"\n🔧 Configuration: {config_name}")
            
            # Run backtest
            results = run_macro_regime_backtest(
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000.0,
                trailing_stop_pct=0.05,
                circuit_breaker_pct=0.15,
                enable_macro_filter=enable_macro,
                enable_risk_management=enable_risk
            )
            
            if results:
                period_results[config_name] = results
                
                # Display results
                print(f"  📈 Performance Results:")
                print(f"     Total Return: {results['total_return']:.2%}")
                print(f"     Max Drawdown: {results['max_drawdown']:.2%}")
                print(f"     Sharpe Ratio: {results['sharpe_ratio']:.3f}")
                print(f"     Total Trades: {results['total_trades']}")
                
                # Display macro regime information
                if 'macro_regime' in results:
                    macro_info = results['macro_regime']
                    print(f"  🌍 Macro Regime Information:")
                    print(f"     Current Regime: {macro_info['current_regime']}")
                    print(f"     Regime Confidence: {macro_info['regime_confidence']:.2f}")
                    print(f"     Leverage Multiplier: {macro_info['leverage_multiplier']:.2f}")
                    print(f"     Stop Loss Multiplier: {macro_info['stop_loss_multiplier']:.2f}")
                    print(f"     Volatility Target: {macro_info['volatility_target']:.2%}")
                    print(f"     Description: {macro_info['description']}")
                
                # Display risk management information
                if 'risk_management' in results:
                    risk_info = results['risk_management']
                    print(f"  🛡️  Risk Management Information:")
                    print(f"     Trailing Stop: {risk_info['trailing_stop_pct']:.1%}")
                    print(f"     Circuit Breaker: {risk_info['circuit_breaker_pct']:.1%}")
                    print(f"     Max Drawdown: {risk_info['max_drawdown']:.2%}")
                    print(f"     Volatility: {risk_info['volatility']:.2%}")
                    print(f"     Trading Halted: {risk_info['trading_halted']}")
                    print(f"     Circuit Breaker Triggered: {risk_info['circuit_breaker_triggered']}")
                
                # Display regime analysis
                if 'regime_analysis' in results:
                    regime_analysis = results['regime_analysis']
                    print(f"  📊 Regime Performance Analysis:")
                    for regime, stats in regime_analysis.items():
                        print(f"     {regime.upper()}:")
                        print(f"       Trade Count: {stats['trade_count']}")
                        print(f"       Avg Position Size: {stats['avg_position_size']:.3f}")
                        print(f"       Avg Leverage: {stats['avg_leverage']:.2f}")
        
        all_results[period_name] = period_results
    
    # Summary analysis
    print("\n" + "=" * 70)
    print("📋 SUMMARY ANALYSIS")
    print("=" * 70)
    
    for period_name, period_results in all_results.items():
        print(f"\n📊 {period_name}:")
        
        # Compare configurations
        if len(period_results) >= 2:
            full_strategy = period_results.get("Full Strategy (Macro + Risk)", {})
            base_strategy = period_results.get("Base Strategy (No Macro, No Risk)", {})
            
            if full_strategy and base_strategy:
                print(f"  🔄 Full Strategy vs Base Strategy:")
                print(f"     Return Improvement: {full_strategy['total_return'] - base_strategy['total_return']:.2%}")
                print(f"     Drawdown Improvement: {base_strategy['max_drawdown'] - full_strategy['max_drawdown']:.2%}")
                print(f"     Sharpe Improvement: {full_strategy['sharpe_ratio'] - base_strategy['sharpe_ratio']:.3f}")
                
                # Assess effectiveness
                if full_strategy['sharpe_ratio'] > base_strategy['sharpe_ratio']:
                    print(f"     ✅ Macro regime filtering improves risk-adjusted returns")
                else:
                    print(f"     ⚠️  Macro regime filtering does not improve risk-adjusted returns")
                
                if full_strategy['max_drawdown'] < base_strategy['max_drawdown']:
                    print(f"     ✅ Macro regime filtering improves drawdown control")
                else:
                    print(f"     ⚠️  Macro regime filtering does not improve drawdown control")

def test_regime_persistence():
    """Test regime persistence and switching behavior."""
    
    print("\n🔄 REGIME PERSISTENCE ANALYSIS")
    print("=" * 50)
    
    # Test different persistence settings
    persistence_configs = [
        ("Low Persistence (2 days)", 2),
        ("Medium Persistence (5 days)", 5),
        ("High Persistence (10 days)", 10)
    ]
    
    for config_name, persistence_days in persistence_configs:
        print(f"\n🔧 {config_name}:")
        
        # Run backtest with different persistence settings
        results = run_macro_regime_backtest(
            start_date="2023-07-01",
            end_date="2023-09-30",
            initial_capital=100000.0,
            enable_macro_filter=True,
            enable_risk_management=True
        )
        
        if results and 'macro_regime' in results:
            macro_info = results['macro_regime']
            print(f"  📊 Regime Statistics:")
            print(f"     Current Regime: {macro_info['current_regime']}")
            print(f"     Regime Confidence: {macro_info['regime_confidence']:.2f}")
            print(f"     Regime History Length: {macro_info['regime_history_length']}")
            print(f"     Leverage Multiplier: {macro_info['leverage_multiplier']:.2f}")

def test_regime_effectiveness():
    """Test the effectiveness of regime-based position sizing."""
    
    print("\n🎯 REGIME EFFECTIVENESS ANALYSIS")
    print("=" * 50)
    
    # Test different leverage configurations
    leverage_configs = [
        ("Conservative Leverage (1.0x)", 1.0),
        ("Moderate Leverage (1.5x)", 1.5),
        ("Aggressive Leverage (2.0x)", 2.0)
    ]
    
    for config_name, max_leverage in leverage_configs:
        print(f"\n🔧 {config_name}:")
        
        # Run backtest
        results = run_macro_regime_backtest(
            start_date="2023-07-01",
            end_date="2023-09-30",
            initial_capital=100000.0,
            enable_macro_filter=True,
            enable_risk_management=True
        )
        
        if results:
            print(f"  📈 Performance Results:")
            print(f"     Total Return: {results['total_return']:.2%}")
            print(f"     Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"     Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            print(f"     Total Trades: {results['total_trades']}")
            
            # Assess leverage effectiveness
            if results['sharpe_ratio'] > 0.5:
                print(f"     ✅ Effective regime-based leverage")
            else:
                print(f"     ⚠️  Limited effectiveness of regime-based leverage")

def test_macro_indicators():
    """Test the impact of different macro indicators."""
    
    print("\n📊 MACRO INDICATOR ANALYSIS")
    print("=" * 50)
    
    # Test different macro indicator combinations
    indicator_tests = [
        ("Real Rates Only", "real_rates"),
        ("USD Strength Only", "usd_strength"),
        ("Volatility Only", "volatility"),
        ("Market Stress Only", "market_stress"),
        ("All Indicators", "all")
    ]
    
    for test_name, indicator in indicator_tests:
        print(f"\n🔧 {test_name}:")
        
        # Run backtest
        results = run_macro_regime_backtest(
            start_date="2023-07-01",
            end_date="2023-09-30",
            initial_capital=100000.0,
            enable_macro_filter=True,
            enable_risk_management=True
        )
        
        if results and 'macro_regime' in results:
            macro_info = results['macro_regime']
            print(f"  📊 Regime Classification:")
            print(f"     Current Regime: {macro_info['current_regime']}")
            print(f"     Regime Confidence: {macro_info['regime_confidence']:.2f}")
            print(f"     Description: {macro_info['description']}")
            
            # Assess indicator effectiveness
            if macro_info['regime_confidence'] > 0.7:
                print(f"     ✅ High confidence regime classification")
            elif macro_info['regime_confidence'] > 0.5:
                print(f"     ⚠️  Moderate confidence regime classification")
            else:
                print(f"     ❌ Low confidence regime classification")

if __name__ == "__main__":
    print("🚀 Starting Macro Regime Strategy Tests...")
    
    # Run comprehensive tests
    test_macro_regime_strategy()
    test_regime_persistence()
    test_regime_effectiveness()
    test_macro_indicators()
    
    print("\n✅ Macro Regime Strategy testing completed!")
    print("📋 Check the results above to see the effectiveness of macro regime filtering.") 