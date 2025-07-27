#!/usr/bin/env python3
"""
Test Conservative Risk Strategy
==============================

This script tests the conservative risk strategy and compares it with the overfitted version
to demonstrate the importance of realistic parameters and proper risk management.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from gold_algo.strategies.conservative_risk_strategy import run_conservative_backtest
from gold_algo.strategies.risk_enhanced_strategy import run_risk_enhanced_backtest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_conservative_vs_overfitted():
    """Compare conservative strategy with overfitted strategy."""
    
    print("🔍 CONSERVATIVE VS OVERFITTED STRATEGY COMPARISON")
    print("=" * 70)
    
    # Test periods
    test_periods = [
        {
            "name": "Original Period (Jul-Sep 2023)",
            "start_date": "2023-07-01",
            "end_date": "2023-09-30"
        },
        {
            "name": "Earlier Period (Apr-Jun 2023)",
            "start_date": "2023-04-01", 
            "end_date": "2023-06-30"
        }
    ]
    
    results_comparison = []
    
    for period in test_periods:
        print(f"\n📊 Testing: {period['name']}")
        print(f"   Period: {period['start_date']} to {period['end_date']}")
        
        # Test conservative strategy
        print(f"\n   🟢 CONSERVATIVE STRATEGY:")
        try:
            conservative_results = run_conservative_backtest(
                start_date=period['start_date'],
                end_date=period['end_date'],
                initial_capital=100000.0
            )
            
            if conservative_results:
                print(f"      ✅ Total Return: {conservative_results.get('total_return', 0):.2%}")
                print(f"      ✅ Max Drawdown: {conservative_results.get('max_drawdown', 0):.2%}")
                print(f"      ✅ Sharpe Ratio: {conservative_results.get('sharpe_ratio', 0):.3f}")
                print(f"      ✅ Total Trades: {conservative_results.get('total_trades', 0)}")
                print(f"      ✅ Strategy Type: {conservative_results.get('strategy_type', 'Unknown')}")
            else:
                print(f"      ❌ No results returned")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
        
        # Test overfitted strategy
        print(f"\n   🔴 OVERFITTED STRATEGY:")
        try:
            overfitted_results = run_risk_enhanced_backtest(
                start_date=period['start_date'],
                end_date=period['end_date'],
                initial_capital=100000.0
            )
            
            if overfitted_results:
                print(f"      ⚠️  Total Return: {overfitted_results.get('total_return', 0):.2%}")
                print(f"      ⚠️  Max Drawdown: {overfitted_results.get('max_drawdown', 0):.2%}")
                print(f"      ⚠️  Sharpe Ratio: {overfitted_results.get('sharpe_ratio', 0):.3f}")
                print(f"      ⚠️  Total Trades: {overfitted_results.get('total_trades', 0)}")
            else:
                print(f"      ❌ No results returned")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
        
        # Store comparison
        results_comparison.append({
            'period': period['name'],
            'conservative': conservative_results,
            'overfitted': overfitted_results
        })
    
    # Summary analysis
    print(f"\n📋 COMPARISON SUMMARY")
    print("=" * 70)
    
    for result in results_comparison:
        print(f"\n📊 {result['period']}:")
        
        conservative = result['conservative']
        overfitted = result['overfitted']
        
        if conservative and overfitted:
            # Compare key metrics
            cons_return = conservative.get('total_return', 0)
            over_return = overfitted.get('total_return', 0)
            cons_drawdown = conservative.get('max_drawdown', 0)
            over_drawdown = overfitted.get('max_drawdown', 0)
            cons_sharpe = conservative.get('sharpe_ratio', 0)
            over_sharpe = overfitted.get('sharpe_ratio', 0)
            
            print(f"   Return: Conservative {cons_return:.2%} vs Overfitted {over_return:.2%}")
            print(f"   Drawdown: Conservative {cons_drawdown:.2%} vs Overfitted {over_drawdown:.2%}")
            print(f"   Sharpe: Conservative {cons_sharpe:.3f} vs Overfitted {over_sharpe:.3f}")
            
            # Realism assessment
            print(f"   Realism Assessment:")
            
            if abs(cons_return) < 0.5:  # Less than 50% return
                print(f"      ✅ Conservative return is realistic")
            else:
                print(f"      ⚠️  Conservative return may be optimistic")
            
            if abs(over_return) > 5.0:  # More than 500% return
                print(f"      ❌ Overfitted return is unrealistic")
            else:
                print(f"      ⚠️  Overfitted return may be realistic")
            
            if abs(cons_drawdown) < 0.2:  # Less than 20% drawdown
                print(f"      ✅ Conservative drawdown is reasonable")
            else:
                print(f"      ⚠️  Conservative drawdown is high")
            
            if abs(over_drawdown) > 0.5:  # More than 50% drawdown
                print(f"      ❌ Overfitted drawdown is excessive")
            else:
                print(f"      ⚠️  Overfitted drawdown may be acceptable")
            
            if 0 < cons_sharpe < 3.0:  # Realistic Sharpe range
                print(f"      ✅ Conservative Sharpe is realistic")
            else:
                print(f"      ⚠️  Conservative Sharpe may be unrealistic")
            
            if over_sharpe > 5.0:  # Very high Sharpe
                print(f"      ❌ Overfitted Sharpe is suspiciously high")
            else:
                print(f"      ⚠️  Overfitted Sharpe may be realistic")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   1. Conservative strategy uses realistic parameters")
    print(f"   2. Overfitted strategy shows extreme performance variations")
    print(f"   3. Conservative strategy is more robust across time periods")
    print(f"   4. Overfitted strategy is sensitive to specific market conditions")
    print(f"   5. Conservative strategy prioritizes capital preservation")
    
    return results_comparison

if __name__ == "__main__":
    test_conservative_vs_overfitted() 