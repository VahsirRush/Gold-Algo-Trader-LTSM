#!/usr/bin/env python3
"""
Test Overfitting Protection System
=================================

This script demonstrates the comprehensive overfitting protection system by:
1. Testing protected vs unprotected strategies
2. Showing overfitting detection capabilities
3. Comparing performance across different time periods
4. Demonstrating protection recommendations
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from gold_algo.strategies.protected_conservative_strategy import run_protected_conservative_backtest
from gold_algo.strategies.risk_enhanced_strategy import run_risk_enhanced_backtest
from overfitting_protection import OverfittingProtection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_overfitting_protection_system():
    """Test the comprehensive overfitting protection system."""
    
    print("🛡️  OVERFITTING PROTECTION SYSTEM TEST")
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
    
    results_summary = []
    
    for period in test_periods:
        print(f"\n📊 Testing: {period['name']}")
        print(f"   Period: {period['start_date']} to {period['end_date']}")
        
        # Test 1: Protected Conservative Strategy
        print(f"\n   🛡️  PROTECTED CONSERVATIVE STRATEGY:")
        try:
            protected_results = run_protected_conservative_backtest(
                start_date=period['start_date'],
                end_date=period['end_date'],
                initial_capital=100000.0,
                enable_protection=True
            )
            
            if protected_results:
                print(f"      ✅ Total Return: {protected_results.get('total_return', 0):.2%}")
                print(f"      ✅ Max Drawdown: {protected_results.get('max_drawdown', 0):.2%}")
                print(f"      ✅ Sharpe Ratio: {protected_results.get('sharpe_ratio', 0):.3f}")
                print(f"      ✅ Total Trades: {protected_results.get('total_trades', 0)}")
                
                # Overfitting protection results
                protection = protected_results.get('overfitting_protection', {})
                if protection:
                    print(f"      🛡️  Overfitting Risk: {protection.get('overfitting_risk', 'UNKNOWN')}")
                    print(f"      🛡️  Overfitting Score: {protection.get('overfitting_score', 0):.3f}")
                    print(f"      🛡️  Safe to Deploy: {protection.get('is_safe_to_deploy', False)}")
                    
                    # Show recommendations
                    recommendations = protection.get('recommendations', [])
                    if recommendations:
                        print(f"      💡 Recommendations:")
                        for rec in recommendations[:3]:  # Show first 3
                            print(f"         - {rec}")
            else:
                print(f"      ❌ No results returned")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
        
        # Test 2: Unprotected Conservative Strategy
        print(f"\n   🔓 UNPROTECTED CONSERVATIVE STRATEGY:")
        try:
            unprotected_results = run_protected_conservative_backtest(
                start_date=period['start_date'],
                end_date=period['end_date'],
                initial_capital=100000.0,
                enable_protection=False
            )
            
            if unprotected_results:
                print(f"      ⚠️  Total Return: {unprotected_results.get('total_return', 0):.2%}")
                print(f"      ⚠️  Max Drawdown: {unprotected_results.get('max_drawdown', 0):.2%}")
                print(f"      ⚠️  Sharpe Ratio: {unprotected_results.get('sharpe_ratio', 0):.3f}")
                print(f"      ⚠️  Total Trades: {unprotected_results.get('total_trades', 0)}")
                print(f"      ⚠️  No overfitting protection applied")
            else:
                print(f"      ❌ No results returned")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
        
        # Test 3: Overfitted Strategy (for comparison)
        print(f"\n   🔴 OVERFITTED STRATEGY (for comparison):")
        try:
            overfitted_results = run_risk_enhanced_backtest(
                start_date=period['start_date'],
                end_date=period['end_date'],
                initial_capital=100000.0
            )
            
            if overfitted_results:
                print(f"      ❌ Total Return: {overfitted_results.get('total_return', 0):.2%}")
                print(f"      ❌ Max Drawdown: {overfitted_results.get('max_drawdown', 0):.2%}")
                print(f"      ❌ Sharpe Ratio: {overfitted_results.get('sharpe_ratio', 0):.3f}")
                print(f"      ❌ Total Trades: {overfitted_results.get('total_trades', 0)}")
                print(f"      ❌ Known overfitting issues")
            else:
                print(f"      ❌ No results returned")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
        
        # Store results for summary
        results_summary.append({
            'period': period['name'],
            'protected': protected_results,
            'unprotected': unprotected_results,
            'overfitted': overfitted_results
        })
    
    # Comprehensive analysis
    print(f"\n📋 COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    for result in results_summary:
        print(f"\n📊 {result['period']}:")
        
        protected = result['protected']
        unprotected = result['unprotected']
        overfitted = result['overfitted']
        
        if protected and unprotected and overfitted:
            # Compare key metrics
            p_return = protected.get('total_return', 0)
            u_return = unprotected.get('total_return', 0)
            o_return = overfitted.get('total_return', 0)
            
            p_drawdown = protected.get('max_drawdown', 0)
            u_drawdown = unprotected.get('max_drawdown', 0)
            o_drawdown = overfitted.get('max_drawdown', 0)
            
            p_sharpe = protected.get('sharpe_ratio', 0)
            u_sharpe = unprotected.get('sharpe_ratio', 0)
            o_sharpe = overfitted.get('sharpe_ratio', 0)
            
            print(f"   Returns: Protected {p_return:.2%} | Unprotected {u_return:.2%} | Overfitted {o_return:.2%}")
            print(f"   Drawdown: Protected {p_drawdown:.2%} | Unprotected {u_drawdown:.2%} | Overfitted {o_drawdown:.2%}")
            print(f"   Sharpe: Protected {p_sharpe:.3f} | Unprotected {u_sharpe:.3f} | Overfitted {o_sharpe:.3f}")
            
            # Protection analysis
            protection = protected.get('overfitting_protection', {})
            if protection:
                risk = protection.get('overfitting_risk', 'UNKNOWN')
                score = protection.get('overfitting_score', 0)
                safe = protection.get('is_safe_to_deploy', False)
                
                print(f"   Protection: Risk={risk}, Score={score:.3f}, Safe={safe}")
                
                # Realism assessment
                print(f"   Realism Assessment:")
                
                if abs(p_return) < 0.5 and 0 < p_sharpe < 3.0:
                    print(f"      ✅ Protected strategy shows realistic performance")
                else:
                    print(f"      ⚠️  Protected strategy may have unrealistic metrics")
                
                if abs(o_return) > 5.0 or o_sharpe > 5.0:
                    print(f"      ❌ Overfitted strategy shows unrealistic performance")
                else:
                    print(f"      ⚠️  Overfitted strategy may be realistic in this period")
                
                if abs(p_drawdown) < abs(o_drawdown):
                    print(f"      ✅ Protected strategy has better risk control")
                else:
                    print(f"      ⚠️  Protected strategy has higher drawdown")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT")
    print("=" * 70)
    
    # Count safe deployments
    safe_deployments = 0
    total_tests = 0
    
    for result in results_summary:
        protected = result['protected']
        if protected:
            protection = protected.get('overfitting_protection', {})
            if protection.get('is_safe_to_deploy', False):
                safe_deployments += 1
            total_tests += 1
    
    if total_tests > 0:
        safety_rate = safe_deployments / total_tests
        print(f"   Safety Rate: {safety_rate:.1%} ({safe_deployments}/{total_tests} tests safe to deploy)")
        
        if safety_rate >= 0.8:
            print(f"   ✅ Excellent protection system - high safety rate")
        elif safety_rate >= 0.6:
            print(f"   ⚠️  Good protection system - moderate safety rate")
        else:
            print(f"   ❌ Poor protection system - low safety rate")
    
    # Key benefits
    print(f"\n💡 KEY BENEFITS OF OVERFITTING PROTECTION:")
    print(f"   1. 🛡️  Automatic detection of overfitting patterns")
    print(f"   2. 📊 Cross-validation testing across multiple folds")
    print(f"   3. 🔍 Parameter sensitivity analysis")
    print(f"   4. 📈 Walk-forward performance validation")
    print(f"   5. 🎯 Realistic performance bounds checking")
    print(f"   6. 💡 Actionable recommendations for improvement")
    print(f"   7. 🚦 Clear deployment safety indicators")
    print(f"   8. 📋 Comprehensive risk assessment")
    
    # Recommendations
    print(f"\n🚀 RECOMMENDATIONS:")
    print(f"   1. Always enable overfitting protection for new strategies")
    print(f"   2. Monitor protection scores over time")
    print(f"   3. Follow protection recommendations")
    print(f"   4. Use multiple time periods for validation")
    print(f"   5. Implement conservative parameters by default")
    print(f"   6. Regular re-validation of strategy performance")
    
    return results_summary

def test_protection_components():
    """Test individual protection components."""
    
    print(f"\n🔧 TESTING INDIVIDUAL PROTECTION COMPONENTS")
    print("=" * 70)
    
    # Create protection system
    protection = OverfittingProtection()
    
    # Test realistic performance bounds
    print(f"\n📊 Testing Realistic Performance Bounds:")
    
    test_cases = [
        {'total_return': 0.15, 'sharpe_ratio': 1.5, 'max_drawdown': -0.10, 'total_trades': 50},
        {'total_return': 5.0, 'sharpe_ratio': 10.0, 'max_drawdown': -0.80, 'total_trades': 500},
        {'total_return': -0.30, 'sharpe_ratio': -0.5, 'max_drawdown': -0.25, 'total_trades': 25}
    ]
    
    for i, case in enumerate(test_cases, 1):
        result = protection.realistic_performance_bounds(case)
        print(f"   Case {i}: Risk={result.get('overfitting_risk')}, Realistic={result.get('is_realistic')}")
        if result.get('violations'):
            print(f"      Violations: {result['violations']}")

if __name__ == "__main__":
    # Run comprehensive test
    results = test_overfitting_protection_system()
    
    # Test individual components
    test_protection_components()
    
    print(f"\n✅ Overfitting protection system test completed!")
    print(f"   Check the results above to see the protection system in action.") 