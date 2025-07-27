#!/usr/bin/env python3
"""
Protected Strategy Results Summary
=================================

This script displays the results of the protected conservative strategy
with overfitting protections in a clean, readable format.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from gold_algo.strategies.protected_conservative_strategy import run_protected_conservative_backtest

def display_protected_strategy_results():
    """Display protected strategy results in a clean format."""
    
    print("🛡️  PROTECTED CONSERVATIVE STRATEGY RESULTS")
    print("=" * 60)
    
    # Run the protected strategy
    results = run_protected_conservative_backtest(
        start_date="2023-07-01",
        end_date="2023-09-30",
        initial_capital=100000.0,
        enable_protection=True
    )
    
    if not results:
        print("❌ No results returned")
        return
    
    # Extract key metrics
    total_return = results.get('total_return', 0)
    max_drawdown = results.get('max_drawdown', 0)
    sharpe_ratio = results.get('sharpe_ratio', 0)
    win_rate = results.get('win_rate', 0)
    total_trades = results.get('total_trades', 0)
    initial_capital = results.get('initial_capital', 100000)
    final_capital = results.get('final_capital', 100000)
    
    # Extract protection results
    protection = results.get('overfitting_protection', {})
    overfitting_score = protection.get('overfitting_score', 1.0)
    overfitting_risk = protection.get('overfitting_risk', 'UNKNOWN')
    is_safe_to_deploy = protection.get('is_safe_to_deploy', False)
    recommendations = protection.get('recommendations', [])
    
    # Display performance metrics
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Max Drawdown: {max_drawdown:.2%}")
    print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Total Trades: {total_trades}")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Capital: ${final_capital:,.2f}")
    print(f"   Net P&L: ${final_capital - initial_capital:,.2f}")
    
    # Display overfitting protection results
    print(f"\n🛡️  OVERFITTING PROTECTION RESULTS:")
    print(f"   Overfitting Score: {overfitting_score:.3f}")
    print(f"   Risk Level: {overfitting_risk}")
    print(f"   Safe to Deploy: {'✅ YES' if is_safe_to_deploy else '❌ NO'}")
    
    # Display recommendations
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Display trade details
    trades = results.get('trades', [])
    if trades:
        print(f"\n📈 TRADE DETAILS:")
        print(f"   Number of Trades: {len(trades)}")
        
        # Show last few trades
        print(f"   Recent Trades:")
        for i, trade in enumerate(trades[-3:], 1):
            direction = "BUY" if trade['direction'] > 0 else "SELL"
            print(f"     {i}. {direction} {trade['shares']:.2f} shares at ${trade['price']:.2f}")
    
    # Display equity curve summary
    equity_curve = results.get('equity_curve', [])
    if equity_curve:
        print(f"\n📊 EQUITY CURVE SUMMARY:")
        print(f"   Data Points: {len(equity_curve)}")
        
        # Calculate some equity curve statistics
        portfolio_values = [point['portfolio_value'] for point in equity_curve]
        if portfolio_values:
            min_value = min(portfolio_values)
            max_value = max(portfolio_values)
            print(f"   Min Portfolio Value: ${min_value:,.2f}")
            print(f"   Max Portfolio Value: ${max_value:,.2f}")
            print(f"   Portfolio Value Range: ${max_value - min_value:,.2f}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    if is_safe_to_deploy:
        print(f"   ✅ Strategy is SAFE to deploy")
        print(f"   ✅ Overfitting protection is working")
        print(f"   ✅ Risk management is effective")
    else:
        print(f"   ⚠️  Strategy needs review before deployment")
        print(f"   ⚠️  Overfitting concerns detected")
    
    if total_return > 0:
        print(f"   ✅ Strategy generated positive returns")
    elif total_return == 0:
        print(f"   ⚠️  Strategy had neutral performance")
    else:
        print(f"   ⚠️  Strategy had negative returns")
    
    if abs(max_drawdown) < 0.05:  # Less than 5%
        print(f"   ✅ Excellent risk control (low drawdown)")
    elif abs(max_drawdown) < 0.10:  # Less than 10%
        print(f"   ✅ Good risk control")
    else:
        print(f"   ⚠️  High drawdown - consider risk management")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Monitor strategy performance regularly")
    print(f"   2. Re-run overfitting protection monthly")
    print(f"   3. Adjust parameters if needed")
    print(f"   4. Scale up gradually if performance is consistent")
    
    return results

if __name__ == "__main__":
    display_protected_strategy_results() 