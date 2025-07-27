#!/usr/bin/env python3
"""
Verify Performance Metrics
=========================

This script verifies the performance metrics to ensure they are calculated correctly.
"""

import numpy as np
import pandas as pd
from gold_algo.strategies.optimized_performance_strategy import run_optimized_performance_backtest

def verify_performance_metrics():
    """Verify the performance metrics are calculated correctly."""
    
    print("🔍 VERIFYING PERFORMANCE METRICS")
    print("=" * 60)
    
    # Run the optimized strategy
    result = run_optimized_performance_backtest('2023-01-01', '2023-12-31')
    
    if not result:
        print("❌ Error: Could not get results")
        return
    
    print(f"\n📊 REPORTED METRICS:")
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"Win Rate: {result['win_rate']:.1%}")
    print(f"Total Trades: {result['total_trades']}")
    
    # Verify using equity curve
    print(f"\n🔍 VERIFICATION USING EQUITY CURVE:")
    
    if 'equity_curve' in result and result['equity_curve']:
        equity_df = pd.DataFrame(result['equity_curve'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Verify total return
        calculated_return = (equity_df['equity'].iloc[-1] - equity_df['equity'].iloc[0]) / equity_df['equity'].iloc[0]
        print(f"Calculated Total Return: {calculated_return:.2%}")
        print(f"Reported Total Return: {result['total_return']:.2%}")
        print(f"Return Match: {'✅' if abs(calculated_return - result['total_return']) < 0.001 else '❌'}")
        
        # Verify max drawdown
        peak = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - peak) / peak
        calculated_max_dd = drawdown.min()
        print(f"Calculated Max Drawdown: {calculated_max_dd:.2%}")
        print(f"Reported Max Drawdown: {result['max_drawdown']:.2%}")
        print(f"Drawdown Match: {'✅' if abs(calculated_max_dd - result['max_drawdown']) < 0.001 else '❌'}")
        
        # Verify Sharpe ratio
        if returns.std() > 0:
            calculated_sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            print(f"Calculated Sharpe Ratio: {calculated_sharpe:.3f}")
            print(f"Reported Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(f"Sharpe Match: {'✅' if abs(calculated_sharpe - result['sharpe_ratio']) < 0.01 else '❌'}")
            
            # Additional Sharpe ratio checks
            print(f"\n📈 SHARPE RATIO COMPONENTS:")
            print(f"Annualized Return: {returns.mean() * 252:.2%}")
            print(f"Annualized Volatility: {returns.std() * np.sqrt(252):.2%}")
            print(f"Daily Return Mean: {returns.mean():.4f}")
            print(f"Daily Return Std: {returns.std():.4f}")
            print(f"Number of Trading Days: {len(returns)}")
            
            # Check for potential issues
            if returns.std() < 0.001:
                print("⚠️  WARNING: Very low volatility - Sharpe ratio may be inflated")
            if abs(returns.mean()) < 0.0001:
                print("⚠️  WARNING: Very low mean return - Sharpe ratio may be unreliable")
                
        else:
            print("❌ ERROR: Zero volatility - Sharpe ratio calculation invalid")
        
        # Verify win rate
        if 'trades' in result and result['trades']:
            trades = result['trades']
            print(f"\n📊 TRADE ANALYSIS:")
            print(f"Total Trades: {len(trades)}")
            
            # Calculate trade P&L if available
            if len(trades) > 1:
                trade_pnls = []
                for i in range(1, len(trades)):
                    prev_trade = trades[i-1]
                    curr_trade = trades[i]
                    
                    # Simple P&L calculation (this is approximate)
                    if prev_trade['action'] != curr_trade['action']:
                        price_change = (curr_trade['price'] - prev_trade['price']) / prev_trade['price']
                        position = prev_trade['position']
                        pnl = position * price_change
                        trade_pnls.append(pnl)
                
                if trade_pnls:
                    winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
                    calculated_win_rate = winning_trades / len(trade_pnls)
                    print(f"Calculated Win Rate: {calculated_win_rate:.1%}")
                    print(f"Reported Win Rate: {result['win_rate']:.1%}")
                    print(f"Win Rate Match: {'✅' if abs(calculated_win_rate - result['win_rate']) < 0.1 else '❌'}")
        
        # Additional verification checks
        print(f"\n🔍 ADDITIONAL VERIFICATION:")
        print(f"Initial Capital: ${equity_df['equity'].iloc[0]:,.2f}")
        print(f"Final Capital: ${equity_df['equity'].iloc[-1]:,.2f}")
        print(f"Peak Capital: ${equity_df['equity'].max():,.2f}")
        print(f"Min Capital: ${equity_df['equity'].min():,.2f}")
        print(f"Capital Growth: {equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0] - 1:.2%}")
        
        # Check for suspicious patterns
        print(f"\n⚠️  SUSPICIOUS PATTERN CHECKS:")
        
        # Check for constant returns
        if returns.std() < 0.0001:
            print("❌ SUSPICIOUS: Near-constant returns detected")
        
        # Check for unrealistic Sharpe ratio
        if result['sharpe_ratio'] > 5:
            print("⚠️  WARNING: Very high Sharpe ratio (>5) - may indicate calculation error")
        
        # Check for zero drawdown
        if abs(result['max_drawdown']) < 0.001:
            print("⚠️  WARNING: Near-zero drawdown - may indicate calculation error")
        
        # Check for low trade count
        if result['total_trades'] < 10:
            print("⚠️  WARNING: Very low trade count - may not be statistically significant")
        
        # Overall assessment
        print(f"\n✅ OVERALL VERIFICATION ASSESSMENT:")
        
        issues_found = 0
        if abs(calculated_return - result['total_return']) > 0.001:
            issues_found += 1
        if abs(calculated_max_dd - result['max_drawdown']) > 0.001:
            issues_found += 1
        if returns.std() > 0 and abs(calculated_sharpe - result['sharpe_ratio']) > 0.01:
            issues_found += 1
        
        if issues_found == 0:
            print("✅ ALL METRICS VERIFIED CORRECTLY")
        else:
            print(f"❌ {issues_found} METRIC(S) HAVE DISCREPANCIES")
            
    else:
        print("❌ ERROR: No equity curve data available for verification")

if __name__ == "__main__":
    verify_performance_metrics() 