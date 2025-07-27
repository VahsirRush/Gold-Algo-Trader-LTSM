#!/usr/bin/env python3
"""
Out-of-Sample Testing for Risk-Enhanced Strategy
================================================

This script tests the strategy on different time periods to check for overfitting
and validate the Sharpe ratio calculation.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from gold_algo.strategies.risk_enhanced_strategy import run_risk_enhanced_backtest
from data_pipeline.databento_collector import DatabentoGoldCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_different_periods():
    """Test the strategy on different time periods to check for overfitting."""
    
    print("🔍 OUT-OF-SAMPLE TESTING FOR OVERFITTING")
    print("=" * 60)
    
    # Define different test periods
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
        },
        {
            "name": "Later Period (Oct-Dec 2023)",
            "start_date": "2023-10-01",
            "end_date": "2023-12-31"
        },
        {
            "name": "Full Year (2023)",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
    ]
    
    results_summary = []
    
    for period in test_periods:
        print(f"\n📊 Testing: {period['name']}")
        print(f"   Period: {period['start_date']} to {period['end_date']}")
        
        try:
            # Run backtest
            results = run_risk_enhanced_backtest(
                start_date=period['start_date'],
                end_date=period['end_date'],
                initial_capital=100000.0
            )
            
            if results:
                # Extract key metrics
                total_return = results.get('total_return', 0)
                max_drawdown = results.get('max_drawdown', 0)
                sharpe_ratio = results.get('sharpe_ratio', 0)
                total_trades = results.get('total_trades', 0)
                
                # Calculate annualized metrics properly
                equity_curve = results.get('equity_curve', [])
                if equity_curve:
                    df = pd.DataFrame(equity_curve)
                    df['returns'] = df['portfolio_value'].pct_change()
                    returns = df['returns'].dropna()
                    
                    # Calculate proper annualized metrics
                    days = len(returns)
                    if days > 0:
                        # Annualized return using compound annual growth rate
                        total_return_actual = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1
                        annualized_return = (1 + total_return_actual) ** (252 / days) - 1
                        
                        # Annualized volatility
                        daily_vol = returns.std()
                        annualized_vol = daily_vol * np.sqrt(252)
                        
                        # Proper Sharpe ratio
                        risk_free_rate = 0.02
                        proper_sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
                        
                        print(f"   ✅ Results:")
                        print(f"      Total Return: {total_return:.2%}")
                        print(f"      Annualized Return: {annualized_return:.2%}")
                        print(f"      Max Drawdown: {max_drawdown:.2%}")
                        print(f"      Annualized Volatility: {annualized_vol:.2%}")
                        print(f"      Sharpe Ratio (Original): {sharpe_ratio:.3f}")
                        print(f"      Sharpe Ratio (Proper): {proper_sharpe:.3f}")
                        print(f"      Total Trades: {total_trades}")
                        print(f"      Trading Days: {days}")
                        
                        results_summary.append({
                            'period': period['name'],
                            'total_return': total_return,
                            'annualized_return': annualized_return,
                            'max_drawdown': max_drawdown,
                            'annualized_volatility': annualized_vol,
                            'sharpe_original': sharpe_ratio,
                            'sharpe_proper': proper_sharpe,
                            'total_trades': total_trades,
                            'trading_days': days
                        })
                    else:
                        print(f"   ❌ No valid returns data")
            else:
                print(f"   ❌ No results returned")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary analysis
    print(f"\n📋 SUMMARY ANALYSIS")
    print("=" * 60)
    
    if results_summary:
        df_summary = pd.DataFrame(results_summary)
        
        print(f"Sharpe Ratio Analysis:")
        print(f"   Original Calculation Range: {df_summary['sharpe_original'].min():.3f} to {df_summary['sharpe_original'].max():.3f}")
        print(f"   Proper Calculation Range: {df_summary['sharpe_proper'].min():.3f} to {df_summary['sharpe_proper'].max():.3f}")
        
        print(f"\nReturn Analysis:")
        print(f"   Total Return Range: {df_summary['total_return'].min():.2%} to {df_summary['total_return'].max():.2%}")
        print(f"   Annualized Return Range: {df_summary['annualized_return'].min():.2%} to {df_summary['annualized_return'].max():.2%}")
        
        print(f"\nRisk Analysis:")
        print(f"   Max Drawdown Range: {df_summary['max_drawdown'].min():.2%} to {df_summary['max_drawdown'].max():.2%}")
        print(f"   Volatility Range: {df_summary['annualized_volatility'].min():.2%} to {df_summary['annualized_volatility'].max():.2%}")
        
        # Overfitting indicators
        print(f"\n🔍 OVERFITTING INDICATORS:")
        
        # 1. Sharpe ratio consistency
        sharpe_std = df_summary['sharpe_proper'].std()
        if sharpe_std > 1.0:
            print(f"   ⚠️  HIGH Sharpe ratio variance ({sharpe_std:.3f}) - possible overfitting")
        else:
            print(f"   ✅ Reasonable Sharpe ratio consistency ({sharpe_std:.3f})")
        
        # 2. Return consistency
        return_std = df_summary['annualized_return'].std()
        if return_std > 0.5:  # 50% standard deviation
            print(f"   ⚠️  HIGH return variance ({return_std:.2%}) - possible overfitting")
        else:
            print(f"   ✅ Reasonable return consistency ({return_std:.2%})")
        
        # 3. Performance degradation
        original_performance = df_summary[df_summary['period'].str.contains('Original')]['sharpe_proper'].iloc[0]
        other_performances = df_summary[~df_summary['period'].str.contains('Original')]['sharpe_proper']
        
        if len(other_performances) > 0:
            avg_other = other_performances.mean()
            degradation = (original_performance - avg_other) / original_performance if original_performance != 0 else 0
            
            if degradation > 0.5:  # 50% degradation
                print(f"   ⚠️  HIGH performance degradation ({degradation:.1%}) - likely overfitting")
            else:
                print(f"   ✅ Reasonable performance consistency (degradation: {degradation:.1%})")
        
        print(f"\n🎯 CONCLUSION:")
        if sharpe_std > 1.0 or return_std > 0.5 or degradation > 0.5:
            print(f"   ❌ LIKELY OVERFITTING DETECTED")
            print(f"   Recommendations:")
            print(f"   - Use more conservative parameters")
            print(f"   - Implement cross-validation")
            print(f"   - Test on longer time periods")
        else:
            print(f"   ✅ NO SIGNIFICANT OVERFITTING DETECTED")
            print(f"   - Strategy shows reasonable consistency across periods")
            print(f"   - Sharpe ratios are in realistic ranges")
    
    return results_summary

if __name__ == "__main__":
    test_different_periods() 