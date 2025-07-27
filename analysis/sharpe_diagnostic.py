#!/usr/bin/env python3
"""
Sharpe Ratio Diagnostic Tool
============================

This script investigates the Sharpe ratio calculation to identify potential issues
like overfitting, calculation errors, or unrealistic volatility values.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from gold_algo.strategies.risk_enhanced_strategy import run_risk_enhanced_backtest
from data_pipeline.databento_collector import DatabentoGoldCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_sharpe_calculation():
    """Analyze the Sharpe ratio calculation in detail."""
    
    print("🔍 SHARPE RATIO DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    # Run the risk-enhanced strategy
    results = run_risk_enhanced_backtest(
        start_date="2023-07-01",
        end_date="2023-09-30",
        initial_capital=100000.0
    )
    
    if not results:
        print("❌ No results returned from backtest")
        return
    
    # Get the equity curve
    equity_curve = results.get('equity_curve', [])
    if not equity_curve:
        print("❌ No equity curve data")
        return
    
    # Convert to DataFrame
    equity_df = pd.DataFrame(equity_curve)
    equity_df.set_index('timestamp', inplace=True)
    
    print(f"📊 EQUITY CURVE ANALYSIS")
    print(f"   Data points: {len(equity_df)}")
    print(f"   Date range: {equity_df.index.min()} to {equity_df.index.max()}")
    print(f"   Initial value: ${equity_df['portfolio_value'].iloc[0]:,.2f}")
    print(f"   Final value: ${equity_df['portfolio_value'].iloc[-1]:,.2f}")
    
    # Calculate returns
    equity_df['returns'] = equity_df['portfolio_value'].pct_change()
    returns = equity_df['returns'].dropna()
    
    print(f"\n📈 RETURNS ANALYSIS")
    print(f"   Number of returns: {len(returns)}")
    print(f"   Mean return: {returns.mean():.6f}")
    print(f"   Std return: {returns.std():.6f}")
    print(f"   Min return: {returns.min():.6f}")
    print(f"   Max return: {returns.max():.6f}")
    print(f"   Skewness: {returns.skew():.3f}")
    print(f"   Kurtosis: {returns.kurtosis():.3f}")
    
    # Check for potential issues
    print(f"\n⚠️  POTENTIAL ISSUES CHECK")
    
    # 1. Check for zero or near-zero volatility
    volatility = returns.std()
    if volatility < 1e-6:
        print(f"   ❌ CRITICAL: Near-zero volatility detected: {volatility:.10f}")
    else:
        print(f"   ✅ Volatility looks reasonable: {volatility:.6f}")
    
    # 2. Check for unrealistic returns
    if abs(returns.max()) > 0.5:  # 50% daily return
        print(f"   ❌ CRITICAL: Unrealistic daily return detected: {returns.max():.2%}")
    else:
        print(f"   ✅ Daily returns look reasonable: max {returns.max():.2%}")
    
    # 3. Check for constant returns (overfitting indicator)
    if returns.std() < 1e-4:
        print(f"   ❌ CRITICAL: Constant returns detected (overfitting)")
    else:
        print(f"   ✅ Returns have reasonable variance")
    
    # 4. Check for perfect returns
    if returns.min() >= 0:
        print(f"   ⚠️  WARNING: No negative returns detected (suspicious)")
    else:
        print(f"   ✅ Negative returns present: {returns.min():.2%}")
    
    # Calculate Sharpe ratio step by step
    print(f"\n🧮 SHARPE RATIO CALCULATION BREAKDOWN")
    
    # Step 1: Total return
    total_return = (1 + returns).prod() - 1
    print(f"   Step 1 - Total return: {total_return:.6f}")
    
    # Step 2: Annualized return
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    print(f"   Step 2 - Annualized return: {annualized_return:.6f}")
    
    # Step 3: Annualized volatility
    annualized_volatility = returns.std() * np.sqrt(252)
    print(f"   Step 3 - Annualized volatility: {annualized_volatility:.6f}")
    
    # Step 4: Risk-free rate
    risk_free_rate = 0.02
    print(f"   Step 4 - Risk-free rate: {risk_free_rate:.6f}")
    
    # Step 5: Sharpe ratio
    if annualized_volatility > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        print(f"   Step 5 - Sharpe ratio: {sharpe_ratio:.6f}")
    else:
        print(f"   Step 5 - Sharpe ratio: Cannot calculate (zero volatility)")
    
    # Alternative Sharpe calculation for comparison
    print(f"\n🔄 ALTERNATIVE CALCULATIONS")
    
    # Method 1: Simple Sharpe (no annualization)
    simple_sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
    print(f"   Simple Sharpe (daily): {simple_sharpe:.6f}")
    
    # Method 2: Annualized simple Sharpe
    annualized_simple_sharpe = simple_sharpe * np.sqrt(252) if returns.std() > 0 else 0
    print(f"   Annualized Simple Sharpe: {annualized_simple_sharpe:.6f}")
    
    # Method 3: Using log returns
    log_returns = np.log(equity_df['portfolio_value'] / equity_df['portfolio_value'].shift(1)).dropna()
    if len(log_returns) > 0 and log_returns.std() > 0:
        log_sharpe = log_returns.mean() / log_returns.std() * np.sqrt(252)
        print(f"   Log Returns Sharpe: {log_sharpe:.6f}")
    else:
        print(f"   Log Returns Sharpe: Cannot calculate")
    
    # Check for overfitting indicators
    print(f"\n🔍 OVERFITTING ANALYSIS")
    
    # 1. Check if strategy trades too frequently
    trades = results.get('trades', [])
    print(f"   Total trades: {len(trades)}")
    print(f"   Trades per day: {len(trades) / len(equity_df):.2f}")
    
    if len(trades) / len(equity_df) > 2:
        print(f"   ⚠️  WARNING: High trade frequency may indicate overfitting")
    else:
        print(f"   ✅ Trade frequency looks reasonable")
    
    # 2. Check for parameter sensitivity
    print(f"\n📊 PARAMETER SENSITIVITY TEST")
    
    # Test with different parameters
    test_params = [
        {"target_volatility": 0.01, "hard_drawdown_limit": 0.03},
        {"target_volatility": 0.03, "hard_drawdown_limit": 0.07},
        {"target_volatility": 0.02, "hard_drawdown_limit": 0.05}
    ]
    
    for i, params in enumerate(test_params):
        print(f"   Test {i+1}: {params}")
        # Note: This would require modifying the strategy to accept parameters
        # For now, just show the current results
    
    print(f"\n🎯 CONCLUSION")
    print(f"   If Sharpe ratio > 3.0, it's likely overfitting or calculation error")
    print(f"   If volatility < 0.001, calculation error likely")
    print(f"   If no negative returns, strategy may be overfitted")
    print(f"   If trade frequency > 2/day, may be overfitting")

if __name__ == "__main__":
    analyze_sharpe_calculation() 