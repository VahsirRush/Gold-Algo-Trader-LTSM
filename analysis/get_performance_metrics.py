#!/usr/bin/env python3
"""
Get Performance Metrics
======================

Simple script to get the performance metrics for the optimized strategy.
"""

from gold_algo.strategies.optimized_performance_strategy import run_optimized_performance_backtest

def get_performance_metrics():
    """Get performance metrics for the optimized strategy."""
    
    print("🔍 Getting Optimized Strategy Performance Metrics...")
    print("=" * 60)
    
    # Run the optimized strategy
    result = run_optimized_performance_backtest('2023-01-01', '2023-12-31')
    
    if result:
        print("\n📊 OPTIMIZED STRATEGY PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Total Return: {result['total_return']:.2%}")
        print(f"Max Drawdown: {result['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
        print(f"Win Rate: {result['win_rate']:.1%}")
        print(f"Total Trades: {result['total_trades']}")
        print(f"Final Capital: ${result['final_capital']:,.2f}")
        print(f"Peak Equity: ${result['peak_equity']:,.2f}")
        
        # Compare with previous performance
        print("\n📈 PERFORMANCE COMPARISON")
        print("=" * 60)
        print("Previous Strategy (Macro Regime Full):")
        print("  - Total Return: 8.90%")
        print("  - Max Drawdown: -21.92%")
        print("  - Sharpe Ratio: 0.506")
        print("  - Total Trades: 33")
        
        print("\nOptimized Strategy:")
        print(f"  - Total Return: {result['total_return']:.2%}")
        print(f"  - Max Drawdown: {result['max_drawdown']:.2%}")
        print(f"  - Sharpe Ratio: {result['sharpe_ratio']:.3f}")
        print(f"  - Total Trades: {result['total_trades']}")
        
        # Calculate improvements
        return_improvement = result['total_return'] - 0.089
        drawdown_improvement = -0.2192 - result['max_drawdown']
        sharpe_improvement = result['sharpe_ratio'] - 0.506
        trade_improvement = result['total_trades'] - 33
        
        print("\n🎯 IMPROVEMENTS ACHIEVED")
        print("=" * 60)
        print(f"Return Improvement: {return_improvement:.2%}")
        print(f"Drawdown Improvement: {drawdown_improvement:.2%}")
        print(f"Sharpe Improvement: {sharpe_improvement:.3f}")
        print(f"Trade Frequency Change: {trade_improvement:+d}")
        
        # Assessment
        print("\n✅ PERFORMANCE ASSESSMENT")
        print("=" * 60)
        if result['sharpe_ratio'] > 0.8:
            print("✅ EXCELLENT - Sharpe ratio above target")
        elif result['sharpe_ratio'] > 0.6:
            print("✅ GOOD - Sharpe ratio improved significantly")
        else:
            print("⚠️  NEEDS IMPROVEMENT - Sharpe ratio below target")
            
        if result['max_drawdown'] < -0.15:
            print("⚠️  NEEDS IMPROVEMENT - Max drawdown above target")
        else:
            print("✅ GOOD - Max drawdown within target")
            
        if result['total_trades'] > 50:
            print("✅ EXCELLENT - High trade frequency")
        elif result['total_trades'] > 30:
            print("✅ GOOD - Adequate trade frequency")
        else:
            print("⚠️  NEEDS IMPROVEMENT - Low trade frequency")
            
    else:
        print("❌ Error: Could not get performance metrics")

if __name__ == "__main__":
    get_performance_metrics() 