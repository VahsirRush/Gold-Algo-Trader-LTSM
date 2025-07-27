#!/usr/bin/env python3
"""
Ensemble Weight Optimizer
Systematically tests different weight combinations to find optimal ensemble configuration
"""

import pandas as pd
import numpy as np
from itertools import product
from shared_utilities import DataFetcher, PerformanceMetrics
from ml_ensemble_strategy import MLEnsembleStrategy

def optimize_ensemble_weights():
    """Systematically optimize ensemble weights for maximum Sharpe ratio."""
    print("🔧 ENSEMBLE WEIGHT OPTIMIZATION")
    print("=" * 50)
    
    # Fetch data
    print("📥 Fetching data...")
    data = DataFetcher.fetch_data('GLD', '3y')
    if data.empty:
        print("❌ Failed to fetch data")
        return
    
    print(f"📊 Data loaded: {len(data)} days")
    print()
    
    # Define weight ranges to test
    weight_ranges = {
        'ml_weight': [0.1, 0.2, 0.3, 0.4, 0.5],
        'rule_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
        'sentiment_weight': [0.05, 0.1, 0.15, 0.2],
        'macro_weight': [0.05, 0.1, 0.15, 0.2]
    }
    
    # Generate valid weight combinations (must sum to 1.0)
    valid_combinations = []
    
    for ml_w, rule_w, sent_w, macro_w in product(
        weight_ranges['ml_weight'],
        weight_ranges['rule_weight'], 
        weight_ranges['sentiment_weight'],
        weight_ranges['macro_weight']
    ):
        total_weight = ml_w + rule_w + sent_w + macro_w
        if abs(total_weight - 1.0) < 0.01:  # Allow small tolerance
            valid_combinations.append({
                'ml_weight': ml_w,
                'rule_weight': rule_w,
                'sentiment_weight': sent_w,
                'macro_weight': macro_w
            })
    
    print(f"🔍 Testing {len(valid_combinations)} weight combinations...")
    print()
    
    # Test each combination
    results = []
    best_sharpe = -np.inf
    best_config = None
    best_result = None
    
    for i, config in enumerate(valid_combinations):
        print(f"🔄 Testing combination {i+1}/{len(valid_combinations)}: {config}")
        
        try:
            # Create strategy with these weights
            strategy = MLEnsembleStrategy(
                use_kelly=True,
                ml_weight=config['ml_weight'],
                rule_weight=config['rule_weight'],
                sentiment_weight=config['sentiment_weight'],
                macro_weight=config['macro_weight']
            )
            
            # Train ML model
            training_results = strategy.train_ml_model(data)
            if not training_results:
                print(f"   ❌ ML training failed")
                continue
            
            # Run backtest
            strategy_results = strategy.run_backtest(data)
            
            # Extract key metrics
            sharpe = strategy_results.get('Sharpe Ratio', -999)
            total_return = strategy_results.get('total_return', 0) * 100
            trades = strategy_results.get('total_trades', 0)
            win_rate = strategy_results.get('win_rate', 0) * 100
            max_drawdown = strategy_results.get('max_drawdown', 0) * 100
            
            # Store results
            result = {
                'config': config,
                'sharpe': sharpe,
                'total_return': total_return,
                'trades': trades,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'strategy_results': strategy_results
            }
            results.append(result)
            
            print(f"   📊 Sharpe: {sharpe:.3f}, Return: {total_return:.2f}%, Trades: {trades}, Win Rate: {win_rate:.1f}%")
            
            # Track best configuration
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_config = config
                best_result = strategy_results
                print(f"   🏆 NEW BEST! Sharpe: {sharpe:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    # Display optimization results
    print("\n" + "=" * 80)
    print("📊 WEIGHT OPTIMIZATION RESULTS")
    print("=" * 80)
    
    if not results:
        print("❌ No valid results found")
        return
    
    # Sort by Sharpe ratio
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    # Display top 10 configurations
    print("\n🏆 TOP 10 CONFIGURATIONS:")
    print("-" * 80)
    
    for i, result in enumerate(results[:10]):
        config = result['config']
        print(f"{i+1:2d}. ML:{config['ml_weight']:.2f} | Rule:{config['rule_weight']:.2f} | "
              f"Sent:{config['sentiment_weight']:.2f} | Macro:{config['macro_weight']:.2f} | "
              f"Sharpe: {result['sharpe']:.3f} | Return: {result['total_return']:.2f}% | "
              f"Trades: {result['trades']}")
    
    # Display best configuration details
    print(f"\n🎯 BEST CONFIGURATION:")
    print("-" * 40)
    print(f"ML Weight: {best_config['ml_weight']:.2f}")
    print(f"Rule Weight: {best_config['rule_weight']:.2f}")
    print(f"Sentiment Weight: {best_config['sentiment_weight']:.2f}")
    print(f"Macro Weight: {best_config['macro_weight']:.2f}")
    print()
    
    print("📊 BEST PERFORMANCE METRICS:")
    PerformanceMetrics.print_metrics(best_result)
    
    # Analyze weight patterns
    print("\n📈 WEIGHT PATTERN ANALYSIS:")
    print("-" * 30)
    
    # Find patterns in top performers
    top_5 = results[:5]
    
    avg_ml = np.mean([r['config']['ml_weight'] for r in top_5])
    avg_rule = np.mean([r['config']['rule_weight'] for r in top_5])
    avg_sent = np.mean([r['config']['sentiment_weight'] for r in top_5])
    avg_macro = np.mean([r['config']['macro_weight'] for r in top_5])
    
    print(f"Average weights in top 5 performers:")
    print(f"  ML: {avg_ml:.3f}")
    print(f"  Rule: {avg_rule:.3f}")
    print(f"  Sentiment: {avg_sent:.3f}")
    print(f"  Macro: {avg_macro:.3f}")
    
    # Check if we reached 2+ Sharpe
    if best_sharpe >= 2.0:
        print(f"\n🎉 TARGET ACHIEVED! 2+ Sharpe ratio reached: {best_sharpe:.3f}")
    else:
        print(f"\n📈 Need {(2.0 - best_sharpe):.3f} more Sharpe to reach target")
    
    return best_config, best_result

def create_optimized_strategy():
    """Create a strategy with the optimized weights."""
    print("\n🚀 CREATING OPTIMIZED STRATEGY")
    print("=" * 40)
    
    # Run optimization
    best_config, best_result = optimize_ensemble_weights()
    
    if best_config is None:
        print("❌ Optimization failed")
        return None
    
    # Create optimized strategy
    optimized_strategy = MLEnsembleStrategy(
        use_kelly=True,
        ml_weight=best_config['ml_weight'],
        rule_weight=best_config['rule_weight'],
        sentiment_weight=best_config['sentiment_weight'],
        macro_weight=best_config['macro_weight']
    )
    
    print(f"\n✅ Optimized strategy created with weights:")
    print(f"   ML: {best_config['ml_weight']:.2f}")
    print(f"   Rule: {best_config['rule_weight']:.2f}")
    print(f"   Sentiment: {best_config['sentiment_weight']:.2f}")
    print(f"   Macro: {best_config['macro_weight']:.2f}")
    
    return optimized_strategy

def test_optimized_strategy():
    """Test the optimized strategy on different time periods."""
    print("\n🧪 TESTING OPTIMIZED STRATEGY")
    print("=" * 40)
    
    # Create optimized strategy
    strategy = create_optimized_strategy()
    if strategy is None:
        return
    
    # Test on different time periods
    periods = ['1y', '2y', '3y', '5y']
    
    for period in periods:
        print(f"\n📊 Testing on {period} data...")
        
        # Fetch data
        data = DataFetcher.fetch_data('GLD', period)
        if data.empty:
            print(f"   ❌ Failed to fetch {period} data")
            continue
        
        # Train ML model
        training_results = strategy.train_ml_model(data)
        if not training_results:
            print(f"   ❌ ML training failed for {period}")
            continue
        
        # Run backtest
        results = strategy.run_backtest(data)
        
        sharpe = results.get('Sharpe Ratio', 0)
        total_return = results.get('total_return', 0) * 100
        trades = results.get('total_trades', 0)
        
        print(f"   📊 Sharpe: {sharpe:.3f}, Return: {total_return:.2f}%, Trades: {trades}")
        
        if sharpe >= 2.0:
            print(f"   🎉 2+ Sharpe achieved on {period} data!")

if __name__ == "__main__":
    # Run weight optimization
    optimize_ensemble_weights()
    
    # Test optimized strategy
    test_optimized_strategy() 
 