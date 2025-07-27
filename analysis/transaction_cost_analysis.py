#!/usr/bin/env python3
"""
Transaction Cost Analysis
========================

This script analyzes the impact of realistic transaction costs including:
1. Commission costs (per trade)
2. Slippage (bid-ask spread impact)
3. Market impact (for larger trades)
4. Different cost scenarios (low, medium, high)

This provides a more realistic assessment of actual trading performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from gold_algo.strategies.optimized_performance_strategy import run_optimized_performance_backtest
from data_pipeline.databento_collector import DatabentoGoldCollector
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransactionCostAnalyzer:
    """Analyzes the impact of transaction costs on strategy performance."""
    
    def __init__(self):
        self.cost_scenarios = {
            'low': {
                'commission_per_share': 0.01,  # $0.01 per share
                'commission_per_trade': 1.00,  # $1.00 per trade
                'slippage_bps': 2,  # 2 basis points
                'market_impact_bps': 1,  # 1 basis point per $10k
                'description': 'Low Cost (Discount Broker)'
            },
            'medium': {
                'commission_per_share': 0.02,  # $0.02 per share
                'commission_per_trade': 2.50,  # $2.50 per trade
                'slippage_bps': 5,  # 5 basis points
                'market_impact_bps': 2,  # 2 basis points per $10k
                'description': 'Medium Cost (Standard Broker)'
            },
            'high': {
                'commission_per_share': 0.05,  # $0.05 per share
                'commission_per_trade': 5.00,  # $5.00 per trade
                'slippage_bps': 10,  # 10 basis points
                'market_impact_bps': 5,  # 5 basis points per $10k
                'description': 'High Cost (Premium Broker)'
            },
            'realistic': {
                'commission_per_share': 0.015,  # $0.015 per share
                'commission_per_trade': 1.50,  # $1.50 per trade
                'slippage_bps': 3,  # 3 basis points
                'market_impact_bps': 1.5,  # 1.5 basis points per $10k
                'description': 'Realistic Cost (Typical Retail)'
            }
        }
    
    def calculate_transaction_costs(self, trade_data, cost_scenario):
        """Calculate transaction costs for a series of trades."""
        
        costs = self.cost_scenarios[cost_scenario]
        
        total_commission = 0
        total_slippage = 0
        total_market_impact = 0
        
        for trade in trade_data:
            # Calculate position value (position is a percentage, not absolute shares)
            # We need to estimate the actual dollar value traded
            # For simplicity, assume average trade size of $15,000 (15% of $100k)
            estimated_trade_value = 15000  # This is approximate
            
            # Calculate shares traded
            shares_traded = estimated_trade_value / trade['price']
            
            # Commission costs
            commission = costs['commission_per_share'] * shares_traded + costs['commission_per_trade']
            total_commission += commission
            
            # Slippage costs (bid-ask spread)
            slippage_cost = trade['price'] * (costs['slippage_bps'] / 10000) * shares_traded
            total_slippage += slippage_cost
            
            # Market impact (for larger trades)
            trade_value = shares_traded * trade['price']
            market_impact = trade_value * (costs['market_impact_bps'] / 10000) * (trade_value / 10000)
            total_market_impact += market_impact
        
        return {
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_market_impact': total_market_impact,
            'total_costs': total_commission + total_slippage + total_market_impact
        }
    
    def analyze_cost_impact(self, start_date="2023-01-01", end_date="2023-12-31"):
        """Analyze the impact of different transaction cost scenarios."""
        
        print("💰 TRANSACTION COST ANALYSIS")
        print("=" * 60)
        
        # Get baseline results without transaction costs
        print("📊 Getting baseline results...")
        baseline_result = run_optimized_performance_backtest(start_date, end_date)
        
        if not baseline_result:
            print("❌ Error: Could not get baseline results")
            return None
        
        print(f"✅ Baseline Return: {baseline_result['total_return']:.2%}")
        print(f"✅ Baseline Sharpe: {baseline_result['sharpe_ratio']:.3f}")
        print(f"✅ Baseline Trades: {baseline_result['total_trades']}")
        
        # Analyze each cost scenario
        results = {}
        
        print(f"\n🔍 Analyzing {len(self.cost_scenarios)} cost scenarios:")
        print("-" * 50)
        
        for scenario_name, costs in self.cost_scenarios.items():
            print(f"\n💰 Testing: {costs['description']}")
            
            # Calculate transaction costs
            if 'trades' in baseline_result and baseline_result['trades']:
                transaction_costs = self.calculate_transaction_costs(
                    baseline_result['trades'], scenario_name
                )
                
                # Adjust performance metrics
                initial_capital = baseline_result.get('initial_capital', 100000)
                final_capital = baseline_result.get('final_capital', initial_capital)
                
                # Subtract transaction costs
                adjusted_final_capital = final_capital - transaction_costs['total_costs']
                adjusted_return = (adjusted_final_capital - initial_capital) / initial_capital
                
                # Calculate cost impact
                cost_impact_bps = (transaction_costs['total_costs'] / initial_capital) * 10000
                return_reduction = baseline_result['total_return'] - adjusted_return
                
                results[scenario_name] = {
                    'baseline_return': baseline_result['total_return'],
                    'adjusted_return': adjusted_return,
                    'return_reduction': return_reduction,
                    'cost_impact_bps': cost_impact_bps,
                    'transaction_costs': transaction_costs,
                    'cost_description': costs['description']
                }
                
                print(f"  📈 Baseline Return: {baseline_result['total_return']:.2%}")
                print(f"  📉 Adjusted Return: {adjusted_return:.2%}")
                print(f"  💸 Return Reduction: {return_reduction:.2%}")
                print(f"  💰 Total Costs: ${transaction_costs['total_costs']:,.2f}")
                print(f"  📊 Cost Impact: {cost_impact_bps:.1f} bps")
                
                # Cost breakdown
                print(f"    - Commission: ${transaction_costs['total_commission']:,.2f}")
                print(f"    - Slippage: ${transaction_costs['total_slippage']:,.2f}")
                print(f"    - Market Impact: ${transaction_costs['total_market_impact']:,.2f}")
        
        return results
    
    def calculate_sharpe_with_costs(self, baseline_result, transaction_costs):
        """Calculate Sharpe ratio adjusted for transaction costs."""
        
        if 'equity_curve' not in baseline_result or not baseline_result['equity_curve']:
            return baseline_result.get('sharpe_ratio', 0)
        
        # Get equity curve
        equity_df = pd.DataFrame(baseline_result['equity_curve'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate daily returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Adjust for transaction costs (distribute evenly across trading days)
        total_costs = transaction_costs['total_costs']
        initial_capital = baseline_result.get('initial_capital', 100000)
        
        # Distribute costs across trading days
        cost_per_day = total_costs / len(returns)
        daily_cost_return = cost_per_day / initial_capital
        
        # Adjust returns
        adjusted_returns = returns - daily_cost_return
        
        # Calculate adjusted Sharpe
        if adjusted_returns.std() > 0:
            adjusted_sharpe = (adjusted_returns.mean() * 252) / (adjusted_returns.std() * np.sqrt(252))
        else:
            adjusted_sharpe = 0
        
        return adjusted_sharpe
    
    def analyze_trade_frequency_impact(self, results):
        """Analyze how trade frequency affects transaction cost impact."""
        
        print(f"\n" + "=" * 60)
        print("📊 TRADE FREQUENCY IMPACT ANALYSIS")
        print("=" * 60)
        
        if not results:
            print("❌ No results to analyze")
            return
        
        print("\n📈 Cost Impact by Trade Frequency:")
        print("-" * 40)
        
        for scenario_name, result in results.items():
            cost_impact = result['cost_impact_bps']
            return_reduction = result['return_reduction']
            
            print(f"\n💰 {result['cost_description']}:")
            print(f"  📊 Cost Impact: {cost_impact:.1f} bps")
            print(f"  📉 Return Reduction: {return_reduction:.2%}")
            
            # Assess impact level
            if cost_impact < 50:
                impact_level = "✅ LOW"
            elif cost_impact < 100:
                impact_level = "⚠️ MODERATE"
            else:
                impact_level = "❌ HIGH"
            
            print(f"  🎯 Impact Level: {impact_level}")
    
    def provide_recommendations(self, results):
        """Provide recommendations based on transaction cost analysis."""
        
        print(f"\n" + "=" * 60)
        print("🔧 RECOMMENDATIONS")
        print("=" * 60)
        
        if not results:
            print("❌ No results to analyze")
            return
        
        # Get realistic scenario
        realistic_result = results.get('realistic', None)
        
        if realistic_result:
            cost_impact = realistic_result['cost_impact_bps']
            return_reduction = realistic_result['return_reduction']
            
            print(f"\n📊 REALISTIC TRANSACTION COSTS:")
            print(f"  💰 Total Cost Impact: {cost_impact:.1f} bps")
            print(f"  📉 Return Reduction: {return_reduction:.2%}")
            print(f"  📈 Net Return: {realistic_result['adjusted_return']:.2%}")
            
            print(f"\n🎯 RECOMMENDATIONS:")
            
            if cost_impact < 50:
                print(f"  ✅ Transaction costs are LOW - strategy is viable")
                print(f"  ✅ Consider live trading with current parameters")
                print(f"  ✅ Focus on execution quality over cost optimization")
                
            elif cost_impact < 100:
                print(f"  ⚠️ Transaction costs are MODERATE - consider optimizations")
                print(f"  ⚠️ Look for lower-cost brokers or execution methods")
                print(f"  ⚠️ Consider reducing trade frequency")
                print(f"  ⚠️ Monitor slippage and market impact")
                
            else:
                print(f"  ❌ Transaction costs are HIGH - strategy may not be viable")
                print(f"  ❌ Significant optimization needed before live trading")
                print(f"  ❌ Consider alternative execution methods")
                print(f"  ❌ Reduce trade frequency or position sizes")
            
            # Specific recommendations
            print(f"\n🔧 SPECIFIC OPTIMIZATIONS:")
            
            if realistic_result['transaction_costs']['total_commission'] > realistic_result['transaction_costs']['total_slippage']:
                print(f"  💡 Commission costs dominate - negotiate better rates")
                print(f"  💡 Consider commission-free brokers")
                
            if realistic_result['transaction_costs']['total_slippage'] > realistic_result['transaction_costs']['total_commission']:
                print(f"  💡 Slippage costs dominate - improve execution timing")
                print(f"  💡 Use limit orders instead of market orders")
                print(f"  💡 Trade during higher liquidity periods")
                
            if realistic_result['transaction_costs']['total_market_impact'] > realistic_result['transaction_costs']['total_commission'] * 0.5:
                print(f"  💡 Market impact is significant - reduce position sizes")
                print(f"  💡 Use algorithmic execution for larger trades")
                print(f"  💡 Consider time-weighted average price (TWAP) orders")
    
    def create_cost_comparison_table(self, results):
        """Create a comparison table of different cost scenarios."""
        
        print(f"\n" + "=" * 80)
        print("📊 TRANSACTION COST COMPARISON TABLE")
        print("=" * 80)
        
        if not results:
            print("❌ No results to display")
            return
        
        # Create table header
        print(f"{'Scenario':<20} {'Baseline':<10} {'Adjusted':<10} {'Reduction':<10} {'Cost (bps)':<12} {'Impact':<10}")
        print("-" * 80)
        
        for scenario_name, result in results.items():
            baseline = result['baseline_return'] * 100
            adjusted = result['adjusted_return'] * 100
            reduction = result['return_reduction'] * 100
            cost_bps = result['cost_impact_bps']
            
            # Determine impact level
            if cost_bps < 50:
                impact = "LOW"
            elif cost_bps < 100:
                impact = "MOD"
            else:
                impact = "HIGH"
            
            print(f"{result['cost_description']:<20} {baseline:>8.2f}% {adjusted:>8.2f}% {reduction:>8.2f}% {cost_bps:>10.1f} {impact:>8}")

def run_transaction_cost_analysis():
    """Run comprehensive transaction cost analysis."""
    
    analyzer = TransactionCostAnalyzer()
    
    # Analyze cost impact
    results = analyzer.analyze_cost_impact()
    
    if results:
        # Analyze trade frequency impact
        analyzer.analyze_trade_frequency_impact(results)
        
        # Create comparison table
        analyzer.create_cost_comparison_table(results)
        
        # Provide recommendations
        analyzer.provide_recommendations(results)
        
        return results
    else:
        print("❌ Failed to complete transaction cost analysis")
        return None

if __name__ == "__main__":
    run_transaction_cost_analysis() 