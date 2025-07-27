"""
Test Risk-Enhanced Strategy
===========================

This script tests the risk-enhanced strategy with volatility targeting and drawdown protection,
comparing it against the baseline ultra-aggressive strategy.
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
from gold_algo.strategies.ultra_aggressive_strategy import UltraAggressiveStrategy
from data_pipeline.databento_collector import DatabentoGoldCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_baseline_strategy(data: pd.DataFrame) -> dict:
    """
    Run baseline ultra-aggressive strategy for comparison.
    
    Args:
        data: OHLCV price data
        initial_capital: Initial capital
        
    Returns:
        Baseline strategy results
    """
    try:
        logger.info("Running baseline ultra-aggressive strategy...")
        
        strategy = UltraAggressiveStrategy()
        results = strategy.run_backtest(data)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running baseline strategy: {e}")
        return {}


def compare_strategies(risk_enhanced_results: dict, baseline_results: dict) -> dict:
    """
    Compare risk-enhanced strategy with baseline strategy.
    
    Args:
        risk_enhanced_results: Results from risk-enhanced strategy
        baseline_results: Results from baseline strategy
        
    Returns:
        Comparison metrics
    """
    try:
        comparison = {
            'metric': [],
            'baseline': [],
            'risk_enhanced': [],
            'improvement': [],
            'improvement_pct': []
        }
        
        # Define metrics to compare
        metrics = [
            ('Total Return', 'total_return'),
            ('Max Drawdown', 'max_drawdown'),
            ('Sharpe Ratio', 'sharpe_ratio'),
            ('Win Rate', 'win_rate'),
            ('Profit Factor', 'profit_factor'),
            ('Total Trades', 'total_trades')
        ]
        
        for metric_name, metric_key in metrics:
            baseline_value = baseline_results.get(metric_key, 0)
            risk_enhanced_value = risk_enhanced_results.get(metric_key, 0)
            
            if metric_key in ['max_drawdown']:
                # For drawdown, lower is better
                improvement = baseline_value - risk_enhanced_value
                improvement_pct = (improvement / abs(baseline_value)) * 100 if baseline_value != 0 else 0
            else:
                # For other metrics, higher is better
                improvement = risk_enhanced_value - baseline_value
                improvement_pct = (improvement / abs(baseline_value)) * 100 if baseline_value != 0 else 0
            
            comparison['metric'].append(metric_name)
            comparison['baseline'].append(baseline_value)
            comparison['risk_enhanced'].append(risk_enhanced_value)
            comparison['improvement'].append(improvement)
            comparison['improvement_pct'].append(improvement_pct)
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        return {}


def generate_comprehensive_report(risk_enhanced_results: dict, 
                                baseline_results: dict, 
                                comparison: dict) -> str:
    """
    Generate comprehensive comparison report.
    
    Args:
        risk_enhanced_results: Risk-enhanced strategy results
        baseline_results: Baseline strategy results
        comparison: Strategy comparison metrics
        
    Returns:
        Formatted report string
    """
    try:
        report = []
        report.append("=" * 80)
        report.append("RISK-ENHANCED STRATEGY COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Risk-Enhanced Strategy Results
        report.append("RISK-ENHANCED STRATEGY RESULTS")
        report.append("-" * 40)
        report.append(f"Total Return: {risk_enhanced_results.get('total_return', 0):.2%}")
        report.append(f"Max Drawdown: {risk_enhanced_results.get('max_drawdown', 0):.2%}")
        report.append(f"Sharpe Ratio: {risk_enhanced_results.get('sharpe_ratio', 0):.3f}")
        report.append(f"Win Rate: {risk_enhanced_results.get('win_rate', 0):.2%}")
        report.append(f"Profit Factor: {risk_enhanced_results.get('profit_factor', 0):.3f}")
        report.append(f"Total Trades: {risk_enhanced_results.get('total_trades', 0)}")
        report.append(f"Initial Capital: ${risk_enhanced_results.get('initial_capital', 0):,.2f}")
        report.append(f"Final Capital: ${risk_enhanced_results.get('final_capital', 0):,.2f}")
        report.append("")
        
        # Risk Metrics
        risk_metrics = risk_enhanced_results.get('risk_metrics', {})
        report.append("RISK MANAGEMENT METRICS")
        report.append("-" * 40)
        report.append(f"Current Drawdown: {risk_metrics.get('current_drawdown', 0):.2%}")
        report.append(f"High Water Mark: {risk_metrics.get('high_water_mark', 1.0):.3f}")
        report.append(f"Position Scale: {risk_metrics.get('position_scale', 1.0):.3f}")
        report.append(f"Soft Stop Triggered: {risk_metrics.get('soft_stop_triggered', False)}")
        report.append(f"Hard Stop Triggered: {risk_metrics.get('hard_stop_triggered', False)}")
        report.append(f"Recovery Mode: {risk_metrics.get('recovery_mode', False)}")
        report.append(f"Average ATR: {risk_metrics.get('avg_atr', 0):.4f}")
        report.append(f"Average Volatility: {risk_metrics.get('avg_volatility', 0):.2%}")
        report.append("")
        
        # Baseline Strategy Results
        report.append("BASELINE STRATEGY RESULTS")
        report.append("-" * 40)
        report.append(f"Total Return: {baseline_results.get('total_return', 0):.2%}")
        report.append(f"Max Drawdown: {baseline_results.get('max_drawdown', 0):.2%}")
        report.append(f"Sharpe Ratio: {baseline_results.get('sharpe_ratio', 0):.3f}")
        report.append(f"Win Rate: {baseline_results.get('win_rate', 0):.2%}")
        report.append(f"Profit Factor: {baseline_results.get('profit_factor', 0):.3f}")
        report.append(f"Total Trades: {baseline_results.get('total_trades', 0)}")
        report.append(f"Initial Capital: ${baseline_results.get('initial_capital', 0):,.2f}")
        report.append(f"Final Capital: ${baseline_results.get('final_capital', 0):,.2f}")
        report.append("")
        
        # Strategy Comparison
        report.append("STRATEGY COMPARISON")
        report.append("-" * 40)
        report.append(f"{'Metric':<20} {'Baseline':<12} {'Risk-Enhanced':<15} {'Improvement':<12} {'% Change':<10}")
        report.append("-" * 80)
        
        for i in range(len(comparison.get('metric', []))):
            metric = comparison['metric'][i]
            baseline = comparison['baseline'][i]
            risk_enhanced = comparison['risk_enhanced'][i]
            improvement = comparison['improvement'][i]
            improvement_pct = comparison['improvement_pct'][i]
            
            # Format values based on metric type
            if 'Return' in metric or 'Rate' in metric:
                baseline_str = f"{baseline:.2%}"
                risk_enhanced_str = f"{risk_enhanced:.2%}"
                improvement_str = f"{improvement:.2%}"
            elif 'Ratio' in metric or 'Factor' in metric:
                baseline_str = f"{baseline:.3f}"
                risk_enhanced_str = f"{risk_enhanced:.3f}"
                improvement_str = f"{improvement:.3f}"
            elif 'Drawdown' in metric:
                baseline_str = f"{baseline:.2%}"
                risk_enhanced_str = f"{risk_enhanced:.2%}"
                improvement_str = f"{improvement:.2%}"
            else:
                baseline_str = f"{baseline:.0f}"
                risk_enhanced_str = f"{risk_enhanced:.0f}"
                improvement_str = f"{improvement:.0f}"
            
            improvement_pct_str = f"{improvement_pct:+.1f}%"
            
            report.append(f"{metric:<20} {baseline_str:<12} {risk_enhanced_str:<15} {improvement_str:<12} {improvement_pct_str:<10}")
        
        report.append("")
        
        # Key Findings
        report.append("KEY FINDINGS")
        report.append("-" * 40)
        
        # Check if objectives were met
        max_drawdown_enhanced = risk_enhanced_results.get('max_drawdown', 0)
        sharpe_enhanced = risk_enhanced_results.get('sharpe_ratio', 0)
        baseline_sharpe = baseline_results.get('sharpe_ratio', 0)
        
        if max_drawdown_enhanced > -0.15:
            report.append("✅ Max Drawdown Objective MET: Below 15% threshold")
        else:
            report.append("❌ Max Drawdown Objective NOT MET: Above 15% threshold")
        
        if sharpe_enhanced > baseline_sharpe:
            report.append("✅ Sharpe Ratio Objective MET: Improved over baseline")
        else:
            report.append("❌ Sharpe Ratio Objective NOT MET: Below baseline")
        
        if sharpe_enhanced > 0.776:
            report.append("✅ Target Sharpe Ratio MET: Above 0.776 threshold")
        else:
            report.append("❌ Target Sharpe Ratio NOT MET: Below 0.776 threshold")
        
        report.append("")
        
        # Risk Management Effectiveness
        report.append("RISK MANAGEMENT EFFECTIVENESS")
        report.append("-" * 40)
        
        soft_stop_triggered = risk_metrics.get('soft_stop_triggered', False)
        hard_stop_triggered = risk_metrics.get('hard_stop_triggered', False)
        
        if soft_stop_triggered:
            report.append("🟡 Soft Stop (-10%) was triggered during backtest")
        else:
            report.append("🟢 Soft Stop (-10%) was not triggered")
        
        if hard_stop_triggered:
            report.append("🔴 HARD STOP (-15%) was triggered - Trading halted")
        else:
            report.append("🟢 Hard Stop (-15%) was not triggered")
        
        position_scale = risk_metrics.get('position_scale', 1.0)
        if position_scale < 1.0:
            report.append(f"🟡 Position scaling active: {position_scale:.1%} of normal size")
        else:
            report.append("🟢 Position scaling: Full size (100%)")
        
        report.append("")
        
        # Conclusion
        report.append("CONCLUSION")
        report.append("-" * 40)
        
        if (max_drawdown_enhanced > -0.15 and 
            sharpe_enhanced > baseline_sharpe and 
            not hard_stop_triggered):
            report.append("🎯 SUCCESS: Risk-enhanced strategy meets all objectives!")
            report.append("   - Max drawdown below 15%")
            report.append("   - Sharpe ratio improved over baseline")
            report.append("   - No hard stop triggered")
            report.append("   - Ready for production deployment")
        else:
            report.append("⚠️  PARTIAL SUCCESS: Some objectives not met")
            report.append("   - Further optimization may be needed")
            report.append("   - Consider adjusting risk parameters")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return f"Error generating report: {e}"


def save_results(risk_enhanced_results: dict, 
                baseline_results: dict, 
                comparison: dict, 
                report: str):
    """
    Save test results to files.
    
    Args:
        risk_enhanced_results: Risk-enhanced strategy results
        baseline_results: Baseline strategy results
        comparison: Strategy comparison metrics
        report: Comprehensive report
    """
    try:
        # Save report
        with open('risk_enhanced_strategy_report.txt', 'w') as f:
            f.write(report)
        
        # Save detailed results
        results_df = pd.DataFrame(comparison)
        results_df.to_csv('strategy_comparison_results.csv', index=False)
        
        # Save equity curves
        if 'equity_curve' in risk_enhanced_results:
            equity_df = pd.DataFrame(risk_enhanced_results['equity_curve'])
            equity_df.to_csv('risk_enhanced_equity_curve.csv', index=False)
        
        if 'equity_curve' in baseline_results:
            baseline_equity_df = pd.DataFrame(baseline_results['equity_curve'])
            baseline_equity_df.to_csv('baseline_equity_curve.csv', index=False)
        
        # Save trades
        if 'trades' in risk_enhanced_results:
            trades_df = pd.DataFrame(risk_enhanced_results['trades'])
            trades_df.to_csv('risk_enhanced_trades.csv', index=False)
        
        logger.info("Results saved to files:")
        logger.info("- risk_enhanced_strategy_report.txt")
        logger.info("- strategy_comparison_results.csv")
        logger.info("- risk_enhanced_equity_curve.csv")
        logger.info("- baseline_equity_curve.csv")
        logger.info("- risk_enhanced_trades.csv")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def main():
    """Main test function."""
    try:
        logger.info("Starting comprehensive risk-enhanced strategy test...")
        
        # Test parameters
        start_date = "2023-07-01"
        end_date = "2023-09-30"
        initial_capital = 100000.0
        
        logger.info(f"Test Period: {start_date} to {end_date}")
        logger.info(f"Initial Capital: ${initial_capital:,.2f}")
        
        # Fetch data
        logger.info("Fetching market data...")
        collector = DatabentoGoldCollector()
        data = collector.fetch_gld(start_date, end_date)
        
        if data is None or len(data) == 0:
            logger.error("No data fetched. Exiting.")
            return
        
        logger.info(f"Fetched {len(data)} data points")
        
        # Run risk-enhanced strategy
        logger.info("Running risk-enhanced strategy...")
        risk_enhanced_results = run_risk_enhanced_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        if not risk_enhanced_results:
            logger.error("Risk-enhanced strategy failed. Exiting.")
            return
        
        # Run baseline strategy
        logger.info("Running baseline strategy...")
        baseline_results = run_baseline_strategy(data)
        
        if not baseline_results:
            logger.error("Baseline strategy failed. Exiting.")
            return
        
        # Compare strategies
        logger.info("Comparing strategies...")
        comparison = compare_strategies(risk_enhanced_results, baseline_results)
        
        # Generate report
        logger.info("Generating comprehensive report...")
        report = generate_comprehensive_report(
            risk_enhanced_results, 
            baseline_results, 
            comparison
        )
        
        # Print report
        print("\n" + report)
        
        # Save results
        logger.info("Saving results...")
        save_results(risk_enhanced_results, baseline_results, comparison, report)
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main test: {e}")


if __name__ == "__main__":
    main() 