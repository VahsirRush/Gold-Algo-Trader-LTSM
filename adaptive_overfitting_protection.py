#!/usr/bin/env python3
"""
ADAPTIVE OVERFITTING PROTECTION SYSTEM
=====================================

Self-correcting overfitting protection that automatically:
- Detects overfitting in real-time
- Adjusts model parameters automatically
- Reduces complexity when needed
- Increases regularization dynamically
- Adapts feature selection
- Adjusts signal thresholds
- Provides continuous self-improvement
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdaptiveOverfittingProtection:
    """Self-correcting overfitting protection system."""
    
    def __init__(self, 
                 initial_regularization: float = 0.05,
                 max_features: int = 50,
                 min_stability_threshold: float = 0.6,
                 adaptation_frequency: int = 21,  # Adapt every 3 weeks
                 max_adaptation_iterations: int = 10):
        
        self.initial_regularization = initial_regularization
        self.max_features = max_features
        self.min_stability_threshold = min_stability_threshold
        self.adaptation_frequency = adaptation_frequency
        self.max_adaptation_iterations = max_adaptation_iterations
        
        # Current adaptive parameters
        self.current_regularization = initial_regularization
        self.current_max_features = max_features
        self.current_threshold_multiplier = 1.0
        self.current_position_size_multiplier = 1.0
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_history = []
        self.overfitting_alerts = []
        self.stability_metrics = {}
        
        # Adaptation counters
        self.adaptation_count = 0
        self.last_adaptation = None
        self.consecutive_high_risk_periods = 0
        
        # Adaptive thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
    
    def adaptive_validation(self, data: pd.DataFrame, strategy) -> Dict:
        """Run adaptive validation with self-correction."""
        print("🔄 ADAPTIVE OVERFITTING VALIDATION")
        print("=" * 60)
        
        # Run comprehensive validation
        validation_results = self._run_comprehensive_validation(data, strategy)
        
        # Check if adaptation is needed
        risk_level = validation_results['overall_assessment']['risk_level']
        risk_score = validation_results['overall_assessment']['risk_score']
        
        print(f"📊 Current Risk Level: {risk_level} (Score: {risk_score:.3f})")
        print(f"🔧 Current Parameters:")
        print(f"   Regularization: {self.current_regularization:.3f}")
        print(f"   Max Features: {self.current_max_features}")
        print(f"   Threshold Multiplier: {self.current_threshold_multiplier:.2f}")
        print(f"   Position Size Multiplier: {self.current_position_size_multiplier:.2f}")
        
        # Determine if adaptation is needed
        if self._should_adapt(risk_level, risk_score):
            print(f"⚠️  ADAPTATION NEEDED - Risk Level: {risk_level}")
            adaptation_result = self._perform_adaptation(validation_results, strategy)
            validation_results['adaptation_result'] = adaptation_result
            
            # Re-validate after adaptation
            if adaptation_result['success']:
                print(f"✅ ADAPTATION SUCCESSFUL - Re-validating...")
                validation_results['post_adaptation'] = self._run_comprehensive_validation(data, strategy)
        
        return validation_results
    
    def _should_adapt(self, risk_level: str, risk_score: float) -> bool:
        """Determine if adaptation is needed."""
        # Check adaptation frequency
        if self.last_adaptation:
            days_since_adaptation = (datetime.now() - self.last_adaptation).days
            if days_since_adaptation < self.adaptation_frequency:
                return False
        
        # Check risk level
        if risk_level == 'CRITICAL':
            return True
        elif risk_level == 'HIGH' and risk_score > self.risk_thresholds['high']:
            return True
        elif risk_level == 'MEDIUM' and self.consecutive_high_risk_periods >= 2:
            return True
        
        # Check adaptation limits
        if self.adaptation_count >= self.max_adaptation_iterations:
            print(f"⚠️  Maximum adaptation iterations reached ({self.max_adaptation_iterations})")
            return False
        
        return False
    
    def _perform_adaptation(self, validation_results: Dict, strategy) -> Dict:
        """Perform automatic adaptation to reduce overfitting."""
        print("🔧 PERFORMING AUTOMATIC ADAPTATION")
        print("-" * 40)
        
        risk_level = validation_results['overall_assessment']['risk_level']
        risk_score = validation_results['overall_assessment']['risk_score']
        
        adaptation_actions = []
        
        # 1. Increase regularization
        if risk_score > 0.7:
            old_reg = self.current_regularization
            self.current_regularization = min(0.3, self.current_regularization * 1.5)
            adaptation_actions.append(f"Increased regularization: {old_reg:.3f} → {self.current_regularization:.3f}")
        
        # 2. Reduce feature complexity
        if risk_score > 0.6:
            old_features = self.current_max_features
            self.current_max_features = max(20, int(self.current_max_features * 0.8))
            adaptation_actions.append(f"Reduced max features: {old_features} → {self.current_max_features}")
        
        # 3. Adjust signal thresholds
        if risk_score > 0.5:
            old_threshold = self.current_threshold_multiplier
            self.current_threshold_multiplier = min(2.0, self.current_threshold_multiplier * 1.2)
            adaptation_actions.append(f"Increased threshold multiplier: {old_threshold:.2f} → {self.current_threshold_multiplier:.2f}")
        
        # 4. Reduce position sizes
        if risk_score > 0.8:
            old_position = self.current_position_size_multiplier
            self.current_position_size_multiplier = max(0.3, self.current_position_size_multiplier * 0.8)
            adaptation_actions.append(f"Reduced position size: {old_position:.2f} → {self.current_position_size_multiplier:.2f}")
        
        # 5. Apply adaptations to strategy
        success = self._apply_adaptations_to_strategy(strategy)
        
        # Record adaptation
        self.adaptation_count += 1
        self.last_adaptation = datetime.now()
        
        if risk_level in ['HIGH', 'CRITICAL']:
            self.consecutive_high_risk_periods += 1
        else:
            self.consecutive_high_risk_periods = 0
        
        adaptation_result = {
            'success': success,
            'actions_taken': adaptation_actions,
            'new_parameters': {
                'regularization': self.current_regularization,
                'max_features': self.current_max_features,
                'threshold_multiplier': self.current_threshold_multiplier,
                'position_size_multiplier': self.current_position_size_multiplier
            },
            'adaptation_count': self.adaptation_count,
            'risk_level': risk_level,
            'risk_score': risk_score
        }
        
        # Print adaptation summary
        print(f"✅ ADAPTATION COMPLETED:")
        for action in adaptation_actions:
            print(f"   • {action}")
        print(f"   Adaptation Count: {self.adaptation_count}/{self.max_adaptation_iterations}")
        
        return adaptation_result
    
    def _apply_adaptations_to_strategy(self, strategy) -> bool:
        """Apply adaptations to the strategy."""
        try:
            # Update strategy parameters
            if hasattr(strategy, 'ml_ensemble'):
                # Update regularization in models
                for name, model in strategy.ml_ensemble.models.items():
                    if hasattr(model, 'alpha'):
                        model.alpha = self.current_regularization
                    elif hasattr(model, 'C'):
                        model.C = 1.0 / self.current_regularization
            
            # Update feature engineer
            if hasattr(strategy, 'feature_engineer'):
                strategy.feature_engineer.max_features = self.current_max_features
            
            # Update signal generation parameters
            if hasattr(strategy, 'current_threshold_multiplier'):
                strategy.current_threshold_multiplier = self.current_threshold_multiplier
            
            if hasattr(strategy, 'current_position_size_multiplier'):
                strategy.current_position_size_multiplier = self.current_position_size_multiplier
            
            return True
        except Exception as e:
            print(f"❌ Error applying adaptations: {e}")
            return False
    
    def _run_comprehensive_validation(self, data: pd.DataFrame, strategy) -> Dict:
        """Run comprehensive validation (simplified version)."""
        # This would integrate with the full ComprehensiveOverfittingProtection
        # For now, we'll create a simplified validation
        
        # Split data for validation
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Train and test
        strategy.train_on_data(train_data)
        test_signals = strategy.generate_signals(test_data)
        test_returns = test_data['Close'].pct_change()
        strategy_returns = test_signals.shift(1) * test_returns
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(strategy_returns)
        
        # Simplified risk assessment
        risk_score = self._calculate_risk_score(metrics, strategy)
        risk_level = self._determine_risk_level(risk_score)
        
        return {
            'overall_assessment': {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'metrics': metrics
            }
        }
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics."""
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {
                'sharpe_ratio': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'volatility': 0
            }
        
        total_return = (1 + returns).prod() - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': volatility
        }
    
    def _calculate_risk_score(self, metrics: Dict, strategy) -> float:
        """Calculate overfitting risk score."""
        risk_factors = []
        
        # Sharpe ratio risk
        if metrics['sharpe_ratio'] < 0:
            risk_factors.append(0.3)
        elif metrics['sharpe_ratio'] > 3.0:
            risk_factors.append(0.2)  # Potentially overfitting
        
        # Drawdown risk
        if abs(metrics['max_drawdown']) > 0.1:
            risk_factors.append(0.2)
        
        # Win rate risk
        if metrics['win_rate'] > 0.9:
            risk_factors.append(0.3)  # Suspiciously high win rate
        elif metrics['win_rate'] < 0.3:
            risk_factors.append(0.2)
        
        # Volatility risk
        if metrics['volatility'] > 0.5:
            risk_factors.append(0.2)
        
        # Model complexity risk
        feature_count = None
        if hasattr(strategy, 'feature_engineer') and strategy.feature_engineer is not None:
            feature_count = getattr(strategy.feature_engineer, 'max_features', None)
            if feature_count is not None and feature_count > 60:
                risk_factors.append(0.2)
        
        # Regularization risk
        if hasattr(strategy, 'ml_ensemble') and strategy.ml_ensemble is not None:
            reg_strength = getattr(strategy.ml_ensemble, 'regularization_strength', None)
            if reg_strength is not None and reg_strength < 0.01:
                risk_factors.append(0.3)
        
        return min(1.0, sum(risk_factors))
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score."""
        if risk_score >= self.risk_thresholds['critical']:
            return 'CRITICAL'
        elif risk_score >= self.risk_thresholds['high']:
            return 'HIGH'
        elif risk_score >= self.risk_thresholds['medium']:
            return 'MEDIUM'
        elif risk_score >= self.risk_thresholds['low']:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def get_adaptive_parameters(self) -> Dict:
        """Get current adaptive parameters."""
        return {
            'regularization': self.current_regularization,
            'max_features': self.current_max_features,
            'threshold_multiplier': self.current_threshold_multiplier,
            'position_size_multiplier': self.current_position_size_multiplier,
            'adaptation_count': self.adaptation_count,
            'consecutive_high_risk_periods': self.consecutive_high_risk_periods
        }
    
    def reset_adaptations(self):
        """Reset adaptations to initial values."""
        self.current_regularization = self.initial_regularization
        self.current_max_features = self.max_features
        self.current_threshold_multiplier = 1.0
        self.current_position_size_multiplier = 1.0
        self.adaptation_count = 0
        self.consecutive_high_risk_periods = 0
        self.last_adaptation = None
        print("🔄 Adaptations reset to initial values")

class AdaptiveGoldStrategy:
    """Gold strategy with adaptive overfitting protection."""
    
    def __init__(self, 
                 target_volatility: float = 0.20,
                 max_position_size: float = 1.0,
                 use_adaptive_thresholds: bool = True,
                 use_regime_filtering: bool = True,
                 regularization_strength: float = 0.05,
                 max_features: int = 50):
        
        self.target_volatility = target_volatility
        self.max_position_size = max_position_size
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.use_regime_filtering = use_regime_filtering
        self.regularization_strength = regularization_strength
        
        # Adaptive parameters
        self.current_threshold_multiplier = 1.0
        self.current_position_size_multiplier = 1.0
        
        # Initialize components (simplified for demonstration)
        self.feature_engineer = None  # Would be initialized with proper feature engineer
        self.scaler = None  # Would be initialized with proper scaler
        self.ml_ensemble = None  # Would be initialized with proper ML ensemble
        
        # Adaptive overfitting protection
        self.adaptive_protection = AdaptiveOverfittingProtection(
            initial_regularization=regularization_strength,
            max_features=max_features
        )
        
        # Performance tracking
        self.training_history = []
        self.validation_results = {}
    
    def train_on_data(self, data: pd.DataFrame):
        """Train the strategy with adaptive protection."""
        # This would contain the actual training logic
        # For demonstration, we'll just track the training
        self.training_history.append({
            'timestamp': datetime.now(),
            'data_length': len(data),
            'adaptive_params': self.adaptive_protection.get_adaptive_parameters()
        })
        print(f"✅ Strategy trained on {len(data)} samples")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals with adaptive thresholds."""
        # This would contain the actual signal generation logic
        # For demonstration, we'll create dummy signals
        signals = pd.Series(0, index=data.index)
        
        # Apply adaptive thresholds
        base_threshold = 0.01 * self.current_threshold_multiplier
        
        # Generate some dummy signals for demonstration
        for i in range(len(signals)):
            if i % 50 == 0:  # Generate signal every 50 days
                signals.iloc[i] = 0.5 if i % 100 == 0 else -0.5
        
        # Apply adaptive position sizing
        signals = signals * self.current_position_size_multiplier
        
        return signals
    
    def run_adaptive_validation(self, data: pd.DataFrame) -> Dict:
        """Run validation with adaptive overfitting protection."""
        print("🚀 ADAPTIVE GOLD STRATEGY VALIDATION")
        print("=" * 60)
        
        # Run adaptive validation
        validation_results = self.adaptive_protection.adaptive_validation(data, self)
        
        # Generate signals for performance calculation
        signals = self.generate_signals(data)
        returns = data['Close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # Calculate final metrics
        metrics = self.adaptive_protection._calculate_performance_metrics(strategy_returns)
        
        # Store results
        self.validation_results = validation_results
        
        return {
            'validation_results': validation_results,
            'final_metrics': metrics,
            'adaptive_parameters': self.adaptive_protection.get_adaptive_parameters()
        }

def test_adaptive_system():
    """Test the adaptive overfitting protection system with Databento OHLCV data."""
    print("[DEBUG] Inside test_adaptive_system()...")
    print("🧪 TESTING ADAPTIVE OVERFITTING PROTECTION SYSTEM (DATABENTO OHLCV DATA)")
    print("=" * 70)
    
    # Import and use the DatabentoGoldCollector
    try:
        from data_pipeline.databento_collector import DatabentoGoldCollector
        print("[INFO] Using DatabentoGoldCollector for real GOLD OHLCV data...")
        collector = DatabentoGoldCollector()
        
        # Fetch aggregated OHLCV data for August 2023
        print("[INFO] Fetching aggregated OHLCV data for August 2023...")
        ohlcv_data = collector.fetch_and_aggregate_gold_mbo_to_ohlcv(
            start_date="2023-08-01", 
            end_date="2023-08-31"
        )
        
        if ohlcv_data.empty:
            print("[ERROR] No OHLCV data fetched from Databento.")
            return
        
        print(f"[INFO] Successfully fetched {len(ohlcv_data)} days of OHLCV data.")
        print(f"[INFO] Data range: {ohlcv_data.index.min()} to {ohlcv_data.index.max()}")
        print(f"[INFO] Sample data:")
        print(ohlcv_data.head())
        
        # Convert to the format expected by the strategy
        data = ohlcv_data.copy()
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Standardize column names
        
    except Exception as e:
        print(f"[ERROR] Failed to import or use DatabentoGoldCollector: {e}")
        return
    
    # Create adaptive strategy
    strategy = AdaptiveGoldStrategy(
        target_volatility=0.20,
        max_position_size=1.0,
        use_adaptive_thresholds=True,
        use_regime_filtering=True,
        regularization_strength=0.05,
        max_features=50
    )
    
    # Train strategy
    print("[INFO] Training strategy on Databento OHLCV data...")
    strategy.train_on_data(data)
    print("[INFO] Strategy trained.")
    
    # Run adaptive validation
    print("[INFO] Running adaptive validation...")
    results = strategy.run_adaptive_validation(data)
    print("[INFO] Adaptive validation complete.")
    
    # Run comprehensive backtest and risk analysis
    print("[INFO] Running comprehensive backtest and risk analysis...")
    backtest_results = run_comprehensive_backtest(data, strategy)
    print("[INFO] Backtest and risk analysis complete.")
    
    # Prepare results string
    output_lines = []
    output_lines.append("\n📊 ADAPTIVE VALIDATION RESULTS:")
    output_lines.append(f"   Sharpe Ratio: {results['final_metrics']['sharpe_ratio']:.3f}")
    output_lines.append(f"   Total Return: {results['final_metrics']['total_return']:.2%}")
    output_lines.append(f"   Max Drawdown: {results['final_metrics']['max_drawdown']:.2%}")
    output_lines.append(f"   Win Rate: {results['final_metrics']['win_rate']:.2%}")
    
    output_lines.append("\n🔧 ADAPTIVE PARAMETERS:")
    adaptive_params = results['adaptive_parameters']
    output_lines.append(f"   Regularization: {adaptive_params['regularization']:.3f}")
    output_lines.append(f"   Max Features: {adaptive_params['max_features']}")
    output_lines.append(f"   Threshold Multiplier: {adaptive_params['threshold_multiplier']:.2f}")
    output_lines.append(f"   Position Size Multiplier: {adaptive_params['position_size_multiplier']:.2f}")
    output_lines.append(f"   Adaptation Count: {adaptive_params['adaptation_count']}")
    
    # Add backtest results
    output_lines.append("\n📈 COMPREHENSIVE BACKTEST RESULTS:")
    output_lines.append(f"   Total Return: {backtest_results['total_return']:.2%}")
    output_lines.append(f"   Annualized Return: {backtest_results['annualized_return']:.2%}")
    output_lines.append(f"   Volatility: {backtest_results['volatility']:.2%}")
    output_lines.append(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
    output_lines.append(f"   Sortino Ratio: {backtest_results['sortino_ratio']:.3f}")
    output_lines.append(f"   Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    output_lines.append(f"   Calmar Ratio: {backtest_results['calmar_ratio']:.3f}")
    output_lines.append(f"   Win Rate: {backtest_results['win_rate']:.2%}")
    output_lines.append(f"   Profit Factor: {backtest_results['profit_factor']:.3f}")
    output_lines.append(f"   Average Win: {backtest_results['avg_win']:.2%}")
    output_lines.append(f"   Average Loss: {backtest_results['avg_loss']:.2%}")
    output_lines.append(f"   Best Day: {backtest_results['best_day']:.2%}")
    output_lines.append(f"   Worst Day: {backtest_results['worst_day']:.2%}")
    
    # Risk metrics
    output_lines.append("\n⚠️  RISK ANALYSIS:")
    output_lines.append(f"   VaR (95%): {backtest_results['var_95']:.2%}")
    output_lines.append(f"   CVaR (95%): {backtest_results['cvar_95']:.2%}")
    output_lines.append(f"   Downside Deviation: {backtest_results['downside_deviation']:.2%}")
    output_lines.append(f"   Skewness: {backtest_results['skewness']:.3f}")
    output_lines.append(f"   Kurtosis: {backtest_results['kurtosis']:.3f}")
    output_lines.append(f"   Beta: {backtest_results['beta']:.3f}")
    output_lines.append(f"   Alpha: {backtest_results['alpha']:.2%}")
    
    # Trading statistics
    output_lines.append("\n📊 TRADING STATISTICS:")
    output_lines.append(f"   Total Trades: {backtest_results['total_trades']}")
    output_lines.append(f"   Winning Trades: {backtest_results['winning_trades']}")
    output_lines.append(f"   Losing Trades: {backtest_results['losing_trades']}")
    output_lines.append(f"   Average Trade Duration: {backtest_results['avg_trade_duration']:.1f} days")
    output_lines.append(f"   Largest Win: {backtest_results['largest_win']:.2%}")
    output_lines.append(f"   Largest Loss: {backtest_results['largest_loss']:.2%}")
    
    if 'adaptation_result' in results['validation_results']:
        adaptation = results['validation_results']['adaptation_result']
        output_lines.append("\n🔄 ADAPTATION PERFORMED:")
        output_lines.append(f"   Success: {adaptation['success']}")
        output_lines.append(f"   Risk Level: {adaptation['risk_level']}")
        output_lines.append(f"   Actions Taken: {len(adaptation['actions_taken'])}")
        for action in adaptation['actions_taken']:
            output_lines.append(f"     • {action}")
    
    # Print results to terminal
    print("[INFO] Printing results to terminal...")
    print("\n".join(output_lines))
    
    # Save results to file with error handling
    try:
        with open("adaptive_results_databento.txt", "w") as f:
            f.write("\n".join(output_lines))
        print("[INFO] Results saved to adaptive_results_databento.txt")
        
        # Also save detailed backtest data
        backtest_results['equity_curve'].to_csv("equity_curve_databento.csv")
        print("[INFO] Equity curve saved to equity_curve_databento.csv")
        
    except Exception as file_err:
        print(f"[ERROR] Could not write results to file: {file_err}")
    
    print("[TEST] test_adaptive_system() complete.")
    return results, backtest_results


def run_comprehensive_backtest(data: pd.DataFrame, strategy) -> Dict:
    """Run comprehensive backtest with detailed risk analysis."""
    print("🔄 Running comprehensive backtest...")
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Calculate returns
    price_returns = data['Close'].pct_change().dropna()
    strategy_returns = signals.shift(1) * price_returns
    strategy_returns = strategy_returns.dropna()
    
    # Calculate equity curve
    equity_curve = (1 + strategy_returns).cumprod()
    
    # Basic performance metrics
    total_return = equity_curve.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Sortino ratio
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate and profit factor
    winning_trades = strategy_returns[strategy_returns > 0]
    losing_trades = strategy_returns[strategy_returns < 0]
    win_rate = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
    profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 and losing_trades.sum() != 0 else 0
    
    # Average win/loss
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    
    # Best/worst days
    best_day = strategy_returns.max()
    worst_day = strategy_returns.min()
    
    # Risk metrics
    var_95 = np.percentile(strategy_returns, 5)
    cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
    
    # Skewness and kurtosis
    skewness = strategy_returns.skew()
    kurtosis = strategy_returns.kurtosis()
    
    # Beta and Alpha (assuming risk-free rate of 2%)
    risk_free_rate = 0.02
    market_returns = price_returns  # Using price returns as market proxy
    market_returns = market_returns[strategy_returns.index]  # Align indices
    
    if len(market_returns) > 0:
        covariance = np.cov(strategy_returns, market_returns)[0, 1]
        market_variance = market_returns.var()
        beta = covariance / market_variance if market_variance > 0 else 0
        alpha = annualized_return - (risk_free_rate + beta * (market_returns.mean() * 252 - risk_free_rate))
    else:
        beta = 0
        alpha = 0
    
    # Trading statistics
    total_trades = len(strategy_returns[strategy_returns != 0])
    winning_trades_count = len(winning_trades)
    losing_trades_count = len(losing_trades)
    
    # Trade duration (simplified - count consecutive non-zero returns)
    trade_durations = []
    current_duration = 0
    for ret in strategy_returns:
        if ret != 0:
            current_duration += 1
        elif current_duration > 0:
            trade_durations.append(current_duration)
            current_duration = 0
    
    avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
    
    # Largest win/loss
    largest_win = winning_trades.max() if len(winning_trades) > 0 else 0
    largest_loss = losing_trades.min() if len(losing_trades) > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'best_day': best_day,
        'worst_day': worst_day,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'downside_deviation': downside_deviation,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'beta': beta,
        'alpha': alpha,
        'total_trades': total_trades,
        'winning_trades': winning_trades_count,
        'losing_trades': losing_trades_count,
        'avg_trade_duration': avg_trade_duration,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'equity_curve': equity_curve,
        'strategy_returns': strategy_returns
    }

if __name__ == "__main__":
    import os
    import sys
    print(f"[DEBUG] Script started: {__file__}")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Python version: {sys.version}")
    try:
        print("[DEBUG] Entering test_adaptive_system() call...")
        results, backtest_results = test_adaptive_system()
        print("[TEST FUNCTION COMPLETED]")
    except Exception as e:
        print(f"[ERROR] Exception during test_adaptive_system: {e}")
