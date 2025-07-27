#!/usr/bin/env python3
"""
Overfitting Protection System
============================

This module provides comprehensive protection against overfitting through:
1. Cross-validation testing
2. Parameter sensitivity analysis
3. Walk-forward analysis
4. Performance degradation monitoring
5. Realistic performance bounds
6. Out-of-sample validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OverfittingProtection:
    """
    Comprehensive overfitting protection system for trading strategies.
    """
    
    def __init__(self, 
                 min_performance_threshold: float = 0.5,
                 max_sharpe_threshold: float = 3.0,
                 max_drawdown_threshold: float = 0.25,
                 cv_folds: int = 5,
                 walk_forward_windows: int = 4):
        """
        Initialize overfitting protection system.
        
        Args:
            min_performance_threshold: Minimum acceptable performance degradation
            max_sharpe_threshold: Maximum realistic Sharpe ratio
            max_drawdown_threshold: Maximum acceptable drawdown
            cv_folds: Number of cross-validation folds
            walk_forward_windows: Number of walk-forward windows
        """
        self.min_performance_threshold = min_performance_threshold
        self.max_sharpe_threshold = max_sharpe_threshold
        self.max_drawdown_threshold = max_drawdown_threshold
        self.cv_folds = cv_folds
        self.walk_forward_windows = walk_forward_windows
        
        # Overfitting indicators
        self.overfitting_score = 0.0
        self.overfitting_flags = []
        self.protection_recommendations = []
    
    def cross_validation_test(self, strategy_func, data: pd.DataFrame, 
                            strategy_params: Dict) -> Dict:
        """
        Perform cross-validation testing to detect overfitting.
        
        Args:
            strategy_func: Function that runs the strategy
            data: Market data
            strategy_params: Strategy parameters
            
        Returns:
            Cross-validation results
        """
        try:
            logger.info("Running cross-validation test...")
            
            # Split data into folds
            fold_size = len(data) // self.cv_folds
            cv_results = []
            
            for fold in range(self.cv_folds):
                # Create train/test split
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size
                
                if fold == self.cv_folds - 1:  # Last fold gets remaining data
                    end_idx = len(data)
                
                # Test data (holdout)
                test_data = data.iloc[start_idx:end_idx]
                
                # Train data (everything else)
                train_data = pd.concat([
                    data.iloc[:start_idx],
                    data.iloc[end_idx:]
                ]).reset_index(drop=True)
                
                if len(train_data) < 50 or len(test_data) < 20:
                    continue
                
                # Run strategy on test data
                try:
                    test_results = strategy_func(test_data, strategy_params)
                    cv_results.append({
                        'fold': fold,
                        'test_size': len(test_data),
                        'train_size': len(train_data),
                        'total_return': test_results.get('total_return', 0),
                        'sharpe_ratio': test_results.get('sharpe_ratio', 0),
                        'max_drawdown': test_results.get('max_drawdown', 0),
                        'total_trades': test_results.get('total_trades', 0)
                    })
                except Exception as e:
                    logger.warning(f"Fold {fold} failed: {e}")
                    continue
            
            # Analyze CV results
            if cv_results:
                returns = [r['total_return'] for r in cv_results]
                sharpes = [r['sharpe_ratio'] for r in cv_results]
                drawdowns = [r['max_drawdown'] for r in cv_results]
                
                cv_analysis = {
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'mean_sharpe': np.mean(sharpes),
                    'std_sharpe': np.std(sharpes),
                    'mean_drawdown': np.mean(drawdowns),
                    'cv_score': self._calculate_cv_score(returns, sharpes, drawdowns),
                    'overfitting_risk': self._assess_cv_overfitting_risk(returns, sharpes, drawdowns)
                }
                
                logger.info(f"CV Score: {cv_analysis['cv_score']:.3f}")
                logger.info(f"Overfitting Risk: {cv_analysis['overfitting_risk']}")
                
                return cv_analysis
            else:
                logger.error("No valid CV results")
                return {}
                
        except Exception as e:
            logger.error(f"Cross-validation test failed: {e}")
            return {}
    
    def parameter_sensitivity_test(self, strategy_func, data: pd.DataFrame,
                                 base_params: Dict, param_ranges: Dict) -> Dict:
        """
        Test parameter sensitivity to detect overfitting.
        
        Args:
            strategy_func: Function that runs the strategy
            data: Market data
            base_params: Base strategy parameters
            param_ranges: Parameter ranges to test
            
        Returns:
            Sensitivity analysis results
        """
        try:
            logger.info("Running parameter sensitivity test...")
            
            sensitivity_results = {}
            
            for param_name, param_range in param_ranges.items():
                param_results = []
                
                for param_value in param_range:
                    # Create modified parameters
                    test_params = base_params.copy()
                    test_params[param_name] = param_value
                    
                    try:
                        # Run strategy with modified parameter
                        results = strategy_func(data, test_params)
                        
                        param_results.append({
                            'param_value': param_value,
                            'total_return': results.get('total_return', 0),
                            'sharpe_ratio': results.get('sharpe_ratio', 0),
                            'max_drawdown': results.get('max_drawdown', 0)
                        })
                    except Exception as e:
                        logger.warning(f"Parameter {param_name}={param_value} failed: {e}")
                        continue
                
                if param_results:
                    # Calculate sensitivity metrics
                    returns = [r['total_return'] for r in param_results]
                    sharpes = [r['sharpe_ratio'] for r in param_results]
                    
                    sensitivity_results[param_name] = {
                        'return_sensitivity': np.std(returns),
                        'sharpe_sensitivity': np.std(sharpes),
                        'optimal_value': param_results[np.argmax(returns)]['param_value'],
                        'performance_range': (min(returns), max(returns)),
                        'is_sensitive': np.std(returns) > 0.1  # 10% threshold
                    }
            
            # Overall sensitivity assessment
            sensitive_params = [name for name, result in sensitivity_results.items() 
                              if result['is_sensitive']]
            
            sensitivity_analysis = {
                'sensitive_parameters': sensitive_params,
                'sensitivity_score': len(sensitive_params) / len(param_ranges),
                'overfitting_risk': 'HIGH' if len(sensitive_params) > len(param_ranges) * 0.5 else 'LOW'
            }
            
            logger.info(f"Sensitivity Score: {sensitivity_analysis['sensitivity_score']:.3f}")
            logger.info(f"Sensitive Parameters: {sensitive_params}")
            
            return sensitivity_analysis
            
        except Exception as e:
            logger.error(f"Parameter sensitivity test failed: {e}")
            return {}
    
    def walk_forward_analysis(self, strategy_func, data: pd.DataFrame,
                            strategy_params: Dict) -> Dict:
        """
        Perform walk-forward analysis to test strategy robustness.
        
        Args:
            strategy_func: Function that runs the strategy
            data: Market data
            strategy_params: Strategy parameters
            
        Returns:
            Walk-forward analysis results
        """
        try:
            logger.info("Running walk-forward analysis...")
            
            window_size = len(data) // self.walk_forward_windows
            walk_forward_results = []
            
            for window in range(self.walk_forward_windows):
                # Define window boundaries
                start_idx = window * window_size
                end_idx = start_idx + window_size
                
                if window == self.walk_forward_windows - 1:
                    end_idx = len(data)
                
                # Test window
                test_data = data.iloc[start_idx:end_idx]
                
                if len(test_data) < 20:
                    continue
                
                try:
                    # Run strategy on test window
                    results = strategy_func(test_data, strategy_params)
                    
                    walk_forward_results.append({
                        'window': window,
                        'start_date': test_data.index[0] if hasattr(test_data.index[0], 'date') else start_idx,
                        'end_date': test_data.index[-1] if hasattr(test_data.index[-1], 'date') else end_idx,
                        'total_return': results.get('total_return', 0),
                        'sharpe_ratio': results.get('sharpe_ratio', 0),
                        'max_drawdown': results.get('max_drawdown', 0),
                        'total_trades': results.get('total_trades', 0)
                    })
                except Exception as e:
                    logger.warning(f"Window {window} failed: {e}")
                    continue
            
            # Analyze walk-forward results
            if walk_forward_results:
                returns = [r['total_return'] for r in walk_forward_results]
                sharpes = [r['sharpe_ratio'] for r in walk_forward_results]
                drawdowns = [r['max_drawdown'] for r in walk_forward_results]
                
                # Calculate performance degradation
                performance_degradation = self._calculate_performance_degradation(returns)
                
                walk_forward_analysis = {
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'mean_sharpe': np.mean(sharpes),
                    'std_sharpe': np.std(sharpes),
                    'performance_degradation': performance_degradation,
                    'consistency_score': self._calculate_consistency_score(returns, sharpes),
                    'overfitting_risk': self._assess_walk_forward_risk(returns, performance_degradation)
                }
                
                logger.info(f"Performance Degradation: {performance_degradation:.3f}")
                logger.info(f"Consistency Score: {walk_forward_analysis['consistency_score']:.3f}")
                
                return walk_forward_analysis
            else:
                logger.error("No valid walk-forward results")
                return {}
                
        except Exception as e:
            logger.error(f"Walk-forward analysis failed: {e}")
            return {}
    
    def realistic_performance_bounds(self, results: Dict) -> Dict:
        """
        Check if performance metrics are within realistic bounds.
        
        Args:
            results: Strategy results
            
        Returns:
            Realism assessment
        """
        try:
            total_return = results.get('total_return', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            max_drawdown = results.get('max_drawdown', 0)
            total_trades = results.get('total_trades', 0)
            
            # Define realistic bounds
            bounds = {
                'total_return': (-0.5, 2.0),  # -50% to +200%
                'sharpe_ratio': (-1.0, 3.0),  # -1.0 to +3.0
                'max_drawdown': (-0.5, 0.0),  # -50% to 0%
                'trades_per_day': (0.01, 2.0)  # 0.01 to 2 trades per day
            }
            
            # Check bounds
            violations = []
            
            if not (bounds['total_return'][0] <= total_return <= bounds['total_return'][1]):
                violations.append(f"Total return {total_return:.2%} outside bounds {bounds['total_return']}")
            
            if not (bounds['sharpe_ratio'][0] <= sharpe_ratio <= bounds['sharpe_ratio'][1]):
                violations.append(f"Sharpe ratio {sharpe_ratio:.3f} outside bounds {bounds['sharpe_ratio']}")
            
            if not (bounds['max_drawdown'][0] <= max_drawdown <= bounds['max_drawdown'][1]):
                violations.append(f"Max drawdown {max_drawdown:.2%} outside bounds {bounds['max_drawdown']}")
            
            # Calculate trades per day (assuming 252 trading days)
            trades_per_day = total_trades / 252 if total_trades > 0 else 0
            if not (bounds['trades_per_day'][0] <= trades_per_day <= bounds['trades_per_day'][1]):
                violations.append(f"Trades per day {trades_per_day:.3f} outside bounds {bounds['trades_per_day']}")
            
            realism_assessment = {
                'is_realistic': len(violations) == 0,
                'violations': violations,
                'realism_score': max(0, 1 - len(violations) / len(bounds)),
                'overfitting_risk': 'HIGH' if len(violations) > 2 else 'LOW'
            }
            
            logger.info(f"Realism Score: {realism_assessment['realism_score']:.3f}")
            if violations:
                logger.warning(f"Performance violations: {violations}")
            
            return realism_assessment
            
        except Exception as e:
            logger.error(f"Realistic performance bounds check failed: {e}")
            return {'is_realistic': False, 'violations': [str(e)], 'realism_score': 0.0}
    
    def comprehensive_overfitting_check(self, strategy_func, data: pd.DataFrame,
                                      strategy_params: Dict, param_ranges: Dict = None) -> Dict:
        """
        Perform comprehensive overfitting check using all protection methods.
        
        Args:
            strategy_func: Function that runs the strategy
            data: Market data
            strategy_params: Strategy parameters
            param_ranges: Parameter ranges for sensitivity testing
            
        Returns:
            Comprehensive overfitting assessment
        """
        try:
            logger.info("Running comprehensive overfitting check...")
            
            # Run all protection tests
            cv_results = self.cross_validation_test(strategy_func, data, strategy_params)
            
            sensitivity_results = {}
            if param_ranges:
                sensitivity_results = self.parameter_sensitivity_test(strategy_func, data, strategy_params, param_ranges)
            
            walk_forward_results = self.walk_forward_analysis(strategy_func, data, strategy_params)
            
            # Run strategy on full dataset for realism check
            full_results = strategy_func(data, strategy_params)
            realism_results = self.realistic_performance_bounds(full_results)
            
            # Calculate overall overfitting score
            overfitting_score = self._calculate_overall_overfitting_score(
                cv_results, sensitivity_results, walk_forward_results, realism_results
            )
            
            # Generate recommendations
            recommendations = self._generate_protection_recommendations(
                cv_results, sensitivity_results, walk_forward_results, realism_results
            )
            
            comprehensive_assessment = {
                'overfitting_score': overfitting_score,
                'overfitting_risk': self._classify_overfitting_risk(overfitting_score),
                'cv_results': cv_results,
                'sensitivity_results': sensitivity_results,
                'walk_forward_results': walk_forward_results,
                'realism_results': realism_results,
                'recommendations': recommendations,
                'is_safe_to_deploy': overfitting_score < 0.7  # 70% threshold
            }
            
            logger.info(f"Overall Overfitting Score: {overfitting_score:.3f}")
            logger.info(f"Overfitting Risk: {comprehensive_assessment['overfitting_risk']}")
            logger.info(f"Safe to Deploy: {comprehensive_assessment['is_safe_to_deploy']}")
            
            return comprehensive_assessment
            
        except Exception as e:
            logger.error(f"Comprehensive overfitting check failed: {e}")
            return {'overfitting_score': 1.0, 'overfitting_risk': 'HIGH', 'is_safe_to_deploy': False}
    
    def _calculate_cv_score(self, returns: List[float], sharpes: List[float], 
                           drawdowns: List[float]) -> float:
        """Calculate cross-validation score."""
        try:
            # Normalize metrics
            return_std = np.std(returns) if returns else 1.0
            sharpe_std = np.std(sharpes) if sharpes else 1.0
            drawdown_std = np.std(drawdowns) if drawdowns else 1.0
            
            # Lower std = better (more consistent)
            cv_score = (return_std + sharpe_std + drawdown_std) / 3
            return min(1.0, cv_score)
        except:
            return 1.0
    
    def _assess_cv_overfitting_risk(self, returns: List[float], sharpes: List[float], 
                                   drawdowns: List[float]) -> str:
        """Assess overfitting risk from CV results."""
        try:
            return_std = np.std(returns) if returns else 1.0
            sharpe_std = np.std(sharpes) if sharpes else 1.0
            
            if return_std > 0.3 or sharpe_std > 1.0:
                return 'HIGH'
            elif return_std > 0.15 or sharpe_std > 0.5:
                return 'MEDIUM'
            else:
                return 'LOW'
        except:
            return 'HIGH'
    
    def _calculate_performance_degradation(self, returns: List[float]) -> float:
        """Calculate performance degradation across windows."""
        try:
            if len(returns) < 2:
                return 0.0
            
            # Calculate degradation from first to last window
            first_half = returns[:len(returns)//2]
            second_half = returns[len(returns)//2:]
            
            if not first_half or not second_half:
                return 0.0
            
            first_performance = np.mean(first_half)
            second_performance = np.mean(second_half)
            
            if first_performance == 0:
                return 0.0
            
            degradation = (first_performance - second_performance) / abs(first_performance)
            return max(0.0, degradation)
        except:
            return 0.0
    
    def _calculate_consistency_score(self, returns: List[float], sharpes: List[float]) -> float:
        """Calculate consistency score across windows."""
        try:
            return_consistency = 1.0 - min(1.0, np.std(returns) * 2)
            sharpe_consistency = 1.0 - min(1.0, np.std(sharpes) * 0.5)
            
            return (return_consistency + sharpe_consistency) / 2
        except:
            return 0.0
    
    def _assess_walk_forward_risk(self, returns: List[float], degradation: float) -> str:
        """Assess overfitting risk from walk-forward results."""
        try:
            return_std = np.std(returns) if returns else 1.0
            
            if degradation > 0.5 or return_std > 0.3:
                return 'HIGH'
            elif degradation > 0.2 or return_std > 0.15:
                return 'MEDIUM'
            else:
                return 'LOW'
        except:
            return 'HIGH'
    
    def _calculate_overall_overfitting_score(self, cv_results: Dict, sensitivity_results: Dict,
                                           walk_forward_results: Dict, realism_results: Dict) -> float:
        """Calculate overall overfitting score."""
        try:
            scores = []
            
            # CV score
            if cv_results:
                scores.append(cv_results.get('cv_score', 1.0))
            
            # Sensitivity score
            if sensitivity_results:
                scores.append(sensitivity_results.get('sensitivity_score', 1.0))
            
            # Walk-forward score
            if walk_forward_results:
                degradation = walk_forward_results.get('performance_degradation', 0.0)
                consistency = walk_forward_results.get('consistency_score', 0.0)
                scores.append((degradation + (1 - consistency)) / 2)
            
            # Realism score
            if realism_results:
                scores.append(1 - realism_results.get('realism_score', 0.0))
            
            if scores:
                return np.mean(scores)
            else:
                return 1.0
        except:
            return 1.0
    
    def _classify_overfitting_risk(self, score: float) -> str:
        """Classify overfitting risk based on score."""
        if score < 0.3:
            return 'LOW'
        elif score < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _generate_protection_recommendations(self, cv_results: Dict, sensitivity_results: Dict,
                                           walk_forward_results: Dict, realism_results: Dict) -> List[str]:
        """Generate protection recommendations."""
        recommendations = []
        
        # CV recommendations
        if cv_results and cv_results.get('overfitting_risk') == 'HIGH':
            recommendations.append("Use cross-validation with more folds")
            recommendations.append("Increase training data size")
        
        # Sensitivity recommendations
        if sensitivity_results and sensitivity_results.get('overfitting_risk') == 'HIGH':
            recommendations.append("Reduce parameter sensitivity")
            recommendations.append("Use more robust parameter selection")
        
        # Walk-forward recommendations
        if walk_forward_results and walk_forward_results.get('overfitting_risk') == 'HIGH':
            recommendations.append("Implement walk-forward optimization")
            recommendations.append("Use shorter optimization windows")
        
        # Realism recommendations
        if realism_results and not realism_results.get('is_realistic', True):
            recommendations.append("Review strategy parameters")
            recommendations.append("Implement more conservative position sizing")
        
        if not recommendations:
            recommendations.append("Strategy appears robust - continue monitoring")
        
        return recommendations


def create_protected_strategy_wrapper(strategy_func, protection_system: OverfittingProtection):
    """
    Create a protected wrapper around a strategy function.
    
    Args:
        strategy_func: Original strategy function
        protection_system: Overfitting protection system
        
    Returns:
        Protected strategy function
    """
    def protected_strategy(data: pd.DataFrame, params: Dict) -> Dict:
        """
        Protected strategy that includes overfitting checks.
        """
        try:
            # Run overfitting protection checks
            protection_results = protection_system.comprehensive_overfitting_check(
                strategy_func, data, params
            )
            
            # Run original strategy
            strategy_results = strategy_func(data, params)
            
            # Add protection results to strategy results
            strategy_results['overfitting_protection'] = protection_results
            
            # Log warnings if overfitting detected
            if protection_results['overfitting_risk'] == 'HIGH':
                logger.warning("HIGH overfitting risk detected!")
                logger.warning(f"Recommendations: {protection_results['recommendations']}")
            
            return strategy_results
            
        except Exception as e:
            logger.error(f"Protected strategy failed: {e}")
            return {'error': str(e), 'overfitting_protection': {'overfitting_risk': 'HIGH'}}
    
    return protected_strategy 