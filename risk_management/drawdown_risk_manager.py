#!/usr/bin/env python3
"""
Drawdown-Based Risk Management System
====================================

This module implements comprehensive drawdown-based risk controls including:
- Trailing stop-losses with dynamic thresholds
- Circuit breakers for extreme drawdowns
- Dynamic position sizing based on drawdown levels
- Real-time drawdown tracking and volatility estimation
- Monte Carlo stress testing capabilities

Key Features:
- 10-day volatility lookback for robustness
- Simple, well-understood risk rules
- Real-time calculation feasibility
- Out-of-sample validation support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DrawdownRiskManager:
    """
    Comprehensive drawdown-based risk management system.
    
    Features:
    - Trailing stop-losses with dynamic thresholds
    - Circuit breakers for extreme drawdowns
    - Dynamic position sizing based on drawdown levels
    - Real-time volatility estimation (10-day lookback)
    - Monte Carlo stress testing
    """
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 trailing_stop_pct: float = 0.05,  # 5% trailing stop
                 circuit_breaker_pct: float = 0.15,  # 15% circuit breaker
                 volatility_lookback: int = 10,  # 10-day volatility
                 max_position_size: float = 0.20,  # 20% max position
                 min_position_size: float = 0.01,  # 1% min position
                 drawdown_scaling: bool = True):
        """
        Initialize the drawdown risk manager.
        
        Args:
            initial_capital: Starting capital
            trailing_stop_pct: Trailing stop percentage
            circuit_breaker_pct: Circuit breaker percentage
            volatility_lookback: Days for volatility calculation
            max_position_size: Maximum position size
            min_position_size: Minimum position size
            drawdown_scaling: Enable drawdown-based position scaling
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # Risk parameters
        self.trailing_stop_pct = trailing_stop_pct
        self.circuit_breaker_pct = circuit_breaker_pct
        self.volatility_lookback = volatility_lookback
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.drawdown_scaling = drawdown_scaling
        
        # State tracking
        self.current_drawdown = 0.0
        self.trailing_stop_level = initial_capital * (1 - trailing_stop_pct)
        self.circuit_breaker_triggered = False
        self.trading_halted = False
        
        # Performance tracking
        self.equity_curve = []
        self.drawdown_history = []
        self.volatility_history = []
        self.position_size_history = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.volatility = 0.0
        self.var_95 = 0.0  # 95% Value at Risk
        
        logger.info(f"DrawdownRiskManager initialized with {trailing_stop_pct:.1%} trailing stop, {circuit_breaker_pct:.1%} circuit breaker")
    
    def update_equity(self, current_value: float, timestamp: datetime) -> Dict[str, any]:
        """
        Update equity and calculate risk metrics.
        
        Args:
            current_value: Current portfolio value
            timestamp: Current timestamp
            
        Returns:
            Dictionary with risk metrics and actions
        """
        try:
            # Update capital
            self.current_capital = current_value
            
            # Update peak capital (trailing high water mark)
            if current_value > self.peak_capital:
                self.peak_capital = current_value
                self.trailing_stop_level = current_value * (1 - self.trailing_stop_pct)
            
            # Calculate current drawdown
            self.current_drawdown = (self.peak_capital - current_value) / self.peak_capital
            
            # Update max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            
            # Check trailing stop
            trailing_stop_triggered = current_value <= self.trailing_stop_level
            
            # Check circuit breaker
            circuit_breaker_triggered = self.current_drawdown >= self.circuit_breaker_pct
            
            # Update trading status
            if circuit_breaker_triggered and not self.circuit_breaker_triggered:
                self.circuit_breaker_triggered = True
                self.trading_halted = True
                logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED at {self.current_drawdown:.2%} drawdown - TRADING HALTED")
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'value': current_value,
                'drawdown': self.current_drawdown,
                'peak': self.peak_capital
            })
            
            # Calculate volatility (10-day lookback)
            if len(self.equity_curve) >= self.volatility_lookback:
                recent_values = [point['value'] for point in self.equity_curve[-self.volatility_lookback:]]
                returns = np.diff(recent_values) / recent_values[:-1]
                self.volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate position size scaling
            position_scale = self._calculate_position_scale()
            
            return {
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'trailing_stop_triggered': trailing_stop_triggered,
                'circuit_breaker_triggered': circuit_breaker_triggered,
                'trading_halted': self.trading_halted,
                'position_scale': position_scale,
                'volatility': self.volatility,
                'peak_capital': self.peak_capital
            }
            
        except Exception as e:
            logger.error(f"Error updating equity: {e}")
            return {}
    
    def _calculate_position_scale(self) -> float:
        """
        Calculate position size scaling based on drawdown.
        
        Returns:
            Position scale factor (0.0 to 1.0)
        """
        if not self.drawdown_scaling or self.trading_halted:
            return 0.0
        
        # Simple linear scaling based on drawdown
        if self.current_drawdown <= 0.02:  # 0-2% drawdown: full position
            return 1.0
        elif self.current_drawdown <= 0.05:  # 2-5% drawdown: 75% position
            return 0.75
        elif self.current_drawdown <= 0.08:  # 5-8% drawdown: 50% position
            return 0.50
        elif self.current_drawdown <= 0.10:  # 8-10% drawdown: 25% position
            return 0.25
        else:  # >10% drawdown: no position
            return 0.0
    
    def calculate_position_size(self, signal_strength: float, base_position_size: float) -> float:
        """
        Calculate final position size with risk controls.
        
        Args:
            signal_strength: Signal strength (-1 to 1)
            base_position_size: Base position size
            
        Returns:
            Final position size
        """
        try:
            if self.trading_halted:
                return 0.0
            
            # Apply drawdown scaling
            position_scale = self._calculate_position_scale()
            
            # Calculate final position size
            final_position = base_position_size * abs(signal_strength) * position_scale
            
            # Apply limits
            final_position = np.clip(final_position, self.min_position_size, self.max_position_size)
            
            # Track position size
            self.position_size_history.append({
                'timestamp': datetime.now(),
                'position_size': final_position,
                'drawdown': self.current_drawdown,
                'scale': position_scale
            })
            
            return final_position
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def should_exit_position(self, current_value: float) -> bool:
        """
        Determine if position should be exited based on risk rules.
        
        Args:
            current_value: Current portfolio value
            
        Returns:
            True if position should be exited
        """
        # Check trailing stop
        if current_value <= self.trailing_stop_level:
            logger.warning(f"Trailing stop triggered at {current_value:.2f}")
            return True
        
        # Check circuit breaker
        if self.current_drawdown >= self.circuit_breaker_pct:
            logger.critical(f"Circuit breaker triggered at {self.current_drawdown:.2%} drawdown")
            return True
        
        return False
    
    def reset_trading(self) -> bool:
        """
        Reset trading after circuit breaker.
        
        Returns:
            True if trading can resume
        """
        if self.circuit_breaker_triggered:
            # Only resume if drawdown improves significantly
            if self.current_drawdown < self.circuit_breaker_pct * 0.5:  # 50% recovery
                self.circuit_breaker_triggered = False
                self.trading_halted = False
                logger.info(f"Trading resumed - drawdown improved to {self.current_drawdown:.2%}")
                return True
            else:
                logger.warning(f"Trading remains halted - drawdown at {self.current_drawdown:.2%}")
                return False
        return True
    
    def get_risk_summary(self) -> Dict[str, any]:
        """
        Get comprehensive risk summary.
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'trading_halted': self.trading_halted,
            'circuit_breaker_triggered': self.circuit_breaker_triggered,
            'peak_capital': self.peak_capital,
            'current_capital': self.current_capital,
            'trailing_stop_level': self.trailing_stop_level,
            'position_scale': self._calculate_position_scale()
        }
    
    def run_monte_carlo_stress_test(self, historical_returns: pd.Series, 
                                   num_simulations: int = 1000) -> Dict[str, any]:
        """
        Run Monte Carlo stress test on historical data.
        
        Args:
            historical_returns: Historical return series
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Stress test results
        """
        try:
            logger.info(f"Running Monte Carlo stress test with {num_simulations} simulations...")
            
            # Calculate historical statistics
            mean_return = historical_returns.mean()
            std_return = historical_returns.std()
            
            # Generate Monte Carlo scenarios
            scenarios = np.random.normal(mean_return, std_return, 
                                       (num_simulations, len(historical_returns)))
            
            # Calculate drawdowns for each scenario
            max_drawdowns = []
            var_95_scenarios = []
            
            for scenario in scenarios:
                # Calculate cumulative returns
                cumulative = np.cumprod(1 + scenario)
                
                # Calculate drawdown
                peak = np.maximum.accumulate(cumulative)
                drawdown = (peak - cumulative) / peak
                max_drawdown = np.max(drawdown)
                max_drawdowns.append(max_drawdown)
                
                # Calculate VaR
                var_95 = np.percentile(scenario, 5)
                var_95_scenarios.append(var_95)
            
            # Calculate statistics
            max_drawdown_mean = np.mean(max_drawdowns)
            max_drawdown_95th = np.percentile(max_drawdowns, 95)
            max_drawdown_99th = np.percentile(max_drawdowns, 99)
            var_95_mean = np.mean(var_95_scenarios)
            
            results = {
                'num_simulations': num_simulations,
                'max_drawdown_mean': max_drawdown_mean,
                'max_drawdown_95th': max_drawdown_95th,
                'max_drawdown_99th': max_drawdown_99th,
                'var_95_mean': var_95_mean,
                'scenarios': max_drawdowns
            }
            
            logger.info(f"Monte Carlo results: Mean max drawdown {max_drawdown_mean:.2%}, 95th percentile {max_drawdown_95th:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo stress test: {e}")
            return {}
    
    def reset(self):
        """Reset the risk manager state."""
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.trailing_stop_level = self.initial_capital * (1 - self.trailing_stop_pct)
        self.circuit_breaker_triggered = False
        self.trading_halted = False
        self.equity_curve = []
        self.drawdown_history = []
        self.volatility_history = []
        self.position_size_history = []
        self.volatility = 0.0
        
        logger.info("DrawdownRiskManager reset") 