"""
Volatility-Targeted Position Sizing and Drawdown Management
==========================================================

This module implements dynamic position sizing based on volatility targeting
and comprehensive drawdown protection mechanisms for the gold trading algorithm.

Key Features:
- Volatility-targeted position sizing using ATR and realized volatility
- Trailing drawdown caps with soft and hard stops
- Rolling high-water mark logic
- Dynamic exposure scaling based on market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolatilityPositionManager:
    """
    Manages position sizing based on volatility targeting and drawdown protection.
    
    Features:
    - Volatility-targeted position sizing using ATR (10-20 period window)
    - Realized volatility calculation for position scaling
    - Drawdown monitoring with soft (-10%) and hard (-15%) stops
    - Rolling high-water mark tracking
    - Dynamic exposure adjustment
    """
    
    def __init__(self, 
                 target_volatility: float = 0.02,  # 2% daily volatility target
                 atr_period: int = 15,             # ATR calculation period
                 vol_window: int = 20,             # Volatility calculation window
                 soft_drawdown_limit: float = 0.10,  # 10% soft stop
                 hard_drawdown_limit: float = 0.12,  # 12% hard stop (reduced from 15%)
                 max_position_size: float = 1.0,     # Maximum position size
                 min_position_size: float = 0.1):    # Minimum position size
        
        self.target_volatility = target_volatility
        self.atr_period = atr_period
        self.vol_window = vol_window
        # Drawdown protection parameters (very conservative)
        self.soft_drawdown_limit = soft_drawdown_limit  # 10% soft stop
        self.hard_drawdown_limit = min(hard_drawdown_limit, 0.05)  # Cap at 5% for maximum protection
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        
        # State variables
        self.high_water_mark = 1.0
        self.current_drawdown = 0.0
        self.position_scale = 1.0
        self.volatility_history = []
        self.atr_history = []
        self.nav_history = []
        
        # Risk state flags
        self.soft_stop_triggered = False
        self.hard_stop_triggered = False
        self.recovery_mode = False
        
        logger.info(f"VolatilityPositionManager initialized with target volatility: {target_volatility}")
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """
        Calculate Average True Range (ATR) for volatility measurement.
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            
        Returns:
            Current ATR value
        """
        try:
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR using exponential moving average
            atr = true_range.ewm(span=self.atr_period, adjust=False).mean().iloc[-1]
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.01  # Default to 1% if calculation fails
    
    def calculate_realized_volatility(self, returns: pd.Series) -> float:
        """
        Calculate realized volatility using rolling window.
        
        Args:
            returns: Price returns series
            
        Returns:
            Realized volatility (annualized)
        """
        try:
            if len(returns) < self.vol_window:
                return 0.02  # Default 2% if insufficient data
            
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=self.vol_window).std()
            current_vol = rolling_vol.iloc[-1]
            
            # Annualize (assuming daily data)
            annualized_vol = current_vol * np.sqrt(252)
            
            return max(annualized_vol, 0.005)  # Minimum 0.5% volatility
            
        except Exception as e:
            logger.error(f"Error calculating realized volatility: {e}")
            return 0.02
    
    def update_drawdown_monitor(self, current_nav: float) -> Dict[str, float]:
        """
        Update drawdown monitoring and return risk metrics.
        
        Args:
            current_nav: Current Net Asset Value
            
        Returns:
            Dictionary with drawdown metrics and risk state
        """
        try:
            # Update NAV history
            self.nav_history.append(current_nav)
            
            # Update high water mark
            if current_nav > self.high_water_mark:
                self.high_water_mark = current_nav
                # Reset recovery mode if we hit new high
                if self.recovery_mode:
                    self.recovery_mode = False
                    self.position_scale = 1.0
                    logger.info("Recovery mode deactivated - new high water mark reached")
            
            # Calculate current drawdown
            self.current_drawdown = (self.high_water_mark - current_nav) / self.high_water_mark
            
            # Check drawdown limits
            risk_state = {
                'current_drawdown': self.current_drawdown,
                'high_water_mark': self.high_water_mark,
                'position_scale': self.position_scale,
                'soft_stop_triggered': False,
                'hard_stop_triggered': False,
                'recovery_mode': self.recovery_mode
            }
            
            # Soft stop logic (-10%)
            if self.current_drawdown >= self.soft_drawdown_limit and not self.soft_stop_triggered:
                self.soft_stop_triggered = True
                self.position_scale = 0.5  # Reduce position size to 50%
                self.recovery_mode = True
                risk_state['soft_stop_triggered'] = True
                risk_state['position_scale'] = self.position_scale
                logger.warning(f"Soft stop triggered at {self.current_drawdown:.2%} drawdown")
            
            # Hard stop logic (-15%)
            if self.current_drawdown >= self.hard_drawdown_limit and not self.hard_stop_triggered:
                self.hard_stop_triggered = True
                self.position_scale = 0.0  # Stop all trading
                risk_state['hard_stop_triggered'] = True
                risk_state['position_scale'] = self.position_scale
                logger.critical(f"HARD STOP triggered at {self.current_drawdown:.2%} drawdown - TRADING HALTED")
            
            # Enhanced recovery logic
            if self.recovery_mode and self.current_drawdown < self.soft_drawdown_limit * 0.7:
                # More responsive recovery - start recovery earlier
                recovery_ratio = 1 - (self.current_drawdown / self.soft_drawdown_limit)
                # Accelerated recovery curve
                recovery_multiplier = recovery_ratio ** 0.5  # Square root for faster recovery
                self.position_scale = min(1.0, 0.5 + (0.5 * recovery_multiplier))
                risk_state['position_scale'] = self.position_scale
                logger.info(f"Recovery mode: position scale increased to {self.position_scale:.2f} (drawdown: {self.current_drawdown:.2%})")
            
            # Full recovery when drawdown is minimal
            if self.recovery_mode and self.current_drawdown < 0.02:  # 2% drawdown
                self.position_scale = 1.0
                self.recovery_mode = False
                risk_state['position_scale'] = self.position_scale
                risk_state['recovery_mode'] = False
                logger.info("Full recovery achieved - normal trading resumed")
            
            return risk_state
            
        except Exception as e:
            logger.error(f"Error in drawdown monitoring: {e}")
            return {
                'current_drawdown': 0.0,
                'high_water_mark': self.high_water_mark,
                'position_scale': 1.0,
                'soft_stop_triggered': False,
                'hard_stop_triggered': False,
                'recovery_mode': False
            }
    
    def calculate_volatility_targeted_position(self, 
                                             signal_strength: float,
                                             current_price: float,
                                             high: pd.Series,
                                             low: pd.Series, 
                                             close: pd.Series,
                                             returns: pd.Series,
                                             current_nav: float) -> Dict[str, float]:
        """
        Calculate volatility-targeted position size with drawdown protection.
        
        Args:
            signal_strength: Trading signal strength (-1 to 1)
            current_price: Current asset price
            high: High prices series
            low: Low prices series
            close: Close prices series
            returns: Price returns series
            current_nav: Current Net Asset Value
            
        Returns:
            Dictionary with position sizing information
        """
        try:
            # Update drawdown monitoring
            risk_state = self.update_drawdown_monitor(current_nav)
            
            # Calculate volatility metrics
            atr = self.calculate_atr(high, low, close)
            realized_vol = self.calculate_realized_volatility(returns)
            
            # Store volatility history
            self.atr_history.append(atr)
            self.volatility_history.append(realized_vol)
            
            # Calculate volatility-adjusted position size
            # Use the higher of ATR-based or realized volatility
            volatility_measure = max(atr / current_price, realized_vol / np.sqrt(252))
            
            # Volatility targeting: scale position to achieve target volatility
            if volatility_measure > 0:
                vol_target_scale = self.target_volatility / volatility_measure
            else:
                vol_target_scale = 1.0
            
            # Apply position size limits (more conservative)
            base_position_size = abs(signal_strength) * vol_target_scale
            base_position_size = np.clip(base_position_size, 
                                       self.min_position_size, 
                                       0.2)  # Cap at 20% instead of 30%
            
            # Apply drawdown protection scaling
            final_position_size = base_position_size * risk_state['position_scale']
            
            # Determine position direction
            position_direction = np.sign(signal_strength)
            
            # Calculate position value and risk metrics
            position_value = final_position_size * current_nav
            expected_daily_risk = position_value * volatility_measure
            
            result = {
                'position_size': final_position_size,
                'position_direction': position_direction,
                'position_value': position_value,
                'volatility_measure': volatility_measure,
                'atr': atr,
                'realized_volatility': realized_vol,
                'vol_target_scale': vol_target_scale,
                'expected_daily_risk': expected_daily_risk,
                'risk_adjusted_signal': signal_strength * risk_state['position_scale'],
                **risk_state
            }
            
            logger.info(f"Position calculated: size={final_position_size:.3f}, "
                       f"direction={position_direction}, vol_scale={vol_target_scale:.3f}, "
                       f"drawdown_scale={risk_state['position_scale']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating volatility-targeted position: {e}")
            return {
                'position_size': 0.0,
                'position_direction': 0,
                'position_value': 0.0,
                'volatility_measure': 0.02,
                'atr': 0.01,
                'realized_volatility': 0.02,
                'vol_target_scale': 1.0,
                'expected_daily_risk': 0.0,
                'risk_adjusted_signal': 0.0,
                'current_drawdown': 0.0,
                'high_water_mark': self.high_water_mark,
                'position_scale': 0.0,
                'soft_stop_triggered': True,
                'hard_stop_triggered': False,
                'recovery_mode': False
            }
    
    def get_risk_summary(self) -> Dict[str, float]:
        """
        Get current risk management summary.
        
        Returns:
            Dictionary with current risk metrics
        """
        return {
            'current_drawdown': self.current_drawdown,
            'high_water_mark': self.high_water_mark,
            'position_scale': self.position_scale,
            'soft_stop_triggered': self.soft_stop_triggered,
            'hard_stop_triggered': self.hard_stop_triggered,
            'recovery_mode': self.recovery_mode,
            'avg_atr': np.mean(self.atr_history[-20:]) if self.atr_history else 0.01,
            'avg_volatility': np.mean(self.volatility_history[-20:]) if self.volatility_history else 0.02
        }
    
    def reset(self):
        """Reset the position manager state."""
        self.high_water_mark = 1.0
        self.current_drawdown = 0.0
        self.position_scale = 1.0
        self.volatility_history = []
        self.atr_history = []
        self.nav_history = []
        self.soft_stop_triggered = False
        self.hard_stop_triggered = False
        self.recovery_mode = False
        logger.info("VolatilityPositionManager reset")


class RiskOverlay:
    """
    Risk overlay that integrates with existing trading strategies.
    Provides volatility targeting and drawdown protection.
    """
    
    def __init__(self, 
                 target_volatility: float = 0.02,
                 atr_period: int = 15,
                 vol_window: int = 20,
                 soft_drawdown_limit: float = 0.10,
                 hard_drawdown_limit: float = 0.05):  # Reduced to 5% for maximum protection
        
        self.position_manager = VolatilityPositionManager(
            target_volatility=target_volatility,
            atr_period=atr_period,
            vol_window=vol_window,
            soft_drawdown_limit=soft_drawdown_limit,
            hard_drawdown_limit=hard_drawdown_limit
        )
        
        logger.info("RiskOverlay initialized")
    
    def apply_risk_controls(self, 
                           signal: float,
                           price_data: pd.DataFrame,
                           current_nav: float) -> Dict[str, float]:
        """
        Apply risk controls to trading signal.
        
        Args:
            signal: Raw trading signal (-1 to 1)
            price_data: DataFrame with OHLCV data
            current_nav: Current Net Asset Value
            
        Returns:
            Risk-adjusted position information
        """
        try:
            # Extract price series
            high = price_data['high']
            low = price_data['low']
            close = price_data['close']
            
            # Calculate returns
            returns = close.pct_change().dropna()
            
            # Apply volatility targeting and drawdown protection
            position_info = self.position_manager.calculate_volatility_targeted_position(
                signal_strength=signal,
                current_price=close.iloc[-1],
                high=high,
                low=low,
                close=close,
                returns=returns,
                current_nav=current_nav
            )
            
            return position_info
            
        except Exception as e:
            logger.error(f"Error applying risk controls: {e}")
            return {
                'position_size': 0.0,
                'position_direction': 0,
                'position_value': 0.0,
                'volatility_measure': 0.02,
                'atr': 0.01,
                'realized_volatility': 0.02,
                'vol_target_scale': 1.0,
                'expected_daily_risk': 0.0,
                'risk_adjusted_signal': 0.0,
                'current_drawdown': 0.0,
                'high_water_mark': 1.0,
                'position_scale': 0.0,
                'soft_stop_triggered': True,
                'hard_stop_triggered': False,
                'recovery_mode': False
            }
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics."""
        return self.position_manager.get_risk_summary()
    
    def reset(self):
        """Reset risk overlay state."""
        self.position_manager.reset() 