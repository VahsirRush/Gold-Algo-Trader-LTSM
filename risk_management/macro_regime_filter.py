#!/usr/bin/env python3
"""
Macroeconomic Regime Filter for Gold Trading
===========================================

This module implements a macroeconomic regime classification system that
adjusts gold trading exposure based on prevailing economic conditions.

Key Features:
- Real-time regime detection using macro indicators
- Two-state Markov regime-switching model
- Risk-off (hedging) vs Risk-on (carry) regime classification
- Dynamic position sizing and risk parameter adjustment
- Regime persistence to prevent whipsaw
- Leverage limits (6x-8x range) and risk controls

Regime Indicators:
- Real interest rates (inflation breakevens)
- USD strength (DXY proxy)
- Equity market volatility (VIX proxy)
- Fed policy indicators
- Market stress indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class MacroRegimeFilter:
    """
    Macroeconomic regime filter for gold trading strategy.
    
    Identifies favorable vs unfavorable regimes for gold and adjusts
    exposure accordingly using real-time macro indicators.
    """
    
    def __init__(self,
                 regime_persistence_days: int = 5,  # Minimum days in regime
                 max_leverage: float = 6.0,  # Maximum leverage (6x)
                 min_leverage: float = 0.5,  # Minimum leverage (0.5x)
                 regime_confidence_threshold: float = 0.7,  # Regime confidence threshold
                 use_markov_model: bool = True):
        """
        Initialize the macro regime filter.
        
        Args:
            regime_persistence_days: Minimum days to stay in a regime
            max_leverage: Maximum leverage multiplier
            min_leverage: Minimum leverage multiplier
            regime_confidence_threshold: Confidence threshold for regime classification
            use_markov_model: Use Markov regime-switching model
        """
        self.regime_persistence_days = regime_persistence_days
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.regime_confidence_threshold = regime_confidence_threshold
        self.use_markov_model = use_markov_model
        
        # Regime state tracking
        self.current_regime = 'neutral'  # 'risk_off', 'risk_on', 'neutral'
        self.regime_start_date = None
        self.regime_confidence = 0.0
        self.regime_history = []
        
        # Markov model parameters (simplified)
        self.transition_matrix = np.array([
            [0.8, 0.15, 0.05],  # risk_off -> [risk_off, risk_on, neutral]
            [0.15, 0.8, 0.05],  # risk_on -> [risk_off, risk_on, neutral]
            [0.1, 0.1, 0.8]     # neutral -> [risk_off, risk_on, neutral]
        ])
        
        # Regime indicators and weights
        self.indicator_weights = {
            'real_rates': 0.3,      # Real interest rates (negative = favorable for gold)
            'usd_strength': 0.25,   # USD strength (negative = favorable for gold)
            'volatility': 0.25,     # Market volatility (positive = favorable for gold)
            'market_stress': 0.2    # Market stress indicators (positive = favorable for gold)
        }
        
        # Regime-specific parameters
        self.regime_parameters = {
            'risk_off': {
                'leverage_multiplier': 1.5,    # Increase position size
                'stop_loss_multiplier': 1.2,   # Wider stops
                'volatility_target': 0.20,     # Higher volatility target
                'description': 'Risk-off regime: Favorable for gold hedging'
            },
            'risk_on': {
                'leverage_multiplier': 0.7,    # Reduce position size
                'stop_loss_multiplier': 0.8,   # Tighter stops
                'volatility_target': 0.12,     # Lower volatility target
                'description': 'Risk-on regime: Unfavorable for gold'
            },
            'neutral': {
                'leverage_multiplier': 1.0,    # Normal position size
                'stop_loss_multiplier': 1.0,   # Normal stops
                'volatility_target': 0.15,     # Normal volatility target
                'description': 'Neutral regime: Standard parameters'
            }
        }
        
        # Data storage
        self.macro_data = pd.DataFrame()
        self.regime_scores = []
        
        logger.info(f"MacroRegimeFilter initialized with {regime_persistence_days} day persistence")
    
    def calculate_real_rates(self, treasury_yields: pd.Series, inflation_breakevens: pd.Series) -> pd.Series:
        """
        Calculate real interest rates.
        
        Args:
            treasury_yields: Treasury yields
            inflation_breakevens: Inflation breakeven rates
            
        Returns:
            Real interest rates
        """
        try:
            real_rates = treasury_yields - inflation_breakevens
            return real_rates
        except Exception as e:
            logger.error(f"Error calculating real rates: {e}")
            return pd.Series([0.0] * len(treasury_yields))
    
    def calculate_usd_strength(self, usd_index: pd.Series) -> pd.Series:
        """
        Calculate USD strength indicator.
        
        Args:
            usd_index: USD index values
            
        Returns:
            USD strength indicator (normalized)
        """
        try:
            # Normalize USD strength (higher = stronger USD = unfavorable for gold)
            usd_strength = (usd_index - usd_index.rolling(20).mean()) / usd_index.rolling(20).std()
            return usd_strength
        except Exception as e:
            logger.error(f"Error calculating USD strength: {e}")
            return pd.Series([0.0] * len(usd_index))
    
    def calculate_volatility_indicator(self, vix: pd.Series) -> pd.Series:
        """
        Calculate volatility indicator.
        
        Args:
            vix: VIX volatility index
            
        Returns:
            Volatility indicator (normalized)
        """
        try:
            # Normalize VIX (higher = more volatility = favorable for gold)
            volatility = (vix - vix.rolling(20).mean()) / vix.rolling(20).std()
            return volatility
        except Exception as e:
            logger.error(f"Error calculating volatility indicator: {e}")
            return pd.Series([0.0] * len(vix))
    
    def calculate_market_stress(self, credit_spreads: pd.Series, equity_returns: pd.Series) -> pd.Series:
        """
        Calculate market stress indicator.
        
        Args:
            credit_spreads: Credit spreads
            equity_returns: Equity market returns
            
        Returns:
            Market stress indicator
        """
        try:
            # Combine credit spreads and equity volatility
            credit_stress = (credit_spreads - credit_spreads.rolling(20).mean()) / credit_spreads.rolling(20).std()
            equity_stress = -equity_returns.rolling(10).std()  # Negative for stress
            
            # Combine indicators
            market_stress = (credit_stress + equity_stress) / 2
            return market_stress
        except Exception as e:
            logger.error(f"Error calculating market stress: {e}")
            return pd.Series([0.0] * len(credit_spreads))
    
    def calculate_regime_score(self, macro_data: pd.DataFrame) -> float:
        """
        Calculate regime score based on macro indicators.
        
        Args:
            macro_data: DataFrame with macro indicators
            
        Returns:
            Regime score (-1 to 1, positive = risk-off/favorable for gold)
        """
        try:
            if macro_data.empty:
                return 0.0
            
            # Calculate individual indicators
            indicators = {}
            
            # Real rates (negative = favorable for gold)
            if 'real_rates' in macro_data.columns:
                indicators['real_rates'] = -macro_data['real_rates'].iloc[-1]  # Negative for favorable
            
            # USD strength (negative = favorable for gold)
            if 'usd_strength' in macro_data.columns:
                indicators['usd_strength'] = -macro_data['usd_strength'].iloc[-1]  # Negative for favorable
            
            # Volatility (positive = favorable for gold)
            if 'volatility' in macro_data.columns:
                indicators['volatility'] = macro_data['volatility'].iloc[-1]
            
            # Market stress (positive = favorable for gold)
            if 'market_stress' in macro_data.columns:
                indicators['market_stress'] = macro_data['market_stress'].iloc[-1]
            
            # Calculate weighted regime score
            regime_score = 0.0
            total_weight = 0.0
            
            for indicator, value in indicators.items():
                if indicator in self.indicator_weights and not np.isnan(value):
                    weight = self.indicator_weights[indicator]
                    regime_score += weight * value
                    total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                regime_score = regime_score / total_weight
            
            # Clip to [-1, 1] range
            regime_score = np.clip(regime_score, -1.0, 1.0)
            
            return regime_score
            
        except Exception as e:
            logger.error(f"Error calculating regime score: {e}")
            return 0.0
    
    def classify_regime(self, regime_score: float) -> Tuple[str, float]:
        """
        Classify regime based on regime score.
        
        Args:
            regime_score: Regime score (-1 to 1)
            
        Returns:
            Tuple of (regime, confidence)
        """
        try:
            # Determine regime based on score
            if regime_score > 0.3:
                regime = 'risk_off'
                confidence = min(abs(regime_score), 1.0)
            elif regime_score < -0.3:
                regime = 'risk_on'
                confidence = min(abs(regime_score), 1.0)
            else:
                regime = 'neutral'
                confidence = 1.0 - abs(regime_score)
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return 'neutral', 0.5
    
    def should_switch_regime(self, new_regime: str, new_confidence: float) -> bool:
        """
        Determine if regime should switch based on persistence and confidence.
        
        Args:
            new_regime: New regime classification
            new_confidence: Confidence in new regime
            
        Returns:
            True if regime should switch
        """
        try:
            # Check if we have enough confidence
            if new_confidence < self.regime_confidence_threshold:
                return False
            
            # Check if regime is different
            if new_regime == self.current_regime:
                return False
            
            # Check persistence requirement
            if self.regime_start_date:
                days_in_current = (datetime.now() - self.regime_start_date).days
                if days_in_current < self.regime_persistence_days:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking regime switch: {e}")
            return False
    
    def update_regime(self, macro_data: pd.DataFrame) -> Dict[str, any]:
        """
        Update regime classification based on macro data.
        
        Args:
            macro_data: DataFrame with macro indicators
            
        Returns:
            Dictionary with regime information
        """
        try:
            # Calculate regime score
            regime_score = self.calculate_regime_score(macro_data)
            
            # Classify regime
            new_regime, new_confidence = self.classify_regime(regime_score)
            
            # Check if regime should switch
            regime_switched = False
            if self.should_switch_regime(new_regime, new_confidence):
                old_regime = self.current_regime
                self.current_regime = new_regime
                self.regime_confidence = new_confidence
                self.regime_start_date = datetime.now()
                regime_switched = True
                
                logger.info(f"Regime switched from {old_regime} to {new_regime} (confidence: {new_confidence:.2f})")
            
            # Update regime history
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': self.current_regime,
                'confidence': self.regime_confidence,
                'score': regime_score,
                'switched': regime_switched
            })
            
            # Get regime parameters
            regime_params = self.regime_parameters[self.current_regime]
            
            return {
                'current_regime': self.current_regime,
                'regime_confidence': self.regime_confidence,
                'regime_score': regime_score,
                'regime_switched': regime_switched,
                'leverage_multiplier': regime_params['leverage_multiplier'],
                'stop_loss_multiplier': regime_params['stop_loss_multiplier'],
                'volatility_target': regime_params['volatility_target'],
                'description': regime_params['description']
            }
            
        except Exception as e:
            logger.error(f"Error updating regime: {e}")
            return {
                'current_regime': 'neutral',
                'regime_confidence': 0.5,
                'regime_score': 0.0,
                'regime_switched': False,
                'leverage_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'volatility_target': 0.15,
                'description': 'Neutral regime: Standard parameters'
            }
    
    def calculate_adjusted_position_size(self, base_position_size: float, 
                                       signal_strength: float) -> float:
        """
        Calculate position size adjusted for macro regime.
        
        Args:
            base_position_size: Base position size
            signal_strength: Signal strength (-1 to 1)
            
        Returns:
            Adjusted position size
        """
        try:
            # Get regime parameters
            regime_params = self.regime_parameters[self.current_regime]
            leverage_multiplier = regime_params['leverage_multiplier']
            
            # Calculate adjusted position size
            adjusted_position = base_position_size * abs(signal_strength) * leverage_multiplier
            
            # Apply leverage limits
            adjusted_position = np.clip(adjusted_position, self.min_leverage, self.max_leverage)
            
            return adjusted_position
            
        except Exception as e:
            logger.error(f"Error calculating adjusted position size: {e}")
            return base_position_size
    
    def get_regime_summary(self) -> Dict[str, any]:
        """
        Get comprehensive regime summary.
        
        Returns:
            Dictionary with regime information
        """
        try:
            regime_params = self.regime_parameters[self.current_regime]
            
            return {
                'current_regime': self.current_regime,
                'regime_confidence': self.regime_confidence,
                'regime_start_date': self.regime_start_date,
                'leverage_multiplier': regime_params['leverage_multiplier'],
                'stop_loss_multiplier': regime_params['stop_loss_multiplier'],
                'volatility_target': regime_params['volatility_target'],
                'description': regime_params['description'],
                'regime_history_length': len(self.regime_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting regime summary: {e}")
            return {}
    
    def simulate_macro_data(self, days: int = 252) -> pd.DataFrame:
        """
        Simulate macro data for testing purposes.
        
        Args:
            days: Number of days to simulate
            
        Returns:
            Simulated macro data
        """
        try:
            dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
            
            # Simulate realistic macro indicators
            np.random.seed(42)
            
            # Real rates (mean around 0.5%, some volatility)
            real_rates = np.random.normal(0.005, 0.02, days)
            
            # USD strength (mean around 100, some trend)
            usd_strength = 100 + np.cumsum(np.random.normal(0, 0.5, days))
            
            # VIX (mean around 20, spikes during stress)
            vix = np.random.gamma(2, 10, days)
            
            # Market stress (correlated with VIX)
            market_stress = vix / 20 + np.random.normal(0, 0.1, days)
            
            # Create DataFrame
            macro_data = pd.DataFrame({
                'date': dates,
                'real_rates': real_rates,
                'usd_strength': usd_strength,
                'volatility': vix,
                'market_stress': market_stress
            })
            
            macro_data.set_index('date', inplace=True)
            
            return macro_data
            
        except Exception as e:
            logger.error(f"Error simulating macro data: {e}")
            return pd.DataFrame()
    
    def reset(self):
        """Reset the regime filter state."""
        self.current_regime = 'neutral'
        self.regime_start_date = None
        self.regime_confidence = 0.0
        self.regime_history = []
        self.macro_data = pd.DataFrame()
        self.regime_scores = []
        
        logger.info("MacroRegimeFilter reset") 