"""
Advanced risk management system for gold trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class RiskMetric(Enum):
    """Risk metrics for monitoring."""
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    EXPOSURE = "exposure"

class PositionType(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

@dataclass
class Position:
    """Position information."""
    type: PositionType
    size: float
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: float
    take_profit: float
    trailing_stop: float
    current_price: float
    pnl: float
    max_favorable_excursion: float
    max_adverse_excursion: float

@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_size: float = 0.02  # 2% of portfolio
    max_portfolio_exposure: float = 0.10  # 10% of portfolio
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_drawdown: float = 0.20  # 20% maximum drawdown
    var_limit: float = 0.03  # 3% Value at Risk limit
    max_correlation: float = 0.7  # Maximum correlation between positions
    min_confidence: float = 0.6  # Minimum confidence for trades

class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 max_position_size: float = 0.1,  # 10% of capital
                 max_portfolio_risk: float = 0.02,  # 2% max risk per trade
                 stop_loss_pct: float = 0.02,  # 2% stop loss
                 take_profit_pct: float = 0.04,  # 4% take profit
                 trailing_stop_pct: float = 0.01,  # 1% trailing stop
                 max_drawdown: float = 0.15,  # 15% max drawdown
                 volatility_lookback: int = 20,
                 correlation_lookback: int = 60):
        """
        Initialize risk manager.
        
        Args:
            initial_capital: Initial trading capital
            max_position_size: Maximum position size as fraction of capital
            max_portfolio_risk: Maximum risk per trade as fraction of capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            trailing_stop_pct: Trailing stop percentage
            max_drawdown: Maximum allowed drawdown
            volatility_lookback: Lookback period for volatility calculation
            correlation_lookback: Lookback period for correlation calculation
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown = max_drawdown
        self.volatility_lookback = volatility_lookback
        self.correlation_lookback = correlation_lookback
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        # Risk metrics
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.daily_returns = []
        self.var_95 = 0.0
        self.var_99 = 0.0
        
        self.logger = logging.getLogger(__name__)
        self.portfolio_history: List[Dict] = []
        self.risk_alerts: List[Dict] = []
        
    def calculate_position_size(self, 
                              signal_strength: float,
                              confidence: float,
                              volatility: float,
                              price: float,
                              available_capital: float) -> float:
        """
        Calculate dynamic position size based on multiple factors.
        
        Args:
            signal_strength: Signal strength (-1 to 1)
            confidence: Confidence score (0 to 1)
            volatility: Current volatility
            price: Current price
            available_capital: Available capital for trading
            
        Returns:
            Position size in units
        """
        # Base position size from signal strength
        base_size = abs(signal_strength)
        
        # Adjust for confidence
        confidence_adjustment = confidence ** 0.5  # Square root for conservative adjustment
        
        # Volatility adjustment (inverse relationship)
        avg_volatility = 0.02  # 2% average volatility
        volatility_adjustment = avg_volatility / max(volatility, 0.001)
        volatility_adjustment = np.clip(volatility_adjustment, 0.5, 2.0)
        
        # Kelly criterion approximation
        win_rate = 0.55  # Estimated win rate
        avg_win = self.take_profit_pct
        avg_loss = self.stop_loss_pct
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = np.clip(kelly_fraction, 0.0, 0.25)  # Cap at 25%
        
        # Combine all factors
        position_fraction = (
            base_size * 
            confidence_adjustment * 
            volatility_adjustment * 
            kelly_fraction
        )
        
        # Apply maximum position size limit
        position_fraction = min(position_fraction, self.max_position_size)
        
        # Calculate position size in units
        position_size = (available_capital * position_fraction) / price
        
        return position_size
    
    def calculate_stop_loss(self, 
                           entry_price: float,
                           position_type: PositionType,
                           volatility: float,
                           atr: float) -> float:
        """
        Calculate dynamic stop loss.
        
        Args:
            entry_price: Entry price
            position_type: Position type (LONG/SHORT)
            volatility: Current volatility
            atr: Average True Range
            
        Returns:
            Stop loss price
        """
        # Base stop loss
        base_stop_distance = entry_price * self.stop_loss_pct
        
        # Volatility-adjusted stop loss
        vol_adjustment = volatility / 0.02  # Normalize to 2% volatility
        vol_adjustment = np.clip(vol_adjustment, 0.5, 3.0)
        
        # ATR-adjusted stop loss
        atr_adjustment = atr / entry_price
        atr_adjustment = np.clip(atr_adjustment, 0.005, 0.05)  # 0.5% to 5%
        
        # Use the larger of volatility-adjusted or ATR-adjusted
        stop_distance = max(base_stop_distance * vol_adjustment, entry_price * atr_adjustment)
        
        if position_type == PositionType.LONG:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(self,
                             entry_price: float,
                             position_type: PositionType,
                             volatility: float,
                             risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate dynamic take profit.
        
        Args:
            entry_price: Entry price
            position_type: Position type
            volatility: Current volatility
            risk_reward_ratio: Risk-reward ratio
            
        Returns:
            Take profit price
        """
        # Calculate stop loss first
        stop_loss = self.calculate_stop_loss(entry_price, position_type, volatility, volatility * entry_price)
        
        # Calculate risk
        if position_type == PositionType.LONG:
            risk = entry_price - stop_loss
        else:
            risk = stop_loss - entry_price
        
        # Calculate reward based on risk-reward ratio
        reward = risk * risk_reward_ratio
        
        # Volatility adjustment
        vol_adjustment = volatility / 0.02
        vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)
        reward *= vol_adjustment
        
        if position_type == PositionType.LONG:
            return entry_price + reward
        else:
            return entry_price - reward
    
    def calculate_trailing_stop(self,
                               position: Position,
                               current_price: float,
                               volatility: float) -> float:
        """
        Calculate trailing stop.
        
        Args:
            position: Current position
            current_price: Current price
            volatility: Current volatility
            
        Returns:
            New trailing stop price
        """
        if position.type == PositionType.LONG:
            # For long positions, trailing stop moves up
            if current_price > position.entry_price:
                # Calculate profit
                profit = current_price - position.entry_price
                
                # Dynamic trailing stop based on profit and volatility
                if profit > position.entry_price * 0.02:  # 2% profit
                    # Tighten trailing stop
                    trailing_distance = current_price * self.trailing_stop_pct * 0.5
                else:
                    # Normal trailing stop
                    trailing_distance = current_price * self.trailing_stop_pct
                
                # Volatility adjustment
                vol_adjustment = volatility / 0.02
                vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)
                trailing_distance *= vol_adjustment
                
                new_trailing_stop = current_price - trailing_distance
                
                # Only move trailing stop up
                return max(new_trailing_stop, position.trailing_stop)
            else:
                return position.trailing_stop
                
        else:
            # For short positions, trailing stop moves down
            if current_price < position.entry_price:
                # Calculate profit
                profit = position.entry_price - current_price
                
                # Dynamic trailing stop based on profit and volatility
                if profit > position.entry_price * 0.02:  # 2% profit
                    # Tighten trailing stop
                    trailing_distance = current_price * self.trailing_stop_pct * 0.5
                else:
                    # Normal trailing stop
                    trailing_distance = current_price * self.trailing_stop_pct
                
                # Volatility adjustment
                vol_adjustment = volatility / 0.02
                vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)
                trailing_distance *= vol_adjustment
                
                new_trailing_stop = current_price + trailing_distance
                
                # Only move trailing stop down
                return min(new_trailing_stop, position.trailing_stop)
            else:
                return position.trailing_stop
    
    def check_exit_signals(self,
                          position: Position,
                          current_price: float,
                          current_time: pd.Timestamp) -> Tuple[bool, str]:
        """
        Check if position should be exited.
        
        Args:
            position: Current position
            current_price: Current price
            current_time: Current timestamp
            
        Returns:
            Tuple of (should_exit, reason)
        """
        # Update position
        position.current_price = current_price
        
        # Calculate current P&L
        if position.type == PositionType.LONG:
            position.pnl = (current_price - position.entry_price) * position.size
        else:
            position.pnl = (position.entry_price - current_price) * position.size
        
        # Update max favorable/adverse excursion
        if position.pnl > position.max_favorable_excursion:
            position.max_favorable_excursion = position.pnl
        if position.pnl < position.max_adverse_excursion:
            position.max_adverse_excursion = position.pnl
        
        # Check stop loss
        if position.type == PositionType.LONG and current_price <= position.stop_loss:
            return True, "stop_loss"
        elif position.type == PositionType.SHORT and current_price >= position.stop_loss:
            return True, "stop_loss"
        
        # Check take profit
        if position.type == PositionType.LONG and current_price >= position.take_profit:
            return True, "take_profit"
        elif position.type == PositionType.SHORT and current_price <= position.take_profit:
            return True, "take_profit"
        
        # Check trailing stop
        if position.type == PositionType.LONG and current_price <= position.trailing_stop:
            return True, "trailing_stop"
        elif position.type == PositionType.SHORT and current_price >= position.trailing_stop:
            return True, "trailing_stop"
        
        # Check time-based exit (optional)
        holding_period = (current_time - position.entry_time).days
        if holding_period > 30:  # Exit after 30 days
            return True, "time_exit"
        
        return False, ""
    
    def enter_position(self,
                      symbol: str,
                      position_type: PositionType,
                      size: float,
                      entry_price: float,
                      entry_time: pd.Timestamp,
                      volatility: float,
                      atr: float) -> bool:
        """
        Enter a new position.
        
        Args:
            symbol: Trading symbol
            position_type: Position type
            size: Position size
            entry_price: Entry price
            entry_time: Entry time
            volatility: Current volatility
            atr: Average True Range
            
        Returns:
            True if position entered successfully
        """
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            return False
        
        # Check capital constraints
        required_capital = size * entry_price
        if required_capital > self.current_capital * self.max_position_size:
            return False
        
        # Calculate stop loss and take profit
        stop_loss = self.calculate_stop_loss(entry_price, position_type, volatility, atr)
        take_profit = self.calculate_take_profit(entry_price, position_type, volatility)
        trailing_stop = stop_loss  # Initial trailing stop equals stop loss
        
        # Create position
        position = Position(
            type=position_type,
            size=size,
            entry_price=entry_price,
            entry_time=entry_time,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            current_price=entry_price,
            pnl=0.0,
            max_favorable_excursion=0.0,
            max_adverse_excursion=0.0
        )
        
        # Add to positions
        self.positions[symbol] = position
        
        # Update capital
        self.current_capital -= required_capital
        
        return True
    
    def exit_position(self,
                     symbol: str,
                     exit_price: float,
                     exit_time: pd.Timestamp,
                     reason: str = "manual") -> Optional[Dict]:
        """
        Exit a position.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_time: Exit time
            reason: Exit reason
            
        Returns:
            Trade summary or None if no position
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate final P&L
        if position.type == PositionType.LONG:
            final_pnl = (exit_price - position.entry_price) * position.size
        else:
            final_pnl = (position.entry_price - exit_price) * position.size
        
        # Update capital
        self.current_capital += (position.size * exit_price) + final_pnl
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Calculate drawdown
        self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        # Create trade summary
        trade_summary = {
            'symbol': symbol,
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'position_type': position.type.value,
            'size': position.size,
            'pnl': final_pnl,
            'pnl_pct': final_pnl / (position.size * position.entry_price),
            'max_favorable_excursion': position.max_favorable_excursion,
            'max_adverse_excursion': position.max_adverse_excursion,
            'exit_reason': reason,
            'holding_period': (exit_time - position.entry_time).days
        }
        
        # Add to trade history
        self.trade_history.append(trade_summary)
        
        # Remove position
        del self.positions[symbol]
        
        return trade_summary
    
    def update_positions(self,
                        current_prices: Dict[str, float],
                        current_time: pd.Timestamp,
                        volatility: float,
                        atr: float) -> List[Dict]:
        """
        Update all positions and check for exits.
        
        Args:
            current_prices: Current prices for all symbols
            current_time: Current timestamp
            volatility: Current volatility
            atr: Average True Range
            
        Returns:
            List of exited trades
        """
        exited_trades = []
        
        for symbol, position in list(self.positions.items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                # Update trailing stop
                position.trailing_stop = self.calculate_trailing_stop(
                    position, current_price, volatility
                )
                
                # Check exit signals
                should_exit, reason = self.check_exit_signals(
                    position, current_price, current_time
                )
                
                if should_exit:
                    trade_summary = self.exit_position(symbol, current_price, current_time, reason)
                    if trade_summary:
                        exited_trades.append(trade_summary)
        
        return exited_trades
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        if not self.trade_history:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'var_95': 0.0,
                'var_99': 0.0,
                'calmar_ratio': 0.0
            }
        
        # Calculate daily returns
        if len(self.daily_returns) > 0:
            returns = pd.Series(self.daily_returns)
            
            # Basic metrics
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Value at Risk
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            # Calmar ratio
            calmar_ratio = total_return / self.current_drawdown if self.current_drawdown > 0 else 0
        else:
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            volatility = 0.0
            sharpe_ratio = 0.0
            var_95 = 0.0
            var_99 = 0.0
            calmar_ratio = total_return / self.current_drawdown if self.current_drawdown > 0 else 0
        
        # Trade-based metrics
        trades_df = pd.DataFrame(self.trade_history)
        
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.current_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'var_95': var_95,
            'var_99': var_99,
            'calmar_ratio': calmar_ratio,
            'current_capital': self.current_capital,
            'total_trades': len(self.trade_history)
        }
    
    def get_position_summary(self) -> Dict[str, Dict]:
        """
        Get summary of current positions.
        
        Returns:
            Dictionary of position summaries
        """
        summary = {}
        
        for symbol, position in self.positions.items():
            summary[symbol] = {
                'type': position.type.value,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'pnl': position.pnl,
                'pnl_pct': position.pnl / (position.size * position.entry_price),
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'trailing_stop': position.trailing_stop,
                'holding_period': (pd.Timestamp.now() - position.entry_time).days
            }
        
        return summary
    
    def reset(self):
        """Reset risk manager to initial state."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.position_history = []
        self.trade_history = []
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.daily_returns = []
        self.var_95 = 0.0
        self.var_99 = 0.0
    
    def calculate_portfolio_risk(self, portfolio_value: float, 
                               market_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics.
        
        Args:
            portfolio_value: Current portfolio value
            market_data: Market data for risk calculations
            
        Returns:
            Dictionary of risk metrics
        """
        risk_metrics = {}
        
        # Calculate current exposure
        total_exposure = sum(p.size * p.current_price for p in self.positions.values())
        risk_metrics['exposure'] = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate position concentration
        if self.positions:
            position_values = [abs(p.size * p.current_price) for p in self.positions.values()]
            risk_metrics['concentration'] = max(position_values) / total_exposure if total_exposure > 0 else 0
        
        # Calculate portfolio volatility if market data available
        if market_data is not None and len(self.portfolio_history) > 1:
            portfolio_returns = pd.Series([h['value'] for h in self.portfolio_history]).pct_change().dropna()
            if len(portfolio_returns) > 0:
                risk_metrics['volatility'] = portfolio_returns.std() * np.sqrt(252)
                
                # Calculate Value at Risk (95% confidence)
                risk_metrics['var_95'] = np.percentile(portfolio_returns, 5)
                
                # Calculate Conditional VaR (Expected Shortfall)
                var_threshold = risk_metrics['var_95']
                risk_metrics['cvar_95'] = portfolio_returns[portfolio_returns <= var_threshold].mean()
        
        # Calculate maximum drawdown
        if len(self.portfolio_history) > 1:
            portfolio_values = pd.Series([h['value'] for h in self.portfolio_history])
            rolling_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - rolling_max) / rolling_max
            risk_metrics['max_drawdown'] = drawdown.min()
            risk_metrics['current_drawdown'] = drawdown.iloc[-1]
        
        return risk_metrics
    
    def check_risk_limits(self, portfolio_value: float, 
                         risk_metrics: Dict[str, float]) -> List[Dict]:
        """
        Check if portfolio violates risk limits.
        
        Args:
            portfolio_value: Current portfolio value
            risk_metrics: Current risk metrics
            
        Returns:
            List of risk alerts
        """
        alerts = []
        
        # Check exposure limit
        if risk_metrics.get('exposure', 0) > self.max_position_size:
            alerts.append({
                'type': 'exposure_limit',
                'message': f"Portfolio exposure {risk_metrics['exposure']:.2%} exceeds limit {self.max_position_size:.2%}",
                'severity': 'high'
            })
        
        # Check drawdown limit
        if risk_metrics.get('current_drawdown', 0) < -self.max_drawdown:
            alerts.append({
                'type': 'drawdown_limit',
                'message': f"Current drawdown {risk_metrics['current_drawdown']:.2%} exceeds limit {self.max_drawdown:.2%}",
                'severity': 'critical'
            })
        
        # Check VaR limit
        if risk_metrics.get('var_95', 0) < -self.var_limit:
            alerts.append({
                'type': 'var_limit',
                'message': f"VaR {risk_metrics['var_95']:.2%} exceeds limit {self.var_limit:.2%}",
                'severity': 'high'
            })
        
        # Check daily loss limit
        if len(self.portfolio_history) >= 2:
            daily_return = (portfolio_value - self.portfolio_history[-2]['value']) / self.portfolio_history[-2]['value']
            if daily_return < -self.max_daily_loss:
                alerts.append({
                    'type': 'daily_loss_limit',
                    'message': f"Daily loss {daily_return:.2%} exceeds limit {self.max_daily_loss:.2%}",
                    'severity': 'critical'
                })
        
        # Store alerts
        self.risk_alerts.extend(alerts)
        
        return alerts
    
    def add_position(self, position: Position):
        """Add a new position."""
        self.positions[position.symbol] = position
    
    def remove_position(self, position: Position):
        """Remove a position."""
        if position.symbol in self.positions:
            del self.positions[position.symbol]
    
    def update_portfolio_history(self, portfolio_value: float, timestamp: datetime):
        """Update portfolio history."""
        self.portfolio_history.append({
            'value': portfolio_value,
            'timestamp': timestamp,
            'num_positions': len(self.positions)
        })
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        if not self.portfolio_history:
            return {'error': 'No portfolio history available'}
        
        current_value = self.portfolio_history[-1]['value']
        risk_metrics = self.calculate_portfolio_risk(current_value)
        
        report = {
            'timestamp': datetime.now(),
            'portfolio_value': current_value,
            'num_positions': len(self.positions),
            'risk_metrics': risk_metrics,
            'risk_limits': {
                'max_position_size': self.max_position_size,
                'max_portfolio_exposure': self.max_portfolio_exposure,
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown,
                'var_limit': self.var_limit
            },
            'positions': [
                {
                    'symbol': p.symbol,
                    'quantity': p.size,
                    'current_value': p.size * p.current_price,
                    'unrealized_pnl': p.size * (p.current_price - p.entry_price),
                    'confidence': p.confidence,
                    'strategy': p.strategy
                }
                for p in self.positions.values()
            ],
            'recent_alerts': self.risk_alerts[-10:] if self.risk_alerts else []
        }
        
        return report
    
    def save_risk_report(self, filename: str):
        """Save risk report to file."""
        import json
        
        report = self.get_risk_report()
        
        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        # Clean report for JSON serialization
        clean_report = json.loads(json.dumps(report, default=convert_datetime))
        
        with open(filename, 'w') as f:
            json.dump(clean_report, f, indent=2)
        
        self.logger.info(f"Risk report saved to {filename}") 