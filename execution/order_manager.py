"""
Live execution order manager for the gold trading framework.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from threading import Thread, Lock
import queue

from .broker_adapter_base import BrokerAdapter, Order, Position, OrderType, OrderSide, OrderStatus
from risk_management.risk_manager import RiskManager, Position as RiskPosition

@dataclass
class Signal:
    """Trading signal from strategy."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    confidence: float
    strategy: str
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    order: Optional[Order] = None
    error_message: Optional[str] = None
    execution_time: Optional[datetime] = None

class OrderManager:
    """Live execution order manager."""
    
    def __init__(self, broker_adapter: BrokerAdapter, risk_manager: RiskManager,
                 config: Dict[str, Any]):
        """
        Initialize order manager.
        
        Args:
            broker_adapter: Broker adapter instance
            risk_manager: Risk manager instance
            config: Configuration dictionary
        """
        self.broker_adapter = broker_adapter
        self.risk_manager = risk_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Signal queue and processing
        self.signal_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.positions: Dict[str, Position] = {}
        
        # Thread safety
        self.lock = Lock()
        
        # Configuration
        self.max_position_size = config.get('MAX_POSITION_SIZE', 0.02)
        self.min_confidence = config.get('MIN_CONFIDENCE', 0.6)
        self.execution_delay = config.get('EXECUTION_DELAY', 1.0)  # seconds
        self.max_retries = config.get('MAX_RETRIES', 3)
        self.retry_delay = config.get('RETRY_DELAY', 5.0)  # seconds
        
        # Callbacks
        self.on_order_placed: Optional[Callable[[Order], None]] = None
        self.on_order_filled: Optional[Callable[[Order], None]] = None
        self.on_order_cancelled: Optional[Callable[[Order], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
    
    def start(self):
        """Start the order manager."""
        if self.running:
            self.logger.warning("Order manager is already running")
            return
        
        # Connect to broker
        if not self.broker_adapter.connect():
            self.logger.error("Failed to connect to broker")
            return
        
        # Start processing thread
        self.running = True
        self.processing_thread = Thread(target=self._process_signals, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Order manager started")
    
    def stop(self):
        """Stop the order manager."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all active orders
        self._cancel_all_orders()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10)
        
        # Disconnect from broker
        self.broker_adapter.disconnect()
        
        self.logger.info("Order manager stopped")
    
    def submit_signal(self, signal: Signal):
        """
        Submit a trading signal for execution.
        
        Args:
            signal: Trading signal
        """
        try:
            self.signal_queue.put(signal, timeout=1.0)
            self.logger.info(f"Signal submitted: {signal.symbol} {signal.side.value} "
                           f"(confidence: {signal.confidence:.3f})")
        except queue.Full:
            self.logger.error("Signal queue is full, dropping signal")
    
    def _process_signals(self):
        """Process signals in the background thread."""
        while self.running:
            try:
                # Get signal from queue
                signal = self.signal_queue.get(timeout=1.0)
                
                # Process signal
                self._execute_signal(signal)
                
                # Small delay to prevent overwhelming the broker
                time.sleep(self.execution_delay)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing signal: {e}")
                if self.on_error:
                    self.on_error(str(e))
    
    def _execute_signal(self, signal: Signal):
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
        """
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return
            
            # Check risk limits
            if not self._check_risk_limits(signal):
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            if position_size <= 0:
                self.logger.warning(f"Position size is zero for {signal.symbol}")
                return
            
            # Check if we already have a position
            current_position = self.broker_adapter.get_position(signal.symbol)
            
            # Determine order type and parameters
            order_type, price, stop_price = self._determine_order_parameters(signal)
            
            # Place order
            order = self.broker_adapter.place_order(
                symbol=signal.symbol,
                side=signal.side,
                order_type=order_type,
                quantity=position_size,
                price=price,
                stop_price=stop_price
            )
            
            if order:
                # Track order
                with self.lock:
                    self.active_orders[order.id] = order
                
                # Call callback
                if self.on_order_placed:
                    self.on_order_placed(order)
                
                self.logger.info(f"Order placed: {order.id} - {signal.side.value} "
                               f"{position_size} {signal.symbol} at {price or 'market'}")
            else:
                self.logger.error(f"Failed to place order for {signal.symbol}")
                
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    def _validate_signal(self, signal: Signal) -> bool:
        """Validate a trading signal."""
        if not signal.symbol or not signal.symbol.strip():
            self.logger.error("Invalid signal: missing symbol")
            return False
        
        if signal.confidence < self.min_confidence:
            self.logger.warning(f"Signal confidence too low: {signal.confidence:.3f} < {self.min_confidence}")
            return False
        
        if signal.side not in [OrderSide.BUY, OrderSide.SELL]:
            self.logger.error(f"Invalid signal side: {signal.side}")
            return False
        
        return True
    
    def _check_risk_limits(self, signal: Signal) -> bool:
        """Check risk limits before executing signal."""
        try:
            # Get account info
            account = self.broker_adapter.get_account()
            if not account:
                self.logger.error("Failed to get account info for risk check")
                return False
            
            # Get current price
            current_price = signal.price or self.broker_adapter.get_current_price(signal.symbol)
            if not current_price:
                self.logger.error(f"Failed to get current price for {signal.symbol}")
                return False
            
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            position_value = position_size * current_price
            
            # Create risk position for validation
            risk_position = RiskPosition(
                symbol=signal.symbol,
                quantity=position_size,
                entry_price=current_price,
                current_price=current_price,
                entry_time=datetime.now(),
                strategy=signal.strategy,
                confidence=signal.confidence
            )
            
            # Check with risk manager
            is_allowed, reason = self.risk_manager.check_position_limits(risk_position, account.portfolio_value)
            
            if not is_allowed:
                self.logger.warning(f"Risk limit check failed: {reason}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    def _calculate_position_size(self, signal: Signal) -> float:
        """Calculate position size based on signal and risk management."""
        try:
            # Get account info
            account = self.broker_adapter.get_account()
            if not account:
                return 0.0
            
            # Get current price
            current_price = signal.price or self.broker_adapter.get_current_price(signal.symbol)
            if not current_price:
                return 0.0
            
            # Calculate position size using risk manager
            position_size = self.risk_manager.calculate_position_size(
                capital=account.buying_power,
                price=current_price,
                confidence=signal.confidence
            )
            
            # Apply maximum position size limit
            max_position_value = account.portfolio_value * self.max_position_size
            max_position_size = max_position_value / current_price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _determine_order_parameters(self, signal: Signal) -> tuple:
        """
        Determine order type and parameters.
        
        Returns:
            Tuple of (order_type, price, stop_price)
        """
        # Default to market order
        order_type = OrderType.MARKET
        price = None
        stop_price = None
        
        # Use limit order if price is provided and confidence is high
        if signal.price and signal.confidence > 0.8:
            order_type = OrderType.LIMIT
            price = signal.price
        
        # Add stop loss if provided
        if signal.stop_loss:
            if order_type == OrderType.LIMIT:
                # Use stop-limit order
                order_type = OrderType.STOP_LIMIT
                stop_price = signal.stop_loss
            else:
                # Use stop order
                order_type = OrderType.STOP
                stop_price = signal.stop_loss
        
        return order_type, price, stop_price
    
    def _cancel_all_orders(self):
        """Cancel all active orders."""
        with self.lock:
            order_ids = list(self.active_orders.keys())
        
        for order_id in order_ids:
            try:
                if self.broker_adapter.cancel_order(order_id):
                    self.logger.info(f"Order cancelled: {order_id}")
                else:
                    self.logger.error(f"Failed to cancel order: {order_id}")
            except Exception as e:
                self.logger.error(f"Error cancelling order {order_id}: {e}")
    
    def update_order_status(self):
        """Update status of all active orders."""
        with self.lock:
            order_ids = list(self.active_orders.keys())
        
        for order_id in order_ids:
            try:
                order = self.broker_adapter.get_order(order_id)
                if order:
                    # Update order
                    with self.lock:
                        self.active_orders[order_id] = order
                    
                    # Check if order is filled or cancelled
                    if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                        # Remove from active orders
                        with self.lock:
                            self.active_orders.pop(order_id, None)
                        
                        # Add to history
                        self.order_history.append(order)
                        
                        # Call appropriate callback
                        if order.status == OrderStatus.FILLED and self.on_order_filled:
                            self.on_order_filled(order)
                        elif order.status == OrderStatus.CANCELLED and self.on_order_cancelled:
                            self.on_order_cancelled(order)
                        
                        self.logger.info(f"Order {order.status.value}: {order_id}")
                
            except Exception as e:
                self.logger.error(f"Error updating order {order_id}: {e}")
    
    def get_active_orders(self) -> List[Order]:
        """Get list of active orders."""
        with self.lock:
            return list(self.active_orders.values())
    
    def get_order_history(self) -> List[Order]:
        """Get order history."""
        return self.order_history.copy()
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        try:
            positions = self.broker_adapter.get_positions()
            self.positions = {pos.symbol: pos for pos in positions}
            return self.positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return self.positions
    
    def get_account_info(self):
        """Get account information."""
        return self.broker_adapter.get_account()
    
    def set_callbacks(self, on_order_placed: Optional[Callable] = None,
                     on_order_filled: Optional[Callable] = None,
                     on_order_cancelled: Optional[Callable] = None,
                     on_error: Optional[Callable] = None):
        """Set callback functions."""
        self.on_order_placed = on_order_placed
        self.on_order_filled = on_order_filled
        self.on_order_cancelled = on_order_cancelled
        self.on_error = on_error
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 