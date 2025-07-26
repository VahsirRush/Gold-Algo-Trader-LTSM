"""
Abstract base class for broker adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class Order:
    """Order information."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    commission: float = 0.0

@dataclass
class Position:
    """Position information."""
    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    cost_basis: float

@dataclass
class Account:
    """Account information."""
    account_id: str
    cash: float
    buying_power: float
    equity: float
    market_value: float
    day_trade_count: int
    pattern_day_trader: bool
    portfolio_value: float
    regt_buying_power: float
    regt_selling_power: float
    long_market_value: float
    short_market_value: float
    initial_margin: float
    maintenance_margin: float
    last_equity: float
    last_maintenance_margin: float

class BrokerAdapter(ABC):
    """Abstract base class for broker adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize broker adapter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.connected = False
        self.logger = None  # Will be set by subclasses
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the broker.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to broker.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def get_account(self) -> Optional[Account]:
        """
        Get account information.
        
        Returns:
            Account information or None if failed
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get current positions.
        
        Returns:
            List of positions
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position information or None if not found
        """
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                   quantity: float, price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> Optional[Order]:
        """
        Place an order.
        
        Args:
            symbol: Symbol to trade
            side: Buy or sell
            order_type: Type of order
            quantity: Number of shares/contracts
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            
        Returns:
            Order object or None if failed
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order information.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object or None if not found
        """
        pass
    
    @abstractmethod
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get list of orders.
        
        Args:
            status: Filter by status (optional)
            
        Returns:
            List of orders
        """
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None, interval: str = '1D') -> Optional[Dict[str, Any]]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            start_date: Start date (optional)
            end_date: End date (optional)
            interval: Data interval (e.g., '1D', '1H', '1m')
            
        Returns:
            Market data dictionary or None if failed
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price or None if failed
        """
        pass
    
    def validate_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                      quantity: float, price: Optional[float] = None,
                      stop_price: Optional[float] = None) -> Tuple[bool, str]:
        """
        Validate order parameters.
        
        Args:
            symbol: Symbol to trade
            side: Buy or sell
            order_type: Type of order
            quantity: Number of shares/contracts
            price: Limit price
            stop_price: Stop price
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic validation
        if not symbol or not symbol.strip():
            return False, "Symbol is required"
        
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if order_type == OrderType.LIMIT and price is None:
            return False, "Price is required for limit orders"
        
        if order_type == OrderType.STOP and stop_price is None:
            return False, "Stop price is required for stop orders"
        
        if order_type == OrderType.STOP_LIMIT and (price is None or stop_price is None):
            return False, "Both price and stop price are required for stop-limit orders"
        
        return True, ""
    
    def calculate_commission(self, order_value: float) -> float:
        """
        Calculate commission for an order.
        
        Args:
            order_value: Total order value
            
        Returns:
            Commission amount
        """
        # Default commission calculation (can be overridden by subclasses)
        return max(1.0, order_value * 0.001)  # $1 minimum or 0.1%
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect() 