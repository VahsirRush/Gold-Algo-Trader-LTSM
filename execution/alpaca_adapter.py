"""
Alpaca broker adapter implementation.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from .broker_adapter_base import (
    BrokerAdapter, Order, Position, Account, OrderType, OrderSide, OrderStatus
)

class AlpacaAdapter(BrokerAdapter):
    """Alpaca broker adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Alpaca adapter.
        
        Args:
            config: Configuration dictionary with Alpaca credentials
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca SDK not available. Install with: pip install alpaca-py")
        
        # Extract configuration
        self.api_key = config.get('ALPACA_API_KEY')
        self.secret_key = config.get('ALPACA_SECRET_KEY')
        self.paper = config.get('ALPACA_PAPER', True)  # Default to paper trading
        self.base_url = config.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets' if self.paper else 'https://api.alpaca.markets')
        
        # Initialize clients
        self.trading_client = None
        self.data_client = None
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret key are required")
    
    def connect(self) -> bool:
        """Connect to Alpaca."""
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper
            )
            
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            # Test connection by getting account info
            account = self.trading_client.get_account()
            if account:
                self.connected = True
                self.logger.info(f"Connected to Alpaca {'Paper' if self.paper else 'Live'} Trading")
                return True
            else:
                self.logger.error("Failed to get account info from Alpaca")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Alpaca."""
        try:
            self.trading_client = None
            self.data_client = None
            self.connected = False
            self.logger.info("Disconnected from Alpaca")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from Alpaca: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to Alpaca."""
        return self.connected and self.trading_client is not None
    
    def get_account(self) -> Optional[Account]:
        """Get Alpaca account information."""
        try:
            if not self.is_connected():
                return None
            
            alpaca_account = self.trading_client.get_account()
            
            return Account(
                account_id=alpaca_account.id,
                cash=float(alpaca_account.cash),
                buying_power=float(alpaca_account.buying_power),
                equity=float(alpaca_account.equity),
                market_value=float(alpaca_account.market_value),
                day_trade_count=alpaca_account.daytrade_count,
                pattern_day_trader=alpaca_account.pattern_day_trader,
                portfolio_value=float(alpaca_account.portfolio_value),
                regt_buying_power=float(alpaca_account.regt_buying_power),
                regt_selling_power=float(alpaca_account.regt_selling_power),
                long_market_value=float(alpaca_account.long_market_value),
                short_market_value=float(alpaca_account.short_market_value),
                initial_margin=float(alpaca_account.initial_margin),
                maintenance_margin=float(alpaca_account.maintenance_margin),
                last_equity=float(alpaca_account.last_equity),
                last_maintenance_margin=float(alpaca_account.last_maintenance_margin)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return None
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        try:
            if not self.is_connected():
                return []
            
            alpaca_positions = self.trading_client.get_all_positions()
            positions = []
            
            for pos in alpaca_positions:
                positions.append(Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    average_price=float(pos.avg_entry_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    realized_pnl=0.0,  # Alpaca doesn't provide this directly
                    cost_basis=float(pos.cost_basis)
                ))
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        try:
            if not self.is_connected():
                return None
            
            alpaca_position = self.trading_client.get_position(symbol)
            
            return Position(
                symbol=alpaca_position.symbol,
                quantity=float(alpaca_position.qty),
                average_price=float(alpaca_position.avg_entry_price),
                market_value=float(alpaca_position.market_value),
                unrealized_pnl=float(alpaca_position.unrealized_pl),
                realized_pnl=0.0,
                cost_basis=float(alpaca_position.cost_basis)
            )
            
        except Exception as e:
            # Position not found or other error
            return None
    
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                   quantity: float, price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> Optional[Order]:
        """Place an order on Alpaca."""
        try:
            if not self.is_connected():
                return None
            
            # Validate order
            is_valid, error_msg = self.validate_order(symbol, side, order_type, quantity, price, stop_price)
            if not is_valid:
                self.logger.error(f"Invalid order: {error_msg}")
                return None
            
            # Convert to Alpaca order side
            alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL
            
            # Create order request based on type
            if order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY
                )
            elif order_type == OrderType.LIMIT:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=price
                )
            elif order_type == OrderType.STOP:
                order_request = StopOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=stop_price
                )
            elif order_type == OrderType.STOP_LIMIT:
                order_request = StopLimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=price,
                    stop_price=stop_price
                )
            else:
                self.logger.error(f"Unsupported order type: {order_type}")
                return None
            
            # Submit order
            alpaca_order = self.trading_client.submit_order(order_request)
            
            # Convert to our Order format
            order = Order(
                id=alpaca_order.id,
                symbol=alpaca_order.symbol,
                side=side,
                order_type=order_type,
                quantity=float(alpaca_order.qty),
                price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                status=self._convert_order_status(alpaca_order.status),
                filled_quantity=float(alpaca_order.filled_qty),
                filled_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                created_at=alpaca_order.created_at,
                filled_at=alpaca_order.filled_at
            )
            
            self.logger.info(f"Order placed: {order.id} - {side.value} {quantity} {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            if not self.is_connected():
                return False
            
            self.trading_client.cancel_order_by_id(order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order information."""
        try:
            if not self.is_connected():
                return None
            
            alpaca_order = self.trading_client.get_order_by_id(order_id)
            
            return Order(
                id=alpaca_order.id,
                symbol=alpaca_order.symbol,
                side=OrderSide.BUY if alpaca_order.side == AlpacaOrderSide.BUY else OrderSide.SELL,
                order_type=self._convert_order_type(alpaca_order.order_type),
                quantity=float(alpaca_order.qty),
                price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                status=self._convert_order_status(alpaca_order.status),
                filled_quantity=float(alpaca_order.filled_qty),
                filled_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                created_at=alpaca_order.created_at,
                filled_at=alpaca_order.filled_at
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get order {order_id}: {e}")
            return None
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get list of orders."""
        try:
            if not self.is_connected():
                return []
            
            # Convert status filter
            alpaca_status = None
            if status:
                alpaca_status = self._convert_status_to_alpaca(status)
            
            alpaca_orders = self.trading_client.get_orders(status=alpaca_status)
            orders = []
            
            for alpaca_order in alpaca_orders:
                order = Order(
                    id=alpaca_order.id,
                    symbol=alpaca_order.symbol,
                    side=OrderSide.BUY if alpaca_order.side == AlpacaOrderSide.BUY else OrderSide.SELL,
                    order_type=self._convert_order_type(alpaca_order.order_type),
                    quantity=float(alpaca_order.qty),
                    price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                    stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                    status=self._convert_order_status(alpaca_order.status),
                    filled_quantity=float(alpaca_order.filled_qty),
                    filled_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                    created_at=alpaca_order.created_at,
                    filled_at=alpaca_order.filled_at
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Failed to get orders: {e}")
            return []
    
    def get_market_data(self, symbol: str, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None, interval: str = '1D') -> Optional[Dict[str, Any]]:
        """Get market data from Alpaca."""
        try:
            if not self.is_connected():
                return None
            
            # Convert interval to Alpaca TimeFrame
            timeframe_map = {
                '1m': TimeFrame.Minute,
                '5m': TimeFrame.Minute5,
                '15m': TimeFrame.Minute15,
                '30m': TimeFrame.Minute30,
                '1H': TimeFrame.Hour,
                '1D': TimeFrame.Day
            }
            
            alpaca_timeframe = timeframe_map.get(interval, TimeFrame.Day)
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_timeframe,
                start=start_date,
                end=end_date
            )
            
            # Get data
            bars = self.data_client.get_stock_bars(request)
            
            # Convert to DataFrame
            if bars and symbol in bars:
                df = bars[symbol].df
                return {
                    'data': df,
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date,
                    'interval': interval
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            # Get latest bar
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            market_data = self.get_market_data(symbol, start_date, end_date, '1D')
            if market_data and not market_data['data'].empty:
                return float(market_data['data']['close'].iloc[-1])
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    def _convert_order_status(self, alpaca_status: str) -> OrderStatus:
        """Convert Alpaca order status to our format."""
        status_map = {
            'new': OrderStatus.PENDING,
            'accepted': OrderStatus.PENDING,
            'pending_new': OrderStatus.PENDING,
            'pending_cancel': OrderStatus.PENDING,
            'pending_replace': OrderStatus.PENDING,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.EXPIRED
        }
        return status_map.get(alpaca_status, OrderStatus.PENDING)
    
    def _convert_status_to_alpaca(self, status: OrderStatus) -> str:
        """Convert our order status to Alpaca format."""
        status_map = {
            OrderStatus.PENDING: 'new',
            OrderStatus.PARTIALLY_FILLED: 'partially_filled',
            OrderStatus.FILLED: 'filled',
            OrderStatus.CANCELLED: 'canceled',
            OrderStatus.REJECTED: 'rejected',
            OrderStatus.EXPIRED: 'expired'
        }
        return status_map.get(status, 'new')
    
    def _convert_order_type(self, alpaca_order_type: str) -> OrderType:
        """Convert Alpaca order type to our format."""
        type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop': OrderType.STOP,
            'stop_limit': OrderType.STOP_LIMIT
        }
        return type_map.get(alpaca_order_type, OrderType.MARKET) 