"""
Gold spot data collector for the trading framework.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from .base import BaseDataCollector

class GoldSpotCollector(BaseDataCollector):
    """Collector for gold spot price data."""
    
    def __init__(self):
        """Initialize gold spot collector."""
        super().__init__('gold_spot', cache_duration=300)  # 5 minutes cache
        self.symbols = {
            'GC=F': 'Gold Futures',
            'GLD': 'SPDR Gold Trust',
            'IAU': 'iShares Gold Trust',
            'SGOL': 'Aberdeen Standard Physical Gold ETF',
            'GLDM': 'SPDR Gold MiniShares Trust',
            'BAR': 'GraniteShares Gold Trust'
        }
    
    def collect_data(self, symbol: str = 'GC=F', period: str = '1y', 
                    interval: str = '1d', **kwargs) -> pd.DataFrame:
        """
        Collect gold spot data.
        
        Args:
            symbol: Gold symbol to collect
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with gold price data
        """
        try:
            # Check cache first
            cache_filename = f"{symbol}_{period}_{interval}.parquet"
            cached_data = self.get_cached_data(cache_filename)
            if cached_data is not None:
                return cached_data
            
            self.logger.info(f"Collecting {symbol} data for period {period}")
            
            # Download data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No data collected for {symbol}")
                return pd.DataFrame()
            
            # Clean and validate data
            data = self.clean_data(data)
            if not self.validate_data(data):
                return pd.DataFrame()
            
            # Save to cache
            self.save_to_cache(data, cache_filename)
            
            self.logger.info(f"Collected {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error collecting {symbol} data: {e}")
            return pd.DataFrame()
    
    def collect_multiple_symbols(self, symbols: list = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple gold symbols.
        
        Args:
            symbols: List of symbols to collect
            
        Returns:
            Dictionary of symbol -> DataFrame mappings
        """
        if symbols is None:
            symbols = list(self.symbols.keys())
        
        results = {}
        for symbol in symbols:
            try:
                data = self.collect_data(symbol=symbol, **kwargs)
                if not data.empty:
                    results[symbol] = data
            except Exception as e:
                self.logger.error(f"Error collecting {symbol}: {e}")
        
        return results
    
    def get_latest_price(self, symbol: str = 'GC=F') -> Optional[float]:
        """
        Get the latest gold price.
        
        Args:
            symbol: Gold symbol
            
        Returns:
            Latest price or None
        """
        try:
            data = self.collect_data(symbol=symbol, period='5d', interval='1d')
            if not data.empty:
                return data['Close'].iloc[-1]
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
        
        return None
    
    def get_required_columns(self) -> list:
        """Get required columns for gold price data."""
        return ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def _check_data_ranges(self, data: pd.DataFrame) -> bool:
        """Check if gold price data is within reasonable ranges."""
        if 'Close' in data.columns:
            close_prices = data['Close'].dropna()
            if len(close_prices) > 0:
                # Gold prices should be between $100 and $10,000
                if close_prices.min() < 100 or close_prices.max() > 10000:
                    self.logger.warning(f"Gold prices outside expected range: {close_prices.min()} - {close_prices.max()}")
                    return False
        
        return True
