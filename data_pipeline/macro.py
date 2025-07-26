"""
Macroeconomic data collector for the gold trading framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
from .base import BaseDataCollector

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logging.warning("FRED API not available. Install fredapi for macroeconomic data.")

class MacroDataCollector(BaseDataCollector):
    """Collector for macroeconomic data."""
    
    def __init__(self, fred_api_key: str = None):
        """
        Initialize macroeconomic data collector.
        
        Args:
            fred_api_key: FRED API key
        """
        super().__init__('macro', cache_duration=3600)  # 1 hour cache
        
        self.fred_api_key = fred_api_key
        if fred_api_key and FRED_AVAILABLE:
            self.fred = Fred(api_key=fred_api_key)
        else:
            self.fred = None
            self.logger.warning("FRED API not configured")
        
        # Define key macroeconomic indicators
        self.indicators = {
            # Interest Rates
            'DGS10': '10-Year Treasury Constant Maturity Rate',
            'DGS2': '2-Year Treasury Constant Maturity Rate',
            'DFF': 'Federal Funds Effective Rate',
            'DGS30': '30-Year Treasury Constant Maturity Rate',
            
            # Inflation
            'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
            'PCEPI': 'Personal Consumption Expenditures Price Index',
            'CPILFESL': 'Consumer Price Index for All Urban Consumers: All Items Less Food & Energy',
            
            # Economic Activity
            'GDP': 'Gross Domestic Product',
            'UNRATE': 'Unemployment Rate',
            'PAYEMS': 'Total Nonfarm Payrolls',
            'INDPRO': 'Industrial Production: Total Index',
            
            # Currency
            'DEXUSEU': 'U.S. / Euro Foreign Exchange Rate',
            'DEXCHUS': 'China / U.S. Foreign Exchange Rate',
            'DGS10': '10-Year Treasury Constant Maturity Rate',
            
            # Commodities
            'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate',
            'DCOILBRENTEU': 'Crude Oil Prices: Brent Europe',
            
            # Gold-specific
            'GOLDPMGBD228NLBM': 'Gold Fixing Price 3:00 P.M. (London time) in London Bullion Market',
        }
    
    def collect_data(self, series_id: str, start_date: str = None, 
                    end_date: str = None, **kwargs) -> pd.DataFrame:
        """
        Collect macroeconomic data from FRED.
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with macroeconomic data
        """
        if not self.fred:
            self.logger.error("FRED API not available")
            return pd.DataFrame()
        
        try:
            # Check cache first
            cache_filename = f"{series_id}_{start_date}_{end_date}.parquet"
            cached_data = self.get_cached_data(cache_filename)
            if cached_data is not None:
                return cached_data
            
            self.logger.info(f"Collecting {series_id} data")
            
            # Get data from FRED
            data = self.fred.get_series(series_id, start=start_date, end=end_date)
            
            if data.empty:
                self.logger.warning(f"No data collected for {series_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['value'])
            df.index.name = 'date'
            df.reset_index(inplace=True)
            
            # Clean and validate data
            df = self.clean_data(df)
            if not self.validate_data(df):
                return pd.DataFrame()
            
            # Save to cache
            self.save_to_cache(df, cache_filename)
            
            self.logger.info(f"Collected {len(df)} records for {series_id}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting {series_id} data: {e}")
            return pd.DataFrame()
    
    def collect_key_indicators(self, start_date: str = None, 
                             end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Collect data for key macroeconomic indicators.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of indicator -> DataFrame mappings
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        results = {}
        for series_id, description in self.indicators.items():
            try:
                data = self.collect_data(series_id, start_date, end_date)
                if not data.empty:
                    results[series_id] = data
                    self.logger.info(f"Collected {description}: {series_id}")
            except Exception as e:
                self.logger.error(f"Error collecting {series_id}: {e}")
        
        return results
    
    def calculate_gold_drivers(self, gold_data: pd.DataFrame, 
                             macro_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate gold-specific macroeconomic drivers.
        
        Args:
            gold_data: Gold price data
            macro_data: Macroeconomic indicators
            
        Returns:
            DataFrame with calculated drivers
        """
        drivers = pd.DataFrame(index=gold_data.index)
        
        try:
            # Real interest rates (if we have inflation data)
            if 'DGS10' in macro_data and 'CPIAUCSL' in macro_data:
                nominal_rate = macro_data['DGS10'].set_index('date')['value']
                inflation = macro_data['CPIAUCSL'].set_index('date')['value']
                
                # Calculate real rate (simplified)
                real_rate = nominal_rate - inflation.pct_change(12) * 100
                drivers['real_interest_rate'] = real_rate
            
            # Dollar strength (if we have DXY data)
            if 'DEXUSEU' in macro_data:
                dollar_strength = macro_data['DEXUSEU'].set_index('date')['value']
                drivers['dollar_strength'] = dollar_strength
            
            # Economic uncertainty (unemployment rate)
            if 'UNRATE' in macro_data:
                unemployment = macro_data['UNRATE'].set_index('date')['value']
                drivers['unemployment_rate'] = unemployment
            
            # Oil prices (commodity correlation)
            if 'DCOILWTICO' in macro_data:
                oil_prices = macro_data['DCOILWTICO'].set_index('date')['value']
                drivers['oil_prices'] = oil_prices
            
        except Exception as e:
            self.logger.error(f"Error calculating gold drivers: {e}")
        
        return drivers
    
    def get_required_columns(self) -> list:
        """Get required columns for macroeconomic data."""
        return ['date', 'value']
    
    def _check_data_ranges(self, data: pd.DataFrame) -> bool:
        """Check if macroeconomic data is within reasonable ranges."""
        if 'value' in data.columns:
            values = data['value'].dropna()
            if len(values) > 0:
                # Check for extreme outliers
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                if len(outliers) > len(values) * 0.1:  # More than 10% outliers
                    self.logger.warning(f"Too many outliers in data: {len(outliers)} out of {len(values)}")
                    return False
        
        return True
