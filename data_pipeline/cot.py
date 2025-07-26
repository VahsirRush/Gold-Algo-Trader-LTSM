"""
CFTC Commitments of Traders (CoT) data collector for the gold trading framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import requests
from .base import BaseDataCollector

class CoTDataCollector(BaseDataCollector):
    """Collector for CFTC Commitments of Traders data."""
    
    def __init__(self):
        """Initialize CoT data collector."""
        super().__init__('cot', cache_duration=604800)  # 1 week cache (weekly data)
        
        # CFTC API endpoints
        self.base_url = "https://publicreporting.cftc.gov/resource"
        
        # Gold futures contract codes
        self.gold_contracts = {
            'GC': 'Gold Futures',
            'SI': 'Silver Futures',
            'PL': 'Platinum Futures',
            'PA': 'Palladium Futures'
        }
        
        # Report types
        self.report_types = {
            'F_ALL': 'All Reportable',
            'F_S': 'Short Form',
            'F_L': 'Long Form',
            'F_T': 'Traders'
        }
    
    def collect_cot_data(self, commodity_code: str = 'GC', 
                        report_type: str = 'F_ALL',
                        start_date: str = None,
                        end_date: str = None,
                        **kwargs) -> pd.DataFrame:
        """
        Collect CFTC CoT data.
        
        Args:
            commodity_code: Commodity code (GC for gold)
            report_type: Report type
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with CoT data
        """
        try:
            # Check cache first
            cache_filename = f"cot_{commodity_code}_{report_type}_{start_date}_{end_date}.parquet"
            cached_data = self.get_cached_data(cache_filename)
            if cached_data is not None:
                return cached_data
            
            self.logger.info(f"Collecting CoT data for {commodity_code}")
            
            # Build API URL
            url = f"{self.base_url}/{commodity_code}_{report_type}.json"
            
            # Add date filters if provided
            params = {}
            if start_date:
                params['$where'] = f"report_date >= '{start_date}'"
            if end_date:
                if '$where' in params:
                    params['$where'] += f" AND report_date <= '{end_date}'"
                else:
                    params['$where'] = f"report_date <= '{end_date}'"
            
            # Make request
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                self.logger.warning(f"No CoT data found for {commodity_code}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Clean and validate data
            df = self.clean_data(df)
            if not self.validate_data(df):
                return pd.DataFrame()
            
            # Save to cache
            self.save_to_cache(df, cache_filename)
            
            self.logger.info(f"Collected {len(df)} CoT records for {commodity_code}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting CoT data for {commodity_code}: {e}")
            return pd.DataFrame()
    
    def collect_gold_cot(self, start_date: str = None, 
                        end_date: str = None) -> pd.DataFrame:
        """
        Collect gold-specific CoT data.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with gold CoT data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        return self.collect_cot_data('GC', 'F_ALL', start_date, end_date)
    
    def calculate_positioning_metrics(self, cot_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate positioning metrics from CoT data.
        
        Args:
            cot_data: Raw CoT data
            
        Returns:
            DataFrame with calculated metrics
        """
        if cot_data.empty:
            return pd.DataFrame()
        
        try:
            # Convert date column
            if 'report_date' in cot_data.columns:
                cot_data['report_date'] = pd.to_datetime(cot_data['report_date'])
                cot_data.set_index('report_date', inplace=True)
            
            # Calculate positioning metrics
            metrics = pd.DataFrame(index=cot_data.index)
            
            # Net positions
            if 'noncomm_positions_long_all' in cot_data.columns and 'noncomm_positions_short_all' in cot_data.columns:
                metrics['net_speculative'] = (
                    cot_data['noncomm_positions_long_all'].astype(float) - 
                    cot_data['noncomm_positions_short_all'].astype(float)
                )
            
            if 'comm_positions_long_all' in cot_data.columns and 'comm_positions_short_all' in cot_data.columns:
                metrics['net_commercial'] = (
                    cot_data['comm_positions_long_all'].astype(float) - 
                    cot_data['comm_positions_short_all'].astype(float)
                )
            
            # Position ratios
            if 'noncomm_positions_long_all' in cot_data.columns and 'noncomm_positions_short_all' in cot_data.columns:
                long_pos = cot_data['noncomm_positions_long_all'].astype(float)
                short_pos = cot_data['noncomm_positions_short_all'].astype(float)
                metrics['spec_long_short_ratio'] = long_pos / short_pos
            
            # Open interest
            if 'open_interest_all' in cot_data.columns:
                metrics['open_interest'] = cot_data['open_interest_all'].astype(float)
            
            # Percentage of open interest
            if 'open_interest_all' in cot_data.columns:
                oi = cot_data['open_interest_all'].astype(float)
                if 'noncomm_positions_long_all' in cot_data.columns:
                    metrics['spec_long_pct_oi'] = (
                        cot_data['noncomm_positions_long_all'].astype(float) / oi * 100
                    )
                if 'noncomm_positions_short_all' in cot_data.columns:
                    metrics['spec_short_pct_oi'] = (
                        cot_data['noncomm_positions_short_all'].astype(float) / oi * 100
                    )
            
            # Momentum indicators
            if 'net_speculative' in metrics.columns:
                metrics['spec_momentum'] = metrics['net_speculative'].diff()
                metrics['spec_momentum_ma'] = metrics['net_speculative'].rolling(window=4).mean()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating positioning metrics: {e}")
            return pd.DataFrame()
    
    def get_extreme_positions(self, cot_data: pd.DataFrame, 
                            threshold: float = 0.8) -> Dict[str, Any]:
        """
        Identify extreme positioning levels.
        
        Args:
            cot_data: CoT data
            threshold: Percentile threshold for extreme positions
            
        Returns:
            Dictionary with extreme position indicators
        """
        if cot_data.empty:
            return {}
        
        try:
            metrics = self.calculate_positioning_metrics(cot_data)
            
            extreme_indicators = {}
            
            # Check for extreme speculative positions
            if 'net_speculative' in metrics.columns:
                net_spec = metrics['net_speculative'].dropna()
                if len(net_spec) > 0:
                    upper_threshold = net_spec.quantile(threshold)
                    lower_threshold = net_spec.quantile(1 - threshold)
                    
                    latest_net_spec = net_spec.iloc[-1]
                    
                    extreme_indicators['extreme_spec_long'] = latest_net_spec > upper_threshold
                    extreme_indicators['extreme_spec_short'] = latest_net_spec < lower_threshold
                    extreme_indicators['spec_position_percentile'] = (
                        (net_spec < latest_net_spec).mean() * 100
                    )
            
            # Check for extreme commercial positions
            if 'net_commercial' in metrics.columns:
                net_comm = metrics['net_commercial'].dropna()
                if len(net_comm) > 0:
                    upper_threshold = net_comm.quantile(threshold)
                    lower_threshold = net_comm.quantile(1 - threshold)
                    
                    latest_net_comm = net_comm.iloc[-1]
                    
                    extreme_indicators['extreme_comm_long'] = latest_net_comm > upper_threshold
                    extreme_indicators['extreme_comm_short'] = latest_net_comm < lower_threshold
            
            return extreme_indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating extreme positions: {e}")
            return {}
    
    def get_required_columns(self) -> list:
        """Get required columns for CoT data."""
        return ['report_date', 'open_interest_all']
    
    def _check_data_ranges(self, data: pd.DataFrame) -> bool:
        """Check if CoT data is within reasonable ranges."""
        if 'open_interest_all' in data.columns:
            oi_values = pd.to_numeric(data['open_interest_all'], errors='coerce').dropna()
            if len(oi_values) > 0:
                # Open interest should be positive and reasonable
                if oi_values.min() < 0 or oi_values.max() > 1000000:
                    self.logger.warning(f"Open interest values outside expected range: {oi_values.min()} - {oi_values.max()}")
                    return False
        
        return True
