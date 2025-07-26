"""
Base data collector class for the gold trading framework.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path

class BaseDataCollector(ABC):
    """Base class for all data collectors."""
    
    def __init__(self, name: str, cache_duration: int = 3600):
        """
        Initialize base data collector.
        
        Args:
            name: Name of the data collector
            cache_duration: Cache duration in seconds
        """
        self.name = name
        self.cache_duration = cache_duration
        self.logger = logging.getLogger(f"data_collector.{name}")
        
        # Create cache directory
        self.cache_dir = Path('cache') / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def collect_data(self, **kwargs) -> pd.DataFrame:
        """
        Collect data from the source.
        
        Returns:
            DataFrame with collected data
        """
        pass
    
    def save_to_cache(self, data: pd.DataFrame, filename: str):
        """Save data to cache."""
        filepath = self.cache_dir / filename
        data.to_parquet(filepath)
        self.logger.info(f"Data saved to cache: {filepath}")
    
    def load_from_cache(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        filepath = self.cache_dir / filename
        if filepath.exists():
            data = pd.read_parquet(filepath)
            self.logger.info(f"Data loaded from cache: {filepath}")
            return data
        return None
    
    def is_cache_valid(self, filename: str) -> bool:
        """Check if cached data is still valid."""
        filepath = self.cache_dir / filename
        if not filepath.exists():
            return False
        
        file_time = filepath.stat().st_mtime
        current_time = datetime.now().timestamp()
        
        return (current_time - file_time) < self.cache_duration
    
    def get_cached_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Get cached data if valid."""
        if self.is_cache_valid(filename):
            return self.load_from_cache(filename)
        return None
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.
        
        Args:
            data: Raw data to clean
            
        Returns:
            Cleaned data
        """
        if data.empty:
            return data
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Sort by index if it's a datetime
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        
        # Forward fill missing values for time series data
        data = data.fillna(method='ffill')
        
        return data
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid
        """
        if data.empty:
            self.logger.warning("Data is empty")
            return False
        
        # Check for required columns
        required_columns = self.get_required_columns()
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for reasonable data ranges
        if not self._check_data_ranges(data):
            return False
        
        return True
    
    @abstractmethod
    def get_required_columns(self) -> list:
        """Get list of required columns for this data type."""
        pass
    
    def _check_data_ranges(self, data: pd.DataFrame) -> bool:
        """Check if data values are within reasonable ranges."""
        # Override in subclasses for specific validation
        return True
