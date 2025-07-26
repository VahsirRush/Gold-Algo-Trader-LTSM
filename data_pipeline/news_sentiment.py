"""
News and sentiment data collector for the gold trading framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import requests
from .base import BaseDataCollector

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Install textblob for sentiment analysis.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER not available. Install vaderSentiment for sentiment analysis.")

class NewsSentimentCollector(BaseDataCollector):
    """Collector for news and sentiment data."""
    
    def __init__(self, news_api_key: str = None):
        """
        Initialize news sentiment collector.
        
        Args:
            news_api_key: News API key
        """
        super().__init__('news_sentiment', cache_duration=1800)  # 30 minutes cache
        
        self.news_api_key = news_api_key
        self.base_url = "https://newsapi.org/v2"
        
        # Initialize sentiment analyzers
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        else:
            self.vader_analyzer = None
        
        # Gold-related keywords
        self.gold_keywords = [
            'gold', 'precious metals', 'bullion', 'XAU/USD', 'GLD', 'IAU',
            'gold mining', 'gold price', 'gold market', 'gold trading',
            'central bank gold', 'gold reserves', 'gold ETF'
        ]
        
        # Economic keywords that affect gold
        self.economic_keywords = [
            'inflation', 'interest rates', 'federal reserve', 'fed',
            'dollar', 'USD', 'treasury', 'bonds', 'yields',
            'economic crisis', 'recession', 'geopolitical', 'trade war'
        ]
    
    def collect_news(self, query: str = 'gold', language: str = 'en',
                    sort_by: str = 'publishedAt', page_size: int = 100,
                    **kwargs) -> pd.DataFrame:
        """
        Collect news articles from News API.
        
        Args:
            query: Search query
            language: Language code
            sort_by: Sort method (relevancy, popularity, publishedAt)
            page_size: Number of articles per request
            
        Returns:
            DataFrame with news data
        """
        if not self.news_api_key:
            self.logger.error("News API key not configured")
            return pd.DataFrame()
        
        try:
            # Check cache first
            cache_filename = f"news_{query}_{datetime.now().strftime('%Y%m%d')}.parquet"
            cached_data = self.get_cached_data(cache_filename)
            if cached_data is not None:
                return cached_data
            
            self.logger.info(f"Collecting news for query: {query}")
            
            # Prepare API request
            url = f"{self.base_url}/everything"
            params = {
                'q': query,
                'language': language,
                'sortBy': sort_by,
                'pageSize': page_size,
                'apiKey': self.news_api_key
            }
            
            # Make request
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'ok':
                self.logger.error(f"News API error: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()
            
            # Process articles
            articles = data.get('articles', [])
            if not articles:
                self.logger.warning("No articles found")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(articles)
            
            # Clean and validate data
            df = self.clean_data(df)
            if not self.validate_data(df):
                return pd.DataFrame()
            
            # Save to cache
            self.save_to_cache(df, cache_filename)
            
            self.logger.info(f"Collected {len(df)} news articles")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting news: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using multiple methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        sentiment_scores = {}
        
        # VADER sentiment analysis
        if self.vader_analyzer:
            try:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                sentiment_scores.update({
                    'vader_compound': vader_scores['compound'],
                    'vader_positive': vader_scores['pos'],
                    'vader_negative': vader_scores['neg'],
                    'vader_neutral': vader_scores['neu']
                })
            except Exception as e:
                self.logger.error(f"VADER sentiment analysis error: {e}")
        
        # TextBlob sentiment analysis
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                sentiment_scores.update({
                    'textblob_polarity': blob.sentiment.polarity,
                    'textblob_subjectivity': blob.sentiment.subjectivity
                })
            except Exception as e:
                self.logger.error(f"TextBlob sentiment analysis error: {e}")
        
        return sentiment_scores
    
    def collect_gold_sentiment(self, days: int = 7) -> pd.DataFrame:
        """
        Collect and analyze gold-related news sentiment.
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame with sentiment data
        """
        try:
            # Collect gold-related news
            gold_news = self.collect_news(query='gold AND (price OR market OR trading)')
            
            if gold_news.empty:
                return pd.DataFrame()
            
            # Analyze sentiment for each article
            sentiment_data = []
            for _, article in gold_news.iterrows():
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                
                # Combine text for analysis
                text = f"{title} {description} {content}"
                
                # Analyze sentiment
                sentiment_scores = self.analyze_sentiment(text)
                
                # Add article metadata
                sentiment_data.append({
                    'published_at': article.get('publishedAt'),
                    'title': title,
                    'source': article.get('source', {}).get('name'),
                    'url': article.get('url'),
                    **sentiment_scores
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(sentiment_data)
            
            # Convert published_at to datetime
            if 'published_at' in df.columns:
                df['published_at'] = pd.to_datetime(df['published_at'])
                df.set_index('published_at', inplace=True)
            
            # Calculate daily sentiment aggregates
            daily_sentiment = self._calculate_daily_sentiment(df)
            
            return daily_sentiment
            
        except Exception as e:
            self.logger.error(f"Error collecting gold sentiment: {e}")
            return pd.DataFrame()
    
    def _calculate_daily_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily sentiment aggregates.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            
        Returns:
            DataFrame with daily sentiment metrics
        """
        if sentiment_df.empty:
            return pd.DataFrame()
        
        # Resample to daily frequency and calculate aggregates
        daily_metrics = {}
        
        # VADER metrics
        vader_columns = [col for col in sentiment_df.columns if col.startswith('vader_')]
        for col in vader_columns:
            daily_metrics[f'{col}_mean'] = sentiment_df[col].resample('D').mean()
            daily_metrics[f'{col}_std'] = sentiment_df[col].resample('D').std()
        
        # TextBlob metrics
        textblob_columns = [col for col in sentiment_df.columns if col.startswith('textblob_')]
        for col in textblob_columns:
            daily_metrics[f'{col}_mean'] = sentiment_df[col].resample('D').mean()
            daily_metrics[f'{col}_std'] = sentiment_df[col].resample('D').std()
        
        # Article count
        daily_metrics['article_count'] = sentiment_df.resample('D').size()
        
        # Create DataFrame
        daily_df = pd.DataFrame(daily_metrics)
        
        return daily_df
    
    def get_required_columns(self) -> list:
        """Get required columns for news data."""
        return ['title', 'description', 'publishedAt']
    
    def _check_data_ranges(self, data: pd.DataFrame) -> bool:
        """Check if news data is within reasonable ranges."""
        # Check for required columns
        required_cols = self.get_required_columns()
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        return True
