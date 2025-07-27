import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import databento - API key should be set via environment variable
import databento as db
import pandas as pd
import yfinance as yf

class DatabentoGoldCollector:
    """Hybrid collector for GLD data - tries Databento first, falls back to yfinance."""
    def __init__(self, api_key: str = None):
        # API key is already set in the environment
        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            raise ValueError("Databento API key must be provided or set as DATABENTO_API_KEY.")
        self.client = db.Historical()

    def discover_datasets(self):
        """List available datasets and their descriptions."""
        print("[Databento] Discovering available datasets...")
        try:
            # Try to get dataset list
            datasets = self.client.metadata.list_datasets()
            print(f"[Databento] Found {len(datasets)} datasets:")
            for dataset in datasets:
                print(f"  - {dataset}")
            return datasets
        except Exception as e:
            print(f"[Databento] Error listing datasets: {e}")
            return []

    def search_gold_symbols(self, dataset: str = "GLBX.MDP3"):
        """Search for gold-related symbols in a dataset."""
        print(f"[Databento] Searching for gold symbols in {dataset}...")
        
        # Common gold symbol patterns to try
        gold_patterns = [
            "GC", "GOLD", "XAU", "GLD", "IAU", "SGOL", "GLDM", "BAR",
            "GCZ3", "GCM4", "GCQ4", "GCV4",  # Common futures contract codes
            "GC=F", "XAUUSD", "XAU=X"  # Common forex/spot symbols
        ]
        
        working_symbols = []
        
        for symbol in gold_patterns:
            try:
                print(f"[Databento] Testing symbol: {symbol}")
                result = self.client.timeseries.get_range(
                    dataset=dataset,
                    symbols=[symbol],
                    schema="ohlcv-1d",
                    start="2024-01-01",
                    end="2024-01-10",  # Short test period
                    limit=10
                )
                
                if hasattr(result, 'to_df'):
                    df = result.to_df()
                    if not df.empty:
                        print(f"  ✅ {symbol} - Found {len(df)} rows")
                        working_symbols.append(symbol)
                    else:
                        print(f"  ❌ {symbol} - No data")
                else:
                    print(f"  ❌ {symbol} - No DataFrame conversion")
                    
            except Exception as e:
                print(f"  ❌ {symbol} - Error: {str(e)[:100]}...")
        
        return working_symbols

    def test_datasets_for_gold(self):
        """Test multiple datasets to find gold data."""
        print("[Databento] Testing multiple datasets for gold data...")
        
        # Common datasets that might have gold data
        test_datasets = [
            "GLBX.MDP3",  # CME Group
            "CME.MDP3",   # CME (alternative)
            "OPRA.PILLAR", # Options (might have GLD options)
            "XNAS.ITCH",   # NASDAQ (might have GLD)
            "XNYS.PILLAR", # NYSE (might have GLD)
            "IFEU.IMPACT", # ICE Europe
            "IFUS.IMPACT", # ICE US
        ]
        
        results = {}
        
        for dataset in test_datasets:
            print(f"\n[Databento] Testing dataset: {dataset}")
            try:
                symbols = self.search_gold_symbols(dataset)
                if symbols:
                    results[dataset] = symbols
                    print(f"✅ {dataset}: Found {len(symbols)} gold symbols")
                else:
                    print(f"❌ {dataset}: No gold symbols found")
            except Exception as e:
                print(f"❌ {dataset}: Error - {str(e)[:100]}...")
        
        return results

    def fetch_gold_databento(self, start: str = "2023-01-01", end: str = None, limit: int = 1000, data_type: str = "etf") -> pd.DataFrame:
        """Try to fetch gold data from Databento using working symbols."""
        if data_type == "etf":
            dataset = "XNAS.ITCH"
            symbol = "GLD"
            print(f"[Databento] Fetching {dataset} {symbol} (ETF) data from {start} to {end or 'latest'}...")
        elif data_type == "futures":
            dataset = "GLBX.MDP3"
            symbol = "GCM4"  # June 2024 Gold Futures
            print(f"[Databento] Fetching {dataset} {symbol} (Futures) data from {start} to {end or 'latest'}...")
        else:
            print(f"[Databento] Unknown data_type: {data_type}. Using ETF.")
            dataset = "XNAS.ITCH"
            symbol = "GLD"
        
        try:
            result = self.client.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                schema="ohlcv-1d",
                start=start,
                end=end,
                limit=limit
            )
            # Convert to DataFrame if possible
            if hasattr(result, 'to_df'):
                df = result.to_df()
            else:
                print("[Databento] Result does not have .to_df(). Returning raw result.")
                return result
            if df.empty:
                print("[Databento] No data returned.")
                return df
            df = df.reset_index()
            print(f"[Databento] Successfully fetched {len(df)} rows of {symbol} from {dataset}.")
            return df
        except Exception as e:
            print(f"[Databento] Exception: {e}")
            return pd.DataFrame()

    def fetch_gold_futures_databento(self, start: str = "2023-01-01", end: str = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch gold futures data from Databento."""
        return self.fetch_gold_databento(start, end, limit, "futures")

    def fetch_gold_etf_databento(self, start: str = "2023-01-01", end: str = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch gold ETF data from Databento."""
        return self.fetch_gold_databento(start, end, limit, "etf")

    def fetch_gold_yfinance(self, start: str = "2023-01-01", end: str = None) -> pd.DataFrame:
        """Fetch GLD data from yfinance as fallback."""
        print(f"[YFinance] Fetching GLD data from {start} to {end or 'latest'}...")
        try:
            ticker = yf.Ticker("GLD")
            df = ticker.history(start=start, end=end)
            if df.empty:
                print("[YFinance] No data returned.")
                return df
            print(f"[YFinance] Successfully fetched {len(df)} rows.")
            return df
        except Exception as e:
            print(f"[YFinance] Exception: {e}")
            return pd.DataFrame()

    def fetch_gld(self, start: str = "2023-01-01", end: str = None, use_databento: bool = True) -> pd.DataFrame:
        """Fetch GLD data, trying Databento first if enabled, then falling back to yfinance."""
        if use_databento:
            df = self.fetch_gold_databento(start, end)
            if not df.empty:
                return df
            print("[INFO] Databento failed, falling back to yfinance...")
        
        return self.fetch_gold_yfinance(start, end)

    def fetch_gold_etf_aberdeen(self, start: str = "2023-01-01", end: str = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch the Aberdeen Standard Physical Gold Shares ETF (GOLD) from XNAS.ITCH."""
        dataset = "XNAS.ITCH"
        symbol = "GOLD"
        print(f"[Databento] Fetching {dataset} {symbol} (Aberdeen ETF) data from {start} to {end or 'latest'}...")
        try:
            result = self.client.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                schema="ohlcv-1d",
                start=start,
                end=end,
                limit=limit
            )
            if hasattr(result, 'to_df'):
                df = result.to_df()
            else:
                print("[Databento] Result does not have .to_df(). Returning raw result.")
                return result
            if df.empty:
                print("[Databento] No data returned.")
                return df
            df = df.reset_index()
            print(f"[Databento] Successfully fetched {len(df)} rows of {symbol} from {dataset}.")
            return df
        except Exception as e:
            print(f"[Databento] Exception: {e}")
            return pd.DataFrame()

    def fetch_gold_etf_aberdeen_mbo(self, start: str = "2023-01-01", end: str = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch the Aberdeen Standard Physical Gold Shares ETF (GOLD) from XNAS.ITCH using the MBO schema."""
        dataset = "XNAS.ITCH"
        symbol = "GOLD"
        print(f"[Databento] Fetching {dataset} {symbol} (Aberdeen ETF, MBO schema) data from {start} to {end or 'latest'}...")
        try:
            result = self.client.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                schema="mbo",
                start=start,
                end=end,
                limit=limit
            )
            if hasattr(result, 'to_df'):
                df = result.to_df()
            else:
                print("[Databento] Result does not have .to_df(). Returning raw result.")
                return result
            if df.empty:
                print("[Databento] No data returned.")
                return df
            df = df.reset_index()
            print(f"[Databento] Successfully fetched {len(df)} rows of {symbol} from {dataset} (MBO schema).")
            return df
        except Exception as e:
            print(f"[Databento] Exception: {e}")
            return pd.DataFrame()

    def try_all_schemas_for_gold_etf(self, start: str = "2024-01-01", end: str = None, limit: int = 1000):
        """Try all available schemas for GOLD in XNAS.ITCH and print the first few rows if any data is found."""
        dataset = "XNAS.ITCH"
        symbol = "GOLD"
        print(f"[Databento] Listing available schemas for {dataset}...")
        try:
            schemas = self.client.metadata.list_schemas(dataset=dataset)
            print(f"[Databento] Schemas for {dataset}: {schemas}")
        except Exception as e:
            print(f"[Databento] Could not list schemas: {e}")
            return
        found_any = False
        for schema in schemas:
            print(f"\n[Databento] Trying schema: {schema} for {symbol}...")
            try:
                result = self.client.timeseries.get_range(
                    dataset=dataset,
                    symbols=[symbol],
                    schema=schema,
                    start=start,
                    end=end,
                    limit=limit
                )
                if hasattr(result, 'to_df'):
                    df = result.to_df()
                else:
                    print("[Databento] Result does not have .to_df(). Skipping.")
                    continue
                if not df.empty:
                    print(f"✅ {schema}: Found {len(df)} rows. Showing first 5:")
                    print(df.head())
                    found_any = True
                else:
                    print(f"❌ {schema}: No data.")
            except Exception as e:
                print(f"❌ {schema}: Exception: {e}")
        if not found_any:
            print("[Databento] No data found for any schema.")

    def try_all_schemas_for_gold_etf_recent_days(self, days: int = 5, limit: int = 1000):
        """Try all schemas for GOLD in XNAS.ITCH for each of the last N business days."""
        import pandas as pd
        from datetime import datetime, timedelta
        dataset = "XNAS.ITCH"
        symbol = "GOLD"
        print(f"[Databento] Listing available schemas for {dataset}...")
        try:
            schemas = self.client.metadata.list_schemas(dataset=dataset)
            print(f"[Databento] Schemas for {dataset}: {schemas}")
        except Exception as e:
            print(f"[Databento] Could not list schemas: {e}")
            return
        # Get last N business days
        today = pd.Timestamp.today().normalize()
        business_days = []
        d = today
        while len(business_days) < days:
            if d.dayofweek < 5:
                business_days.append(d)
            d -= pd.Timedelta(days=1)
        business_days = sorted(business_days)
        found_any = False
        for day in business_days:
            start = end = day.strftime("%Y-%m-%d")
            print(f"\n[Databento] Trying date: {start}")
            for schema in schemas:
                print(f"  [Databento] Trying schema: {schema} for {symbol}...")
                try:
                    result = self.client.timeseries.get_range(
                        dataset=dataset,
                        symbols=[symbol],
                        schema=schema,
                        start=start,
                        end=end,
                        limit=limit
                    )
                    if hasattr(result, 'to_df'):
                        df = result.to_df()
                    else:
                        print("    [Databento] Result does not have .to_df(). Skipping.")
                        continue
                    if not df.empty:
                        print(f"    ✅ {schema}: Found {len(df)} rows. Showing first 5:")
                        print(df.head())
                        found_any = True
                    else:
                        print(f"    ❌ {schema}: No data.")
                except Exception as e:
                    print(f"    ❌ {schema}: Exception: {e}")
        if not found_any:
            print("[Databento] No data found for any schema on any recent day.")

    def fetch_gold_etf_aberdeen_mbo_example(self):
        """Fetch GOLD from XNAS.ITCH using mbo schema and 2023-08-17, like the SPY example."""
        import pandas as pd
        dataset = "XNAS.ITCH"
        symbol = "GOLD"
        schema = "mbo"
        date = "2023-08-17"
        print(f"[Databento] Fetching {symbol} from {dataset} using {schema} schema on {date}...")
        try:
            result = self.client.timeseries.get_range(
                dataset=dataset,
                schema=schema,
                symbols=symbol,
                start=date,
                limit=10_000,
            )
            df = result.to_df()
            if not df.empty:
                print(f"[Databento] Found {len(df)} rows. Showing key columns:")
                print(df[[c for c in ["symbol", "order_id", "action", "side", "price", "size"] if c in df.columns]].head())
            else:
                print("[Databento] No data returned.")
        except Exception as e:
            print(f"[Databento] Exception: {e}")

    def fetch_and_aggregate_gold_mbo_to_ohlcv(self, start_date: str, end_date: str, limit_per_day: int = 10000) -> pd.DataFrame:
        """Fetch and aggregate GOLD MBO data from XNAS.ITCH to daily OHLCV bars over a date range."""
        import pandas as pd
        from datetime import datetime, timedelta
        dataset = "XNAS.ITCH"
        symbol = "GOLD"
        schema = "mbo"
        print(f"[Databento] Aggregating {symbol} MBO data from {start_date} to {end_date}...")
        days = pd.bdate_range(start=start_date, end=end_date)
        all_ohlcv = []
        for day in days:
            date_str = day.strftime("%Y-%m-%d")
            next_day_str = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"  [Databento] Fetching {date_str}...")
            try:
                result = self.client.timeseries.get_range(
                    dataset=dataset,
                    schema=schema,
                    symbols=symbol,
                    start=date_str,
                    end=next_day_str,
                    limit=limit_per_day,
                )
                df = result.to_df()
                if df.empty:
                    print(f"    [Databento] No data for {date_str}.")
                    continue
                if 'action' in df.columns and 'price' in df.columns and 'size' in df.columns:
                    # Databento uses 'T' for trades/executions, not 'E'
                    trades = df[df['action'] == 'T']
                    if trades.empty:
                        print(f"    [Databento] No trades for {date_str}.")
                        continue
                    trades['price'] = trades['price'].astype(float)
                    trades['size'] = trades['size'].astype(float)
                    ohlcv = {
                        'date': date_str,
                        'open': trades['price'].iloc[0],
                        'high': trades['price'].max(),
                        'low': trades['price'].min(),
                        'close': trades['price'].iloc[-1],
                        'volume': trades['size'].sum()
                    }
                    all_ohlcv.append(ohlcv)
                    print(f"    [Databento] OHLCV for {date_str}: {ohlcv}")
                else:
                    print(f"    [Databento] MBO data missing required columns for {date_str}.")
            except Exception as e:
                print(f"    [Databento] Exception for {date_str}: {e}")
        if all_ohlcv:
            df_ohlcv = pd.DataFrame(all_ohlcv)
            df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date'])
            df_ohlcv.set_index('date', inplace=True)
            print(f"[Databento] Aggregated OHLCV DataFrame shape: {df_ohlcv.shape}")
            print(df_ohlcv.head())
            return df_ohlcv
        else:
            print("[Databento] No OHLCV data aggregated.")
            return pd.DataFrame()

if __name__ == "__main__":
    collector = DatabentoGoldCollector()
    
    print("=== DATABENTO GOLD DATA TESTING ===")
    
    print("\n1. Testing Databento ETF (GOLD, Aberdeen, MBO schema)...")
    df_gold_aberdeen_mbo = collector.fetch_gold_etf_aberdeen_mbo(start="2024-01-01")
    print(f"Aberdeen ETF Databento (MBO) result shape: {df_gold_aberdeen_mbo.shape}")
    if not df_gold_aberdeen_mbo.empty:
        print(df_gold_aberdeen_mbo.tail())
    
    print("\n2. Testing Databento Futures (GCM4)...")
    df_futures_databento = collector.fetch_gold_futures_databento(start="2024-01-01")
    print(f"Futures Databento result shape: {df_futures_databento.shape}")
    if not df_futures_databento.empty:
        print(df_futures_databento.tail())
    
    print("\n3. Testing YFinance comparison...")
    df_yfinance = collector.fetch_gold_yfinance(start="2024-01-01")
    print(f"YFinance result shape: {df_yfinance.shape}")
    print(df_yfinance.tail())
    
    print("\n=== SUMMARY ===")
    print(f"Databento Aberdeen ETF (MBO): {len(df_gold_aberdeen_mbo)} rows")
    print(f"Databento Futures: {len(df_futures_databento)} rows")
    print(f"YFinance: {len(df_yfinance)} rows")
    
    if not df_gold_aberdeen_mbo.empty:
        print("✅ Databento Aberdeen ETF (MBO) data working!")
    if not df_futures_databento.empty:
        print("✅ Databento Futures data working!")
    if not df_yfinance.empty:
        print("✅ YFinance fallback working!") 

    print("=== DATABENTO GOLD DATA SCHEMA TESTING ===")
    collector.try_all_schemas_for_gold_etf(start="2024-01-01") 

    print("=== DATABENTO GOLD DATA RECENT DAYS SCHEMA TESTING ===")
    collector.try_all_schemas_for_gold_etf_recent_days(days=5) 

    print("=== DATABENTO GOLD ETF MBO EXAMPLE ===")
    collector.fetch_gold_etf_aberdeen_mbo_example() 

    print("=== DATABENTO GOLD MBO DIAGNOSTIC ANALYSIS ===")
    
    # First, let's diagnose what's in the MBO data for a single day
    print("\n" + "-"*40)
    print("DIAGNOSTIC: Analyzing MBO data structure")
    print("-"*40)
    
    test_date = "2023-08-17"
    print(f"\n[Databento] Fetching MBO data for {test_date} for diagnosis...")
    
    try:
        result = collector.client.timeseries.get_range(
            dataset="XNAS.ITCH",
            symbols=["GOLD"],
            schema="mbo",
            start=test_date,
            end="2023-08-18",  # Next day
            limit=10000
        )
        mbo_data = result.to_df()
        
        if mbo_data is not None and not mbo_data.empty:
            print(f"[Databento] Got {len(mbo_data)} rows for {test_date}")
            print(f"[Databento] Columns: {list(mbo_data.columns)}")
            
            # Check unique actions
            if 'action' in mbo_data.columns:
                unique_actions = mbo_data['action'].unique()
                print(f"[Databento] Unique actions: {unique_actions}")
                
                # Count each action
                action_counts = mbo_data['action'].value_counts()
                print(f"[Databento] Action counts:")
                for action, count in action_counts.items():
                    print(f"  {action}: {count}")
            
            # Check for non-null prices
            if 'price' in mbo_data.columns:
                non_null_prices = mbo_data[mbo_data['price'].notna()]
                print(f"[Databento] Rows with non-null prices: {len(non_null_prices)}")
                
                if len(non_null_prices) > 0:
                    print(f"[Databento] Sample rows with prices:")
                    print(non_null_prices[['action', 'price', 'size', 'ts_event']].head(10))
            
            # Check for any execution-like actions
            execution_actions = ['E', 'T', 'X']  # Common execution action codes
            for action in execution_actions:
                if action in mbo_data['action'].values:
                    exec_data = mbo_data[mbo_data['action'] == action]
                    print(f"[Databento] Found {len(exec_data)} rows with action '{action}'")
                    if len(exec_data) > 0:
                        print(f"[Databento] Sample execution data:")
                        print(exec_data[['action', 'price', 'size', 'ts_event']].head(5))
        
        else:
            print(f"[Databento] No data for {test_date}")
            
    except Exception as e:
        print(f"[Databento] Error in diagnosis: {e}")

    print("\n=== DATABENTO GOLD MBO TO OHLCV AGGREGATION (AUG 2023) ===")
    df_ohlcv = collector.fetch_and_aggregate_gold_mbo_to_ohlcv(start_date="2023-08-01", end_date="2023-08-31")
    print("\nFinal OHLCV DataFrame:")
    print(df_ohlcv) 