import requests 
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import logging
import sqlite3
import matplotlib.pyplot as plt
import time
import json
import concurrent.futures
import requests.adapters

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinnhubETL:
    def __init__(self, api_key, exchange='US', batch_size=50):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.exchange = exchange
        self.batch_size = batch_size
        
    def fetch_symbols(self):
        """Fetch all stock symbols for the configured exchange"""
        try:
            params = {
                "exchange": self.exchange,
                "token": self.api_key
            }
            
            logger.info(f"Making request to Finnhub API...")
            response = requests.get(f"{self.base_url}/stock/symbol", params=params)
            
            if response.status_code != 200:
                logger.error(f"API Error: {response.text}")
                return []
            
            data = response.json()
            logger.info(f"Retrieved {len(data)} total symbols from API")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            logger.info(f"Initial DataFrame shape: {df.shape}")
            
            # Log unique types
            if 'type' in df.columns:
                logger.info(f"Unique types in data: {df['type'].unique()}")
            
            # Filter for common stock types
            stock_types = ['Common Stock', 'Stock', 'Equity']
            if 'type' in df.columns:
                df = df[df['type'].isin(stock_types)]
                logger.info(f"Filtered DataFrame shape: {df.shape}")
            
            symbols = df["symbol"].tolist()
            logger.info(f"Final symbol count: {len(symbols)}")
            if symbols:
                logger.info(f"Sample symbols: {symbols[:5]}")
            
            return symbols
        except Exception as e:
            logger.error(f"Error fetching symbols: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            return []
            
    def extract(self):
        """Extract data for multiple stocks in batches"""
        try:
            symbols = self.fetch_symbols()
            if not symbols:
                logger.error("No symbols to process")
                return []
            
            all_data = []
            total_batches = (len(symbols) + self.batch_size - 1) // self.batch_size
            
            def fetch_quote(session, symbol):
                try:
                    params = {
                        'symbol': symbol,
                        'token': self.api_key
                    }
                    response = session.get(f"{self.base_url}/quote", params=params)
                    
                    if response.status_code == 429:  # Rate limit hit
                        logger.warning("Rate limit reached, sleeping for 60 seconds...")
                        time.sleep(60)  # Wait for 60 seconds before retrying
                        response = session.get(f"{self.base_url}/quote", params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and data.get('c'):  # Check if we have a current price
                            data['symbol'] = symbol
                            data['t'] = int(datetime.now().timestamp())
                            return data
                    else:
                        logger.warning(f"Failed to get quote for {symbol}: {response.text}")
                        return None
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {str(e)}")
                    return None
            
            # Create a session with connection pooling
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,  # Reduced from 100
                pool_maxsize=10,     # Reduced from 100
                max_retries=3,
                pool_block=False
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            for i in range(0, len(symbols), self.batch_size):
                batch = symbols[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")
                
                # Process batch with smaller number of concurrent requests
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  # Reduced from 10
                    futures = [
                        executor.submit(fetch_quote, session, symbol)
                        for symbol in batch
                    ]
                    
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            all_data.append(result)
                
                # Increased pause between batches
                if batch_num < total_batches:
                    logger.info("Rate limiting pause...")
                    time.sleep(60)  # Increased from 1 second to 60 seconds
                    
            logger.info(f"Successfully extracted data for {len(all_data)} symbols")
            return all_data
            
        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            raise
            
    def transform(self, data):
        """
        Transform the extracted data with comprehensive data preparation steps
        
        This method implements:
        - Data validation and cleaning
        - Type conversion and normalization
        - Feature engineering
        - Data quality checks
        - Performance optimization
        """
        try:
            if not data:
                logger.warning("No data to transform")
                return pd.DataFrame()
            
            # Create DataFrame with the data
            df = pd.DataFrame(data)
            logger.info(f"Initial DataFrame shape: {df.shape}")
            
            # Step 1: Data Validation and Initial Cleaning
            def validate_data(df):
                """Validate data quality and remove invalid entries"""
                initial_rows = len(df)
                
                # Remove rows where all numeric columns are 0 or null
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                df = df.dropna(subset=numeric_cols, how='all')
                
                # Remove duplicates
                df = df.drop_duplicates()
                
                rows_removed = initial_rows - len(df)
                if rows_removed > 0:
                    logger.warning(f"Removed {rows_removed} invalid or duplicate rows")
                
                return df
            
            # Step 2: Type Conversion and Normalization
            def normalize_data(df):
                """Convert data types and normalize values"""
                # Rename columns to be more descriptive
                column_mapping = {
                    'c': 'current_price',
                    'h': 'high_price',
                    'l': 'low_price',
                    'o': 'open_price',
                    'pc': 'previous_close',
                    't': 'timestamp',
                    'd': 'daily_change',
                    'dp': 'daily_change_percent'
                }
                
                # Only rename columns that exist
                existing_columns = set(df.columns)
                mapping_to_use = {k: v for k, v in column_mapping.items() if k in existing_columns}
                df = df.rename(columns=mapping_to_use)
                
                # Convert timestamp to datetime if it exists
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Ensure numeric columns are properly typed
                numeric_columns = ['current_price', 'high_price', 'low_price', 'open_price', 
                                 'previous_close', 'daily_change', 'daily_change_percent']
                
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            
            # Step 3: Feature Engineering
            def engineer_features(df):
                """Create derived features for analysis"""
                try:
                    # Calculate price volatility (high-low spread)
                    if all(col in df.columns for col in ['high_price', 'low_price', 'current_price']):
                        df['price_volatility'] = (df['high_price'] - df['low_price']) / df['current_price']
                    
                    # Calculate price momentum (current vs previous close)
                    if all(col in df.columns for col in ['current_price', 'previous_close']):
                        df['price_momentum'] = (df['current_price'] - df['previous_close']) / df['previous_close']
                    
                    # Add trading session indicator
                    if 'timestamp' in df.columns:
                        df['trading_session'] = df['timestamp'].dt.hour.map(
                            lambda x: 'Pre-Market' if x < 9.5 else
                            ('Market-Hours' if x < 16 else 'After-Hours')
                        )
                    
                    logger.info(f"Added derived features: {[col for col in df.columns if col not in data[0].keys()]}")
                    
                except Exception as e:
                    logger.error(f"Error in feature engineering: {str(e)}")
                
                return df
            
            # Step 4: Data Quality Checks
            def check_data_quality(df):
                """Perform data quality checks and log issues"""
                # Check for missing values
                missing_values = df.isnull().sum()
                if missing_values.any():
                    logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")
                
                # Check for outliers in numeric columns
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
                    if len(outliers) > 0:
                        logger.warning(f"Outliers detected in {col}: {len(outliers)} values")
                
                return df
            
            # Execute transformation pipeline
            logger.info("Starting data transformation pipeline...")
            
            df = validate_data(df)
            logger.info("Data validation complete")
            
            df = normalize_data(df)
            logger.info("Data normalization complete")
            
            df = engineer_features(df)
            logger.info("Feature engineering complete")
            
            df = check_data_quality(df)
            logger.info("Data quality checks complete")
            
            # Final validation
            if len(df) == 0:
                logger.error("No data remained after transformation")
                return pd.DataFrame()
            
            logger.info(f"Successfully transformed data with {len(df)} records")
            logger.info(f"Final columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error during transformation: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            raise
            
    def load(self, transformed_data):
        """Load data into the database"""
        try:
            # Create SQLite database engine
            engine = create_engine('sqlite:///finnhub_data.db')
            
            # Check if table exists and get current schema
            conn = sqlite3.connect('finnhub_data.db')
            cursor = conn.cursor()
            
            # Get existing columns (if table exists)
            cursor.execute("SELECT * FROM sqlite_master WHERE type='table' AND name='stock_quotes';")
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # If table doesn't exist, create it with all columns
                transformed_data.to_sql(
                    name='stock_quotes',
                    con=engine,
                    if_exists='append',
                    index=False
                )
                logger.info("Created new table and loaded initial data")
            else:
                # Get existing columns
                cursor.execute("PRAGMA table_info(stock_quotes);")
                existing_columns = [row[1] for row in cursor.fetchall()]
                
                # Get new columns from transformed data
                new_columns = transformed_data.columns.tolist()
                
                # Find missing columns
                missing_columns = set(new_columns) - set(existing_columns)
                
                # Add any missing columns
                for col in missing_columns:
                    dtype = transformed_data[col].dtype
                    sql_type = 'REAL' if dtype in ['float64', 'int64'] else 'TEXT'
                    cursor.execute(f"ALTER TABLE stock_quotes ADD COLUMN {col} {sql_type};")
                    logger.info(f"Added new column: {col}")
                
                # Commit changes
                conn.commit()
                
                # Now we can safely append the new data
                transformed_data.to_sql(
                    name='stock_quotes',
                    con=engine,
                    if_exists='append',
                    index=False
                )
            
            conn.close()
            logger.info(f"Successfully loaded {len(transformed_data)} records into database")
            
        except Exception as e:
            logger.error(f"Error during loading: {str(e)}")
            raise
            
    def create_timestamp_index(self):
        """Create an index on the timestamp column for better query performance"""
        try:
            conn = sqlite3.connect('finnhub_data.db')
            cursor = conn.cursor()
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON stock_quotes (timestamp);
            """)
            conn.commit()
            conn.close()
            logger.info("Created timestamp index")
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
            
    def run_pipeline(self):
        """Execute the full ETL pipeline"""
        try:
            logger.info("Starting ETL pipeline")
            raw_data = self.extract()
            transformed_data = self.transform(raw_data)
            self.load(transformed_data)
            self.create_timestamp_index()
            logger.info("ETL pipeline completed successfully")
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    api_key = 'your_api_key'  # Your actual API key
    
    logger.info("Starting script...")
    
    try:
        etl = FinnhubETL(
            api_key=api_key,
            exchange='US',
            batch_size=50
        )
        
        # Test symbol fetching
        symbols = etl.fetch_symbols()
        if not symbols:
            logger.error("No symbols retrieved. Exiting.")
            exit(1)
            
        logger.info(f"Successfully retrieved {len(symbols)} symbols")
        logger.info(f"First few symbols: {symbols[:5]}")
        
        while True:
            etl.run_pipeline()
            logger.info("Waiting 5 minutes until next run...")
            time.sleep(300)
    except KeyboardInterrupt:
        logger.info("Script stopped by user")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        logger.error(f"Error type: {type(e)}")