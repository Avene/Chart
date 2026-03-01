import logging
from typing import List, Dict, Protocol
import requests
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class StockDataProvider(Protocol):
    def get_daily_prices(self, code: str, days: int = 1825) -> pd.DataFrame: ...
    def get_company_name(self, code: str) -> str: ...

class JQuantsService:
    """Handles J-Quants API interactions."""
    BASE_URL = "https://api.jquants.com/v2"

    def __init__(self, api_key: str) -> None:
        self.headers = {'x-api-key': api_key}

    def get_daily_prices(self, code: str, days: int = 1825) -> pd.DataFrame:
        code5 = code if len(code) == 5 else code + '0'
        base_date = pd.Timestamp.now()
        params = {
            'code': code5,
            'from': (base_date - pd.Timedelta(days=days)).strftime('%Y-%m-%d'),
            'to': base_date.strftime('%Y-%m-%d'),
        }
        
        url = f"{self.BASE_URL}/equities/bars/daily"
        res = requests.get(url, params=params, headers=self.headers)
        res.raise_for_status()
        
        d = res.json()
        data = d["data"]
        while "pagination_key" in d:
            params["pagination_key"] = d["pagination_key"]
            res = requests.get(url, params=params, headers=self.headers)
            d = res.json()
            data += d["data"]
            
        df = pd.DataFrame(data)
        if df.empty: return df
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        cols = {'AdjO': 'Open', 'AdjH': 'High', 'AdjL': 'Low', 'AdjC': 'Close', 'AdjVo': 'Volume'}
        df = df.rename(columns=cols)
        df[list(cols.values())] = df[list(cols.values())].astype(float)
        return df

    def get_company_name(self, code: str) -> str:
        code5 = code if len(code) == 5 else code + '0'
        url = f"{self.BASE_URL}/equities/master"
        try:
            res = requests.get(url, params={"code": code5}, headers=self.headers)
            res.raise_for_status()
            data = res.json().get("data", [])
            if data:
                return data[0]['CoNameEn']
        except Exception as e:
            logger.warning(f"Failed to get name for {code}: {e}")
        return code

class YFinanceService:
    """Handles Yahoo Finance API interactions for US stocks."""
    
    def get_daily_prices(self, code: str, days: int = 1825) -> pd.DataFrame:
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            ticker = yf.Ticker(code)
            df = ticker.history(start=start_date)
            if df.empty: return pd.DataFrame()
            
            # yfinance returns timezone-aware index, remove it for consistency
            df.index = df.index.tz_localize(None)
            df.index.name = 'Date'
            
            # Ensure columns exist and return specific columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df.columns for col in required_cols):
                return df[required_cols]
            return pd.DataFrame()
        except Exception as e:
            logger.exception(f"YFinance error for {code}: {e}")
            return pd.DataFrame()

    def get_company_name(self, code: str) -> str:
        try:
            return yf.Ticker(code).info.get('shortName', code)
        except:
            return code

    def get_bulk_daily_prices(self, codes: List[str], days: int = 5) -> Dict[str, float]:
        """Fetches latest prices for multiple tickers in a single request."""
        if not codes:
            return {}
        
        original_code_map = {code.upper(): code for code in codes}
        tickers_str = ' '.join(codes)
        
        try:
            df = yf.download(tickers_str, period=f"{days}d", progress=False, threads=True)
            if df.empty:
                return {}
            
            # Handle 'Close' column extraction safely for both MultiIndex and Flat DataFrames
            close_data = None
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.get_level_values(0):
                    close_data = df['Close']
            elif 'Close' in df.columns:
                close_data = df['Close']
            
            if close_data is None or close_data.empty:
                return {}

            # Get latest prices (last row)
            latest = close_data.iloc[-1]
            
            result = {}
            
            # Case 1: Multiple tickers (or single ticker returned as MultiIndex) -> Series
            if isinstance(latest, pd.Series):
                for ticker, price in latest.items():
                    # Guard against duplicate columns resulting in Series values
                    if isinstance(price, pd.Series):
                        price = price.iloc[0]
                        
                    if pd.notna(price):
                        ticker_upper = str(ticker).upper()
                        original_code = original_code_map.get(ticker_upper)
                        if original_code:
                            result[original_code] = float(price)
                            
            # Case 2: Single ticker flat DataFrame -> Scalar
            elif pd.notna(latest):
                # If we only requested one code, map it
                if len(codes) == 1:
                    result[codes[0]] = float(latest)
            
            return result
        except Exception as e:
            logger.error(f"YFinance bulk download error: {e}")
            return {}

    def get_bulk_company_names(self, codes: List[str]) -> Dict[str, str]:
        """Fetches company names for multiple tickers sequentially to avoid throttling."""
        if not codes:
            return {}
        
        original_code_map = {code.upper(): code for code in codes}
        # yf.Tickers is efficient for managing multiple ticker objects
        tickers = yf.Tickers(' '.join(codes))
        names = {}

        # Iterate sequentially to avoid sending too many parallel requests for .info
        for ticker_symbol, ticker_obj in tickers.tickers.items():
            original_code = original_code_map.get(ticker_symbol, ticker_symbol)
            try:
                # Accessing .info triggers a network request for each ticker
                names[original_code] = ticker_obj.info.get('shortName', original_code)
            except Exception as e:
                # If fetching info fails for one, log it and move on
                logger.warning(f"Could not fetch .info for {original_code}: {e}")
                names[original_code] = original_code
        return names