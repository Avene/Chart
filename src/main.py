import os
import logging
import json
import io
import tempfile
import concurrent.futures
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Protocol

import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import yfinance as yf
import google.auth
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from google import genai
from dotenv import load_dotenv

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Config ---
@dataclass
class AppConfig:
    GEMINI_API_KEY: str
    JQUANTS_API_KEY: str
    PROMPT_URI: Optional[str]
    GEMINI_MODEL_NAME: str
    STOCK_LIST_SHEET_URL: Optional[str]
    GOOGLE_SERVICE_ACCOUNT_JSON: Optional[str]
    REPORT_SPREADSHEET_ID: Optional[str]
    
    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            GEMINI_API_KEY=os.getenv('GEMINI_API_KEY', ''),
            JQUANTS_API_KEY=os.getenv('JQUANTS_API_KEY', ''),
            PROMPT_URI=os.getenv('PROMPT_URI'),
            GEMINI_MODEL_NAME=os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash-exp'),
            STOCK_LIST_SHEET_URL=os.getenv('STOCK_LIST_SHEET_URL'),
            GOOGLE_SERVICE_ACCOUNT_JSON=os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON'),
            REPORT_SPREADSHEET_ID=os.getenv('REPORT_SPREADSHEET_ID'),
        )

# --- Services ---

class GoogleService:
    """Handles Google Auth, Drive, Docs, and Sheets interactions."""
    SCOPES = [
        'https://www.googleapis.com/auth/drive.readonly',
        'https://www.googleapis.com/auth/spreadsheets',
    ]

    def __init__(self, sa_json: Optional[str]):
        self.sa_json = sa_json
        self._creds: Optional[Credentials] = None

    def _get_credentials(self) -> Credentials:
        if self._creds and self._creds.valid:
            return self._creds
            
        # 1. Service Account
        if self.sa_json:
            try:
                info = json.loads(self.sa_json)
                self._creds = service_account.Credentials.from_service_account_info(info, scopes=self.SCOPES)
                return self._creds
            except Exception as e:
                logger.debug(f"Service Account auth failed: {e}")

        # 2. ADC
        self._creds, _ = google.auth.default(scopes=self.SCOPES)
        return self._creds

    def get_auth_headers(self) -> Dict[str, str]:
        try:
            creds = self._get_credentials()
            creds.refresh(GoogleAuthRequest())
            return {'Authorization': f'Bearer {creds.token}'}
        except Exception as e:
            logger.warning(f"Google Auth failed: {e}")
            return {}

    def fetch_text_content(self, uri: str) -> str:
        """Fetches text from a URL or Google Doc."""
        if not uri: return ""
        
        headers = {}
        target_url = uri
        
        if "docs.google.com/document/d/" in uri:
            headers = self.get_auth_headers()
            doc_id = uri.split('/d/')[1].split('/')[0]
            target_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
            
        try:
            res = requests.get(target_url, headers=headers)
            res.raise_for_status()
            return res.text
        except Exception as e:
            logger.exception(f"Failed to fetch content from {uri}: {e}")
            return ""

    def fetch_stock_list(self, uri: str, default_code: str = '79740') -> List[Dict[str, str]]:
        """Fetches stock list from CSV URL or Google Sheet."""
        if not uri: return [{'code': default_code, 'market': 'JP'}]
        
        headers = {}
        target_url = uri

        if "docs.google.com/spreadsheets" in uri:
            headers = self.get_auth_headers()
            doc_id = uri.split('/d/')[1].split('/')[0]
            target_url = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"
            
        try:
            res = requests.get(target_url, headers=headers)
            res.raise_for_status()
            df = pd.read_csv(io.StringIO(res.text), dtype=str)
            if not df.empty:
                # Normalize columns
                df.columns = [c.strip() for c in df.columns]
                stocks = []
                has_market = 'Market' in df.columns
                
                for _, row in df.iterrows():
                    code = str(row.iloc[0]).strip()
                    if not code or code.lower() == 'nan': continue
                    
                    market = 'JP'
                    if has_market:
                        val = str(row['Market']).strip().upper()
                        if val and val != 'NAN':
                            market = val
                    stocks.append({'code': code, 'market': market})
                return stocks
        except Exception as e:
            logger.exception(f"Failed to fetch stock list: {e}")
        
        return [{'code': default_code, 'market': 'JP'}]

    def setup_summary_sheet(self, spreadsheet_id: str, new_title: str = "Summary"):
        """Renames the default first sheet (ID 0) to the given title."""
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}:batchUpdate"
        headers = self.get_auth_headers()
        body = {
            "requests": [{"updateSheetProperties": {"properties": {"sheetId": 0, "title": new_title}, "fields": "title"}}]
        }
        try:
            requests.post(url, headers=headers, json=body)
        except Exception as e:
            logger.warning(f"Failed to rename summary sheet: {e}")

    def add_sheet(self, spreadsheet_id: str, title: str):
        """Adds a new sheet (tab) to the spreadsheet."""
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}:batchUpdate"
        headers = self.get_auth_headers()
        # Sheet titles must be <= 31 chars and no special chars
        safe_title = "".join(c for c in title if c not in r"*?:/\[\]").strip()[:31]
        body = {"requests": [{"addSheet": {"properties": {"title": safe_title}}}]}
        try:
            requests.post(url, headers=headers, json=body)
        except Exception as e:
            logger.warning(f"Failed to add sheet '{title}': {e}")

    def get_sheet_values(self, spreadsheet_id: str, range_name: str) -> List[List[str]]:
        """Reads values from a specific range."""
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}"
        headers = self.get_auth_headers()
        try:
            res = requests.get(url, headers=headers)
            res.raise_for_status()
            return res.json().get('values', [])
        except Exception as e:
            logger.exception(f"Failed to get values from '{range_name}': {e}")
            return []

    def load_watchlist(self, spreadsheet_id: str, sheet_name: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Reading {sheet_name} from {spreadsheet_id}...")
        rows = self.get_sheet_values(spreadsheet_id, sheet_name)
        
        if not rows:
            logger.warning(f"No data found in {sheet_name}.")
            return None

        headers = rows[0]
        # Helper to find column index or append if missing
        def get_or_create_col(name):
            if name in headers:
                return headers.index(name)
            headers.append(name)
            return len(headers) - 1

        # Identify columns
        try:
            idx_ticker = headers.index('Ticker')
            idx_market = headers.index('Market')
        except ValueError:
            logger.error("Sheet must contain 'Ticker' and 'Market' columns.")
            return None

        indices = {
            'ticker': idx_ticker,
            'market': idx_market,
            'name': get_or_create_col('Name'),
            'price': get_or_create_col('CurrentPrice'),
            'updated': get_or_create_col('PriceUpdated'),
        }

        # Separate US and JP stocks for different processing strategies
        us_to_process = {}
        jp_to_process = {}
        for row in rows[1:]: # Skip header
            try:
                if len(row) > idx_market:
                    market = row[idx_market].strip().upper()
                    ticker = row[idx_ticker]
                    match market:
                        case 'US':
                            us_to_process[ticker] = row
                        case 'JP':
                            jp_to_process[ticker] = row
            except IndexError:
                continue # Skip malformed rows
        
        return {
            'rows': rows,
            'headers': headers,
            'indices': indices,
            'us_to_process': us_to_process,
            'jp_to_process': jp_to_process,
        }

    def update_sheet_values(self, spreadsheet_id: str, sheet_name: str, values: List[List[str]]):
        """Writes values to the specified sheet starting at A1."""
        safe_title = "".join(c for c in sheet_name if c not in r"*?:/\[\]").strip()[:31]
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/'{safe_title}'!A1?valueInputOption=RAW"
        # Use USER_ENTERED to allow formulas like =IMAGE(...)
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/'{safe_title}'!A1?valueInputOption=USER_ENTERED"
        headers = self.get_auth_headers()
        try:
            requests.put(url, headers=headers, json={"values": values})
        except Exception as e:
            logger.exception(f"Failed to update values in '{sheet_name}': {e}")

class StockDataProvider(Protocol):
    def get_daily_prices(self, code: str, days: int = 1825) -> pd.DataFrame: ...
    def get_company_name(self, code: str) -> str: ...

class JQuantsService:
    """Handles J-Quants API interactions."""
    BASE_URL = "https://api.jquants.com/v2"

    def __init__(self, api_key: str):
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
            
            # If only one ticker, columns are not multi-level
            if len(codes) == 1:
                if 'Close' in df.columns and not df.empty:
                    latest_price = df['Close'].iloc[-1]
                    return {codes[0]: latest_price} if pd.notna(latest_price) else {}
                return {}

            latest_prices_series = df['Close'].iloc[-1]
            
            result = {}
            for upper_ticker, price in latest_prices_series.items():
                original_ticker = original_code_map.get(upper_ticker)
                if original_ticker and pd.notna(price):
                    result[original_ticker] = price
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

class ChartService:
    """Handles Technical Analysis and Chart Plotting."""
    
    @staticmethod
    def add_indicators(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        if df.empty: return df
        
        # MA
        for p in periods:
            df[f'MA{p}'] = df['Close'].rolling(window=p).mean()
            
        # Bollinger Bands
        bb_period = periods[1] if len(periods) > 1 else 20
        sma = df['Close'].rolling(window=bb_period).mean()
        std = df['Close'].rolling(window=bb_period).std()
        
        for sigma in [1, 2, 3]:
            df[f'BB_UP_{sigma}'] = sma + (sigma * std)
            df[f'BB_LOW_{sigma}'] = sma - (sigma * std)
            
        # Alias for 2 sigma (backward compatibility if needed)
        df['BB_UP'] = df['BB_UP_2']
        df['BB_LOW'] = df['BB_LOW_2']
        return df

    @staticmethod
    def create_chart_image(df: pd.DataFrame, filename: str, title: str) -> str:
        if df.empty: return ""
        
        ap = []
        # MA
        ma_cols = sorted([c for c in df.columns if c.startswith('MA')], key=lambda x: int(x[2:]))
        colors = ['orange', 'blue', 'green']
        for i, col in enumerate(ma_cols):
            if not df[col].isna().all():
                ap.append(mpf.make_addplot(df[col], color=colors[i % len(colors)], width=1.0))
        
        # BB
        bb_colors = {1: "#EFB2D8", 2: "#DC72B5", 3: "#BA358A"}
        style = dict(width=0.6, alpha=0.8, linestyle='dashdot')
        
        for sigma, color in bb_colors.items():
            up, low = f'BB_UP_{sigma}', f'BB_LOW_{sigma}'
            if up in df.columns and not df[up].isna().all():
                ap.append(mpf.make_addplot(df[up], color=color, **style))
            if low in df.columns and not df[low].isna().all():
                ap.append(mpf.make_addplot(df[low], color=color, **style))

        kwargs = {'addplot': ap} if ap else {}
        mpf.plot(df, type='candle', volume=True, style='yahoo', savefig=filename, title=title, **kwargs)
        logger.info(f"✅ Chart saved: {filename}")
        return filename

class GeminiService:
    """Handles Gemini API interactions."""
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_ms = 300000

    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def _generate(self, client, contents):
        logger.info(f"Generating content with model {self.model_name}")
        return client.models.generate_content(model=self.model_name, contents=contents)

    def analyze(self, image_paths: List[str], csv_data: str, csv_prefix: str, prompt: str) -> str:
        client = genai.Client(api_key=self.api_key, http_options={'timeout': self.timeout_ms})
        try:
            imgs = [client.files.upload(file=p, config={'http_options': {'timeout': self.timeout_ms}}) for p in image_paths]
            
            csv_file = None
            if csv_data:
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, prefix=csv_prefix, suffix='.csv', encoding='utf-8') as tmp:
                    tmp.write(csv_data)
                    tmp_path = tmp.name
                try:
                    csv_file = client.files.upload(file=tmp_path, config={'mime_type': 'text/csv', 'http_options': {'timeout': self.timeout_ms}})
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            contents = [prompt] + imgs
            if csv_file:
                contents.append(csv_file)

            response = self._generate(client, contents)
            return response.text
        except Exception as e:
            logger.exception(f"Gemini Analysis Failed: {e}")
            return "Error during analysis."

    def summarize(self, results: List[str], prompt: str) -> str:
        client = genai.Client(api_key=self.api_key, http_options={'timeout': self.timeout_ms})
        try:
            combined_text = "\n\n---\n\n".join(results)
            response = self._generate(client, [prompt, combined_text])
            return response.text
        except Exception as e:
            logger.exception(f"Gemini Summary Failed: {e}")
            return "Error during summary."

# --- Main Workflow ---

def main():
    os.makedirs('output', exist_ok=True)
    # Load Config
    config = AppConfig.from_env()
    
    google_svc = GoogleService(config.GOOGLE_SERVICE_ACCOUNT_JSON)
    jq_svc = JQuantsService(config.JQUANTS_API_KEY)
    yf_svc = YFinanceService()

    sid = config.REPORT_SPREADSHEET_ID
    if not sid:
        logger.error("REPORT_SPREADSHEET_ID is not set.")
        return
    sheet_name = "Sheet1"
    # 1. Read the spreadsheet (Sheet1)
    data = google_svc.load_watchlist(sid, sheet_name)
    if not data:
        return

    headers = data['headers']
    indices = data['indices']

    idx_ticker = indices['ticker']
    idx_name = indices['name']
    idx_price = indices['price']
    idx_updated = indices['updated']

    # 2. Process rows
    now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # --- Process US Stocks in Bulk ---
    if (us_to_process := data['us_to_process']):
        logger.info(f"Processing {len(us_to_process)} US stocks in bulk...")
        us_tickers_all = list(us_to_process.keys())
        us_tickers_needing_name = [
            t for t, row in us_to_process.items() if len(row) <= idx_name or not row[idx_name]
        ]
        
        names_map = yf_svc.get_bulk_company_names(us_tickers_needing_name)
        prices_map = yf_svc.get_bulk_daily_prices(us_tickers_all, days=5)
        
        for ticker, row in us_to_process.items():
            while len(row) < len(headers): row.append("") # Pad row
            
            # Update name only if it was fetched (i.e., it was empty before)
            if ticker in names_map:
                row[idx_name] = names_map[ticker]
            if ticker in prices_map:
                row[idx_price] = str(prices_map[ticker])
                row[idx_updated] = now_str

    # --- Process JP Stocks concurrently ---
    def fetch_jp_data(row_data):
        while len(row_data) < len(headers): row_data.append("")
        ticker = row_data[idx_ticker]
        if not ticker: return
        
        try:
            # Only fetch and update name if the column is empty
            if len(row_data) <= idx_name or not row_data[idx_name]:
                row_data[idx_name] = jq_svc.get_company_name(ticker)
            df = jq_svc.get_daily_prices(ticker, days=5)
            if not df.empty:
                row_data[idx_price] = str(df.iloc[-1]['Close'])
                row_data[idx_updated] = now_str
        except Exception as e:
            logger.error(f"Error fetching JP ticker {ticker}: {e}")

    if (jp_to_process := data['jp_to_process']):
        logger.info(f"Processing {len(jp_to_process)} JP stocks concurrently...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_jp_data, row) for row in jp_to_process.values()]
            concurrent.futures.wait(futures)

    # 3. Assemble final table and write back
    logger.info("Writing updated data back to sheet...")
    google_svc.update_sheet_values(sid, sheet_name, data['rows'])
    logger.info("Done.")

if __name__ == "__main__":
    main()
