import os
import logging
import json
import io
import tempfile
import threading
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
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

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

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            GEMINI_API_KEY=os.getenv('GEMINI_API_KEY', ''),
            JQUANTS_API_KEY=os.getenv('JQUANTS_API_KEY', ''),
            PROMPT_URI=os.getenv('PROMPT_URI'),
            GEMINI_MODEL_NAME=os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash-exp'),
            STOCK_LIST_SHEET_URL=os.getenv('STOCK_LIST_SHEET_URL'),
            GOOGLE_SERVICE_ACCOUNT_JSON=os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        )

# --- Services ---

class GoogleService:
    """Handles Google Auth, Drive, Docs, and Sheets interactions."""
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

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
            logger.error(f"Failed to fetch content from {uri}: {e}")
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
            logger.error(f"Failed to fetch stock list: {e}")
        
        return [{'code': default_code, 'market': 'JP'}]

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
            logger.error(f"YFinance error for {code}: {e}")
            return pd.DataFrame()

    def get_company_name(self, code: str) -> str:
        try:
            return yf.Ticker(code).info.get('shortName', code)
        except:
            return code

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
            logger.error(f"Gemini Analysis Failed: {e}")
            return "Error during analysis."

    def summarize(self, results: List[str], prompt: str) -> str:
        client = genai.Client(api_key=self.api_key, http_options={'timeout': self.timeout_ms})
        try:
            combined_text = "\n\n---\n\n".join(results)
            response = self._generate(client, [prompt, combined_text])
            return response.text
        except Exception as e:
            logger.error(f"Gemini Summary Failed: {e}")
            return "Error during summary."

# --- Main Workflow ---

def process_stock(code: str, config: AppConfig, provider: StockDataProvider, chart: ChartService, google_svc: GoogleService, gemini: GeminiService, chart_lock: threading.Lock) -> Optional[str]:
    logger.info(f"Processing {code}...")
    
    # 1. データ取得
    df = provider.get_daily_prices(code)
    if df.empty:
        logger.warning(f"No data for {code}")
        return None
    name = provider.get_company_name(code)

    # 2. Charts
    charts_meta = [
        # (Data Slice, Filename, Title, MA Periods)
        (
            chart.add_indicators(df.copy(), [5, 25, 75]).tail(130),
            f'output/chart_daily_{code}.png',
            f'{name} ({code}) Daily'
        ),
        (
            chart.add_indicators(df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna(), [13, 26, 52]).tail(104),
            f'output/chart_weekly_{code}.png',
            f'{name} ({code}) Weekly'
        ),
        # (df_monthly, 'output/chart_monthly.png', 'Monthly Chart')
    ]
    
    paths = []
    try:
        with chart_lock:
            for d, fname, title in charts_meta:
                if os.path.exists(fname): os.remove(fname)
                chart.create_chart_image(d, fname, title)
                paths.append(fname)

        # 3. Analysis
        prompt = google_svc.fetch_text_content(config.PROMPT_URI) or "Analyze these charts."
        csv_text = df.to_csv()
        csv_prefix = f"{code}_"
        result = gemini.analyze(paths, csv_text, csv_prefix, prompt)
        
        print(f"\n{'='*30}\nAnalysis Result for {code}\n{result}\n{'='*30}")
        return f"Stock: {code} ({name})\nAnalysis:\n{result}"
    finally:
        for p in paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

def main():
    os.makedirs('output', exist_ok=True)
    config = AppConfig.from_env()
    
    google_svc = GoogleService(config.GOOGLE_SERVICE_ACCOUNT_JSON)
    jq_svc = JQuantsService(config.JQUANTS_API_KEY)
    yf_svc = YFinanceService()
    chart_svc = ChartService()
    gemini_svc = GeminiService(config.GEMINI_API_KEY, config.GEMINI_MODEL_NAME)
    
    stock_list = google_svc.fetch_stock_list(config.STOCK_LIST_SHEET_URL)

    stock_list = [{'code': 'GOOG', 'market': 'US'}, {'code': '4237', 'market': 'JP'}]
    logger.info(f"Target stocks: {len(stock_list)}")
    
    chart_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_code = {}
        for item in stock_list:
            code = item['code']
            market = item['market']
            provider = yf_svc if market == 'US' else jq_svc
            
            future = executor.submit(process_stock, code, config, provider, chart_svc, google_svc, gemini_svc, chart_lock)
            future_to_code[future] = code
            
        results = []
        for future in concurrent.futures.as_completed(future_to_code):
            code = future_to_code[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                logger.error(f"Error processing {code}: {e}", exc_info=True)

    if results:
        logger.info("Generating summary table...")
        summary_prompt = "Based on the following individual stock analyses, create a summary table. The table should include columns for Stock Code/Name, Overall Trend (Bullish/Bearish/Neutral), Key Technical Signals, and Recommended Action (Buy/Sell/Wait)."
        summary = gemini_svc.summarize(results, summary_prompt)
        print(f"\n{'='*30}\nSUMMARY TABLE\n{summary}\n{'='*30}")

if __name__ == "__main__":
    main()
