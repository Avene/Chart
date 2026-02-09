import os
import logging
import json
import io
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import requests
import pandas as pd
import mplfinance as mpf
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

    def fetch_stock_list(self, uri: str, default_code: str = '79740') -> List[str]:
        """Fetches stock list from CSV URL or Google Sheet."""
        if not uri: return [default_code]
        
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
                return df.iloc[:, 0].astype(str).str.strip().tolist()
        except Exception as e:
            logger.error(f"Failed to fetch stock list: {e}")
        
        return [default_code]

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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def _generate(self, client, contents):
        return client.models.generate_content(model=self.model_name, contents=contents)

    def analyze(self, image_paths: List[str], prompt: str) -> str:
        client = genai.Client(api_key=self.api_key, http_options={'timeout': self.timeout_ms})
        try:
            imgs = [client.files.upload(file=p, config={'http_options': {'timeout': self.timeout_ms}}) for p in image_paths]
            response = self._generate(client, [prompt] + imgs)
            return response.text
        except Exception as e:
            logger.error(f"Gemini Analysis Failed: {e}")
            return "Error during analysis."

# --- Main Workflow ---

def process_stock(code: str, config: AppConfig, jq: JQuantsService, chart: ChartService, google_svc: GoogleService, gemini: GeminiService):
    logger.info(f"Processing {code}...")
    
    # 1. データ取得
    df = jq.get_daily_prices(code)
    if df.empty:
        logger.warning(f"No data for {code}")
        return
    name = jq.get_company_name(code)

    # 2. Charts
    charts_meta = [
        # (Data Slice, Filename, Title, MA Periods)
        (
            chart.add_indicators(df.copy(), [5, 25, 75]).tail(130),
            'output/chart_daily.png',
            f'{name} ({code}) Daily'
        ),
        (
            chart.add_indicators(df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna(), [13, 26, 52]).tail(104),
            'output/chart_weekly.png',
            f'{name} ({code}) Weekly'
        ),
        # (df_monthly, 'output/chart_monthly.png', 'Monthly Chart')
    ]
    
    paths = []
    for d, fname, title in charts_meta:
        if os.path.exists(fname): os.remove(fname)
        chart.create_chart_image(d, fname, title)
        paths.append(fname)

    # 3. Analysis
    prompt = google_svc.fetch_text_content(config.PROMPT_URI) or "Analyze these charts."
    result = gemini.analyze(paths, prompt)
    
    print(f"\n{'='*30}\nAnalysis Result for {code}\n{result}\n{'='*30}")

def main():
    os.makedirs('output', exist_ok=True)
    config = AppConfig.from_env()
    
    google_svc = GoogleService(config.GOOGLE_SERVICE_ACCOUNT_JSON)
    jq_svc = JQuantsService(config.JQUANTS_API_KEY)
    chart_svc = ChartService()
    gemini_svc = GeminiService(config.GEMINI_API_KEY, config.GEMINI_MODEL_NAME)
    
    codes = google_svc.fetch_stock_list(config.STOCK_LIST_SHEET_URL)
    logger.info(f"Target stocks: {codes}")
    
    for code in codes:
        try:
            process_stock(code, config, jq_svc, chart_svc, google_svc, gemini_svc)
        except Exception as e:
            logger.error(f"Error processing {code}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
