import json
import logging
import io
from typing import List, Optional, Dict, Any

import requests
import pandas as pd
import google.auth
from google.oauth2 import service_account
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest

logger = logging.getLogger(__name__)

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
            if len(row) <= idx_market:
                logger.warning(f"Skipping row with missing Market column: {row}")
                continue # Skip if Market column is missing in this row

            match row[idx_market].strip().upper():
                case 'US':
                    us_to_process[row[idx_ticker]] = row
                case 'JP':
                    jp_to_process[row[idx_ticker]] = row
                case _:
                    logger.warning(f"Unknown market '{row[idx_market]}' for ticker '{row[idx_ticker]}', skipping.   Row: {row}")
                    continue 
        
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
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/'{safe_title}'!A1?valueInputOption=USER_ENTERED"
        headers = self.get_auth_headers()
        try:
            requests.put(url, headers=headers, json={"values": values})
        except Exception as e:
            logger.exception(f"Failed to update values in '{sheet_name}': {e}")