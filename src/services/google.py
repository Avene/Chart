import json
import logging
import io
from typing import List, Dict, Any, TypeVar
from enum import Enum

import requests
import pandas as pd
import google.auth
from google.oauth2 import service_account
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator
from pydantic.json_schema import SkipJsonSchema

logger = logging.getLogger(__name__)

class StatusEnum(str, Enum):
    WATCHING = 'Watching'
    LONG_TERM_HOLD = 'LongTermHold'
    SHORT_TERM_HOLD = 'ShortTermHold'

class MarketEnum(str, Enum):
    US = 'US'
    JP = 'JP'

class ActionPlanEnum(str, Enum):
    StrongBuy = 'StrongBuy'
    Buy = 'Buy'
    Wait = 'Wait'
    Sell = 'Sell'
    StrongSell = 'StrongSell'


class WatchlistItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    ticker: str = Field(alias='Ticker', default="")
    market: MarketEnum | None = Field(alias='Market', default = None)
    name: str = Field(alias='Name', default="")
    status: StatusEnum | None = Field(alias='Status', default=None)
    route: str = Field(alias='Route', default="")
    current_price: str = Field(alias='CurrentPrice', default="")
    price_updated: str = Field(alias='PriceUpdated', default="")
    action_plan: ActionPlanEnum | None = Field(alias='ActionPlan', default=None)
    score: str = Field(alias='Score', default='', json_schema_extra={'example': ['10/10', '1/10']})
    loss_cut_target: str = Field(alias='LossCutTarget', default="", json_schema_extra={'example': ['5000円もしくは短期移動平均下抜け']})
    entry_target: str = Field(alias='EntryTarget', default="", json_schema_extra={'example': ['短期移動平均上抜けもしくは下ヒゲ陽線']})
    profit_take_target: str = Field(alias='ProfitTakeTarget', default="", json_schema_extra={'example': ['5000円もしくは3σ上抜けで半分利確']})
    comment: str = Field(alias='Comment', default="")
    plan_updated: str = Field(alias='PlanUpdated', default="")
    memo: str = Field(alias='Memo', default="")

class WatchlistItemWritable(WatchlistItem):
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        json_schema = handler(core_schema)
        json_schema = handler.resolve_ref_schema(json_schema)
        
        exclude = {
            'ticker', 'Ticker', 'market', 'Market', 'name', 'Name', 
            'status', 'Status', 'route', 'Route', 
            'current_price', 'CurrentPrice', 'price_updated', 'PriceUpdated', 'memo', 'Memo'
        }
        
        properties = json_schema.get('properties', {})
        for field in list(properties.keys()):
            if field in exclude:
                del properties[field]
        
        if 'required' in json_schema:
            json_schema['required'] = [f for f in json_schema['required'] if f not in exclude]
            
        return json_schema



T = TypeVar("T", bound="WatchlistItem")

class WatchlistRows(list[T]):
    def us(self) -> "WatchlistRows[T]":
        return WatchlistRows([r for r in self if r.market.strip().upper() == 'US'])

    def jp(self) -> "WatchlistRows[T]":
        return WatchlistRows([r for r in self if r.market.strip().upper() == 'JP'])

    def watching(self) -> "WatchlistRows[T]":
        return WatchlistRows([r for r in self if r.status == StatusEnum.WATCHING])

    def long_term_hold(self) -> "WatchlistRows[T]":
        return WatchlistRows([r for r in self if r.status == StatusEnum.LONG_TERM_HOLD])

    def short_term_hold(self) -> "WatchlistRows[T]":
        return WatchlistRows([r for r in self if r.status == StatusEnum.SHORT_TERM_HOLD])

class WatchlistData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    stocks: WatchlistRows[WatchlistItem]
    headers: List[str]
    indices: Dict[str, int]

    @field_validator('stocks', mode='before')
    @classmethod
    def _validate_rows(cls, v: Any) -> WatchlistRows:
        return WatchlistRows(v)

    def to_sheet_values(self) -> List[List[str]]:
        values = [self.headers]
        for row in self.stocks:
            d = row.model_dump(by_alias=True, mode='json')
            values.append([str(d.get(h, "") or "") for h in self.headers])
        return values

class GoogleService:
    """Handles Google Auth, Drive, Docs, and Sheets interactions."""
    SCOPES = [
        'https://www.googleapis.com/auth/drive.readonly',
        'https://www.googleapis.com/auth/spreadsheets',
    ]

    def __init__(self, sa_json: str | None):
        self.sa_json = sa_json
        self._creds: Credentials | None = None

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

    def setup_summary_sheet(self, spreadsheet_id: str, new_title: str = "Summary") -> None:
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

    def add_sheet(self, spreadsheet_id: str, title: str) -> None:
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

    def load_watchlist(self, spreadsheet_id: str, sheet_name: str) -> WatchlistData | None:
        logger.info(f"Reading {sheet_name} from {spreadsheet_id}...")
        rows = self.get_sheet_values(spreadsheet_id, sheet_name)
        
        if not rows:
            logger.warning(f"No data found in {sheet_name}.")
            return None

        headers = rows[0]
        data_rows = []        
        for r in rows[1:]:
            # Pad row with empty strings to match headers length for zipping
            r_padded = r + [""] * (len(headers) - len(r))
            row_dict = dict(zip(headers, r_padded))
            try:
                data_rows.append(WatchlistItem(**row_dict))
            except ValidationError as e:
                logger.warning(f"Skipping row due to validation error: {row_dict.get('Ticker', 'Unknown')} - {e}")
                continue

        # Identify columns
        try:

            indices = {
                'ticker': headers.index('Ticker'),
                'market': headers.index('Market'),
                'name': headers.index('Name'),
                'status': headers.index('Status'),
                'route': headers.index('Route'),
                'price': headers.index('CurrentPrice')                                   ,
                'updated': headers.index('PriceUpdated'),
                'action_plan': headers.index('ActionPlan'),
                'score': headers.index('Score'),
                'loss_cut': headers.index('LossCutTarget'),
                'entry': headers.index('EntryTarget'),
                'profit_take': headers.index('ProfitTakeTarget'),
                'comment': headers.index('Comment'),
                'plan_updated': headers.index('PlanUpdated'),
                'memo': headers.index('Memo'),
            }            
        except ValueError:
            logger.error("Sheet must contain 'Ticker' and 'Market' columns.")
            return None

        return WatchlistData(
            stocks=data_rows,
            headers=headers,
            indices=indices,
        )

    def update_sheet_values(self, spreadsheet_id: str, sheet_name: str, values: List[List[str]]) -> None:
        """Writes values to the specified sheet starting at A1."""
        safe_title = "".join(c for c in sheet_name if c not in r"*?:/\[\]").strip()[:31]
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/'{safe_title}'!A1?valueInputOption=USER_ENTERED"
        headers = self.get_auth_headers()
        try:
            requests.put(url, headers=headers, json={"values": values})
        except Exception as e:
            logger.exception(f"Failed to update values in '{sheet_name}': {e}")