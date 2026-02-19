import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

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