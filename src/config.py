import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class AppConfig:
    GEMINI_API_KEY: str
    JQUANTS_API_KEY: str
    GOOGLE_SERVICE_ACCOUNT_JSON: str

    GEMINI_MODEL_NAME: str


    PROMPT_URI_SHORT_TERM: str
    CHECKLIST_URI_SHORT_TERM: str

    PROMPT_URI_LONG_TERM: str
    CHECKLIST_URI_LONG_TERM: str

    PROMPT_URI_WATCHING: str
    CHECKLIST_URI_WATCHING: str

    STOCK_LIST_SHEET_URL: str

    REPORT_SPREADSHEET_ID: str
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        return cls(
            GEMINI_API_KEY=os.getenv('GEMINI_API_KEY', ''),
            JQUANTS_API_KEY=os.getenv('JQUANTS_API_KEY', ''),
            PROMPT_URI_SHORT_TERM=os.getenv('PROMPT_URI_SHORT_TERM', os.getenv('PROMPT_URI', '')),
            CHECKLIST_URI_SHORT_TERM=os.getenv('CHECKLIST_URI_SHORT_TERM', os.getenv('CHECKLIST_URI', '')),
            PROMPT_URI_LONG_TERM=os.getenv('PROMPT_URI_LONG_TERM', ''),
            CHECKLIST_URI_LONG_TERM=os.getenv('CHECKLIST_URI_LONG_TERM', ''),
            PROMPT_URI_WATCHING=os.getenv('PROMPT_URI_WATCHING', ''),
            CHECKLIST_URI_WATCHING=os.getenv('CHECKLIST_URI_WATCHING', ''),
            GEMINI_MODEL_NAME=os.getenv('GEMINI_MODEL_NAME', ''),
            STOCK_LIST_SHEET_URL=os.getenv('STOCK_LIST_SHEET_URL', ''),
            GOOGLE_SERVICE_ACCOUNT_JSON=os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON', ''),
            REPORT_SPREADSHEET_ID=os.getenv('REPORT_SPREADSHEET_ID', ''),
        )