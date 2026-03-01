from dataclasses import dataclass
import os
import logging
import concurrent.futures
import pandas as pd
from typing import Dict

from config import AppConfig
from services.google import GoogleService, WatchlistData, WatchlistItem
from services.market import JQuantsService, YFinanceService
from services.analysis import ChartService, GeminiService, StockAnalysisAsset

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- Main Workflow ---

def main() -> None:
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
        logger.error("Watchlist data is empty")
        return

    # 2. Process rows
    data = update_prices(google_svc, jq_svc, yf_svc, sid, sheet_name, data)

    gemini_svc = GeminiService(config.GEMINI_API_KEY, config.GEMINI_MODEL_NAME)
    
    # --- Pre-generate Charts ---
    logger.info("Generating charts for all target stocks...")
    ChartService.clear_output_dir('output')

    stock_assets_short_term = ChartService.generate_stock_assets(data.stocks.short_term_hold(), jq_svc, yf_svc)
    stock_assets_long_term = ChartService.generate_stock_assets(data.stocks.long_term_hold(), jq_svc, yf_svc)
    stock_assets_watching = ChartService.generate_stock_assets(data.stocks.watching(), jq_svc, yf_svc)

    # --- Analysis ---
    analysis_groups = [
        (stock_assets_short_term, config.PROMPT_URI_SHORT_TERM, config.CHECKLIST_URI_SHORT_TERM, "short_term"),
        (stock_assets_long_term, config.PROMPT_URI_LONG_TERM, config.CHECKLIST_URI_LONG_TERM,  "long_term"),
        (stock_assets_watching, config.PROMPT_URI_WATCHING, config.CHECKLIST_URI_WATCHING, "watching"),
    ]

    def process_group(assets: Dict[str, StockAnalysisAsset], prompt_uri: str, checklist_uri: str, key_suffix: str) -> None:
        if not assets: return
        prompt_text = google_svc.fetch_text_content(prompt_uri)
        checklist_text = google_svc.fetch_text_content(checklist_uri) if checklist_uri else None
        gemini_svc.analyze_stocks(assets, prompt_text, checklist_text, key_suffix)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_group, *group) for group in analysis_groups]
        concurrent.futures.wait(futures)

    logger.info("All analyses complete. Writing final results back to sheet...")
    google_svc.update_sheet_values(sid, sheet_name, data.to_sheet_values())
    logger.info("Analysis complete.")
            
def update_prices(
    google_svc: GoogleService, 
    jq_svc: JQuantsService, 
    yf_svc: YFinanceService, 
    sid: str, 
    sheet_name: str, 
    data: WatchlistData
) -> WatchlistData:
    now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # --- Process US Stocks in Bulk ---
    if (us_stocks := data.stocks.us()):
        logger.info(f"Processing {len(us_stocks)} US stocks in bulk...")
        us_tickers_all = [r.ticker for r in us_stocks]
        us_tickers_needing_name = [
            r.ticker for r in us_stocks if not r.name
        ]
        
        names_map = yf_svc.get_bulk_company_names(us_tickers_needing_name)
        prices_map = yf_svc.get_bulk_daily_prices(us_tickers_all, days=5)
        
        for row in us_stocks:
            ticker = row.ticker
            # Update name only if it was fetched (i.e., it was empty before)
            if ticker in names_map:
                row.name = names_map[ticker]
            if ticker in prices_map:
                row.current_price = str(prices_map[ticker])
                row.price_updated = now_str

    # --- Process JP Stocks concurrently ---
    def fetch_jp_data(row: WatchlistItem) -> None:
        ticker = row.ticker
        if not ticker: return
        
        try:
            # Only fetch and update name if the column is empty
            if not row.name:
                row.name = jq_svc.get_company_name(ticker)
            df = jq_svc.get_daily_prices(ticker, days=5)
            if not df.empty:
                row.current_price = str(df.iloc[-1]['Close'])
                row.price_updated = now_str
        except Exception as e:
            logger.error(f"Error fetching JP ticker {ticker}: {e}")

    if (jp_stocks := data.stocks.jp()):
        logger.info(f"Processing {len(jp_stocks)} JP stocks concurrently...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_jp_data, row) for row in jp_stocks]
            concurrent.futures.wait(futures)

    # 3. Assemble final table and write back
    logger.info("Writing updated data back to sheet...")
    google_svc.update_sheet_values(sid, sheet_name, data.to_sheet_values())
    logger.info("Done.")
    return data

if __name__ == "__main__":
    main()
