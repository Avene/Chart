from dataclasses import dataclass
import os
import logging
import concurrent.futures
import pandas as pd
from typing import Dict

from config import AppConfig
from services.google import GoogleService, WatchlistData, WatchlistItem
from services.market import JQuantsService, YFinanceService
from services.analysis import ChartService, GeminiService, StockAsset

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

    for assets, prompt_uri, checklist_uri, key_suffix in analysis_groups:
        analyze_stocks(assets, prompt_uri, checklist_uri, google_svc, gemini_svc, key_suffix)
            
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



def analyze_stocks(
    assets: Dict[str, StockAsset], 
    prompt_uri: str, 
    checklist_uri: str, 
    google_svc: GoogleService, 
    gemini_svc: GeminiService, 
    cache_key_suffix: str
) -> None:
    if not assets:
        return

    logger.info(f"Starting analysis for {len(assets)} stocks ({cache_key_suffix})...")
    
    prompt = google_svc.fetch_text_content(prompt_uri)

    checklist_str = None
    if checklist_uri:
        checklist_str = google_svc.fetch_text_content(checklist_uri)
    
    # TODO  Consider caching prompt and checklist content if the same URIs are used across groups to avoid redundant fetches
    for asset in assets.values():
        ticker = asset.watchlist_item.ticker
        try:
            analysis = gemini_svc.analyze(asset, prompt, checklist_str)
            
            asset.watchlist_item.score = analysis.score
            asset.watchlist_item.action_plan = analysis.action_plan
            asset.watchlist_item.loss_cut_target = analysis.loss_cut_target
            asset.watchlist_item.entry_target = analysis.entry_target
            asset.watchlist_item.profit_take_target = analysis.profit_take_target
            asset.watchlist_item.comment = analysis.comment
            asset.watchlist_item.memo = analysis.memo
            asset.watchlist_item.plan_updated = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

            print(f"\n--- Analysis for {ticker} ---\n{analysis.comment}\n")            
        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")
    # logger.info("Writing analysis results back to sheet...")
    # google_svc.update_sheet_values(sid, sheet_name, data.to_sheet_values())
    # logger.info("Analysis complete.")

if __name__ == "__main__":
    main()
