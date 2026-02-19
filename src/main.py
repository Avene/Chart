import os
import logging
import concurrent.futures
import pandas as pd

from config import AppConfig
from services.google import GoogleService
from services.market import JQuantsService, YFinanceService

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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
