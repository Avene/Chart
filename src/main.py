import os
import logging
import concurrent.futures
import pandas as pd

from config import AppConfig
from services.google import GoogleService
from services.market import JQuantsService, YFinanceService
from services.analysis import ChartService, GeminiService

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
        logger.error("Watchlist data is empty")
        return

    # 2. Process rows
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
    def fetch_jp_data(row):
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

    # --- Analysis for Short Term Hold ---
    short_term_stocks = data.stocks.short_term_hold()
    if not short_term_stocks:
        return

    logger.info(f"Starting analysis for {len(short_term_stocks)} short-term hold stocks...")
    gemini_svc = GeminiService(config.GEMINI_API_KEY, config.GEMINI_MODEL_NAME)
    
    prompt = "Analyze this chart for a short-term trade setup."
    if config.PROMPT_URI:
        p = google_svc.fetch_text_content(config.PROMPT_URI)
        if p: prompt = p

    for row in short_term_stocks:
        ticker = row.ticker
        market = row.market.strip().upper()
        
        try:
            df = pd.DataFrame()
            if market == 'US':
                df = yf_svc.get_daily_prices(ticker, days=180)
            elif market == 'JP':
                df = jq_svc.get_daily_prices(ticker, days=180)
            
            if df.empty:
                logger.warning(f"No data for {ticker}")
                continue

            df = ChartService.add_indicators(df, periods=[5, 25, 75])
            chart_path = os.path.join('output', f"{ticker}.png")
            ChartService.create_chart_image(df, chart_path, f"{ticker} - {row.name}")
            
            analysis = gemini_svc.analyze([chart_path], df.tail(10).to_csv(), ticker, prompt)
            print(f"\n--- Analysis for {ticker} ---\n{analysis}\n")
            
        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")

if __name__ == "__main__":
    main()
