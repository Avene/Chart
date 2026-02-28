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
    stock_assets = generate_stock_assets(data.stocks, jq_svc, yf_svc)

    # --- Analysis ---
    analysis_groups = [
        (data.stocks.short_term_hold(), config.PROMPT_URI_SHORT_TERM, config.CHECKLIST_URI_SHORT_TERM, "Analyze this chart for a short-term trade setup.", "short_term"),
        (data.stocks.long_term_hold(), config.PROMPT_URI_LONG_TERM, config.CHECKLIST_URI_LONG_TERM, "Analyze this chart for a long-term investment.", "long_term"),
        (data.stocks.watching(), config.PROMPT_URI_WATCHING, config.CHECKLIST_URI_WATCHING, "Analyze this chart to see if it is worth watching.", "watching"),
    ]

    for stocks, prompt_uri, checklist_uri, default_prompt, key_suffix in analysis_groups:
        analyze_stocks(stocks, stock_assets, prompt_uri, checklist_uri, default_prompt, google_svc, gemini_svc, key_suffix)
            
def update_prices(google_svc, jq_svc, yf_svc, sid, sheet_name, data):
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
    return data

def generate_stock_assets(stocks, jq_svc, yf_svc):
    ChartService.clear_output_dir('output')

    assets = {}
    for row in stocks:
        ticker = row.ticker
        try:
            df = pd.DataFrame()
            match row.market:
                case 'US':
                    df = yf_svc.get_daily_prices(ticker, days=1825)
                case 'JP':
                    df = jq_svc.get_daily_prices(ticker, days=1825)
                        
            if df.empty:
                logger.warning(f"No data for {ticker}")
                continue

            chart_paths, df_daily = ChartService.generate_charts(ticker, df, 'output')
            assets[ticker] = {
                'paths': chart_paths,
                'csv': df_daily.tail(100).to_csv()
            }
        except Exception as e:
            logger.error(f"Failed to generate assets for {ticker}: {e}")
    return assets

def analyze_stocks(stocks, assets, prompt_uri, checklist_uri, default_prompt, google_svc, gemini_svc, cache_key_suffix):
    if not stocks:
        return

    logger.info(f"Starting analysis for {len(stocks)} stocks ({cache_key_suffix})...")
    
    prompt = default_prompt
    if prompt_uri:
        p = google_svc.fetch_text_content(prompt_uri)
        if p: prompt = p

    checklist_str = None
    if checklist_uri:
        checklist_str = google_svc.fetch_text_content(checklist_uri)
    
    cache_name = gemini_svc.setup_cache(prompt, checklist_str, cache_key=f"chart_analysis_{cache_key_suffix}")

    for row in stocks:
        ticker = row.ticker
        if ticker not in assets:
            continue
            
        asset = assets[ticker]
        try:
            analysis = gemini_svc.analyze(asset['paths'], asset['csv'], ticker, prompt, checklist_str, cached_content=cache_name)
            print(f"\n--- Analysis for {ticker} ({cache_key_suffix}) ---\n{analysis}\n")
            
        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")

if __name__ == "__main__":
    main()
