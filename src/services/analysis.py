from dataclasses import dataclass
import os
import concurrent.futures
import logging
import shutil
import tempfile
from typing import List, Any, Tuple, Dict
import pandas as pd
import matplotlib

from services.google import WatchlistItem, WatchlistItemWritable, WatchlistRows
from services.market import JQuantsService, YFinanceService

matplotlib.use('Agg')
import mplfinance as mpf
from google import genai
from google.genai.types import ContentListUnionDict, File, GenerateContentConfigOrDict

logger = logging.getLogger(__name__)


@dataclass
class StockAnalysisAsset:
    paths: List[str]
    last_100_pricing_data_csv: str
    watchlist_item: WatchlistItem

class ChartService:
    """Handles Technical Analysis and Chart Plotting."""

    @staticmethod
    def generate_stock_assets(
        stocks: WatchlistRows[WatchlistItem], 
        jq_svc: JQuantsService, 
        yf_svc: YFinanceService
    ) -> Dict[str, StockAnalysisAsset]:

        assets: Dict[str, StockAnalysisAsset] = {}
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
                assets[ticker] = StockAnalysisAsset(
                    paths=chart_paths,
                    last_100_pricing_data_csv=df_daily.tail(100).to_csv(),
                    watchlist_item=row
                )
            except Exception as e:
                logger.error(f"Failed to generate assets for {ticker}: {e}")
        
        return assets
    
    @staticmethod
    def clear_output_dir(output_dir: str) -> None:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        if df.empty: return df
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        return df.resample(rule).agg(agg_dict).dropna()

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

        # Custom Style: Black bg, Red Up, Green Down
        mc = mpf.make_marketcolors(up='red', down='green', edge='inherit', wick='inherit', volume='in')
        s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, facecolor='black', figcolor='black', gridcolor='#333333')
        
        ap = []
        # MA
        ma_cols = sorted([c for c in df.columns if c.startswith('MA')], key=lambda x: int(x[2:]))
        colors = ['cyan', 'yellow', 'orange']
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
        mpf.plot(df, type='candle', volume=True, style=s, savefig=filename, title=title, **kwargs)
        logger.info(f"✅ Chart saved: {filename}")
        return filename

    @staticmethod
    def generate_charts(ticker: str, df: pd.DataFrame, output_dir: str) -> Tuple[List[str], pd.DataFrame]:
        """Generates Daily, Weekly, and Monthly charts and returns paths and the daily DataFrame."""
        # Prepare DataFrames
        df_daily = ChartService.add_indicators(df.copy(), periods=[5, 25, 75])
        df_weekly = ChartService.add_indicators(ChartService.resample(df, 'W-FRI'), periods=[13, 26, 52])
        df_monthly = ChartService.add_indicators(ChartService.resample(df, 'ME'), periods=[9, 24, 60])

        paths = []
        # Daily: Last ~9 months (180 bars)
        path_d = os.path.join(output_dir, f"{ticker}_daily.png")
        ChartService.create_chart_image(df_daily.tail(180), path_d, f"{ticker} - Daily")
        paths.append(path_d)
        
        # Weekly: Last ~3 years (150 bars)
        path_w = os.path.join(output_dir, f"{ticker}_weekly.png")
        ChartService.create_chart_image(df_weekly.tail(150), path_w, f"{ticker} - Weekly")
        paths.append(path_w)
        
        # Monthly: Last ~5 years (60 bars)
        path_m = os.path.join(output_dir, f"{ticker}_monthly.png")
        ChartService.create_chart_image(df_monthly.tail(60), path_m, f"{ticker} - Monthly")
        paths.append(path_m)

        return paths, df_daily

class GeminiService:
    """Handles Gemini API interactions."""
    def __init__(self, api_key: str, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_ms = 300000
        self.client = genai.Client(api_key=self.api_key, http_options={'timeout': self.timeout_ms})

    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def _generate(self, contents: ContentListUnionDict, config: GenerateContentConfigOrDict | None = None) -> Any:
        logger.info(f"Generating content with model {self.model_name}")
        return self.client.models.generate_content(model=self.model_name, contents=contents, config=config)

    def analyze(self, asset: StockAnalysisAsset, prompt: str, checklist_content: str | None) -> WatchlistItemWritable:
        image_paths = asset.paths
        price_history_data = asset.last_100_pricing_data_csv

        tmp_path = None
        try:
            if price_history_data:
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, prefix=asset.watchlist_item.ticker, suffix='.csv', encoding='utf-8') as tmp:
                    tmp.write(price_history_data)
                    tmp_path = tmp.name

            def upload_helper(path: str, mime_type: str | None = None):
                config = {'http_options': {'timeout': self.timeout_ms}}
                if mime_type:
                    config['mime_type'] = mime_type
                return self.client.files.upload(file=path, config=config)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                img_futures = [executor.submit(upload_helper, p) for p in image_paths]
                csv_future = executor.submit(upload_helper, tmp_path, 'text/csv') if tmp_path else None
                
                chart_imgs = [f.result() for f in img_futures]
                price_history_csv = csv_future.result() if csv_future else None

            contents: List[str | File] = []
            contents.append(prompt)            
            contents.extend(chart_imgs)
            
            if price_history_csv:
                contents.append(price_history_csv)

            if checklist_content:
                contents.append(checklist_content)

            config = {
                'response_mime_type': 'application/json',
                'response_schema': WatchlistItemWritable
            }

            response = self._generate(contents, config=config)

            if response.usage_metadata:
                logger.info(f"Gemini Usage: {response.usage_metadata}")

            return WatchlistItemWritable.model_validate_json(response.text)

        except Exception as e:
            logger.exception(f"Gemini Analysis Failed: {e}")
            return WatchlistItemWritable(Comment="Error during analysis.")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def analyze_stocks(
        self,
        assets: Dict[str, StockAnalysisAsset], 
        prompt: str, 
        checklist_content: str | None, 
        cache_key_suffix: str
    ) -> None:
        if not assets:
            return

        logger.info(f"Starting analysis for {len(assets)} stocks ({cache_key_suffix})...")
        
        def process_asset(asset: StockAnalysisAsset) -> None:
            ticker = asset.watchlist_item.ticker
            try:
                analysis = self.analyze(asset, prompt, checklist_content)
                
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_asset, asset) for asset in assets.values()]
            concurrent.futures.wait(futures)

        logger.info(f"Finished analysis for {len(assets)} stocks ({cache_key_suffix}).")