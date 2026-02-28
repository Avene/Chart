import os
import logging
import shutil
import tempfile
from typing import List, Any, Tuple
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
from google import genai

logger = logging.getLogger(__name__)

class ChartService:
    """Handles Technical Analysis and Chart Plotting."""
    
    @staticmethod
    def clear_output_dir(output_dir: str):
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
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_ms = 300000
        self.client = genai.Client(api_key=self.api_key, http_options={'timeout': self.timeout_ms})

    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def _generate(self, contents, config=None):
        logger.info(f"Generating content with model {self.model_name}")
        return self.client.models.generate_content(model=self.model_name, contents=contents, config=config)

    def get_or_create_cache(self, cache_key: str, contents: List[Any], ttl_minutes: int = 5) -> str:
        try:
            for c in self.client.caches.list():
                if c.display_name == cache_key:
                    logger.info(f"Found cached content: {c.name}")
                    return c.name
        except Exception as e:
            logger.warning(f"Failed to list caches: {e}")

        logger.info(f"Creating cache: {cache_key}")
        return self.client.caches.create(
            model=self.model_name,
            contents=contents,
            config={'display_name': cache_key, 'ttl': f'{ttl_minutes}m'}
        ).name

    def setup_cache(self, prompt: str, checklist_content: str | None, cache_key: str = "chart_analysis_context", ttl_minutes: int = 10) -> str | None:
        try:
            contents = [prompt]
            if checklist_content:
                contents.append(checklist_content)
            return self.get_or_create_cache(cache_key, contents, ttl_minutes)
        except Exception as e:
            logger.warning(f"Cache setup failed: {e}")
            return None

    def analyze(self, image_paths: List[str], csv_data: str, csv_prefix: str, prompt: str, checklist_content: str | None, cached_content: str | None = None) -> str:
        try:
            imgs = [self.client.files.upload(file=p, config={'http_options': {'timeout': self.timeout_ms}}) for p in image_paths]
            
            csv_file = None
            if csv_data:
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, prefix=csv_prefix, suffix='.csv', encoding='utf-8') as tmp:
                    tmp.write(csv_data)
                    tmp_path = tmp.name
                try:
                    csv_file = self.client.files.upload(file=tmp_path, config={'mime_type': 'text/csv', 'http_options': {'timeout': self.timeout_ms}})
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            contents = []
            if not cached_content:
                contents.append(prompt)
            
            contents.extend(imgs)
            
            if csv_file:
                contents.append(csv_file)

            if checklist_content and not cached_content:
                contents.append(checklist_content)

            config = {'cached_content': cached_content} if cached_content else None
            response = self._generate(contents, config=config)
            
            if response.usage_metadata:
                logger.info(f"Gemini Usage: {response.usage_metadata}")
            return response.text
        except Exception as e:
            logger.exception(f"Gemini Analysis Failed: {e}")
            return "Error during analysis."

    def summarize(self, results: List[str], prompt: str) -> str:
        try:
            combined_text = "\n\n---\n\n".join(results)
            response = self._generate([prompt, combined_text])
            if response.usage_metadata:
                logger.info(f"Gemini Usage: {response.usage_metadata}")
            return response.text
        except Exception as e:
            logger.exception(f"Gemini Summary Failed: {e}")
            return "Error during summary."