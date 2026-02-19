import os
import logging
import tempfile
from typing import List
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
from google import genai

logger = logging.getLogger(__name__)

class ChartService:
    """Handles Technical Analysis and Chart Plotting."""
    
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
        
        ap = []
        # MA
        ma_cols = sorted([c for c in df.columns if c.startswith('MA')], key=lambda x: int(x[2:]))
        colors = ['orange', 'blue', 'green']
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
        mpf.plot(df, type='candle', volume=True, style='yahoo', savefig=filename, title=title, **kwargs)
        logger.info(f"✅ Chart saved: {filename}")
        return filename

class GeminiService:
    """Handles Gemini API interactions."""
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_ms = 300000

    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def _generate(self, client, contents):
        logger.info(f"Generating content with model {self.model_name}")
        return client.models.generate_content(model=self.model_name, contents=contents)

    def analyze(self, image_paths: List[str], csv_data: str, csv_prefix: str, prompt: str) -> str:
        client = genai.Client(api_key=self.api_key, http_options={'timeout': self.timeout_ms})
        try:
            imgs = [client.files.upload(file=p, config={'http_options': {'timeout': self.timeout_ms}}) for p in image_paths]
            
            csv_file = None
            if csv_data:
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, prefix=csv_prefix, suffix='.csv', encoding='utf-8') as tmp:
                    tmp.write(csv_data)
                    tmp_path = tmp.name
                try:
                    csv_file = client.files.upload(file=tmp_path, config={'mime_type': 'text/csv', 'http_options': {'timeout': self.timeout_ms}})
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            contents = [prompt] + imgs
            if csv_file:
                contents.append(csv_file)

            response = self._generate(client, contents)
            return response.text
        except Exception as e:
            logger.exception(f"Gemini Analysis Failed: {e}")
            return "Error during analysis."

    def summarize(self, results: List[str], prompt: str) -> str:
        client = genai.Client(api_key=self.api_key, http_options={'timeout': self.timeout_ms})
        try:
            combined_text = "\n\n---\n\n".join(results)
            response = self._generate(client, [prompt, combined_text])
            return response.text
        except Exception as e:
            logger.exception(f"Gemini Summary Failed: {e}")
            return "Error during summary."