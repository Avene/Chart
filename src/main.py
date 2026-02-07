import os
import requests
import pandas as pd
import mplfinance as mpf
from google import genai
from dotenv import load_dotenv

# .env を読み込む
load_dotenv()

# --- 設定 ---
STOCK_CODE = '79740'  # J-Quantsは末尾0が必要な場合が多い
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
JQUANTS_API_KEY = os.getenv('JQUANTS_API_KEY')

# --- 1. J-Quants データ取得 (簡易版) ---
def get_stock_data(code: str, days: int = 180) -> pd.DataFrame:
    # データ取得 (過去100日分)
    # V2 URL (コードは5桁推奨)
    code5 = code if len(code) == 5 else code + '0'
    headers = {'x-api-key': JQUANTS_API_KEY} 
    from_date = (pd.Timestamp.now() - pd.Timedelta(days=270)).strftime('%Y-%m-%d')
    to_date = (pd.Timestamp.now() - pd.Timedelta(days=150)).strftime('%Y-%m-%d')
    params = {
        'code': code5,
        'from': from_date,
        'to': to_date,
    }

    url = f"https://api.jquants.com/v2/equities/bars/daily"
      
    res = requests.get(url, params=params, headers=headers)
    res.raise_for_status()

    d = res.json()
    data = d["data"]
    while "pagination_key" in d:
        params["pagination_key"] = d["pagination_key"]
        res = requests.get(url, params=params, headers=headers)
        d = res.json()
        data += d["data"]

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # 数値型に変換
    cols = ['O', 'H', 'L', 'C', 'Vo']
    df[cols] = df[cols].astype(float)
    
    # mplfinance用にカラム名を変更
    df = df.rename(columns={'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close', 'Vo': 'Volume'})
    
    return df.tail(days) 

# --- 2. チャート作成 (mplfinance) ---
def create_chart(df: pd.DataFrame, filename: str) -> str:
    # スタイル指定でプロっぽいチャートに
    # mav=(5, 20, 75) で移動平均線を3本描画
    # volume=True で出来高表示
    # type='candle' でローソク足
    # style='yahoo' や 'binance' など選べます
    mpf.plot(df, type='candle', mav=(5, 20, 75), volume=True, 
             style='yahoo', savefig=filename)
    print(f"✅ チャート保存完了: {filename}")

    return filename

# --- 3. Gemini 分析 ---
def analyze_chart(image_path: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    img = client.files.upload(file=image_path)
    
    prompt = "この株価チャート（日足）を見て、スイングトレード視点で分析してください。移動平均線は20日です。"
    response = client.models.generate_content(
        model='gemini-3-flash-preview', contents=[prompt, img]
    )
    
    return response.text

# --- メイン実行 ---
if __name__ == "__main__":
    # 1. データ取得
    print("データ取得中...")
    df = get_stock_data(STOCK_CODE)  # J-Quantsからデータ取得に切り替えた場合はこちらを有効化

    # 2. チャート作成
    chart_path = 'output/chart.png'
    if os.path.exists(chart_path):
        os.remove(chart_path)
    print("チャート作成中...")  
    create_chart(df, chart_path)

    # 3. 分析
    print("Gemini分析中...")
    result = analyze_chart(chart_path)
    
    print("\n" + "="*30)
    print(result)
    print("="*30)
