import os
import logging
import json
import requests
import pandas as pd
import mplfinance as mpf
import google.auth
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from google import genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

# .env を読み込む
load_dotenv()

# --- ログ設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 設定 ---
STOCK_CODE = '79740'  # J-Quantsは末尾0が必要な場合が多い
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
JQUANTS_API_KEY = os.getenv('JQUANTS_API_KEY')
PROMPT_URI = os.getenv('PROMPT_URI')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME')

logger.info(f'model name : {os.getenv("GEMINI_MODEL_NAME")}')


# --- 1. J-Quants データ取得 (簡易版) ---
def get_stock_data(code: str, days: int = 1825) -> pd.DataFrame:
    # データ取得 (デフォルト過去5年分に短縮: API制限回避)
    # V2 URL (コードは5桁推奨)
    code5 = code if len(code) == 5 else code + '0'
    headers = {'x-api-key': JQUANTS_API_KEY} 
    base_date = pd.Timestamp.now()
    from_date = (base_date - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    to_date = base_date.strftime('%Y-%m-%d')
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
    cols = ['AdjO', 'AdjH', 'AdjL', 'AdjC', 'AdjVo']
    df[cols] = df[cols].astype(float)
    
    # mplfinance用にカラム名を変更
    df = df.rename(columns={'AdjO': 'Open', 'AdjH': 'High', 'AdjL': 'Low', 'AdjC': 'Close', 'AdjVo': 'Volume'})
    
    return df

# --- テクニカル指標追加 (MA計算) ---
def add_technical_indicators(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    for p in periods:
        df[f'MA{p}'] = df['Close'].rolling(window=p).mean()
    return df

# --- 2. チャート作成 (mplfinance) ---
def create_chart(df: pd.DataFrame, filename: str, title: str) -> str:
    # スタイル指定でプロっぽいチャートに
    # addplotで計算済みのMAを描画 (mav引数だと表示期間のみで計算されてしまうため)
    ap = []
    
    # MAカラムを抽出して期間順にソート
    ma_cols = sorted([c for c in df.columns if c.startswith('MA')], key=lambda x: int(x[2:]))
    colors = ['orange', 'blue', 'green']
    
    for i, col in enumerate(ma_cols):
        if not df[col].isna().all():
            # 色は期間が短い順に Orange, Blue, Green を割り当て
            color = colors[i % len(colors)]
            ap.append(mpf.make_addplot(df[col], color=color, width=1.0))
    
    # volume=True で出来高表示
    # type='candle' でローソク足
    kwargs = {'addplot': ap} if ap else {}
    mpf.plot(df, type='candle', volume=True, style='yahoo', savefig=filename, title=title, **kwargs)
    logger.info(f"✅ チャート保存完了: {filename}")

    return filename

def get_drive_credentials() -> Credentials:
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    # 1. Service Account (環境変数) - CI/CD用
    sa_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
    if sa_json:
        try:
            info = json.loads(sa_json)
            return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        except Exception:
            pass

    # 2. ADC (gcloud auth application-default login) へのフォールバック
    # ローカル開発(gcloud CLI) や GCP環境(Cloud Run等) で有効
    creds, _ = google.auth.default(scopes=SCOPES)
        
    return creds

# --- プロンプト取得 ---
def get_external_prompt(uri: str | None) -> str:
    default_prompt = "これらの株価チャート（日足・週足・月足）を見て、スイングトレード視点で分析してください。"
    
    if not uri:
        return default_prompt
        
    target_url = uri
    headers = {}
    auth_email = None
    
    # Google Docsの場合、テキストエクスポートURLに変換
    if "docs.google.com/document/d/" in uri:
        try:
            # 認証情報の取得を試みる (非公開ドキュメント対応)
            try:
                creds = get_drive_credentials()
                if hasattr(creds, 'service_account_email'):
                    auth_email = creds.service_account_email
                
                creds.refresh(GoogleAuthRequest())
                headers['Authorization'] = f'Bearer {creds.token}'
            except Exception:
                logger.warning('Failed to Authenticate to Google Docs. Trying public access...')

        except IndexError:
            logger.error(f"Failed to fetch prompt from Google Docs: {uri}")
            logger.info("Using default prompt.")
            return default_prompt

    try:
        # https://docs.google.com/document/d/DOC_ID/edit... -> DOC_ID
        doc_id = uri.split('/d/')[1].split('/')[0]
        target_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"            
        response = requests.get(target_url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.error(f"⛔ 403 Forbidden: Google Docへのアクセス権限がありません。")
            if auth_email:
                logger.info(f"👉 このメールアドレスをGoogle Docの「共有」に追加してください: {auth_email}")
            else:
                logger.info("👉 ドキュメントが公開されているか、認証情報が正しいか確認してください。")
        else:
            logger.error(f"⚠️ プロンプトの取得に失敗しました (HTTP {e.response.status_code}): {e}")
        logger.info("Using default prompt.")
        return default_prompt
    except Exception as e:
        logger.error(f"⚠️ プロンプトの取得に失敗しました: {e}")
        logger.info("Using default prompt.")
        return default_prompt

# --- Gemini API リトライ用関数 ---
# 503エラー (Overloaded) 対策: 指数バックオフでリトライ (4s, 8s, 16s... 最大60s待機, 5回試行)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=before_sleep_log(logger, logging.INFO)
)
def generate_content_with_retry(client, model, contents):
    return client.models.generate_content(
        model=model, contents=contents
    )

# --- 3. Gemini 分析 ---
def analyze_chart(image_paths: list[str]) -> str:
    # タイムアウトを延長 (WinError 10053対策: 5分)
    # google-genaiの仕様に合わせて整数(ミリ秒想定)で指定: 300000 = 5分
    timeout_ms = 300000
    client = genai.Client(api_key=GEMINI_API_KEY, http_options={'timeout': timeout_ms})
    
    try:
        # 画像をアップロード
        imgs = []
        for path in image_paths:
            imgs.append(client.files.upload(file=path, config={'http_options': {'timeout': timeout_ms}}))
        
        logger.info(f'Using prompt: {PROMPT_URI}')
        prompt = get_external_prompt(PROMPT_URI)
        logger.info(f'Using model: {GEMINI_MODEL_NAME}')
        response = generate_content_with_retry(client, GEMINI_MODEL_NAME, [prompt] + imgs)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return "分析中にエラーが発生しました。"

# --- メイン実行 ---
if __name__ == "__main__":
    # 出力ディレクトリを作成 (CI環境用)
    os.makedirs('output', exist_ok=True)

    # 1. データ取得
    logger.info("データ取得中...")
    df = get_stock_data(STOCK_CODE)

    # データ加工 (日足・週足・月足)
    # 日足: 直近6ヶ月 (約130営業日)
    df_daily = add_technical_indicators(df.copy(), [5, 25, 75])
    df_daily = df_daily.tail(130)
    
    # 週足: 直近24ヶ月 (約104週)
    df_weekly = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    df_weekly = add_technical_indicators(df_weekly, [13, 26, 52])
    df_weekly = df_weekly.tail(104)
    
    # 月足: 直近60ヶ月
    df_monthly = df.resample('ME').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    df_monthly = add_technical_indicators(df_monthly, [9, 24])
    df_monthly = df_monthly.tail(60)

    # 2. チャート作成 (3種類)
    chart_paths = []
    charts_config = [
        (df_daily, 'output/chart_daily.png', 'Daily Chart'),
        (df_weekly, 'output/chart_weekly.png', 'Weekly Chart'),
        (df_monthly, 'output/chart_monthly.png', 'Monthly Chart')
    ]

    logger.info("チャート作成中...")  
    for d, path, title in charts_config:
        if os.path.exists(path): os.remove(path)
        create_chart(d, path, title)
        chart_paths.append(path)

    # 3. 分析
    logger.info("Gemini分析中...")
    result = analyze_chart(chart_paths)
    
    print("\n" + "="*30)
    print(result)
    print("="*30)
