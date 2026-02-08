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
    default_prompt = "この株価チャート（日足）を見て、スイングトレード視点で分析してください。移動平均線は20日です。"
    
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
def analyze_chart(image_path: str) -> str:
    # タイムアウトを延長 (WinError 10053対策: 5分)
    # google-genaiの仕様に合わせて整数(ミリ秒想定)で指定: 300000 = 5分
    timeout_ms = 300000
    client = genai.Client(api_key=GEMINI_API_KEY, http_options={'timeout': timeout_ms})
    
    try:
        # アップロード時にも設定を適用
        img = client.files.upload(file=image_path, config={'http_options': {'timeout': timeout_ms}})
        
        prompt = get_external_prompt(PROMPT_URI)
        logger.info(f'Using model: {GEMINI_MODEL_NAME}')
        response = generate_content_with_retry(client, GEMINI_MODEL_NAME, [prompt, img])
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

    # 2. チャート作成
    chart_path = 'output/chart.png'
    if os.path.exists(chart_path):
        os.remove(chart_path)
    logger.info("チャート作成中...")  
    create_chart(df, chart_path)

    # 3. 分析
    logger.info("Gemini分析中...")
    result = analyze_chart(chart_path)
    
    print("\n" + "="*30)
    print(result)
    print("="*30)
