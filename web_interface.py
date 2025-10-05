"""
股票量化交易系統 - Web 介面
提供股票清單管理、快速執行、排程管理和歷史記錄功能
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import threading
import schedule
import time


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

try:
    from quant_trading import fetch, features, rolling, pretidy, predict, reporting  # type: ignore[import]  # noqa: E402
except ImportError as exc:  # pragma: no cover - guard for misconfigured PYTHONPATH
    raise SystemExit("找不到 quant_trading 模組，請確認 src 目錄是否存在。") from exc

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# 配置文件路徑
CONFIG_FILE = 'config/stocks_config.json'
HISTORY_FILE = 'log/execution_history.json'
SCHEDULE_FILE = 'config/schedule_config.json'

# 確保目錄存在
os.makedirs('config', exist_ok=True)
os.makedirs('log', exist_ok=True)

# 全局變量
current_schedule = None
schedule_thread = None
is_running = False


def load_config():
    """載入股票配置"""
    defaults = {
        'stocks': ["00631L.TW", "0050.TW", "2330.TW", "AAPL", "ETH-USD", "BTC-USD"],
        'years': 10,
        'webhook_url': ''
    }

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}

        stocks = data.get('stocks', defaults['stocks'])
        if isinstance(stocks, list):
            cleaned_stocks = []
            for stock in stocks:
                if isinstance(stock, str) and stock.strip():
                    cleaned_stocks.append(stock.strip().upper())
            stocks = cleaned_stocks
        else:
            stocks = defaults['stocks']

        years = data.get('years', defaults['years'])
        try:
            years = int(years)
        except (TypeError, ValueError):
            years = defaults['years']
        years = max(1, min(years, 20))

        webhook_url = data.get('webhook_url', defaults['webhook_url'])
        if not isinstance(webhook_url, str):
            webhook_url = ''

        return {
            'stocks': stocks,
            'years': years,
            'webhook_url': webhook_url.strip(),
        }

    return defaults


def save_config(config):
    """保存股票配置"""
    stocks = config.get('stocks', []) if isinstance(config, dict) else []
    if isinstance(stocks, list):
        stocks = [stock.strip().upper() for stock in stocks if isinstance(stock, str) and stock.strip()]
    else:
        stocks = []

    years = config.get('years', 10) if isinstance(config, dict) else 10
    try:
        years = int(years)
    except (TypeError, ValueError):
        years = 10
    years = max(1, min(years, 20))

    webhook_url = ''
    if isinstance(config, dict):
        raw_webhook = config.get('webhook_url', '')
        webhook_url = raw_webhook.strip() if isinstance(raw_webhook, str) else ''

    payload = {
        'stocks': stocks,
        'years': years,
        'webhook_url': webhook_url,
    }

    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_history():
    """載入執行歷史"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_history(history):
    """保存執行歷史"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_history_record(stock, status, message):
    """添加歷史記錄"""
    history = load_history()
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock': stock,
        'status': status,
        'message': message
    }
    history.insert(0, record)  # 最新記錄在前
    # 只保留最近100條記錄
    if len(history) > 100:
        history = history[:100]
    save_history(history)


def load_schedule_config():
    """載入排程配置"""
    if os.path.exists(SCHEDULE_FILE):
        with open(SCHEDULE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'send_report' not in data:
            data['send_report'] = data.get('sendReport', True)
        else:
            data['send_report'] = bool(data['send_report'])
        data.pop('sendReport', None)
        return data
    return {
        'enabled': False,
        'time': '20:00',
        'stocks': [],
        'send_report': True
    }


def save_schedule_config(config):
    """保存排程配置"""
    with open(SCHEDULE_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def execute_stock_analysis(stock, years, send_report=True, webhook_url=None):
    """執行股票分析流程"""

    data_dir = Path("data")
    log_dir = Path("log")
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    configured_webhook = (webhook_url or '').strip()

    if send_report and not configured_webhook:
        logging.info("[%s] 未在配置中設定 Discord Webhook，將改用環境變數進行推送。", stock)

    def step_fetch() -> None:
        df = fetch.fetch_stock_data(stock, years=years)
        if df is None or df.empty:
            raise RuntimeError("抓取數據結果為空")
        df.to_csv(data_dir / "data.csv", index=False)

    steps.append(("抓取原始數據", step_fetch))

    def step_feature() -> None:
        config = features.FeatureEngineeringConfig(
            input_path=data_dir / "data.csv",
            output_path=data_dir / "feature.csv",
        )
        features.run_feature_engineering(config)

    steps.append(("特徵工程", step_feature))

    def step_rolling() -> None:
        config = rolling.RollingWindowConfig(
            input_path=data_dir / "feature.csv",
            output_path=data_dir / "rolling_window_100day_4bins.csv",
        )
        rolling.run_rolling_window(config)

    steps.append(("時間窗口切分", step_rolling))

    def step_pretidy() -> None:
        config = pretidy.PreTidyConfig(
            input_path=data_dir / "rolling_window_100day_4bins.csv",
            output_path=data_dir / "final_data.csv",
        )
        pretidy.run_pretidy(config)

    steps.append(("資料預處理", step_pretidy))

    def step_predict() -> None:
        config = predict.PredictionConfig(
            input_path=data_dir / "final_data.csv",
            report_output_path=log_dir / "daily_trading_report.txt",
            report_metadata_path=log_dir / "daily_trading_report.json",
        )
        predict.run_prediction_pipeline(config)

    steps.append(("預測隔日信號", step_predict))

    if send_report:
        def step_report() -> None:
            report_config = reporting.DiscordReportConfig(
                report_path=log_dir / "daily_trading_report.txt",
                metadata_path=log_dir / "daily_trading_report.json",
                stock_name=stock,
                webhook_url=configured_webhook or None,
            )
            reporting.run_report_dispatch(report_config)

        steps.append(("發送 Discord 報告", step_report))

    for description, action in steps:
        try:
            logging.info("[%s] 開始: %s", stock, description)
            action()
            logging.info("[%s] 完成: %s", stock, description)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{description} 失敗: {str(exc)[:500]}"
            add_history_record(stock, 'failed', error_msg)
            return False, error_msg

    add_history_record(stock, 'success', '所有步驟執行成功')
    return True, '執行成功'


def schedule_job():
    """排程任務"""
    config = load_schedule_config()
    if config['enabled'] and config['stocks']:
        print(f"[{datetime.now()}] 執行排程任務...")
        stock_config = load_config()
        years = stock_config.get('years', 10)
        send_report = config.get('send_report', True)
        webhook_url = (stock_config.get('webhook_url', '') or '').strip()
        
        for stock in config['stocks']:
            execute_stock_analysis(stock, years, send_report=send_report, webhook_url=webhook_url)


def run_schedule():
    """運行排程循環"""
    global is_running
    while is_running:
        schedule.run_pending()
        time.sleep(1)


@app.route('/')
def index():
    """首頁"""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """獲取配置"""
    return jsonify(load_config())


@app.route('/api/config', methods=['POST'])
def update_config():
    """更新配置"""
    data = request.json
    save_config(data)
    return jsonify({'success': True, 'message': '配置已更新'})


@app.route('/api/execute', methods=['POST'])
def execute():
    """執行分析"""
    data = request.json
    stocks = data.get('stocks', [])
    years = data.get('years', 10)
    send_report = bool(data.get('send_report', data.get('sendReport', True)))
    stock_config = load_config()
    webhook_url = (stock_config.get('webhook_url', '') or '').strip()
    
    if not stocks:
        return jsonify({'success': False, 'message': '請選擇至少一支股票'})
    
    # 在後台線程執行
    def run_analysis():
        for stock in stocks:
            execute_stock_analysis(stock, years, send_report=send_report, webhook_url=webhook_url)
    
    thread = threading.Thread(target=run_analysis)
    thread.start()
    
    return jsonify({'success': True, 'message': f'開始分析 {len(stocks)} 支股票'})


@app.route('/api/history', methods=['GET'])
def get_history():
    """獲取歷史記錄"""
    return jsonify(load_history())


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """清除歷史記錄"""
    save_history([])
    return jsonify({'success': True, 'message': '歷史記錄已清除'})


@app.route('/api/schedule', methods=['GET'])
def get_schedule():
    """獲取排程配置"""
    return jsonify(load_schedule_config())


@app.route('/api/schedule', methods=['POST'])
def update_schedule():
    """更新排程配置"""
    global is_running, schedule_thread
    
    data = request.json
    normalized = {
        'enabled': bool(data.get('enabled', False)),
        'time': data.get('time', '20:00'),
        'stocks': data.get('stocks', []),
        'send_report': bool(data.get('send_report', data.get('sendReport', True)))
    }
    save_schedule_config(normalized)
    
    # 重新設置排程
    schedule.clear()
    
    if normalized['enabled']:
        schedule.every().day.at(normalized['time']).do(schedule_job)
        
        if not is_running:
            is_running = True
            schedule_thread = threading.Thread(target=run_schedule, daemon=True)
            schedule_thread.start()
    else:
        is_running = False
    
    return jsonify({'success': True, 'message': '排程已更新'})


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 股票量化交易系統 Web 介面")
    print("=" * 60)
    print("📡 服務啟動中...")
    print("🌐 請在瀏覽器訪問: http://localhost:5000")
    print("=" * 60)
    
    # 初始化排程
    schedule_config = load_schedule_config()
    if schedule_config['enabled']:
        schedule.every().day.at(schedule_config['time']).do(schedule_job)
        is_running = True
        schedule_thread = threading.Thread(target=run_schedule, daemon=True)
        schedule_thread.start()
        print(f"⏰ 排程已啟用，執行時間: {schedule_config['time']}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
