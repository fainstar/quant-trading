"""
股票量化交易系統 - Web 介面
提供股票清單管理、快速執行、排程管理和歷史記錄功能
"""

from flask import Flask, render_template, request, jsonify, send_file
import subprocess
import json
import os
import sys
from datetime import datetime
import threading
import schedule
import time

# 設定編碼
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'stocks': ["00631L.TW", "0050.TW", "2330.TW", "AAPL", "ETH-USD", "BTC-USD"],
        'years': 10
    }


def save_config(config):
    """保存股票配置"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


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
            return json.load(f)
    return {
        'enabled': False,
        'time': '20:00',
        'stocks': []
    }


def save_schedule_config(config):
    """保存排程配置"""
    with open(SCHEDULE_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def execute_stock_analysis(stock, years):
    """執行股票分析流程"""
    try:
        steps = [
            ("01_fetch_stock_data.py", ["-t", stock, "-y", str(years)]),
            ("02_feature_engineering.py", []),
            ("03_time_window_4bins.py", []),
            ("04_pretidy.py", []),
            ("05_predict_next_day_signal.py", []),
            ("06_send_report_to_discord.py", ["-s", stock])
        ]
        
        for script, args in steps:
            result = subprocess.run(
                ["python", script] + args,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'  # 處理編碼錯誤
            )
            if result.returncode != 0:
                error_msg = f"執行 {script} 失敗: {result.stderr[:200]}"  # 限制錯誤訊息長度
                add_history_record(stock, 'failed', error_msg)
                return False, error_msg
        
        add_history_record(stock, 'success', '所有步驟執行成功')
        return True, '執行成功'
    except Exception as e:
        error_msg = f'執行過程發生錯誤: {str(e)[:200]}'
        add_history_record(stock, 'error', error_msg)
        return False, error_msg


def schedule_job():
    """排程任務"""
    config = load_schedule_config()
    if config['enabled'] and config['stocks']:
        print(f"[{datetime.now()}] 執行排程任務...")
        stock_config = load_config()
        years = stock_config.get('years', 10)
        
        for stock in config['stocks']:
            execute_stock_analysis(stock, years)


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
    
    if not stocks:
        return jsonify({'success': False, 'message': '請選擇至少一支股票'})
    
    # 在後台線程執行
    def run_analysis():
        for stock in stocks:
            execute_stock_analysis(stock, years)
    
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
    save_schedule_config(data)
    
    # 重新設置排程
    schedule.clear()
    
    if data['enabled']:
        schedule.every().day.at(data['time']).do(schedule_job)
        
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
