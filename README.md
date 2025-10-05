# 股票量化交易系統

## 📊 專案簡介

這是一個基於貝葉斯分類器的股票量化交易系統，提供自動化的股票數據抓取、特徵工程、交易信號預測以及 Discord 報告推送功能。系統整合了 Web 介面，方便管理股票清單、執行任務和排程設定。

## ✨ 主要功能

- **自動化數據抓取**：支援台股、美股、加密貨幣等多種金融商品
- **智能特徵工程**：自動生成技術指標和滾動窗口特徵
- **貝葉斯交易策略**：基於統計學習的交易信號預測
- **回測系統**：完整的策略回測與績效分析
- **Web 管理介面**：友善的圖形化操作界面
- **Discord 推送**：每日交易報告自動推送
- **排程執行**：可設定定時自動執行任務

## 🚀 快速開始

### 環境需求

- Python 3.8+
- Windows PowerShell
- Docker (可選，用於容器化部署)

### 安裝方式

#### 方式一：Docker 部署（推薦）🐳

**1. 構建 Docker 映像**

```powershell
docker build -t oomaybeoo/quant-trading:latest .
```

**2. 運行容器**

```powershell
docker run -d `
  --name quant-trading `
  -p 5000:5000 `
  -v ${PWD}/config:/app/config `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/log:/app/log `
  -e TZ=Asia/Taipei `
  --restart unless-stopped `
  oomaybeoo/quant-trading:latest
```

**3. 訪問 Web 介面**

```
http://localhost:5000
```

**優點：**
- ✅ 無需手動安裝依賴
- ✅ 環境隔離，不影響系統
- ✅ 數據持久化，容器重啟不丟失
- ✅ 跨平台支援

**Docker 常用命令：**

```powershell
# 查看容器狀態
docker ps

# 查看日誌
docker logs -f quant-trading

# 停止容器
docker stop quant-trading

# 啟動容器
docker start quant-trading

# 重啟容器
docker restart quant-trading

# 刪除容器
docker rm quant-trading

# 進入容器 shell
docker exec -it quant-trading bash
```

**推送到 Docker Hub（可選）：**

```powershell
# 構建並推送
docker build -t oomaybeoo/quant-trading:latest .
docker push oomaybeoo/quant-trading:latest
```

#### 方式二：本地安裝

**1. 安裝依賴**

```powershell
pip install -r requirements.txt
```

或手動安裝：

```powershell
pip install yfinance pandas numpy flask schedule matplotlib seaborn requests
```

**2. 啟動應用**

### 快速啟動

#### 使用 Web 介面（推薦）

```powershell
python web_interface.py
```

然後在瀏覽器中訪問 `http://localhost:5000`

#### 命令列執行

```powershell
# 執行完整流程（自動處理配置中的所有股票）
python 00_main.py

# 執行單一股票分析
python 01_fetch_stock_data.py -t 2330.TW -y 10
python 02_feature_engineering.py
python 03_time_window_4bins.py
python 04_pretidy.py
python 05_predict_next_day_signal.py
python 06_send_report_to_discord.py -s 2330.TW
```

## 📁 專案結構

```
API_0924/
├── 00_main.py                          # 主程式（排程執行）
├── 01_fetch_stock_data.py              # 股票數據抓取
├── 02_feature_engineering.py           # 特徵工程
├── 03_time_window_4bins.py             # 滾動窗口分箱
├── 04_pretidy.py                       # 數據預處理
├── 05_predict_next_day_signal.py       # 交易信號預測
├── 06_send_report_to_discord.py        # Discord 報告推送
├── A05_bayesian_trading_strategy.py    # 貝葉斯交易策略
├── A06_bayesian_backtest.py            # 策略回測
├── B19_simulate_last_10_days.py        # 最近10天模擬
├── web_interface.py                    # Web 管理介面
├── Dockerfile                          # Docker 映像配置
├── requirements.txt                    # Python 依賴清單
├── .dockerignore                       # Docker 忽略文件
├── README.md                           # 專案說明文件
├── config/                             # 配置文件
│   ├── stocks_config.json              # 股票清單配置
│   └── schedule_config.json            # 排程配置
├── data/                               # 數據存儲
│   ├── data.csv                        # 原始數據
│   ├── feature.csv                     # 特徵數據
│   ├── final_data.csv                  # 最終數據
│   ├── rolling_window_100day_4bins.csv # 滾動窗口數據
│   ├── bayesian_trading_signals.csv    # 交易信號
│   └── *.png                           # 回測圖表
├── log/                                # 日誌記錄
│   ├── daily_trading_report.txt        # 每日交易報告
│   └── execution_history.json          # 執行歷史
└── templates/                          # Web 模板
    └── index.html                      # Web 介面首頁
```

## 🔧 核心模組說明

### 1. 數據抓取模組 (`01_fetch_stock_data.py`)

使用 yfinance API 抓取股票歷史數據，支援：
- 台灣股票（例：`2330.TW`、`0050.TW`）
- 美國股票（例：`AAPL`、`TSLA`）
- 加密貨幣（例：`BTC-USD`、`ETH-USD`）

**參數：**
- `-t, --ticker`：股票代號
- `-y, --years`：抓取年數（預設 5 年）

### 2. 特徵工程模組 (`02_feature_engineering.py`)

自動生成技術指標，包括：
- 移動平均線（MA5, MA10, MA20, MA60）
- 相對強弱指標（RSI）
- 移動平均收斂發散指標（MACD）
- 布林通道（Bollinger Bands）
- 成交量相關指標
- 波動率指標

### 3. 滾動窗口分箱 (`03_time_window_4bins.py`)

使用滾動窗口技術將連續型特徵轉換為離散型特徵（4 個分箱），提高貝葉斯分類器的預測效能。

### 4. 貝葉斯交易策略 (`A05_bayesian_trading_strategy.py`)

實現樸素貝葉斯分類器，根據歷史特徵預測未來交易信號：
- **買入信號**：預期未來 5 天價格上漲 > 3%
- **賣出信號**：預期未來 5 天價格下跌 > 3%
- **持有信號**：價格變動在 ±3% 之間

### 5. 策略回測 (`A06_bayesian_backtest.py`)

完整的回測系統，包含：
- 交易績效分析
- 年度收益統計
- 勝率計算
- 最大回撤分析
- 圖表可視化

### 6. Web 管理介面 (`web_interface.py`)

提供友善的 Web UI，功能包括：
- 股票清單管理（新增/刪除）
- 快速執行交易流程
- 排程設定（每日定時執行）
- 執行歷史查看
- 日誌下載

## ⚙️ 配置說明

### 股票配置 (`config/stocks_config.json`)

```json
{
  "stocks": [
    "00631L.TW",
    "0050.TW",
    "2330.TW",
    "AAPL",
    "ETH-USD",
    "BTC-USD"
  ],
  "years": 10
}
```

### 排程配置 (`config/schedule_config.json`)

```json
{
  "enabled": true,
  "time": "20:00",
  "stocks": ["2330.TW", "0050.TW"]
}
```

## 📈 使用範例

### 範例 1：分析台積電 (2330.TW)

```powershell
python 01_fetch_stock_data.py -t 2330.TW -y 10
python 02_feature_engineering.py
python 03_time_window_4bins.py
python 04_pretidy.py
python 05_predict_next_day_signal.py
python 06_send_report_to_discord.py -s 2330.TW
```

### 範例 2：回測策略

```powershell
# 生成交易信號
python A05_bayesian_trading_strategy.py

# 執行回測
python A06_bayesian_backtest.py
```

### 範例 3：模擬最近 10 天

```powershell
python B19_simulate_last_10_days.py
```

## 📊 輸出結果

系統會生成以下輸出：

1. **數據檔案** (`data/` 目錄)
   - 原始數據、特徵數據、最終數據
   - 交易信號 CSV

2. **視覺化圖表** (`data/` 目錄)
   - 回測績效曲線
   - 年度收益柱狀圖

3. **交易報告** (`log/` 目錄)
   - 每日交易建議
   - 執行歷史記錄

## 🔔 Discord 推送

系統支援將交易報告推送至 Discord 頻道。請在 `06_send_report_to_discord.py` 中配置您的 Discord Webhook URL。

## ⏰ 排程執行

使用 `00_main.py` 可設定每日定時執行：

```python
# 預設每天 20:00 執行
schedule.every().day.at("20:00").do(job)
```

也可透過 Web 介面動態調整排程時間。

## 📝 注意事項

1. **數據來源**：本系統使用 yfinance 抓取數據，請確保網路連線穩定
2. **免責聲明**：本系統僅供學習和研究使用，不構成任何投資建議
3. **風險警告**：股票交易有風險，投資需謹慎
4. **數據延遲**：即時數據可能有延遲，請勿用於高頻交易

## 🛠️ 故障排除

### 問題 1：無法抓取股票數據

- 檢查股票代號是否正確
- 確認網路連線
- 嘗試更新 yfinance：`pip install --upgrade yfinance`

### 問題 2：Web 介面無法啟動

- 確認 5000 端口未被佔用
- 檢查 Flask 是否正確安裝
- 查看終端錯誤訊息

### 問題 3：編碼錯誤

- 確保所有 Python 檔案以 UTF-8 編碼保存
- Windows 系統已自動處理編碼問題

### 問題 4：Docker 端口被佔用

```powershell
# 查找佔用 5000 端口的程序
netstat -ano | findstr :5000

# 停止佔用的程序或使用其他端口
docker run -d -p 8080:5000 ... # 改用 8080 端口
```

### 問題 5：Docker 容器無法訪問

```powershell
# 檢查容器是否正在運行
docker ps

# 查看容器日誌
docker logs quant-trading

# 檢查容器內部網路
docker exec quant-trading curl http://localhost:5000
```

## 🔄 系統流程圖

```
數據抓取 → 特徵工程 → 滾動窗口分箱 → 數據預處理 → 信號預測 → Discord推送
   ↓           ↓            ↓            ↓           ↓
data.csv → feature.csv → rolling_window → final_data → trading_signals
```

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

本專案僅供個人學習使用。

## 📧 聯絡方式

如有問題或建議，歡迎聯絡專案維護者。

---

**最後更新日期：2025年10月5日**

**版本：v1.0**
