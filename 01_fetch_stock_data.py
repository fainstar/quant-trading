import yfinance as yf
import pandas as pd
import os
import argparse
import sys

# 設定輸出編碼為 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fetch_stock_data(ticker_symbol: str, years: int = 5):
    """
    抓取指定股票的歷史資料，修正格式後返回 DataFrame。

    :param ticker_symbol: 股票代號 (例如 'AAPL', '2330.TW', 'ETH-USD')
    :param years: 要抓取的年數 (預設 5)
    :return: DataFrame 包含股票數據
    """
    try:
        print(f"正在抓取股票 {ticker_symbol} 的 {years} 年歷史資料...")
    except:
        print(f"Fetching {ticker_symbol} data for {years} years...")

    try:
        # 抓取資料
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period=f"{years}y")

        if data.empty:
            try:
                print(f"警告: 無法獲取股票 {ticker_symbol} 的資料")
            except:
                print(f"Warning: Cannot fetch data for {ticker_symbol}")
            return None

        # 轉換 index 為欄位
        data.reset_index(inplace=True)

        # 修正日期格式：移除時間和時區，只保留日期
        data['Date'] = data['Date'].astype(str).str.split(' ').str[0]

        # 將價格欄位取到小數點第六位
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].round(6)

        try:
            print(f"成功抓取 {len(data)} 條數據 (從 {data['Date'].min()} 到 {data['Date'].max()})")
        except:
            print(f"Successfully fetched {len(data)} records")
        return data

    except Exception as e:
        try:
            print(f"抓取股票 {ticker_symbol} 資料時發生錯誤: {e}")
        except:
            print(f"Error fetching {ticker_symbol}: {e}")
        return None


def get_user_input():
    """
    獲取用戶輸入的股票代號和年數
    """
    try:
        print("=" * 50)
        print("📈 股票資料抓取工具")
        print("=" * 50)
    except:
        print("=" * 50)
        print("Stock Data Fetcher")
        print("=" * 50)

    # 股票代號輸入
    while True:
        try:
            ticker = input("請輸入股票代號 (例如: AAPL, 2330.TW, ETH-USD): ").strip().upper()
        except:
            ticker = input("Enter ticker symbol (e.g. AAPL, 2330.TW, ETH-USD): ").strip().upper()
        if ticker:
            break
        try:
            print("股票代號不能為空，請重新輸入")
        except:
            print("Ticker cannot be empty")

    # 年數輸入
    while True:
        try:
            try:
                years_input = input("請輸入要抓取的年數 (預設5年，直接按Enter使用預設值): ").strip()
            except:
                years_input = input("Enter years (default 5, press Enter for default): ").strip()
            if not years_input:
                years = 5
                break
            years = int(years_input)
            if years > 0 and years <= 20:
                break
            else:
                try:
                    print("年數必須在1-20之間")
                except:
                    print("Years must be between 1-20")
        except ValueError:
            try:
                print("請輸入有效的數字")
            except:
                print("Please enter a valid number")

    return ticker, years


def main():
    """
    主函數：處理命令行參數或用戶輸入
    """
    parser = argparse.ArgumentParser(description='抓取股票歷史資料')
    parser.add_argument('-t', '--ticker', type=str, help='股票代號 (例如: AAPL, 2330.TW, ETH-USD)')
    parser.add_argument('-y', '--years', type=int, default=5, help='要抓取的年數 (預設: 5)')
    parser.add_argument('-o', '--output', type=str, default='data/data.csv', help='輸出文件路徑 (預設: data/data.csv)')
    parser.add_argument('--interactive', action='store_true', help='使用互動模式輸入參數')

    args = parser.parse_args()

    # 決定使用哪種輸入方式
    if args.interactive or (not args.ticker):
        # 互動模式
        ticker, years = get_user_input()
        filename = args.output
    else:
        # 命令行參數模式
        ticker = args.ticker.upper()
        years = args.years
        filename = args.output

    # 確保目錄存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 抓取數據
    data = fetch_stock_data(ticker, years=years)

    if data is not None:
        # 保存原始數據
        data.to_csv(filename, index=False)
        try:
            print(f"✅ 股票 {ticker} 的 {years} 年原始資料已成功抓取，並保存到 {filename}")
            print(f"📊 數據概覽: {len(data)} 條記錄，日期範圍: {data['Date'].min()} - {data['Date'].max()}")
        except:
            print(f"[SUCCESS] Stock {ticker} data saved to {filename}")
            print(f"Records: {len(data)}, Date range: {data['Date'].min()} - {data['Date'].max()}")
    else:
        try:
            print(f"❌ 無法獲取股票 {ticker} 的資料，請檢查股票代號是否正確")
        except:
            print(f"[FAILED] Cannot fetch data for {ticker}")


if __name__ == "__main__":
    main()
