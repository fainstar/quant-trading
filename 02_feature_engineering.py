import pandas as pd
import numpy as np

def calculate_rsi(data, period=14):
    """
    計算相對強弱指標 (RSI)
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic_oscillator(data, k_period=14, d_period=3):
    """
    計算隨機震盪指標 (%K 和 %D)
    """
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()

    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))

    # %D = Simple Moving Average of %K
    d_percent = k_percent.rolling(window=d_period).mean()

    return k_percent, d_percent

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    計算MACD指標
    """
    # EMA計算
    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()

    # MACD線 = Fast EMA - Slow EMA
    macd_line = ema_fast - ema_slow

    # 信號線 = EMA of MACD線
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # MACD柱狀圖 = MACD線 - 信號線
    macd_histogram = macd_line - signal_line

    return macd_line, signal_line, macd_histogram

def calculate_atr(data, period=14):
    """
    計算平均真實波幅 (ATR)
    """
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_williams_r(data, period=14):
    """
    計算威廉指標 (%R)
    """
    highest_high = data['High'].rolling(window=period).max()
    lowest_low = data['Low'].rolling(window=period).min()
    
    williams_r = -100 * ((highest_high - data['Close']) / (highest_high - lowest_low))
    return williams_r

def calculate_cci(data, period=20):
    """
    計算順勢指標 (CCI)
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mad_tp = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=False)
    
    cci = (typical_price - sma_tp) / (0.015 * mad_tp)
    return cci

def calculate_obv(data):
    """
    計算能量潮指標 (OBV)
    """
    obv = pd.Series(index=data.index, dtype=float)
    obv.iloc[0] = 0
    
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_adx(data, period=14):
    """
    計算平均趨向指標 (ADX)
    """
    # 計算真實波幅 (TR)
    tr = pd.concat([
        data['High'] - data['Low'],
        np.abs(data['High'] - data['Close'].shift(1)),
        np.abs(data['Low'] - data['Close'].shift(1))
    ], axis=1).max(axis=1)
    
    # 計算方向運動 (+DM, -DM)
    dm_plus = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']), 
                       np.maximum(data['High'] - data['High'].shift(1), 0), 0)
    dm_minus = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)), 
                        np.maximum(data['Low'].shift(1) - data['Low'], 0), 0)
    
    # 平滑處理
    atr = tr.rolling(window=period).mean()
    di_plus = 100 * (pd.Series(dm_plus).rolling(window=period).mean() / atr)
    di_minus = 100 * (pd.Series(dm_minus).rolling(window=period).mean() / atr)
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()
    
    return adx, di_plus, di_minus
    
def add_technical_indicators(data):
    """
    添加技術指標到數據中
    """
    # 確保數據按日期排序
    data = data.sort_values('Date').copy()

    # RSI with different periods
    data['RSI_3'] = calculate_rsi(data, period=3)
    data['RSI_5'] = calculate_rsi(data, period=5)
    data['RSI_7'] = calculate_rsi(data, period=7)
    data['RSI_10'] = calculate_rsi(data, period=10)
    data['RSI_14'] = calculate_rsi(data, period=14)

    # Stochastic Oscillator
    data['%K'], data['%D'] = calculate_stochastic_oscillator(data)

    # MACD
    data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = calculate_macd(data)

    # 簡單移動平均線 (SMA)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # 指數移動平均線 (EMA)
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # 布林帶 (Bollinger Bands)
    sma_20 = data['Close'].rolling(window=20).mean()
    std_20 = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = sma_20 + (std_20 * 2)
    data['BB_Lower'] = sma_20 - (std_20 * 2)
    data['BB_Middle'] = sma_20

    # ATR (Average True Range)
    data['ATR_14'] = calculate_atr(data, period=14)

    # Williams %R
    data['Williams_R'] = calculate_williams_r(data, period=14)

    # CCI (Commodity Channel Index)
    data['CCI_20'] = calculate_cci(data, period=20)

    # OBV (On-Balance Volume)
    data['OBV'] = calculate_obv(data)

    # ADX (Average Directional Index)
    data['ADX'], data['DI_Plus'], data['DI_Minus'] = calculate_adx(data, period=14)

    # 價格變化百分比
    data['Price_Change_Pct'] = data['Close'].pct_change() * 100

    # 成交量移動平均線
    data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()

    # 價格與移動平均線的差距
    data['Price_SMA_20_Diff'] = data['Close'] - data['SMA_20']
    data['Price_EMA_12_Diff'] = data['Close'] - data['EMA_12']

    return data
    # 讀取數據

if __name__ == "__main__":
    data = pd.read_csv('data/data.csv')

    # 添加技術指標
    data_with_indicators = add_technical_indicators(data)

    # 保存帶有指標的數據
    output_filename = 'data/feature.csv'
    data_with_indicators.to_csv(output_filename, index=False)

    print(f"技術指標已成功添加到數據中，並保存到 {output_filename}")

