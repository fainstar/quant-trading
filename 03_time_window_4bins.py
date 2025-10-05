"""
滾動窗口特徵分箱處理程式
對時間序列數據進行滾動窗口分析，並在每個窗口內對技術指標進行分箱處理。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定義
BASIC_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

FEATURES_TO_BIN = {
    # 價格數據
    'Open': 'open_bin',
    'High': 'high_bin',
    'Low': 'low_bin',
    'Close': 'close_bin',
    'Volume': 'volume_bin',

    # RSI指標
    'RSI_3': 'rsi_3_bin',
    'RSI_5': 'rsi_5_bin',
    'RSI_7': 'rsi_7_bin',
    'RSI_10': 'rsi_10_bin',
    'RSI_14': 'rsi_14_bin',

    # 隨機指標
    '%K': 'stoch_k_bin',
    '%D': 'stoch_d_bin',

    # MACD指標
    'MACD': 'macd_bin',
    'MACD_Signal': 'macd_signal_bin',
    'MACD_Histogram': 'macd_hist_bin',

    # 移動平均線
    'SMA_5': 'sma_5_bin',
    'SMA_10': 'sma_10_bin',
    'SMA_20': 'sma_20_bin',
    'SMA_50': 'sma_50_bin',
    'EMA_5': 'ema_5_bin',
    'EMA_12': 'ema_12_bin',
    'EMA_26': 'ema_26_bin',

    # 布林帶
    'BB_Upper': 'bb_upper_bin',
    'BB_Lower': 'bb_lower_bin',
    'BB_Middle': 'bb_middle_bin',

    # 其他技術指標
    'ATR_14': 'atr_14_bin',
    'Williams_R': 'williams_r_bin',
    'CCI_20': 'cci_20_bin',
    'OBV': 'obv_bin',
    'ADX': 'adx_bin',
    'DI_Plus': 'di_plus_bin',
    'DI_Minus': 'di_minus_bin',

    # 自定義特徵
    'Price_Change_Pct': 'price_change_pct_bin',
    'Volume_SMA_20': 'volume_sma_20_bin',
    'Price_SMA_20_Diff': 'price_sma_20_diff_bin',
    'Price_EMA_12_Diff': 'price_ema_12_diff_bin'
}


class RollingWindowBinner:
    """滾動窗口分箱處理器"""

    def __init__(self, features_to_bin: Dict[str, str] = None):
        self.features_to_bin = features_to_bin or FEATURES_TO_BIN
        self.basic_columns = BASIC_COLUMNS

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """準備和預處理數據"""
        logger.info("準備數據...")
        df = data.copy()

        # 確保Date列是datetime格式並按時間排序
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        logger.info(f"數據準備完成，共 {len(df)} 條記錄")
        return df

    def create_bins_for_window(self, window_data: pd.DataFrame, num_bins: int) -> pd.DataFrame:
        """在單個窗口內對所有特徵進行分箱"""
        df = window_data.copy()

        for feature, bin_column in self.features_to_bin.items():
            if feature in df.columns and len(df) >= num_bins:
                try:
                    # 使用分位數分箱
                    df[bin_column] = pd.qcut(
                        df[feature],
                        q=num_bins,
                        labels=[f'Q{i+1}' for i in range(num_bins)],
                        duplicates='drop'
                    )
                except ValueError as e:
                    logger.warning(f"特徵 {feature} 使用分位數分箱失敗，改用等頻分箱: {e}")
                    # 如果無法分成num_bins個箱子，使用等頻分箱
                    df[bin_column] = pd.cut(
                        df[feature],
                        bins=num_bins,
                        labels=[f'Q{i+1}' for i in range(num_bins)]
                    )

        return df

    def add_window_features(self, window_data: pd.DataFrame) -> pd.DataFrame:
        """添加窗口級別的特徵"""
        df = window_data.copy()

        # 基本窗口統計
        df['window_size'] = len(df)

        # 價格相關特徵
        if 'Close' in df.columns:
            df['window_price_range'] = df['Close'].max() - df['Close'].min()
            df['window_price_volatility'] = df['Close'].std()

        # 成交量相關特徵
        if 'Volume' in df.columns:
            df['window_avg_volume'] = df['Volume'].mean()

        return df

    def create_rolling_windows(self, data: pd.DataFrame, window_size: int) -> List[pd.DataFrame]:
        """創建滾動窗口"""
        logger.info(f"創建 {window_size} 天的滾動窗口...")

        rolling_windows = []
        start_date = data['Date'].min()
        end_date = data['Date'].max()

        current_date = start_date
        window_id = 0

        while current_date <= end_date:
            window_end = current_date + pd.Timedelta(days=window_size-1)

            # 獲取當前窗口的數據
            window_data = data[(data['Date'] >= current_date) & (data['Date'] <= window_end)].copy()

            if len(window_data) > 0:
                # 添加窗口標識
                window_data['window_id'] = window_id
                window_data['window_start'] = current_date
                window_data['window_end'] = min(window_end, end_date)

                # 在窗口內進行分箱
                window_data = self.create_bins_for_window(window_data, num_bins=4)

                # 添加窗口特徵
                window_data = self.add_window_features(window_data)

                rolling_windows.append(window_data)
                window_id += 1

            # 移動到下一個窗口
            current_date = current_date + pd.Timedelta(days=window_size)

        logger.info(f"創建了 {len(rolling_windows)} 個滾動窗口")
        return rolling_windows

    def process(self, data: pd.DataFrame, window_size: int = 100) -> pd.DataFrame:
        """主處理函數"""
        logger.info("開始滾動窗口特徵分箱處理...")

        # 準備數據
        prepared_data = self.prepare_data(data)

        # 創建滾動窗口
        rolling_windows = self.create_rolling_windows(prepared_data, window_size)

        # 合併所有窗口
        if rolling_windows:
            result = pd.concat(rolling_windows, ignore_index=True)
            logger.info(f"處理完成，共 {len(result)} 條記錄")
            return result
        else:
            logger.warning("沒有創建任何窗口，返回空DataFrame")
            return pd.DataFrame()


def save_results_to_csv(data: pd.DataFrame, window_size: int, num_bins: int, output_dir: str = 'data') -> str:
    """保存結果到CSV文件"""
    output_file = f"{output_dir}/rolling_window_{window_size}day_{num_bins}bins.csv"
    data.to_csv(output_file, index=False)
    logger.info(f"結果已保存到: {output_file}")
    return output_file


def print_statistics(data: pd.DataFrame, features_to_bin: Dict[str, str] = None):
    """打印統計信息"""
    features_to_bin = features_to_bin or FEATURES_TO_BIN

    print(f"\n滾動窗口特徵分箱統計:")
    print(f"總窗口數量: {data['window_id'].nunique()}")
    print(f"總數據點數量: {len(data)}")

    for feature, bin_column in features_to_bin.items():
        if bin_column in data.columns:
            print(f"\n{feature} 分箱分佈:")
            bin_counts = data[bin_column].value_counts().sort_index()
            for bin_name, count in bin_counts.items():
                print(f"  {bin_name}: {count}")


def main():
    """主函數"""
    try:
        # 讀取數據
        data_file = 'data/feature.csv'
        logger.info(f"讀取數據文件: {data_file}")
        data = pd.read_csv(data_file)

        # 創建處理器
        processor = RollingWindowBinner()

        # 處理數據
        result = processor.process(data, window_size=100)

        if not result.empty:
            # 保存結果
            output_file = save_results_to_csv(result, window_size=100, num_bins=4)

            # 打印統計信息
            print_statistics(result)

            print("\n滾動窗口特徵4箱分箱完成！")
            print(f"創建的文件: {output_file}")
            print("\n分箱說明：")
            print("- 每個100天滾動窗口內對每個特徵進行4箱分位數分箱")
            print("- 分箱標籤格式：Q1/Q2/Q3/Q4")
            print("- Q1表示窗口內最低25%，Q4表示窗口內最高25%")
            print("- 數據按時間順序排序")
        else:
            logger.error("處理失敗，沒有生成結果")

    except FileNotFoundError as e:
        logger.error(f"數據文件未找到: {e}")
    except Exception as e:
        logger.error(f"處理過程中發生錯誤: {e}")
        raise


if __name__ == "__main__":
    main()