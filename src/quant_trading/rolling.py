from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


logger = logging.getLogger(__name__)


BASIC_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

FEATURES_TO_BIN = {
    'Open': 'open_bin',
    'High': 'high_bin',
    'Low': 'low_bin',
    'Close': 'close_bin',
    'Volume': 'volume_bin',
    'RSI_3': 'rsi_3_bin',
    'RSI_5': 'rsi_5_bin',
    'RSI_7': 'rsi_7_bin',
    'RSI_10': 'rsi_10_bin',
    'RSI_14': 'rsi_14_bin',
    '%K': 'stoch_k_bin',
    '%D': 'stoch_d_bin',
    'MACD': 'macd_bin',
    'MACD_Signal': 'macd_signal_bin',
    'MACD_Histogram': 'macd_hist_bin',
    'SMA_5': 'sma_5_bin',
    'SMA_10': 'sma_10_bin',
    'SMA_20': 'sma_20_bin',
    'SMA_50': 'sma_50_bin',
    'EMA_5': 'ema_5_bin',
    'EMA_12': 'ema_12_bin',
    'EMA_26': 'ema_26_bin',
    'BB_Upper': 'bb_upper_bin',
    'BB_Lower': 'bb_lower_bin',
    'BB_Middle': 'bb_middle_bin',
    'ATR_14': 'atr_14_bin',
    'Williams_R': 'williams_r_bin',
    'CCI_20': 'cci_20_bin',
    'OBV': 'obv_bin',
    'ADX': 'adx_bin',
    'DI_Plus': 'di_plus_bin',
    'DI_Minus': 'di_minus_bin',
    'Price_Change_Pct': 'price_change_pct_bin',
    'Volume_SMA_20': 'volume_sma_20_bin',
    'Price_SMA_20_Diff': 'price_sma_20_diff_bin',
    'Price_EMA_12_Diff': 'price_ema_12_diff_bin'
}


@dataclass
class RollingWindowConfig:
    input_path: Path = Path('data/feature.csv')
    output_path: Path = Path('data/rolling_window_100day_4bins.csv')
    window_size: int = 100
    num_bins: int = 4


class RollingWindowBinner:
    def __init__(self, features_to_bin: Dict[str, str] | None = None):
        self.features_to_bin = features_to_bin or FEATURES_TO_BIN
        self.basic_columns = BASIC_COLUMNS

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("準備數據...")
        df = data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        logger.info("數據準備完成，共 %d 條記錄", len(df))
        return df

    def create_bins_for_window(self, window_data: pd.DataFrame, num_bins: int) -> pd.DataFrame:
        df = window_data.copy()
        for feature, bin_column in self.features_to_bin.items():
            if feature in df.columns and len(df) >= num_bins:
                try:
                    df[bin_column] = pd.qcut(
                        df[feature],
                        q=num_bins,
                        labels=[f'Q{i+1}' for i in range(num_bins)],
                        duplicates='drop'
                    )
                except ValueError as exc:
                    logger.warning("特徵 %s 使用分位數分箱失敗，改用等頻分箱: %s", feature, exc)
                    df[bin_column] = pd.cut(
                        df[feature],
                        bins=num_bins,
                        labels=[f'Q{i+1}' for i in range(num_bins)]
                    )
        return df

    def add_window_features(self, window_data: pd.DataFrame) -> pd.DataFrame:
        df = window_data.copy()
        df['window_size'] = len(df)
        if 'Close' in df.columns:
            df['window_price_range'] = df['Close'].max() - df['Close'].min()
            df['window_price_volatility'] = df['Close'].std()
        if 'Volume' in df.columns:
            df['window_avg_volume'] = df['Volume'].mean()
        return df

    def create_rolling_windows(
        self, data: pd.DataFrame, window_size: int, num_bins: int
    ) -> List[pd.DataFrame]:
        logger.info("創建 %d 天的滾動窗口...", window_size)
        rolling_windows: List[pd.DataFrame] = []
        start_date = data['Date'].min()
        end_date = data['Date'].max()

        current_date = start_date
        window_id = 0

        while current_date <= end_date:
            window_end = current_date + pd.Timedelta(days=window_size - 1)
            window_data = data[(data['Date'] >= current_date) & (data['Date'] <= window_end)].copy()

            if len(window_data) > 0:
                window_data['window_id'] = window_id
                window_data['window_start'] = current_date
                window_data['window_end'] = min(window_end, end_date)
                window_data = self.create_bins_for_window(window_data, num_bins=num_bins)
                window_data = self.add_window_features(window_data)
                rolling_windows.append(window_data)
                window_id += 1

            current_date = current_date + pd.Timedelta(days=window_size)

        logger.info("創建了 %d 個滾動窗口", len(rolling_windows))
        return rolling_windows

    def process(self, data: pd.DataFrame, window_size: int = 100, num_bins: int = 4) -> pd.DataFrame:
        logger.info("開始滾動窗口特徵分箱處理...")
        prepared_data = self.prepare_data(data)
        rolling_windows = self.create_rolling_windows(prepared_data, window_size, num_bins)
        if rolling_windows:
            result = pd.concat(rolling_windows, ignore_index=True)
            logger.info("處理完成，共 %d 條記錄", len(result))
            return result
        logger.warning("沒有創建任何窗口，返回空DataFrame")
        return pd.DataFrame()


def load_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到輸入檔案: {path}")
    logger.info("讀取滾動窗口來源資料: %s", path)
    return pd.read_csv(path)


def save_results(data: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    logger.info("處理結果已輸出至: %s", path)
    return path


def summarize_bins(
    data: pd.DataFrame, features_to_bin: Dict[str, str] | None = None
) -> Dict[str, pd.Series]:
    features_to_bin = features_to_bin or FEATURES_TO_BIN
    summary: Dict[str, pd.Series] = {}
    for feature, bin_column in features_to_bin.items():
        if bin_column in data.columns:
            summary[feature] = data[bin_column].value_counts().sort_index()
    return summary


def run_rolling_window(config: RollingWindowConfig) -> pd.DataFrame:
    data = load_feature_table(config.input_path)
    processor = RollingWindowBinner()
    result = processor.process(data, window_size=config.window_size, num_bins=config.num_bins)
    if result.empty:
        raise ValueError("滾動窗口處理未產生任何結果")
    save_results(result, config.output_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="將特徵資料進行滾動窗口分箱")
    parser.add_argument("--input", type=Path, default=RollingWindowConfig.input_path, help="來源特徵檔案")
    parser.add_argument("--output", type=Path, default=RollingWindowConfig.output_path, help="輸出結果檔案")
    parser.add_argument("--window-size", type=int, default=RollingWindowConfig.window_size, help="窗口大小")
    parser.add_argument("--bins", type=int, default=RollingWindowConfig.num_bins, help="分箱數量")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日誌層級",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=getattr(logging, args.log_level))

    config = RollingWindowConfig(
        input_path=args.input,
        output_path=args.output,
        window_size=args.window_size,
        num_bins=args.bins,
    )

    try:
        result = run_rolling_window(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("已分箱 %d 個窗口、共 %d 筆資料", result['window_id'].nunique(), len(result))
    return 0


__all__ = [
    "RollingWindowConfig",
    "RollingWindowBinner",
    "run_rolling_window",
    "main",
]
