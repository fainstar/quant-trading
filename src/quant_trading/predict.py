from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


MODEL_NAME_ZH = '樸素貝葉斯'
MODEL_NAME_EN = 'Categorical Naive Bayes'

# 交易訊號策略參數
FUTURE_DAYS = 5
PCT_CHANGE_THRESHOLD = 0.03

def define_signal(price_change: float) -> str:
    """
    根據價格變化定義交易訊號
    
    Args:
        price_change: 價格變化百分比
        
    Returns:
        str: 交易訊號 - '買入', '賣出' 或 '持有'
    """
    if price_change > PCT_CHANGE_THRESHOLD:
        return '買入'
    elif price_change < -PCT_CHANGE_THRESHOLD:
        return '賣出'
    else:
        return '持有'


class NaiveBayesClassifier:
    """簡單的類別型樸素貝葉斯分類器，支援拉普拉斯平滑。"""

    def __init__(self, laplace_smoothing: float = 1.0) -> None:
        if laplace_smoothing <= 0:
            raise ValueError('laplace_smoothing 必須為正數')
        self.laplace_smoothing = float(laplace_smoothing)
        self.priors: Dict[int, float] = {}
        self.likelihoods: dict[str, dict[int, dict[float, float]]] = defaultdict(lambda: defaultdict(dict))
        self.features: list[str] = []
        self.classes_: list[int] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NaiveBayesClassifier':
        if len(X) != len(y):
            raise ValueError('特徵與標籤長度不一致')

        y = pd.Series(y).astype(int)
        self.features = list(X.columns)
        # 確保類別包含 -1, 0, 1，即使數據中可能暫時沒有
        # 這對於三態交易訊號（買入、持有、賣出）很重要
        unique_classes = sorted(y.unique().tolist())
        self.classes_ = sorted(list(set(unique_classes + [-1, 0, 1])))
        n_samples = len(y)

        if not self.classes_:
            raise ValueError('訓練資料中沒有任何類別標籤')

        for action in self.classes_:
            action_count = int((y == action).sum())
            self.priors[action] = (action_count + self.laplace_smoothing) / (
                n_samples + self.laplace_smoothing * len(self.classes_)
            )

        for feature in self.features:
            feature_series = X[feature]
            unique_values = pd.Series(feature_series.unique()).dropna().tolist()
            if not unique_values:
                unique_values = [0.0]
            num_unique = len(unique_values)

            for action in self.classes_:
                action_mask = y == action
                feature_values_in_action = feature_series[action_mask]
                value_counts = feature_values_in_action.value_counts()
                total_in_action = len(feature_values_in_action)

                denominator = total_in_action + num_unique * self.laplace_smoothing
                if denominator == 0:
                    denominator = num_unique * self.laplace_smoothing

                for value in unique_values:
                    value_key = float(value)
                    count = value_counts.get(value, 0)
                    probability = (count + self.laplace_smoothing) / denominator
                    self.likelihoods[feature][action][value_key] = probability

                # 若遇到未見值，給予極小平滑機率備用
                self.likelihoods[feature][action]['__unknown__'] = self.laplace_smoothing / (
                    denominator if denominator > 0 else self.laplace_smoothing * num_unique
                )

        return self

    def _compute_log_posteriors(self, row: pd.Series) -> Dict[int, float]:
        posteriors: Dict[int, float] = {}
        for action in self.classes_:
            log_posterior = float(np.log(self.priors.get(action, 1e-12)))
            for feature in self.features:
                value_key = float(row.get(feature, 0.0))
                feature_likelihoods = self.likelihoods.get(feature, {}).get(action, {})
                probability = feature_likelihoods.get(value_key)
                if probability is None or probability <= 0:
                    probability = feature_likelihoods.get('__unknown__', 1e-12)
                log_posterior += float(np.log(probability))
            posteriors[action] = log_posterior
        return posteriors

    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        log_probas: list[list[float]] = []
        for _, row in X.iterrows():
            posteriors = self._compute_log_posteriors(row)
            log_probas.append([posteriors[action] for action in self.classes_])
        return np.array(log_probas)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        log_probas = self.predict_log_proba(X)
        # 使用 log-sum-exp 轉換為實際機率
        max_log = log_probas.max(axis=1, keepdims=True)
        stabilized = np.exp(log_probas - max_log)
        sums = stabilized.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            probas = np.divide(stabilized, sums, where=sums > 0)
        return probas

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probas = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return np.array([self.classes_[idx] for idx in class_indices])


@dataclass
class PredictionConfig:
    input_path: Path = Path('data/final_data.csv')
    model_output_path: Path = Path('data/trained_model.pkl')
    report_output_path: Path = Path('log/daily_trading_report.txt')
    report_metadata_path: Path = Path('log/daily_trading_report.json')
    test_size: float = 0.2
    random_state: int = 42
    rolling_window_size: int = 500


def load_final_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到最終整理資料檔: {path}")
    logger.info("載入整理後資料: %s", path)
    return pd.read_csv(path)


def create_labels(data: pd.DataFrame) -> pd.Series:
    if 'Close' not in data.columns:
        raise ValueError("資料中缺少 Close 欄位，無法生成標籤")
    
    # 使用全局變量定義的交易訊號策略參數
    # 計算未來價格變動
    future_close = data['Close'].shift(-FUTURE_DAYS)
    price_change = (future_close - data['Close']) / data['Close']
    
    # 使用 define_signal 函數將價格變化映射為交易訊號
    # 然後將文本訊號轉換為數值：1=買入, -1=賣出, 0=持有
    signal_mapping = {'買入': 1, '賣出': -1, '持有': 0}
    labels = price_change.apply(define_signal).map(signal_mapping)
    
    # 因為使用了future_days天的未來數據，需要裁剪相應的尾部數據
    return pd.Series(labels, name='next_day_signal').iloc[:-FUTURE_DAYS]


def prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("準備特徵與標籤...")
    label_series = create_labels(data)
    # 根據標籤長度調整特徵長度，確保特徵與標籤長度一致
    feature_frame = data.iloc[:-FUTURE_DAYS].drop(columns=['Date', 'Ticker'], errors='ignore')
    encoded_features = pd.get_dummies(feature_frame, dummy_na=False)
    encoded_features = encoded_features.fillna(0)
    return encoded_features, label_series


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> NaiveBayesClassifier:
    logger.info("訓練 %s (%s) 模型...", MODEL_NAME_ZH, MODEL_NAME_EN)
    model = NaiveBayesClassifier()
    model.fit(X_train, y_train)
    return model


def rolling_train_and_evaluate(
    features: pd.DataFrame,
    labels: pd.Series,
    window_size: int
) -> Tuple[NaiveBayesClassifier, Dict[str, float | str | List[int] | List[float]]]:
    if window_size <= 0:
        raise ValueError('滾動窗口大小必須為正整數')

    total_samples = len(features)
    if total_samples <= 1:
        raise ValueError('可用樣本數不足，至少需要 2 筆資料')

    if total_samples < window_size:
        logger.warning(
            "可用樣本數 (%d) 小於設定的滾動窗口大小 (%d)，將改用 %d 筆資料視窗",
            total_samples,
            window_size,
            total_samples - 1,
        )
        window_size = total_samples - 1

    predictions: List[int] = []
    actuals: List[int] = []
    probabilities: List[float] = []
    model: NaiveBayesClassifier | None = None

    logger.info("以 %d 筆資料窗口執行每日滾動訓練...", window_size)
    for idx in range(window_size, total_samples):
        X_train = features.iloc[idx - window_size:idx]
        y_train = labels.iloc[idx - window_size:idx]
        model = train_model(X_train, y_train)

        X_test = features.iloc[idx:idx + 1]
        y_true = int(labels.iloc[idx])
        y_pred = int(model.predict(X_test)[0])
        predictions.append(y_pred)
        actuals.append(y_true)

        if hasattr(model, 'predict_proba') and len(model.classes_) > 1:
            # 獲取預測類別的概率作為信心值
            proba_dist = model.predict_proba(X_test)[0]
            predicted_class_idx = np.argmax(proba_dist)
            probability = float(proba_dist[predicted_class_idx])
        else:
            probability = 1.0 if y_pred == y_true else 0.0
        probabilities.append(probability)

    if model is None:
        # 若沒有產生任何測試點 (資料量恰為窗口大小)，至少訓練出最後模型
        X_train = features.iloc[-window_size:]
        y_train = labels.iloc[-window_size:]
        model = train_model(X_train, y_train)

    accuracy = accuracy_score(actuals, predictions) if actuals else 0.0
    report = (
        classification_report(actuals, predictions, zero_division=0)
        if actuals
        else '樣本不足，無法生成滾動驗證報告'
    )

    evaluation = {
        'accuracy': accuracy,
        'classification_report': report,
        'rolling_predictions': predictions,
        'rolling_actuals': actuals,
        'rolling_probabilities': probabilities,
    }

    return model, evaluation


def prepare_latest_features(latest_row: pd.Series, feature_columns: List[str]) -> pd.DataFrame:
    latest_features = latest_row.drop(labels=['Date', 'Ticker'], errors='ignore')
    latest_df = pd.DataFrame([latest_features])
    latest_encoded = pd.get_dummies(latest_df, dummy_na=False)
    latest_encoded = latest_encoded.reindex(columns=feature_columns, fill_value=0)
    latest_encoded = latest_encoded.fillna(0)
    return latest_encoded


def generate_trading_signal(model: NaiveBayesClassifier, latest_features: pd.DataFrame) -> Tuple[int, float, Dict[int, float]]:
    """
    根據模型生成交易訊號和信心值
    
    Args:
        model: 訓練好的分類模型
        latest_features: 最新的特徵數據
        
    Returns:
        Tuple: 包含三項元素
            - int: 交易訊號（1=買入, -1=賣出, 0=持有）
            - float: 信心值（最高類別的概率）
            - Dict: 各類別的概率分布
    """
    prediction = model.predict(latest_features)[0]
    # 獲取所有類別的概率分布
    proba_dist = model.predict_proba(latest_features)[0]
    # 創建類別與概率的映射
    class_probs = {cls: float(prob) for cls, prob in zip(model.classes_, proba_dist)}
    # 獲取最高概率值作為信心值
    probability = float(proba_dist.max())
    
    # 三態交易訊號：1=買入, -1=賣出, 0=持有
    signal = int(prediction)
    
    return signal, probability, class_probs


def format_report(
    evaluation: Dict[str, float | str],
    signal: int,
    probability: float,
    latest_price: float,
    latest_date: str,
    ticker: str,
    generated_at: str,
    class_probs: Dict[int, float] = None
) -> str:
    # 根據訊號確定建議和表情符號
    if signal == 1:
        recommendation = '買進'
        signal_emoji = '🟢'
        signal_description = f"預期未來{FUTURE_DAYS}天內價格上漲超過{PCT_CHANGE_THRESHOLD*100}%"
    elif signal == -1:
        recommendation = '賣出'
        signal_emoji = '🔴'
        signal_description = f"預期未來{FUTURE_DAYS}天內價格下跌超過{PCT_CHANGE_THRESHOLD*100}%"
    else:  # signal == 0
        recommendation = '持有'
        signal_emoji = '⚪'
        signal_description = f"預期未來{FUTURE_DAYS}天內價格波動不超過{PCT_CHANGE_THRESHOLD*100}%"

    # 信心值詳細說明
    confidence_section = f"🎯 信心值：{probability:.2%}"
    
    # 如果提供了詳細的類別概率，添加概率分布信息
    if class_probs:
        signal_mapping = {1: '買進', -1: '賣出', 0: '持有'}
        probs_text = []
        for cls in sorted(class_probs.keys(), reverse=True):
            if cls in signal_mapping:
                probs_text.append(f"  {signal_mapping[cls]}: {class_probs[cls]:.2%}")
        
        if probs_text:
            confidence_section += "\n📊 概率分布：\n" + "\n".join(probs_text)

    sections = [
        f"{signal_emoji} 每日量化交易報告 - {ticker}",
        f"📊 {MODEL_NAME_ZH}模型量化分析結果",
        f"🏢 股票名稱：{ticker}",
        f"🕒 報告生成時間：{generated_at}",
        f"💵 收盤價｜{latest_price:,.2f}",
        f"📈 建議操作｜{recommendation}",
        f"🔮 訊號說明｜{signal_description}",
        confidence_section,
        "",
        "—— 模型評估摘要 ——",
        f"Accuracy：{evaluation['accuracy']:.2%}"
    ]

    sections.append("⚠️ 本報告為量化模型分析結果，非投資建議。請審慎評估風險。")

    return "\n".join(section for section in sections if section is not None)


def save_report(report: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding='utf-8')
    logger.info("報告已儲存至: %s", path)
    return path


def save_report_metadata(metadata: Dict[str, str | float | int], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info("報告中繼資料已儲存至: %s", path)
    return path


def run_prediction_pipeline(config: PredictionConfig) -> Dict[str, str | float | int]:
    data = load_final_data(config.input_path)
    features, labels = prepare_features(data)
    feature_columns = features.columns.tolist()
    model, evaluation = rolling_train_and_evaluate(features, labels, config.rolling_window_size)
    logger.info(
        "滾動驗證完成：共 %d 筆樣本外預測，準確率 %.2f%%",
        len(evaluation['rolling_predictions']),
        evaluation['accuracy'] * 100,
    )
    logger.info("使用 %d 天未來視窗，%.2f%% 價格變化閾值的交易訊號策略", FUTURE_DAYS, PCT_CHANGE_THRESHOLD*100)

    latest_row = data.iloc[-1]
    latest_date = latest_row.get('Date', '未知日期')
    latest_price = float(latest_row.get('Close', float('nan')))
    ticker = latest_row.get('Ticker', '未知標的')
    latest_features = prepare_latest_features(latest_row, feature_columns)
    signal, probability, class_probs = generate_trading_signal(model, latest_features)

    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report_text = format_report(
        evaluation=evaluation,
        signal=signal,
        probability=probability,
        latest_price=latest_price,
        latest_date=latest_date,
        ticker=ticker,
        generated_at=generated_at,
        class_probs=class_probs,
    )
    save_report(report_text, config.report_output_path)

    # 根據訊號確定建議
    if signal == 1:
        recommendation = '買進'
    elif signal == -1:
        recommendation = '賣出'
    else:  # signal == 0
        recommendation = '持有'
    
    # 準備類別概率的字符串表示，用於元數據
    class_probs_dict = {str(k): v for k, v in class_probs.items()}
    
    report_metadata = {
        'ticker': ticker,
        'generated_at': generated_at,
        'latest_date': latest_date,
        'latest_price': latest_price,
        'signal': signal,
        'signal_text': recommendation,  # 添加文字訊號
        'recommendation': recommendation,
        'probability': probability,
        'class_probabilities': class_probs_dict,  # 添加類別概率分布
        'accuracy': evaluation['accuracy'],
        'rolling_window_size': config.rolling_window_size,
        'future_days': FUTURE_DAYS,
        'pct_change_threshold': PCT_CHANGE_THRESHOLD,
        'model_name': MODEL_NAME_ZH,
        'model_name_en': MODEL_NAME_EN,
    }
    save_report_metadata(report_metadata, config.report_metadata_path)

    if signal == 1:
        recommendation_text = '買進'
    elif signal == -1:
        recommendation_text = '賣出'
    else:  # signal == 0
        recommendation_text = '持有'
    logger.info("當日建議: %s", recommendation_text)

    return {
        'signal': signal,
        'probability': probability,
        'class_probabilities': class_probs,  # 添加類別概率分布
        'accuracy': evaluation['accuracy'],
        'classification_report': evaluation['classification_report'],
        'report': report_text,
        'latest_price': latest_price,
        'latest_date': latest_date,
        'ticker': ticker,
        'generated_at': generated_at,
        'report_metadata_path': str(config.report_metadata_path),
        'rolling_window_size': config.rolling_window_size,
        'rolling_prediction_count': len(evaluation['rolling_predictions']),
        'model_name': MODEL_NAME_ZH,
        'model_name_en': MODEL_NAME_EN,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="訓練模型並生成隔日交易訊號")
    parser.add_argument('--input', type=Path, default=PredictionConfig.input_path, help='整理後資料輸入檔')
    parser.add_argument('--model-output', type=Path, default=PredictionConfig.model_output_path, help='模型輸出路徑 (預留)')
    parser.add_argument('--report-output', type=Path, default=PredictionConfig.report_output_path, help='日報輸出路徑')
    parser.add_argument('--metadata-output', type=Path, default=PredictionConfig.report_metadata_path, help='日報中繼資料輸出路徑')
    parser.add_argument('--test-size', type=float, default=PredictionConfig.test_size, help='測試集比例')
    parser.add_argument('--random-state', type=int, default=PredictionConfig.random_state, help='隨機種子')
    parser.add_argument('--rolling-window-size', type=int, default=PredictionConfig.rolling_window_size, help='滾動訓練窗口大小')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='日誌層級')
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=getattr(logging, args.log_level))

    config = PredictionConfig(
        input_path=args.input,
        model_output_path=args.model_output,
        report_output_path=args.report_output,
        report_metadata_path=args.metadata_output,
        test_size=args.test_size,
        random_state=args.random_state,
        rolling_window_size=args.rolling_window_size,
    )

    try:
        result = run_prediction_pipeline(config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("操作完成。模型準確率: %.2f%%", result['accuracy'] * 100)
    return 0


__all__ = [
    'PredictionConfig',
    'run_prediction_pipeline',
    'main',
    'FUTURE_DAYS',
    'PCT_CHANGE_THRESHOLD',
    'define_signal',
]
