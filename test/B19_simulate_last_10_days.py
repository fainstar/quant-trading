import pandas as pd
import numpy as np
from collections import defaultdict
import os
from typing import Dict, List, Tuple

# --- 這是模擬過去交易日預測結果的腳本 ---

# 設定模擬參數
simulation_days = 30

# 1. 建立與正式腳本(predict.py)中完全相同的貝葉斯分類器
class NaiveBayesClassifier:
    """簡單的類別型樸素貝葉斯分類器，支援拉普拉斯平滑。"""

    def __init__(self, laplace_smoothing: float = 1.0) -> None:
        if laplace_smoothing <= 0:
            raise ValueError('laplace_smoothing 必須為正數')
        self.laplace_smoothing = float(laplace_smoothing)
        self.priors = {}
        self.likelihoods = defaultdict(lambda: defaultdict(dict))
        self.features = []
        self.classes_ = []

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError('特徵與標籤長度不一致')

        y = pd.Series(y)
        self.features = list(X.columns)
        # 確保類別包含 -1, 0, 1，即使數據中可能暫時沒有
        # 這對於三態交易訊號（買入、持有、賣出）很重要
        unique_classes = sorted(y.unique().tolist())
        # 將類別統一為字串型態
        if isinstance(unique_classes[0], str):
            self.classes_ = sorted(list(set(unique_classes + ['買入', '持有', '賣出'])))
        else:
            self.classes_ = sorted(list(set(unique_classes)))
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
                    value_key = value
                    count = value_counts.get(value, 0)
                    probability = (count + self.laplace_smoothing) / denominator
                    self.likelihoods[feature][action][value_key] = probability

                # 若遇到未見值，給予極小平滑機率備用
                self.likelihoods[feature][action]['__unknown__'] = self.laplace_smoothing / (
                    denominator if denominator > 0 else self.laplace_smoothing * num_unique
                )

        return self

    def _compute_log_posteriors(self, row):
        posteriors = {}
        for action in self.classes_:
            log_posterior = float(np.log(self.priors.get(action, 1e-12)))
            for feature in self.features:
                value_key = row.get(feature)
                feature_likelihoods = self.likelihoods.get(feature, {}).get(action, {})
                probability = feature_likelihoods.get(value_key)
                if probability is None or probability <= 0:
                    probability = feature_likelihoods.get('__unknown__', 1e-12)
                log_posterior += float(np.log(probability))
            posteriors[action] = log_posterior
        return posteriors

    def predict_log_proba(self, X):
        log_probas = []
        for _, row in X.iterrows():
            posteriors = self._compute_log_posteriors(row)
            log_probas.append([posteriors[action] for action in self.classes_])
        return np.array(log_probas)

    def predict_proba(self, X):
        log_probas = self.predict_log_proba(X)
        # 使用 log-sum-exp 轉換為實際機率
        max_log = log_probas.max(axis=1, keepdims=True)
        stabilized = np.exp(log_probas - max_log)
        sums = stabilized.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            probas = np.divide(stabilized, sums, where=sums > 0)
        return probas

    def predict(self, X):
        if isinstance(X, pd.Series):
            X = pd.DataFrame([X])
        probas = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return np.array([self.classes_[idx] for idx in class_indices])

# 定義與正式模型相同的交易訊號函數
FUTURE_DAYS = 5
PCT_CHANGE_THRESHOLD = 0.03

def define_signal(price_change):
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

def get_prediction_for_data(dataframe):
    """
    接收一個 DataFrame，訓練模型並回傳對其最後一筆資料的預測。
    返回預測訊號、日期、信心值和所有類別概率
    """
    # 準備訓練資料
    feature_columns = [col for col in dataframe.columns if col.endswith('_bin')]
    df_train = dataframe[['Date'] + feature_columns + ['Close']].copy()

    df_train['Future_Close'] = df_train['Close'].shift(-FUTURE_DAYS)
    df_train['Price_Change'] = (df_train['Future_Close'] - df_train['Close']) / df_train['Close']

    df_train.dropna(subset=['Future_Close'], inplace=True)
    df_train['Action'] = df_train['Price_Change'].apply(define_signal)

    # 訓練模型
    X_train_full = df_train[feature_columns]
    y_train_full = df_train['Action']
    
    if len(X_train_full) == 0:
        return "資料不足無法預測", None, None, None

    model = NaiveBayesClassifier()
    model.fit(X_train_full, y_train_full)

    # 準備預測資料 (使用傳入 dataframe 的最後一筆)
    latest_data = dataframe.iloc[-1]
    X_predict = latest_data[feature_columns]
    
    # 創建預測用的 DataFrame
    X_predict_df = pd.DataFrame([X_predict])
    
    # 進行預測
    predicted_signal = model.predict(X_predict_df)[0]
    
    # 計算信心值和概率分布
    probas = model.predict_proba(X_predict_df)[0]
    confidence = float(np.max(probas))
    
    # 創建類別概率字典
    class_probs = {cls: float(prob) for cls, prob in zip(model.classes_, probas)}
    
    return predicted_signal, latest_data['Date'], confidence, class_probs

# 2. 載入完整的歷史資料
try:
    # 載入特徵資料
    full_df = pd.read_csv('data/final_data.csv', parse_dates=['Date'])
    if len(full_df) < simulation_days:
        print(f"警告：可用數據只有 {len(full_df)} 筆，少於請求的 {simulation_days} 天。")
        print(f"將模擬天數調整為 {len(full_df)-1} 天。")
        simulation_days = len(full_df) - 1
    
    # 載入開盤價格資料
    price_df = pd.read_csv('data/data.csv', parse_dates=['Date'])
    print(f"成功載入原始價格資料，共 {len(price_df)} 筆記錄")
    
    # 確保兩個數據框的日期格式一致
    print(f"final_data.csv 日期範例: {full_df['Date'].iloc[0]}")
    print(f"data.csv 日期範例: {price_df['Date'].iloc[0]}")
    
except FileNotFoundError as e:
    print(f"錯誤：檔案找不到。{e}")
    exit()
except Exception as e:
    print(f"載入數據時發生錯誤：{e}")
    exit()

# 3. 迴圈模擬過去交易日的預測
print(f"--- 開始模擬過去 {simulation_days} 天的每日預測 ---")
print(f"使用交易訊號策略: 未來{FUTURE_DAYS}天價格變化閾值 {PCT_CHANGE_THRESHOLD*100:.1f}%\n")

# 檢查數據欄位
available_columns = full_df.columns.tolist()
print(f"可用數據欄位: {', '.join(available_columns)}")

# 準備結果輸出文件
output_file_path = 'data/prediction_history_30days.txt'
output_lines = []
output_lines.append(f"--- 模擬過去 {simulation_days} 天的每日預測 ---")
output_lines.append(f"使用交易訊號策略: 未來{FUTURE_DAYS}天價格變化閾值 {PCT_CHANGE_THRESHOLD*100:.1f}%\n")

for i in range(simulation_days, 0, -1):
    # 每次都從完整資料中切割，模擬當時的資料狀態
    # 例如 i=30，代表取到倒數第30筆資料為止，來預測倒數第29天的訊號
    end_index = len(full_df) - i
    if end_index < 1:
        continue
        
    temp_df = full_df.iloc[:end_index]
    
    # 執行預測
    signal, last_date, confidence, class_probs = get_prediction_for_data(temp_df)
    
    # 取得價格資訊
    last_close = temp_df.iloc[-1]['Close']
    predict_day_data = full_df.iloc[end_index]
    predict_for_date = predict_day_data['Date']
    
    # 尋找對應日期的開盤價格
    matching_price = price_df[price_df['Date'] == predict_for_date]
    if not matching_price.empty and 'Open' in matching_price.columns:
        predict_next_open = matching_price.iloc[0]['Open']
        print(f"找到 {predict_for_date} 的開盤價: {predict_next_open}")
        predict_next_close = predict_next_open  # 將開盤價用於預測
    else:
        predict_next_close = predict_day_data['Close']  # 如果找不到開盤價，使用收盤價替代

    # 確定表情符號
    if signal == '買入':
        signal_emoji = '🟢'
    elif signal == '賣出':
        signal_emoji = '🔴'
    else:  # signal == '持有'
        signal_emoji = '⚪'

    # 計算實際價格變化
    if end_index + FUTURE_DAYS < len(full_df):
        future_data = full_df.iloc[end_index + FUTURE_DAYS]
        future_date = future_data['Date']
        
        # 尋找未來日期的開盤價格
        future_price_data = price_df[price_df['Date'] == future_date]
        if not future_price_data.empty and 'Open' in future_price_data.columns:
            future_open = future_price_data.iloc[0]['Open']
            # print(f"找到 {future_date} 的開盤價: {future_open}")
            actual_change = (future_open - predict_next_close) / predict_next_close
        else:
            future_close = future_data['Close']  # 如果找不到開盤價，使用收盤價
            actual_change = (future_close - predict_next_close) / predict_next_close
            
        actual_pct = actual_change * 100
        actual_result = f"實際至 {future_date} 的變化: {actual_pct:.2f}%"
    else:
        actual_result = "尚無足夠未來資料計算實際變化"

    # 打印預測結果
    result_line = f"{signal_emoji} 基於 {last_date} (收盤價: {last_close:.2f})，預測 {predict_for_date} (開盤價: {predict_next_close:.2f}) 的訊號為：【 {signal} 】"
    confidence_line = f"   信心值: {confidence:.2%}"
    
    # 概率分布
    probs_line = "   概率分布: "
    for cls, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
        probs_line += f"{cls}: {prob:.2%} "
    
    # 實際結果
    actual_line = f"   {actual_result}"
    
    # 打印到控制台
    print(result_line)
    print(confidence_line)
    print(probs_line)
    print(actual_line)
    print()
    
    # 添加到輸出列表
    output_lines.append(result_line)
    output_lines.append(confidence_line)
    output_lines.append(probs_line)
    output_lines.append(actual_line)
    output_lines.append("")

# 將結果寫入檔案
output_lines.append("\n--- 模擬結束 ---")
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print("\n--- 模擬結束 ---")
print(f"結果已儲存至: {output_file_path}")
