import pandas as pd
import numpy as np
from collections import defaultdict
import os

# --- 這是模擬過去10天預測結果的腳本 ---

# 1. 建立與正式腳本(18_...)中完全相同的貝葉斯分類器
class NaiveBayesClassifier:
    def __init__(self):
        self.priors = defaultdict(float)
        self.likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.features = []
        self.laplace_smoothing = 1

    def fit(self, X, y):
        self.features = X.columns
        n_samples = len(y)
        action_counts = y.value_counts()
        for action, count in action_counts.items():
            self.priors[action] = count / n_samples

        for feature in self.features:
            unique_values_in_feature = X[feature].unique()
            num_unique_values = len(unique_values_in_feature)
            for action in self.priors:
                action_mask = (y == action)
                feature_values_in_action = X.loc[action_mask, feature]
                value_counts = feature_values_in_action.value_counts()
                total_in_action = len(feature_values_in_action)
                for value in unique_values_in_feature:
                    self.likelihoods[feature][value][action] = \
                        (value_counts.get(value, 0) + self.laplace_smoothing) / (total_in_action + num_unique_values * self.laplace_smoothing)

    def predict(self, X):
        posteriors = defaultdict(float)
        for action in self.priors:
            posterior = np.log(self.priors.get(action, 1e-9))
            for feature in self.features:
                value = X[feature]
                prob = self.likelihoods[feature].get(value, {}).get(action, 1e-9)
                posterior += np.log(prob)
            posteriors[action] = posterior
        return max(posteriors, key=posteriors.get)

def get_prediction_for_data(dataframe):
    """
    接收一個 DataFrame，訓練模型並回傳對其最後一筆資料的預測。
    """
    # 準備訓練資料
    feature_columns = [col for col in dataframe.columns if col.endswith('_bin')]
    df_train = dataframe[['Date'] + feature_columns + ['Close']].copy()

    future_days = 5
    pct_change_threshold = 0.03
    df_train['Future_Close'] = df_train['Close'].shift(-future_days)
    df_train['Price_Change'] = (df_train['Future_Close'] - df_train['Close']) / df_train['Close']

    def define_signal(price_change):
        if price_change > pct_change_threshold:
            return '買入'
        elif price_change < -pct_change_threshold:
            return '賣出'
        else:
            return '持有'

    df_train.dropna(subset=['Future_Close'], inplace=True)
    df_train['Action'] = df_train['Price_Change'].apply(define_signal)

    # 訓練模型
    X_train_full = df_train[feature_columns]
    y_train_full = df_train['Action']
    
    if len(X_train_full) == 0:
        return "資料不足無法預測"

    model = NaiveBayesClassifier()
    model.fit(X_train_full, y_train_full)

    # 準備預測資料 (使用傳入 dataframe 的最後一筆)
    latest_data = dataframe.iloc[-1]
    X_predict = latest_data[feature_columns]
    
    # 進行預測
    predicted_signal = model.predict(X_predict)
    return predicted_signal, latest_data['Date']

# 2. 載入完整的歷史資料
try:
    full_df = pd.read_csv('data/filtered_file.csv')
except FileNotFoundError:
    print("錯誤：'data/filtered_file.csv' 找不到。")
    exit()

# 3. 迴圈模擬過去10天的預測
simulation_days = 10
print(f"--- 開始模擬過去 {simulation_days} 天的每日預測 ---")

for i in range(simulation_days, 0, -1):
    # 每次都從完整資料中切割，模擬當時的資料狀態
    # 例如 i=10，代表取到倒數第10筆資料為止，來預測倒數第9天的訊號
    end_index = len(full_df) - i
    if end_index < 1:
        continue
        
    temp_df = full_df.iloc[:end_index]
    
    # 執行預測
    signal, last_date = get_prediction_for_data(temp_df)
    
    # 取得價格資訊
    last_close = temp_df.iloc[-1]['Close']
    predict_day_data = full_df.iloc[end_index]
    predict_for_date = predict_day_data['Date']
    predict_open = predict_day_data['Open']

    print(f"基於 {last_date} (收盤價: {last_close:.2f})，預測 {predict_for_date} (開盤價: {predict_open:.2f}) 的訊號為：【 {signal} 】")

print("\n--- 模擬結束 ---")
