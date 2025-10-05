import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import sys

# 設定編碼
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# --- 這是正式使用的預測腳本 ---
# 目標：根據到目前為止的所有歷史資料，預測下一個交易日的決策。

# 1. 建立與訓練腳本(16_...)中完全相同的貝葉斯分類器
class NaiveBayesClassifier:
    def __init__(self):
        self.priors = defaultdict(float)
        self.likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.features = []
        self.laplace_smoothing = 1 # 拉普拉斯平滑 alpha

    def fit(self, X, y):
        self.features = X.columns
        n_samples = len(y)
        
        # 計算先驗機率 P(Action)
        action_counts = y.value_counts()
        for action, count in action_counts.items():
            self.priors[action] = count / n_samples

        # 計算概似機率 P(Feature_Value | Action)
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
        # 這個版本的 predict 只處理單筆資料 (一個 DataFrame row)
        posteriors = defaultdict(float)
        for action in self.priors:
            posterior = np.log(self.priors.get(action, 1e-9))
            
            for feature in self.features:
                value = X[feature]
                prob = self.likelihoods[feature].get(value, {}).get(action, 1e-9)
                posterior += np.log(prob)
            
            posteriors[action] = posterior
            
        return max(posteriors, key=posteriors.get)

# 2. 載入完整的特徵資料
try:
    df = pd.read_csv('data/final_data.csv')
except FileNotFoundError:
    print("錯誤：'data/final_data.csv' 找不到。請確認檔案路徑是否正確。")
    exit()

# 3. 準備訓練資料 (與訓練腳本相同的目標定義)
feature_columns = [col for col in df.columns if col.endswith('_bin')]
df_train = df[['Date'] + feature_columns + ['Close']].copy()

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

# 移除因計算未來價格而產生的 NaN 值，確保訓練資料乾淨
df_train.dropna(subset=['Future_Close'], inplace=True)
df_train['Action'] = df_train['Price_Change'].apply(define_signal)

# 4. 訓練最終模型
# 使用 df_train 的所有資料來訓練，以做出最全面的預測
X_train_full = df_train[feature_columns]
y_train_full = df_train['Action']

print("正在使用所有歷史資料訓練最終模型...")
final_model = NaiveBayesClassifier()
final_model.fit(X_train_full, y_train_full)
print("模型訓練完成。")

# 5. 準備要預測的最新資料
# 我們使用原始 df 的最後一筆資料，因為它代表了最新的市場狀態
latest_data = df.iloc[-1]
X_predict = latest_data[feature_columns]

# 6. 進行預測並輸出結果
print("\n--- 貝葉斯策略預測 ---")
predicted_signal = final_model.predict(X_predict)
print(f"基於日期 {latest_data['Date']} (收盤價: {latest_data['Close']:.2f}) 的資料進行預測：")
print(f"對下一個交易日的建議：【 {predicted_signal} 】")

# 7. 生成文字報告
report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
report_content = f"""
==================================================
      每日交易策略報告 (貝葉斯模型)
==================================================

報告生成時間: {report_time}

--------------------------------------------------
            預 測 摘 要
--------------------------------------------------

* 預測基準日期: {latest_data['Date']}
* 基準日收盤價: {latest_data['Close']:.2f}

--------------------------------------------------
            明日交易建議
--------------------------------------------------

              >>       {predicted_signal.upper():^10}       <<

==================================================
免責聲明: 本報告僅為基於歷史數據的量化分析結果，
不構成任何投資建議。所有交易決策請謹慎評估風險。
==================================================
"""

report_path = 'log/daily_trading_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"\n詳細報告已生成並儲存至: {report_path}")
