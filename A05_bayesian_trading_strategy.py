import pandas as pd
import numpy as np
from collections import defaultdict

# 1. 載入資料並選擇特徵
try:
    df = pd.read_csv('data/final_data.csv')
except FileNotFoundError:
    print("錯誤：'data/final_data.csv' 找不到。請確認檔案路徑是否正確。")
    exit()

# 選取所有 '_bin' 結尾的欄位以及 'Close'
feature_columns = [col for col in df.columns if col.endswith('_bin')]
df_features = df[['Date', 'Close'] + feature_columns].copy()
df_features['Date'] = pd.to_datetime(df_features['Date']) # 確保 Date 是 datetime 物件

# 2. 定義交易訊號 (目標變數)
# 策略：如果未來5天內價格上漲 > 3%，則為'買入'；如果下跌 > 3%，則為'賣出'
future_days = 5
pct_change_threshold = 0.03

df_features['Future_Close'] = df_features['Close'].shift(-future_days)
df_features['Price_Change'] = (df_features['Future_Close'] - df_features['Close']) / df_features['Close']

def define_signal(price_change):
    if price_change > pct_change_threshold:
        return '買入'
    elif price_change < -pct_change_threshold:
        return '賣出'
    else:
        return '持有'

df_features.dropna(subset=['Future_Close'], inplace=True)
df_features['Action'] = df_features['Price_Change'].apply(define_signal)


# 3. 建立簡易的貝葉斯分類器
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
                
                # 使用拉普拉斯平滑避免零機率
                for value in unique_values_in_feature:
                    self.likelihoods[feature][value][action] = \
                        (value_counts.get(value, 0) + self.laplace_smoothing) / (total_in_action + num_unique_values * self.laplace_smoothing)

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            posteriors = defaultdict(float)
            for action in self.priors:
                # 從先驗機率開始
                posterior = np.log(self.priors.get(action, 1e-9)) # 使用 log 加總避免浮點數下溢
                
                # 乘以每個特徵的概似機率
                for feature in self.features:
                    value = row[feature]
                    # 如果測試資料中的值在訓練時未見過，給一個極小的機率
                    prob = self.likelihoods[feature].get(value, {}).get(action, 1e-9)
                    posterior += np.log(prob)
                
                posteriors[action] = posterior
            
            # 選擇後驗機率最高的類別
            predictions.append(max(posteriors, key=posteriors.get))
        return predictions

# 4. 執行固定窗口滾動訓練與預測
X = df_features[feature_columns]
y = df_features['Action']

# 設定固定訓練窗口大小（天數）
train_window_size = 500  # 約 1 年的交易日
min_train_size = 100  # 最小訓練集大小

# 建立一個空的 Series 來儲存樣本外預測結果
bayesian_signals = pd.Series(index=df_features.index, dtype=object)

print(f"開始執行滾動窗口訓練與預測 (訓練窗口: {train_window_size} 天)...")
total_predictions = len(X) - max(train_window_size, min_train_size)
print(f"總共需要預測 {total_predictions} 個交易日")

prediction_count = 0
for i in range(len(X)):
    # 確定訓練集的起始和結束位置
    if i < min_train_size:
        continue  # 跳過資料不足的前期
    
    # 使用固定窗口大小
    if i >= train_window_size:
        train_start = i - train_window_size
        train_end = i
    else:
        # 如果資料不足一個完整窗口，使用所有可用資料
        train_start = 0
        train_end = i
    
    # 獲取訓練集
    X_train = X.iloc[train_start:train_end]
    y_train = y.iloc[train_start:train_end]
    
    # 獲取測試集（當前這一筆）
    X_test = X.iloc[i:i+1]
    
    # 訓練模型
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train, y_train)
    
    # 進行預測
    prediction = nb_classifier.predict(X_test)
    bayesian_signals.iloc[i] = prediction[0]
    
    # 顯示進度
    prediction_count += 1
    if prediction_count % 50 == 0 or prediction_count == 1:
        progress = (prediction_count / total_predictions) * 100
        print(f"進度: {prediction_count}/{total_predictions} ({progress:.1f}%) - 訓練集大小: {len(X_train)} 天")

print(f"\n滾動窗口訓練完成，共產生 {prediction_count} 個預測。")

# 移除最開始沒有被預測到的部分 (第一折的訓練集)
bayesian_signals.dropna(inplace=True)
df_results = df_features.loc[bayesian_signals.index].copy()
df_results['Bayesian_Signal'] = bayesian_signals

# 5. 儲存結果
output_path = 'data/bayesian_trading_signals.csv'
df_results.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n滾動窗口交叉驗證完成。")
print(f"貝葉斯交易訊號已生成並儲存至 '{output_path}'")
print("\n結果預覽：")
print(df_results[['Date', 'Close', 'Action', 'Bayesian_Signal']].tail())
