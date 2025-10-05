import pandas as pd

# 讀取原始 CSV
df = pd.read_csv("data/rolling_window_100day_4bins.csv")

# 保留的基本欄位
base_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]

# 抓出所有 bin 欄位
bin_cols = [col for col in df.columns if "_bin" in col]

# 最終欄位
final_cols = base_cols + bin_cols

# 篩選
filtered_df = df[final_cols].copy()

# 存回新的 CSV
filtered_df.to_csv("data/final_data.csv", index=False)

print("Done! 已輸出 final_data.csv，包含交易訊號")
