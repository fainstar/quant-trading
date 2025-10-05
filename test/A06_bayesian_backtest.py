import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import fontManager

# 解決 Matplotlib 中文顯示問題
try:
    fontManager.addfont('C:/Windows/Fonts/msjh.ttc')
    matplotlib.rc('font', family='Microsoft JhengHei')
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"載入中文字體失敗: {e}\n將使用預設字體。")


# 1. 載入帶有貝葉斯訊號的資料
try:
    df = pd.read_csv('data/bayesian_trading_signals.csv', parse_dates=['Date'])
except FileNotFoundError:
    print("錯誤：'data/bayesian_trading_signals.csv' 找不到。請先執行貝葉斯策略腳本。")
    exit()

# 載入原始資料以獲取開盤價
try:
    df_original = pd.read_csv('data/data.csv', parse_dates=['Date'])
    df_original.set_index('Date', inplace=True)
    # 將開盤價合併到 df 中
    df = df.merge(df_original[['Open']], left_on='Date', right_index=True, how='left')
except FileNotFoundError:
    print("錯誤：'data/data.csv' 找不到。無法取得開盤價資料。")
    exit()

df.set_index('Date', inplace=True)

# 2. 初始化回測參數
initial_capital = 100000.0
cash = initial_capital
position = 0  # 持有股數
portfolio_value = initial_capital
portfolio_history = []
trades = []

# 交易成本設定
commission_rate = 0.001425  # 手續費 0.1425%
tax_rate = 0.003  # 證交稅 0.3% (僅賣出時收取)

# 3. 執行回測（使用隔天開盤價）
for i in range(len(df) - 1):  # 注意：改為 len(df) - 1，因為需要看下一天的開盤價
    date = df.index[i]
    signal = df['Bayesian_Signal'].iloc[i]
    close_price = df['Close'].iloc[i]
    
    # 取得隔天的開盤價
    next_open_price = df['Open'].iloc[i + 1]
    next_date = df.index[i + 1]

    # 買入邏輯 (使用隔天開盤價)
    if signal == '買入' and position == 0:
        # 計算可買入的股數（考慮手續費）
        max_shares = int(cash / (next_open_price * (1 + commission_rate)))
        
        if max_shares > 0:
            shares_to_buy = max_shares
            buy_cost = shares_to_buy * next_open_price
            commission = buy_cost * commission_rate
            total_cost = buy_cost + commission
            
            position += shares_to_buy
            cash -= total_cost
            
            trades.append({
                'Date': next_date,
                'Type': 'BUY',
                'Price': next_open_price,
                'Shares': shares_to_buy,
                'Commission': commission,
                'Tax': 0,
                'Total_Cost': total_cost
            })

    # 賣出邏輯 (使用隔天開盤價)
    elif signal == '賣出' and position > 0:
        sell_revenue = position * next_open_price
        commission = sell_revenue * commission_rate
        tax = sell_revenue * tax_rate
        total_revenue = sell_revenue - commission - tax
        
        cash += total_revenue
        
        trades.append({
            'Date': next_date,
            'Type': 'SELL',
            'Price': next_open_price,
            'Shares': position,
            'Commission': commission,
            'Tax': tax,
            'Total_Revenue': total_revenue
        })
        
        position = 0

    # 更新每日投資組合價值（使用當日收盤價計算持股市值）
    current_portfolio_value = cash + position * close_price
    portfolio_history.append({'Date': date, 'Portfolio_Value': current_portfolio_value})

# 加入最後一天的投資組合價值
last_date = df.index[-1]
last_close = df['Close'].iloc[-1]
final_portfolio_value = cash + position * last_close
portfolio_history.append({'Date': last_date, 'Portfolio_Value': final_portfolio_value})

# 4. 建立回測結果 DataFrame
portfolio_df = pd.DataFrame(portfolio_history)
portfolio_df.set_index('Date', inplace=True)

trades_df = pd.DataFrame(trades)
if not trades_df.empty:
    trades_df.set_index('Date', inplace=True)
else:
    # 如果沒有交易，建立一個空的 DataFrame 以避免後續錯誤
    trades_df = pd.DataFrame(columns=['Type', 'Price', 'Shares'])
    trades_df.index.name = 'Date'

# 5. 計算回測指標
# 總回報率
total_return = (portfolio_df['Portfolio_Value'].iloc[-1] / initial_capital - 1) * 100

# 年化回報率
days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
# 避免除以零
annual_return = ((1 + total_return / 100) ** (365.0 / days) - 1) * 100 if days > 0 else 0

# 夏普比率 (假設無風險利率為 0)
portfolio_df['Daily_Return'] = portfolio_df['Portfolio_Value'].pct_change()
if portfolio_df['Daily_Return'].std() != 0:
    sharpe_ratio = (portfolio_df['Daily_Return'].mean() / portfolio_df['Daily_Return'].std()) * np.sqrt(252)
else:
    sharpe_ratio = 0

# 最大回檔
portfolio_df['Peak'] = portfolio_df['Portfolio_Value'].cummax()
portfolio_df['Drawdown'] = (portfolio_df['Portfolio_Value'] - portfolio_df['Peak']) / portfolio_df['Peak']
max_drawdown = portfolio_df['Drawdown'].min() * 100

# 勝率與交易成本統計
if len(trades_df) > 1:
    buy_prices = trades_df[trades_df['Type'] == 'BUY']['Price']
    sell_prices = trades_df[trades_df['Type'] == 'SELL']['Price']
    
    # 確保交易是成對的
    num_trades = min(len(buy_prices), len(sell_prices))
    if num_trades > 0:
        wins = np.sum(sell_prices[:num_trades].values > buy_prices[:num_trades].values)
        win_rate = (wins / num_trades) * 100
    else:
        win_rate = 0.0
    
    # 計算總交易成本
    total_commission = trades_df['Commission'].sum() if 'Commission' in trades_df.columns else 0
    total_tax = trades_df['Tax'].sum() if 'Tax' in trades_df.columns else 0
    total_transaction_cost = total_commission + total_tax
else:
    win_rate = 0.0
    num_trades = 0
    total_commission = 0
    total_tax = 0
    total_transaction_cost = 0

# 6. 輸出回測結果
print("=" * 50)
print("      貝葉斯策略回測結果（含交易成本）")
print("=" * 50)
print(f"回測期間: {df.index[0].date()} 到 {df.index[-1].date()}")
print(f"初始資金: ${initial_capital:,.2f}")
print(f"最終資產: ${portfolio_df['Portfolio_Value'].iloc[-1]:,.2f}")
print("-" * 50)
print(f"總回報率: {total_return:.2f}%")
print(f"年化回報率: {annual_return:.2f}%")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"最大回檔: {max_drawdown:.2f}%")
print("-" * 50)
print(f"總交易次數 (買/賣對): {num_trades}")
print(f"勝率: {win_rate:.2f}%")
print("-" * 50)
print("交易成本明細:")
print(f"  手續費總計: ${total_commission:,.2f}")
print(f"  證交稅總計: ${total_tax:,.2f}")
print(f"  交易成本合計: ${total_transaction_cost:,.2f}")
print(f"  成本佔初始資金: {(total_transaction_cost/initial_capital)*100:.2f}%")
print("=" * 50)

# 7. 繪製結果圖
plt.figure(figsize=(15, 8))

# 繪製投資組合價值
ax1 = plt.subplot(1, 1, 1)
portfolio_df['Portfolio_Value'].plot(ax=ax1, color='b', label='投資組合價值')
ax1.set_ylabel('投資組合價值', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('貝葉斯策略回測表現 vs. 股價')
ax1.legend(loc='upper left')

# 繪製股價
ax2 = ax1.twinx()
df['Close'].plot(ax=ax2, color='gray', alpha=0.8, label='收盤價')
ax2.set_ylabel('股價', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.legend(loc='upper right')

# 標示買賣點
if not trades_df.empty:
    buy_points = trades_df[trades_df['Type'] == 'BUY']
    sell_points = trades_df[trades_df['Type'] == 'SELL']
    ax2.plot(buy_points.index, df.loc[buy_points.index]['Close'], '^', markersize=10, color='g', label='買入點')
    ax2.plot(sell_points.index, df.loc[sell_points.index]['Close'], 'v', markersize=10, color='r', label='賣出點')

ax1.grid(True)
plt.legend()
plt.tight_layout()
output_image_path = 'data/bayesian_backtest_performance.png'
plt.savefig(output_image_path)
print(f"回測表現圖已儲存至 '{output_image_path}'")
# plt.show()

# 8. 繪製每年度的走勢圖
portfolio_df['Year'] = portfolio_df.index.year
unique_years = sorted(portfolio_df['Year'].unique())

# 計算需要的行數和列數
n_years = len(unique_years)
n_cols = 2
n_rows = (n_years + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5), squeeze=False)
axes = axes.flatten() # 將 2D 陣列轉換為 1D

for i, year in enumerate(unique_years):
    ax = axes[i]
    
    # 篩選當年度資料
    portfolio_year = portfolio_df[portfolio_df['Year'] == year]
    stock_year = df[df.index.year == year]
    
    # 只在有交易資料時才篩選
    if not trades_df.empty and hasattr(trades_df.index, 'year'):
        trades_year = trades_df[trades_df.index.year == year]
    else:
        trades_year = pd.DataFrame()

    # 繪製投資組合價值
    ax.plot(portfolio_year.index, portfolio_year['Portfolio_Value'], color='b', label='投資組合價值')
    ax.set_ylabel('投資組合價值', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    
    # 繪製股價
    ax2 = ax.twinx()
    ax2.plot(stock_year.index, stock_year['Close'], color='gray', alpha=0.8, label='收盤價')
    ax2.set_ylabel('股價', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # 標示買賣點
    if not trades_year.empty:
        buy_points = trades_year[trades_year['Type'] == 'BUY']
        sell_points = trades_year[trades_year['Type'] == 'SELL']
        # 確保索引存在於 stock_year 中
        valid_buy_points = buy_points[buy_points.index.isin(stock_year.index)]
        valid_sell_points = sell_points[sell_points.index.isin(stock_year.index)]
        ax2.plot(valid_buy_points.index, stock_year.loc[valid_buy_points.index]['Close'], '^', markersize=8, color='g', label='買入點')
        ax2.plot(valid_sell_points.index, stock_year.loc[valid_sell_points.index]['Close'], 'v', markersize=8, color='r', label='賣出點')

    ax.set_title(f'{year}年 策略表現')
    ax.grid(True)
    
    # 合併圖例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    
    # 自動調整日期格式
    fig.autofmt_xdate(rotation=45)

# 隱藏多餘的子圖
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=3.0)
yearly_image_path = 'data/bayesian_backtest_yearly_performance.png'
plt.savefig(yearly_image_path)
print(f"每年度回測表現圖已儲存至 '{yearly_image_path}'")
plt.show()
