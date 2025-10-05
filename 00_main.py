import schedule
import time
import subprocess


def job():
    for stock in ["00631L.TW", "0050.TW", "2330.TW", "AAPL", "ETH-USD", "BTC-USD"]: # 可以在這裡添加更多股票代號, 
        subprocess.run(["python", "01_fetch_stock_data.py", "-t", stock, "-y", "10"])
        subprocess.run(["python", "02_feature_engineering.py"])
        subprocess.run(["python", "03_time_window_4bins.py"])
        subprocess.run(["python", "04_pretidy.py"])
        subprocess.run(["python", "05_predict_next_day_signal.py"])
        subprocess.run(["python", "06_send_report_to_discord.py", "-s", stock])
    print("所有任務已完成！")

# 先執行一次
job()

# 再設定排程（例如每天 20:00 執行一次）
schedule.every().day.at("20:00").do(job)

while True: 
    schedule.run_pending()
    time.sleep(1)
