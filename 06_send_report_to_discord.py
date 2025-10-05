import requests
import json
import argparse
from datetime import datetime, timezone, timedelta
import os
import sys

# 設定編碼
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ====== 基本設定 ======
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1424046158799704136/MQ0Yy_G6iv8JkTT4Zq9vPFHzNXynr0AWbYEGagkcvnod-bM6Y7VBgTv6NmyG1umT3bHT"
REPORT_PATH = "log/daily_trading_report.txt"

# ====== 核心功能 ======
class DiscordReportSender:
    def __init__(self, webhook_url, report_path):
        self.webhook_url = webhook_url
        self.report_path = report_path

    def _read_report(self):
        """讀取報告內容並回傳文字"""
        if not os.path.exists(self.report_path):
            raise FileNotFoundError(f"找不到報告檔案: {self.report_path}")
        with open(self.report_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _parse_report(self, content):
        """解析報告文字內容，提取日期、價格、訊號等資訊"""
        lines = content.split('\n')
        data = {
            "report_time": "",
            "predict_date": "",
            "close_price": "",
            "signal": ""
        }

        for line in lines:
            if "報告生成時間:" in line:
                data["report_time"] = line.split("報告生成時間:")[1].strip()
            elif "預測基準日期:" in line:
                data["predict_date"] = line.split("預測基準日期:")[1].strip()
            elif "基準日收盤價:" in line:
                data["close_price"] = line.split("基準日收盤價:")[1].strip()
            elif ">>" in line and "<<" in line:
                data["signal"] = line.split(">>")[1].split("<<")[0].strip()

        return data

    def _build_embed(self, data, stock_name):
        """根據訊號決定顏色與表情符號"""
        signal = data["signal"]
        if "買入" in signal:
            color = 0x2ECC71  # 綠
            emoji = "🟢"
        elif "賣出" in signal:
            color = 0xE74C3C  # 紅
            emoji = "🔴"
        else:
            color = 0xF1C40F  # 黃
            emoji = "⚪"

        # 取得當地時間 (台北時區)
        now_tpe = datetime.now(timezone(timedelta(hours=8)))

        embed = {
            "title": f"{emoji} 每日量化交易報告 - {stock_name}",
            "description": (
                f"📊 **貝葉斯模型量化分析結果**\n"
                f"🏢 股票名稱：`{stock_name}`\n"
                f"🕒 報告生成時間：`{data['report_time']}`\n"
                f"📅 預測基準日期：`{data['predict_date']}`"
            ),
            "color": color,
            "fields": [
                {
                    "name": "💰 收盤價",
                    "value": f"```{data['close_price']}```",
                    "inline": True
                },
                {
                    "name": "📈 建議操作",
                    "value": f"```{signal}```",
                    "inline": True
                },
            ],
            "footer": {
                "text": "⚠️ 本報告為量化模型分析結果，非投資建議。請審慎評估風險。"
            },
            "timestamp": now_tpe.isoformat()
        }

        return embed

    def send(self, stock_name="未指定"):
        """讀取報告、構建 embed 並發送至 Discord"""
        try:
            report_text = self._read_report()
            data = self._parse_report(report_text)
            embed = self._build_embed(data, stock_name)

            payload = {
                "username": "量化策略機器人",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
                "embeds": [embed]
            }

            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 204:
                print("✅ Discord 報告傳送成功！")
                return True
            else:
                print(f"❌ 傳送失敗 ({response.status_code}) - {response.text}")
                return False

        except FileNotFoundError as e:
            print(f"🚫 錯誤：{e}")
        except Exception as e:
            print(f"💥 發送過程中出錯：{str(e)}")
        return False


# ====== 主執行區 ======
def main():
    # 設置命令列參數解析器
    parser = argparse.ArgumentParser(
        description='將交易策略報告傳送到 Discord',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用範例:
  python 20_send_report_to_discord.py --stock "台積電 (2330)"
  python 20_send_report_to_discord.py -s "聯發科 (2454)"
        '''
    )
    
    parser.add_argument(
        '--stock', '-s',
        type=str,
        default='未指定股票',
        help='股票名稱（例如：台積電 (2330)）'
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("📤 Discord 量化報告傳送工具 v2")
        print("=" * 60)
        print(f"🏢 股票名稱: {args.stock}")
        print("=" * 60)
    except:
        print("=" * 60)
        print("Discord Report Sender v2")
        print("=" * 60)
        print(f"Stock: {args.stock}")
        print("=" * 60)

    if not DISCORD_WEBHOOK_URL.startswith("https://discord.com/api/webhooks/"):
        print("\n⚠️ 請先設定正確的 Webhook URL！")
        return

    sender = DiscordReportSender(DISCORD_WEBHOOK_URL, REPORT_PATH)
    success = sender.send(stock_name=args.stock)

    if success:
        print("\n✅ 報告已推送到 Discord！")
    else:
        print("\n❌ 傳送失敗，請檢查網路或報告格式。")


if __name__ == "__main__":
    main()
