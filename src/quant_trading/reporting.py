from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import requests

logger = logging.getLogger(__name__)


def read_report(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"找不到日報檔案: {path}")
    logger.info("讀取日報內容: %s", path)
    return path.read_text(encoding='utf-8')


def read_report_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"找不到日報中繼資料: {path}")
    logger.info("讀取報告中繼資料: %s", path)
    return json.loads(path.read_text(encoding='utf-8'))


def build_discord_embed(metadata: dict, stock_name: Optional[str] = None) -> dict:
    ticker = stock_name or metadata.get('ticker') or '未知標的'
    recommendation = metadata.get('recommendation', '未知')
    signal = metadata.get('signal')
    emoji = '🟢' if recommendation == '買進' or signal == 1 else '🔴'
    color = 0x22C55E if emoji == '🟢' else 0xEF4444

    generated_at = metadata.get('generated_at') or '--'
    latest_date = metadata.get('latest_date') or '--'
    latest_price = metadata.get('latest_price')
    price_text = f"{latest_price:,.2f}" if isinstance(latest_price, (int, float)) else str(latest_price or '--')
    probability = metadata.get('probability')
    probability_text = f"{float(probability):.2%}" if isinstance(probability, (int, float)) else None

    model_name = (metadata.get('model_name') or '樸素貝葉斯').strip()
    description_lines = [
        f"📊 {model_name}模型量化分析結果",
        f"🏢 股票名稱：{ticker}",
        f"🕒 報告生成時間：{generated_at}",
    ]

    fields = [
        {
            'name': '💵 收盤價',
            'value': f"`{price_text}`",
            'inline': True,
        },
        {
            'name': '📈 建議操作',
            'value': f"**{recommendation}**",
            'inline': True,
        },
    ]

    if probability_text:
        fields.append({
            'name': '📌 模型信心',
            'value': probability_text,
            'inline': True,
        })

    description_lines.append("⚠️ 本報告為量化模型分析結果，非投資建議。請審慎評估風險。")

    footer_text = 'Quant Trading Bot'
    model_name_en = metadata.get('model_name_en')
    if model_name_en:
        footer_text = f"{footer_text} · {model_name_en}"

    embed = {
        'title': f"{emoji} 每日量化交易報告 - {ticker}",
        'description': "\n".join(description_lines),
        'color': color,
        'fields': fields,
        'footer': {
            'text': footer_text
        },
    }

    try:
        if generated_at and generated_at != '--':
            parsed_time = datetime.strptime(generated_at, '%Y-%m-%d %H:%M:%S')
            embed['timestamp'] = parsed_time.isoformat()
    except ValueError:
        logger.warning("報告時間格式無法解析: %s", generated_at)

    return embed

def send_to_discord(webhook_url: str, payload: dict, timeout: int = 10) -> requests.Response:
    logger.debug("傳送訊息至 Discord...")
    if not payload.get('content') and not payload.get('embeds'):
        raise ValueError('Discord payload 必須包含 content 或 embeds')
    response = requests.post(webhook_url, json=payload, timeout=timeout)
    if response.status_code >= 400:
        logger.error("Discord 回應錯誤 %s: %s", response.status_code, response.text)
        response.raise_for_status()
    logger.info("已成功傳送至 Discord")
    return response


@dataclass
class DiscordReportConfig:
    report_path: Path = Path('log/daily_trading_report.txt')
    metadata_path: Path = Path('log/daily_trading_report.json')
    webhook_url: Optional[str] = None
    webhook_env: Optional[str] = None
    timeout: int = 10
    stock_name: Optional[str] = None


def resolve_webhook_url(provided: Optional[str], env_name: Optional[str]) -> str:
    if provided:
        return provided

    if env_name:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value

    webhook = os.getenv('DISCORD_WEBHOOK_URL')
    if not webhook:
        raise ValueError('未提供 Discord Webhook URL，可使用 --webhook 或環境變數 DISCORD_WEBHOOK_URL')
    return webhook


def run_report_dispatch(config: DiscordReportConfig) -> str:
    payload: dict
    try:
        metadata = read_report_metadata(config.metadata_path)
        payload = {'embeds': [build_discord_embed(metadata, config.stock_name)]}
        content = json.dumps(metadata, ensure_ascii=False)
    except FileNotFoundError:
        logger.warning("找不到報告中繼資料，將改用純文字內容傳送: %s", config.metadata_path)
        content = read_report(config.report_path)
        payload = {'content': content}

    if config.stock_name:
        logger.info("即將傳送 %s 的報告", config.stock_name)
    webhook = resolve_webhook_url(config.webhook_url, config.webhook_env)
    send_to_discord(webhook, payload, timeout=config.timeout)
    return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='將交易日報傳送至 Discord')
    parser.add_argument('--report', type=Path, default=DiscordReportConfig.report_path, help='日報檔案路徑')
    parser.add_argument('--metadata', type=Path, default=DiscordReportConfig.metadata_path, help='日報中繼資料檔案路徑')
    parser.add_argument(
        '--webhook',
        '--webhook-url',
        dest='webhook',
        default=None,
        help='Discord Webhook URL，可改以環境變數指定',
    )
    parser.add_argument(
        '--webhook-env',
        default=None,
        help='指定環境變數名稱以讀取 Discord Webhook URL (預設為 DISCORD_WEBHOOK_URL)',
    )
    parser.add_argument('--timeout', type=int, default=DiscordReportConfig.timeout, help='HTTP 請求逾時秒數')
    parser.add_argument('--stock', '-s', default=None, help='股票名稱 (選填，僅用於日誌)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='日誌層級')
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=getattr(logging, args.log_level))

    config = DiscordReportConfig(
        report_path=args.report,
        metadata_path=args.metadata,
        webhook_url=args.webhook,
        webhook_env=args.webhook_env,
        timeout=args.timeout,
        stock_name=args.stock,
    )

    try:
        run_report_dispatch(config)
    except (FileNotFoundError, ValueError, requests.RequestException) as exc:
        logger.error("傳送報告失敗: %s", exc)
        return 1

    logger.info("報告已傳送")
    return 0


__all__ = [
    'DiscordReportConfig',
    'run_report_dispatch',
    'main',
]
