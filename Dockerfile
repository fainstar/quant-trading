# 使用 Python 3.11 官方映像作為基礎
FROM python:3.11-slim

# 設定維護者資訊
LABEL maintainer="stock-trading-system"
LABEL description="股票量化交易系統"
LABEL version="1.1"
LABEL created="2025-10-06"

# 設定工作目錄
WORKDIR /app

# 設定環境變數
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=Asia/Taipei \
    MPLBACKEND=Agg \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1

# 安裝系統依賴和中文字體
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    tzdata \
    curl \
    ca-certificates \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    # 清理 apt 緩存以減小映像大小
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # 移除不必要的 pip 緩存
    rm -rf ~/.cache/pip

# 創建必要的目錄
RUN mkdir -p config data log templates

# 複製專案文件到容器（排除不需要的目錄/文件）
COPY --chown=1000:1000 src/ /app/src/
COPY --chown=1000:1000 *.py /app/
COPY --chown=1000:1000 templates/ /app/templates/
COPY --chown=1000:1000 config/ /app/config/

# 設定權限（使用更精確的權限設置）
RUN chmod -R 755 /app && \
    chmod -R 775 /app/config /app/data /app/log

# 暴露 Flask Web 介面端口
EXPOSE 5000

# 健康檢查（增強版本）
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# 建立非特權使用者
RUN adduser --disabled-password --gecos "" --home /app appuser && \
    chown -R appuser:appuser /app

# 切換到非特權使用者
USER appuser

# 設定預設啟動命令
CMD ["python", "web_interface.py"]
