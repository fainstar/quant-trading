# 使用 Python 3.11 官方映像作為基礎
FROM python:3.11-slim

# 設定維護者資訊
LABEL maintainer="stock-trading-system"
LABEL description="股票量化交易系統"

# 設定工作目錄
WORKDIR /app

# 設定環境變數
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=Asia/Taipei \
    MPLBACKEND=Agg

# 安裝系統依賴和中文字體
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    tzdata \
    curl \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 複製 requirements.txt 並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製專案文件到容器
COPY . .

# 創建必要的目錄並設定權限
RUN mkdir -p config data log templates && \
    chmod -R 755 /app

# 暴露 Flask Web 介面端口
EXPOSE 5000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/', timeout=5)" || exit 1

# 設定預設啟動命令
CMD ["python", "web_interface.py"]
