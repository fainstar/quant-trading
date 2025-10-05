# Docker 部署指南（純 Docker 命令）

## 🐳 快速部署

### 1. 構建映像

```powershell
docker build -t oomaybeoo/quant-trading:latest .
```

### 2. 運行容器

```powershell
docker run -d `
  --name quant-trading `
  -p 5000:5000 `
  -v ${PWD}/config:/app/config `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/log:/app/log `
  -e TZ=Asia/Taipei `
  --restart unless-stopped `
  oomaybeoo/quant-trading:latest
```

### 3. 訪問系統

瀏覽器訪問：`http://localhost:5000`

---

## 📋 常用命令

### 容器管理

```powershell
# 查看運行中的容器
docker ps

# 查看所有容器（包括停止的）
docker ps -a

# 停止容器
docker stop quant-trading

# 啟動容器
docker start quant-trading

# 重啟容器
docker restart quant-trading

# 刪除容器
docker rm quant-trading

# 強制刪除運行中的容器
docker rm -f quant-trading
```

### 日誌查看

```powershell
# 查看實時日誌
docker logs -f quant-trading

# 查看最近 100 行日誌
docker logs --tail 100 quant-trading

# 查看帶時間戳的日誌
docker logs -t quant-trading
```

### 容器操作

```powershell
# 進入容器 shell
docker exec -it quant-trading bash

# 在容器內執行命令
docker exec quant-trading python 01_fetch_stock_data.py -t 2330.TW -y 10

# 查看容器資源使用情況
docker stats quant-trading

# 查看容器詳細信息
docker inspect quant-trading
```

### 映像管理

```powershell
# 查看本地映像
docker images

# 刪除映像
docker rmi oomaybeoo/quant-trading:latest

# 清理未使用的映像
docker image prune -a

# 推送到 Docker Hub
docker push oomaybeoo/quant-trading:latest

# 從 Docker Hub 拉取
docker pull oomaybeoo/quant-trading:latest
```

---

## 🔧 進階配置

### 自定義端口

如果 5000 端口被佔用，可以使用其他端口：

```powershell
docker run -d `
  --name quant-trading `
  -p 8080:5000 `
  -v ${PWD}/config:/app/config `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/log:/app/log `
  -e TZ=Asia/Taipei `
  --restart unless-stopped `
  oomaybeoo/quant-trading:latest
```

然後訪問：`http://localhost:8080`

### 資源限制

限制容器使用的 CPU 和內存：

```powershell
docker run -d `
  --name quant-trading `
  -p 5000:5000 `
  --memory="2g" `
  --cpus="2.0" `
  -v ${PWD}/config:/app/config `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/log:/app/log `
  -e TZ=Asia/Taipei `
  --restart unless-stopped `
  oomaybeoo/quant-trading:latest
```

### 環境變數

添加額外的環境變數：

```powershell
docker run -d `
  --name quant-trading `
  -p 5000:5000 `
  -v ${PWD}/config:/app/config `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/log:/app/log `
  -e TZ=Asia/Taipei `
  -e DISCORD_WEBHOOK_URL="your_webhook_url" `
  -e FLASK_ENV=production `
  --restart unless-stopped `
  oomaybeoo/quant-trading:latest
```

### 只讀文件系統（增強安全性）

```powershell
docker run -d `
  --name quant-trading `
  -p 5000:5000 `
  -v ${PWD}/config:/app/config `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/log:/app/log `
  --tmpfs /tmp `
  --read-only `
  -e TZ=Asia/Taipei `
  --restart unless-stopped `
  oomaybeoo/quant-trading:latest
```

---

## 📊 Volume 說明

系統使用三個 Volume 來持久化數據：

| 容器路徑 | 宿主機路徑 | 用途 |
|---------|-----------|------|
| `/app/config` | `./config` | 配置文件（stocks_config.json, schedule_config.json） |
| `/app/data` | `./data` | 數據文件和圖表 |
| `/app/log` | `./log` | 日誌文件 |

### 備份數據

```powershell
# 備份到 zip 文件
$date = Get-Date -Format "yyyyMMdd_HHmmss"
Compress-Archive -Path .\config,.\data,.\log -DestinationPath "backup_$date.zip"
```

### 恢復數據

```powershell
# 從備份恢復
Expand-Archive -Path "backup_20250105_120000.zip" -DestinationPath .
```

---

## 🔄 更新容器

當代碼有更新時：

```powershell
# 1. 停止並刪除舊容器
docker stop quant-trading
docker rm quant-trading

# 2. 重新構建映像
docker build -t oomaybeoo/quant-trading:latest .

# 3. 運行新容器
docker run -d `
  --name quant-trading `
  -p 5000:5000 `
  -v ${PWD}/config:/app/config `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/log:/app/log `
  -e TZ=Asia/Taipei `
  --restart unless-stopped `
  oomaybeoo/quant-trading:latest
```

或使用單行命令：

```powershell
docker stop quant-trading; docker rm quant-trading; docker build -t oomaybeoo/quant-trading:latest .; docker run -d --name quant-trading -p 5000:5000 -v ${PWD}/config:/app/config -v ${PWD}/data:/app/data -v ${PWD}/log:/app/log -e TZ=Asia/Taipei --restart unless-stopped oomaybeoo/quant-trading:latest
```

---

## 🐛 故障排除

### 容器無法啟動

```powershell
# 查看詳細錯誤
docker logs quant-trading

# 檢查端口是否被佔用
netstat -ano | findstr :5000
```

### 無法訪問 Web 介面

```powershell
# 檢查容器狀態
docker ps | Select-String quant-trading

# 測試容器內部服務
docker exec quant-trading curl http://localhost:5000
```

### Volume 掛載問題

```powershell
# 檢查 Volume 掛載
docker inspect quant-trading | Select-String -Pattern "Mounts" -Context 0,20

# 確保目錄存在
New-Item -ItemType Directory -Force -Path .\config, .\data, .\log
```

### 清理所有相關容器和映像

```powershell
# 停止並刪除容器
docker stop quant-trading
docker rm quant-trading

# 刪除映像
docker rmi oomaybeoo/quant-trading:latest

# 清理未使用的資源
docker system prune -a
```

---

## 📈 性能監控

### 查看資源使用

```powershell
# 實時監控
docker stats quant-trading

# 查看容器進程
docker top quant-trading
```

### 健康檢查

```powershell
# 檢查容器健康狀態
docker inspect --format='{{.State.Health.Status}}' quant-trading

# 查看健康檢查日誌
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' quant-trading
```

---

## 🚀 生產環境建議

### 1. 使用特定版本標籤

```powershell
docker build -t oomaybeoo/quant-trading:v1.0.0 .
docker build -t oomaybeoo/quant-trading:latest .
```

### 2. 配置重啟策略

- `no`: 不自動重啟（默認）
- `on-failure`: 僅在失敗時重啟
- `always`: 總是重啟
- `unless-stopped`: 除非手動停止，否則總是重啟

```powershell
docker run -d --restart unless-stopped ...
```

### 3. 配置日誌輪轉

```powershell
docker run -d `
  --log-driver json-file `
  --log-opt max-size=10m `
  --log-opt max-file=3 `
  ...
```

### 4. 使用網路隔離

```powershell
# 創建自定義網路
docker network create quant-network

# 在自定義網路中運行
docker run -d `
  --name quant-trading `
  --network quant-network `
  ...
```

---

## 📦 Docker Hub 部署

### 推送到 Docker Hub

```powershell
# 登入 Docker Hub
docker login

# 構建並標記
docker build -t oomaybeoo/quant-trading:latest .
docker tag oomaybeoo/quant-trading:latest oomaybeoo/quant-trading:v1.0.0

# 推送
docker push oomaybeoo/quant-trading:latest
docker push oomaybeoo/quant-trading:v1.0.0
```

### 從 Docker Hub 拉取

```powershell
# 拉取映像
docker pull oomaybeoo/quant-trading:latest

# 運行
docker run -d `
  --name quant-trading `
  -p 5000:5000 `
  -v ${PWD}/config:/app/config `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/log:/app/log `
  -e TZ=Asia/Taipei `
  --restart unless-stopped `
  oomaybeoo/quant-trading:latest
```

---

**最後更新：2025年10月5日**
