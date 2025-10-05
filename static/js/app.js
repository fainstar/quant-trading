const HISTORY_REFRESH_INTERVAL_MS = 10000;

// 全局變量
let config = { stocks: [], years: 10, webhook_url: '' };
let selectedStocks = new Set();
let historyRefreshInterval = null;
let manualSendReport = true;
let scheduleSendReport = true;

// 頁面載入初始化
window.onload = function() {
    updateDateTime();
    setInterval(updateDateTime, 1000);
    loadConfig();
    loadHistory();
    loadSchedule();
    startHistoryAutoRefresh();
    setupEventListeners();
    updateStats();
};

function setupEventListeners() {
    const manualToggle = document.getElementById('sendReportToggle');
    if (manualToggle) {
        manualToggle.checked = manualSendReport;
        manualToggle.addEventListener('change', (event) => {
            manualSendReport = event.target.checked;
            updateDiscordStatusNote();
        });
        updateDiscordStatusNote();
    }

    const scheduleToggle = document.getElementById('scheduleSendReport');
    if (scheduleToggle) {
        scheduleToggle.checked = scheduleSendReport;
        scheduleToggle.addEventListener('change', (event) => {
            scheduleSendReport = event.target.checked;
        });
    }

    const webhookInput = document.getElementById('webhookUrl');
    if (webhookInput) {
        webhookInput.addEventListener('input', (event) => {
            config.webhook_url = event.target.value.trim();
            updateDiscordStatusNote();
        });
    }

    // 股票輸入框按下Enter鍵觸發添加
    const newStockInput = document.getElementById('newStock');
    if (newStockInput) {
        newStockInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                addStock();
            }
        });
    }
}

function updateDiscordStatusNote() {
    const note = document.getElementById('discordStatusNote');
    if (!note) return;

    if (manualSendReport) {
        if (!config.webhook_url) {
            note.className = 'alert alert-warning';
            note.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i>
                <div>
                    <strong>尚未設定 Discord Webhook</strong><br>
                    系統將改用環境變數 <code>DISCORD_WEBHOOK_URL</code>，若未配置將導致推送失敗
                </div>
            `;
        } else {
            note.className = 'alert alert-success';
            note.innerHTML = `
                <i class="fas fa-check-circle"></i>
                <div>
                    <strong>Discord 自動推送</strong><br>
                    分析完成後會自動將報告發送到設定的頻道
                </div>
            `;
        }
    } else {
        note.className = 'alert alert-warning';
        note.innerHTML = `
            <i class="fas fa-bell-slash"></i>
            <div>
                <strong>Discord 推送已停用</strong><br>
                本次分析完成後將不會發送 Discord 報告
            </div>
        `;
    }
}

// 更新日期時間
function updateDateTime() {
    const now = new Date();
    const dateOptions = { year: 'numeric', month: 'long', day: 'numeric', weekday: 'long' };
    const timeOptions = { hour: '2-digit', minute: '2-digit', second: '2-digit' };
    
    document.getElementById('currentDate').textContent = now.toLocaleDateString('zh-TW', dateOptions);
    document.getElementById('currentTime').textContent = now.toLocaleTimeString('zh-TW', timeOptions);
}

// 啟動歷史記錄自動刷新
function startHistoryAutoRefresh() {
    if (historyRefreshInterval) {
        clearInterval(historyRefreshInterval);
    }
    historyRefreshInterval = setInterval(() => {
        loadHistory();
        updateStats();
    }, HISTORY_REFRESH_INTERVAL_MS);
}

// 頁面卸載時清除定時器
window.onbeforeunload = function() {
    if (historyRefreshInterval) {
        clearInterval(historyRefreshInterval);
    }
};

// 更新統計資料
function updateStats() {
    document.getElementById('totalStocks').textContent = config.stocks.length;
    
    fetch('/api/history')
        .then(res => res.json())
        .then(history => {
            document.getElementById('historyCount').textContent = history.length;
            const successCount = history.filter(item => item.status === 'success').length;
            document.getElementById('successCount').textContent = successCount;
        });

    fetch('/api/schedule')
        .then(res => res.json())
        .then(schedule => {
            const statusEl = document.getElementById('scheduleStatus');
            if (!statusEl) return;
            if (schedule.enabled) {
                statusEl.textContent = schedule.send_report === false || schedule.sendReport === false
                    ? '已啟用（停用推送）'
                    : '已啟用';
            } else {
                statusEl.textContent = '未啟用';
            }
        });
}

// 載入配置
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        config = await response.json();
        config.webhook_url = config.webhook_url || '';
        document.getElementById('years').value = config.years;
        const webhookInput = document.getElementById('webhookUrl');
        if (webhookInput) {
            webhookInput.value = config.webhook_url;
        }
        renderStockList();
        updateDiscordStatusNote();
        updateStats();
    } catch (error) {
        showNotification('載入配置失敗', 'error');
    }
}

// 將股票分類為台股、美股、虛擬貨幣
function categorizeStocks(stocks) {
    return stocks.reduce((acc, stock) => {
        if (stock.includes('.TW')) {
            acc.taiwan.push(stock);
        } else if (stock.includes('-USD')) {
            acc.crypto.push(stock);
        } else {
            acc.us.push(stock);
        }
        return acc;
    }, { taiwan: [], us: [], crypto: [] });
}

// 渲染股票清單
function renderStockList() {
    const container = document.getElementById('stockList');
    
    if (config.stocks.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox"></i>
                <h3>尚無股票</h3>
                <p>請使用下方表單新增股票代號</p>
            </div>
        `;
        return;
    }

    // 將股票分類
    const categorized = categorizeStocks(config.stocks);
    
    // 生成HTML
    let html = '';
    
    // 台股
    if (categorized.taiwan.length > 0) {
        html += createCategorySection('台股', categorized.taiwan, 'badge-tw', 'fa-building');
    }
    
    // 美股
    if (categorized.us.length > 0) {
        html += createCategorySection('美股', categorized.us, 'badge-us', 'fa-chart-line');
    }
    
    // 虛擬貨幣
    if (categorized.crypto.length > 0) {
        html += createCategorySection('虛擬貨幣', categorized.crypto, 'badge-crypto', 'fa-coins');
    }
    
    container.innerHTML = html;
    
    // 重新設定已選擇的股票狀態
    selectedStocks.forEach(stock => {
        const checkbox = document.getElementById(`stock-${stock}`);
        if (checkbox) checkbox.checked = true;
    });
}

// 創建分類部分
function createCategorySection(title, stocks, badgeClass, icon) {
    return `
        <div class="stock-category">
            <div class="category-header">
                <h4><i class="fas ${icon}"></i> ${title}</h4>
                <span class="category-badge ${badgeClass}">${stocks.length}</span>
            </div>
            <div class="stock-items">
                ${stocks.map(stock => createStockItem(stock, badgeClass)).join('')}
            </div>
        </div>
    `;
}

// 創建股票項目
function createStockItem(stock, badgeClass) {
    const badgeText = getBadgeText(badgeClass);
    return `
        <div class="stock-item">
            <input type="checkbox" id="stock-${stock}" onchange="toggleStock('${stock}')">
            <div class="stock-item-info">
                <span class="stock-code">${stock}</span>
                <span class="stock-badge ${badgeClass}">${badgeText}</span>
            </div>
            <button class="stock-delete" onclick="removeStock('${stock}')">
                <i class="fas fa-trash"></i>
            </button>
        </div>
    `;
}

// 根據類別獲取標籤文字
function getBadgeText(badgeClass) {
    switch(badgeClass) {
        case 'badge-tw': return '台股';
        case 'badge-us': return '美股';
        case 'badge-crypto': return '加密貨幣';
        default: return '';
    }
}

// 獲取股票標籤
function getStockBadge(stock) {
    if (stock.includes('.TW')) return { class: 'badge-tw', text: '台股' };
    if (stock.includes('-USD')) return { class: 'badge-crypto', text: '加密貨幣' };
    return { class: 'badge-us', text: '美股' };
}

// 切換股票選擇
function toggleStock(stock) {
    const checkbox = document.getElementById(`stock-${stock}`);
    if (checkbox.checked) {
        selectedStocks.add(stock);
    } else {
        selectedStocks.delete(stock);
    }
    updateScheduleDisplay();
}

// 全選
function selectAll() {
    config.stocks.forEach(stock => {
        const checkbox = document.getElementById(`stock-${stock}`);
        if (checkbox) {
            checkbox.checked = true;
            selectedStocks.add(stock);
        }
    });
    updateScheduleDisplay();
    showNotification('已選擇所有股票', 'success');
}

// 選擇所有台股
function selectTaiwan() {
    const categorized = categorizeStocks(config.stocks);
    selectStocksByCategory(categorized.taiwan);
    showNotification('已選擇所有台股', 'success');
}

// 選擇所有美股
function selectUS() {
    const categorized = categorizeStocks(config.stocks);
    selectStocksByCategory(categorized.us);
    showNotification('已選擇所有美股', 'success');
}

// 選擇所有虛擬貨幣
function selectCrypto() {
    const categorized = categorizeStocks(config.stocks);
    selectStocksByCategory(categorized.crypto);
    showNotification('已選擇所有虛擬貨幣', 'success');
}

// 按類別選擇股票
function selectStocksByCategory(stocks) {
    stocks.forEach(stock => {
        const checkbox = document.getElementById(`stock-${stock}`);
        if (checkbox) {
            checkbox.checked = true;
            selectedStocks.add(stock);
        }
    });
    updateScheduleDisplay();
}

// 取消全選
function deselectAll() {
    config.stocks.forEach(stock => {
        const checkbox = document.getElementById(`stock-${stock}`);
        if (checkbox) {
            checkbox.checked = false;
        }
    });
    selectedStocks.clear();
    updateScheduleDisplay();
    showNotification('已取消所有選擇', 'success');
}

// 新增股票
function addStock() {
    const input = document.getElementById('newStock');
    const stock = input.value.trim().toUpperCase();

    if (!stock) {
        showNotification('請輸入股票代號', 'error');
        return;
    }

    if (config.stocks.includes(stock)) {
        showNotification('股票已存在', 'error');
        return;
    }

    config.stocks.push(stock);
    input.value = '';
    renderStockList();
    updateStats();
    showNotification(`已新增 ${stock}`, 'success');
}

// 刪除股票
function removeStock(stock) {
    if (confirm(`確定要刪除 ${stock} 嗎？`)) {
        config.stocks = config.stocks.filter(s => s !== stock);
        selectedStocks.delete(stock);
        renderStockList();
        updateStats();
        showNotification(`已刪除 ${stock}`, 'success');
    }
}

// 儲存配置
async function saveConfig() {
    try {
        config.years = parseInt(document.getElementById('years').value);
        const webhookInput = document.getElementById('webhookUrl');
        if (webhookInput) {
            config.webhook_url = webhookInput.value.trim();
        }
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const result = await response.json();
        showNotification(result.message, 'success');
        updateStats();
    } catch (error) {
        showNotification('儲存配置失敗', 'error');
    }
}

// 執行分析
async function executeAnalysis() {
    if (selectedStocks.size === 0) {
        showNotification('請至少選擇一支股票', 'error');
        return;
    }

    const btn = event.target;
    const originalHTML = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> 執行中...';

    const statusDiv = document.getElementById('executionStatus');
    statusDiv.className = 'execution-status show';
    statusDiv.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span class="spinner"></span>
            <span>正在分析 ${selectedStocks.size} 支股票...</span>
        </div>
    `;

    try {
        if (manualSendReport && !config.webhook_url) {
            showNotification('尚未設定 Discord Webhook，系統將改用環境變數 DISCORD_WEBHOOK_URL', 'info');
        }

        const response = await fetch('/api/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                stocks: Array.from(selectedStocks),
                years: config.years,
                sendReport: manualSendReport
            })
        });
        const result = await response.json();

        showNotification(result.message, result.success ? 'success' : 'error');
        statusDiv.innerHTML = `
            <i class="fas fa-${result.success ? 'check-circle' : 'exclamation-circle'}"></i>
            ${result.message}
        `;

        setTimeout(() => {
            loadHistory();
            updateStats();
            statusDiv.className = 'execution-status';
        }, 3000);
    } catch (error) {
        showNotification('執行失敗', 'error');
        statusDiv.className = 'execution-status';
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalHTML;
    }
}

// 載入歷史記錄
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const history = await response.json();

        const container = document.getElementById('historyList');
        
        if (history.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-clock"></i>
                    <h3>暫無執行記錄</h3>
                    <p>執行分析後將顯示歷史記錄</p>
                </div>
            `;
            return;
        }

        container.innerHTML = history.map(item => `
            <div class="history-item ${item.status}">
                <div class="history-header">
                    <div>
                        <div class="history-stock">
                            <i class="fas fa-chart-line"></i> ${item.stock}
                            <span class="status-badge ${item.status}">
                                ${getStatusIcon(item.status)} ${getStatusText(item.status)}
                            </span>
                        </div>
                        <div class="history-time">
                            <i class="fas fa-clock"></i> ${item.timestamp}
                        </div>
                    </div>
                </div>
                <div class="history-message">
                    <i class="fas fa-info-circle"></i> ${item.message}
                </div>
            </div>
        `).join('');

        updateStats();
    } catch (error) {
        showNotification('載入歷史記錄失敗', 'error');
    }
}

// 清除歷史
async function clearHistory() {
    if (!confirm('確定要清除所有歷史記錄嗎？')) return;

    try {
        const response = await fetch('/api/history/clear', { method: 'POST' });
        const result = await response.json();
        showNotification(result.message, 'success');
        loadHistory();
        updateStats();
    } catch (error) {
        showNotification('清除失敗', 'error');
    }
}

// 載入排程配置
async function loadSchedule() {
    try {
        const response = await fetch('/api/schedule');
        const schedule = await response.json();

        document.getElementById('scheduleEnabled').checked = schedule.enabled;
        document.getElementById('scheduleTime').value = schedule.time;
        scheduleSendReport = !(schedule.send_report === false || schedule.sendReport === false);
        const scheduleToggle = document.getElementById('scheduleSendReport');
        if (scheduleToggle) {
            scheduleToggle.checked = scheduleSendReport;
        }

        schedule.stocks.forEach(stock => {
            const checkbox = document.getElementById(`stock-${stock}`);
            if (checkbox) {
                checkbox.checked = true;
                selectedStocks.add(stock);
            }
        });

        updateScheduleDisplay();
        updateStats();
    } catch (error) {
        showNotification('載入排程配置失敗', 'error');
    }
}

// 儲存排程
async function saveSchedule() {
    try {
        const scheduleConfig = {
            enabled: document.getElementById('scheduleEnabled').checked,
            time: document.getElementById('scheduleTime').value,
            stocks: Array.from(selectedStocks),
            sendReport: scheduleSendReport
        };

        const response = await fetch('/api/schedule', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(scheduleConfig)
        });
        const result = await response.json();
        showNotification(result.message, 'success');
        updateStats();
    } catch (error) {
        showNotification('儲存排程失敗', 'error');
    }
}

// 更新排程顯示
function updateScheduleDisplay() {
    const container = document.getElementById('scheduleStocks');
    if (selectedStocks.size === 0) {
        container.className = 'schedule-display';
        container.innerHTML = '<i class="fas fa-info-circle"></i> 將在儲存時自動使用目前選中的股票';
    } else {
        container.className = 'schedule-display active';
        container.innerHTML = `<i class="fas fa-check-circle"></i> 已選擇: ${Array.from(selectedStocks).join(', ')}`;
    }
}

// 顯示通知
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    const icon = type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle';
    notification.innerHTML = `
        <i class="fas fa-${icon}" style="font-size: 1.1rem;"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// 獲取狀態圖標
function getStatusIcon(status) {
    const icons = {
        'success': '<i class="fas fa-check"></i>',
        'failed': '<i class="fas fa-times"></i>',
        'error': '<i class="fas fa-exclamation"></i>'
    };
    return icons[status] || '';
}

// 獲取狀態文字
function getStatusText(status) {
    const texts = {
        'success': '成功',
        'failed': '失敗',
        'error': '錯誤'
    };
    return texts[status] || status;
}