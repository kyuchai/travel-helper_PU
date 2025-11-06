# ==========================================
# PowerShell 一鍵建立 & 啟動 Python 3.11 venv
# ==========================================

# 設定專案資料夾路徑
$projectPath = "C:\Users\user\Desktop\final"
$venvPath = "$projectPath\venv"

# 進入專案資料夾
Set-Location $projectPath

# 檢查虛擬環境是否存在
if (-Not (Test-Path $venvPath)) {
    Write-Host "虛擬環境不存在，正在建立..."
    python -m venv venv
    Write-Host "虛擬環境建立完成。"
}
else {
    Write-Host "虛擬環境已存在。"
}

# 臨時允許執行 PowerShell 指令碼
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# 啟動虛擬環境
$activateScript = "$venvPath\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "啟動虛擬環境..."
    & $activateScript
}
else {
    Write-Host "找不到 Activate.ps1，請確認 venv 是否建立成功。"
    exit
}

# 升級 pip
Write-Host "升級 pip..."
python -m pip install --upgrade pip

# 安裝 requirements.txt 套件（如果有）
$reqFile = "$projectPath\requirements.txt"
if (Test-Path $reqFile) {
    Write-Host "安裝 requirements.txt 套件..."
    python -m pip install -r $reqFile
}
else {
    Write-Host "沒有找到 requirements.txt，跳過套件安裝。"
}

Write-Host "`n虛擬環境已啟動，並完成 pip 升級與套件安裝！"
Write-Host "提示符前會出現 (venv)，表示已成功進入 Python 3.11 環境。"
