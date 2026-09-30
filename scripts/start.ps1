$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$streamlit = Join-Path $projectRoot '.venv\Scripts\streamlit.exe'
if (-not (Test-Path -LiteralPath $streamlit)) { throw '未找到 Streamlit，请先按 README 安装依赖。' }
Push-Location $projectRoot
try { & $streamlit run app.py --server.address 127.0.0.1 --server.port 8502 } finally { Pop-Location }

