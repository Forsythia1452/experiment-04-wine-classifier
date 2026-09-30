$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw '未找到 .venv，请先按 README 安装依赖。' }
Push-Location $projectRoot
try { & $python -m src.train } finally { Pop-Location }

