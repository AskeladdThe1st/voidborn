$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv/Scripts/python.exe') -or -not (Test-Path 'frontend/dist/index.html')) {
    throw 'Run .\setup.ps1 first.'
}
Write-Host 'Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop. Ya shyte!'
& ./.venv/Scripts/python.exe main.py