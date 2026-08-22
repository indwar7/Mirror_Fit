# Launch the LivePortrait backend on port 7862.
# Run setup.ps1 first if you haven't.
#
# The env's interpreter is invoked directly rather than via `conda activate`:
# activation only works in a conda-initialised shell, and when it silently
# fails the script runs `python server.py` against base - which has no CUDA
# torch, so the server starts and then dies on the first import.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $env:USERPROFILE "miniconda3\envs\liveportrait\python.exe"
if (-not (Test-Path $py)) { $py = "C:\miniconda3\envs\liveportrait\python.exe" }
if (-not (Test-Path $py)) {
    Write-Host "[start] conda env 'liveportrait' not found. Run: powershell .\setup.ps1" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path (Join-Path $PSScriptRoot "LivePortrait\src"))) {
    Write-Host "[start] LivePortrait/ is missing or incomplete. Run: powershell .\setup.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "[start] LivePortrait backend -> http://0.0.0.0:7862 (health: /health)" -ForegroundColor Cyan
& $py server.py
