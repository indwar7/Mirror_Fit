# =======================================================================
#  Mirror Fit - start every backend
#
#  Usage (PowerShell, on the GPU box):
#      powershell -ExecutionPolicy Bypass .\start_all.ps1
#      powershell -ExecutionPolicy Bypass .\start_all.ps1 -Force     # restart even if already up
#      powershell -ExecutionPolicy Bypass .\start_all.ps1 -Only tryon
#
#  By default it only starts what is DOWN. A backend that is already
#  serving is left alone, so running this cannot interrupt a session
#  someone else is in the middle of.
#
#  Three of the eight tabs - AR Avatar, Fit Analyser, Returns Guard -
#  need no backend at all. They run entirely in the browser.
# =======================================================================
param(
  [switch]$Force,
  [string]$Only = ""
)

$ErrorActionPreference = "Continue"
$root   = $PSScriptRoot
$logDir = "C:\logs"
$conda  = "C:\miniconda3"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# name, port, folder, entry script, python interpreter, what it serves
$svc = @(
  @{ name="face-swap";     port=7860; dir="face_swap_backend";    entry="main.py";
     py="$conda\python.exe";                            tabs="Live Face Swap, My Avatar" }
  @{ name="live-portrait"; port=7862; dir="liveportrait_backend"; entry="server.py";
     py="$conda\envs\liveportrait\python.exe";          tabs="Live Portrait" }
  @{ name="try-on";        port=8000; dir="tryon_backend";        entry="server.py";
     py="$conda\python.exe";                            tabs="Live Try-On" }
  @{ name="ai-twin";       port=7863; dir="sadtalker_backend";    entry="server.py";
     py="$conda\envs\sadtalker\python.exe";             tabs="AI Twin" }
)

function Norm($t) { ($t -replace '[^a-z0-9]', '').ToLower() }

function Test-Port($p) {
  try { (New-Object Net.Sockets.TcpClient).Connect("127.0.0.1", $p); $true } catch { $false }
}

Write-Host "`nMirror Fit - backend startup" -ForegroundColor Cyan
Write-Host ("root: {0}`n" -f $root)

# The try-on backend needs a Hugging Face token to pull the CatVTON /
# inpainting weights. Without it the model falls back to a tier that does
# not mask properly, which looks like a rendering bug rather than a
# missing credential - so fail loudly here instead.
if (-not $env:HF_TOKEN) {
  Write-Host "HF_TOKEN is not set. Try-On needs it to fetch model weights." -ForegroundColor Yellow
  Write-Host 'Set it first:  $env:HF_TOKEN = "hf_..."' -ForegroundColor Yellow
  Write-Host ""
}

$selected = @($svc | Where-Object { -not $Only -or (Norm $_.name).Contains((Norm $Only)) })
if ($Only -and $selected.Count -eq 0) {
  Write-Host ("  no service matches -Only '{0}'. Known: {1}" -f $Only, (($svc | ForEach-Object { $_.name }) -join ", ")) -ForegroundColor Red
  exit 1
}

# Only wait on services this run actually tried to start. Waiting on one that
# was skipped (no conda env) burned the whole 120s timeout printing
# "3 of 4 listening" while nothing was ever going to change.
$expected = @()

foreach ($s in $selected) {

  $dir = Join-Path $root $s.dir
  $up  = Test-Port $s.port

  if (-not (Test-Path $dir)) {
    Write-Host ("  SKIP  {0,-14} :{1}  folder missing" -f $s.name, $s.port) -ForegroundColor DarkGray
    continue
  }
  if (-not (Test-Path $s.py)) {
    Write-Host ("  SKIP  {0,-14} :{1}  interpreter missing: {2}" -f $s.name, $s.port, $s.py) -ForegroundColor Yellow
    Write-Host ("        run {0}\setup.ps1 on this box first" -f $s.dir) -ForegroundColor DarkGray
    continue
  }
  if ($up -and -not $Force) {
    Write-Host ("  UP    {0,-14} :{1}  already serving - left alone" -f $s.name, $s.port) -ForegroundColor Green
    $expected += $s
    continue
  }
  if ($up -and $Force) {
    Write-Host ("  KILL  {0,-14} :{1}" -f $s.name, $s.port) -ForegroundColor DarkYellow
    Get-NetTCPConnection -LocalPort $s.port -State Listen -ErrorAction SilentlyContinue |
      ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
  }

  $expected += $s
  New-NetFirewallRule -DisplayName ("Allow " + $s.port) -Direction Inbound `
      -LocalPort $s.port -Protocol TCP -Action Allow -ErrorAction SilentlyContinue | Out-Null

  Start-Process -FilePath $s.py -ArgumentList $s.entry -WorkingDirectory $dir `
      -RedirectStandardOutput (Join-Path $logDir ($s.name + ".log")) `
      -RedirectStandardError  (Join-Path $logDir ($s.name + ".err.log")) `
      -WindowStyle Hidden
  Write-Host ("  START {0,-14} :{1}  -> {2}\{3}.log" -f $s.name, $s.port, $logDir, $s.name) -ForegroundColor Cyan
}

# Models load lazily; give the heaviest a fair chance before reporting.
Write-Host "`nwaiting for models to load (up to 120s)..." -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds(120)
do {
  Start-Sleep -Seconds 5
  $pending = @($expected | Where-Object { -not (Test-Port $_.port) })
  Write-Host ("  {0} of {1} listening" -f ($expected.Count - $pending.Count), $expected.Count)
} while ($pending.Count -gt 0 -and (Get-Date) -lt $deadline)

Write-Host "`n--------------- status ---------------" -ForegroundColor Cyan
foreach ($s in $selected) {
  $ok = Test-Port $s.port
  $c  = if ($ok) { "Green" } else { "Red" }
  Write-Host ("  {0,-14} :{1}  {2,-8} {3}" -f $s.name, $s.port,
              $(if ($ok) { "UP" } else { "DOWN" }), $s.tabs) -ForegroundColor $c
  if (-not $ok) {
    $err = Join-Path $logDir ($s.name + ".err.log")
    if (Test-Path $err) {
      Get-Content $err -Tail 3 | ForEach-Object { Write-Host ("        " + $_) -ForegroundColor DarkGray }
    }
  }
}
Write-Host "`n  AR Avatar / Fit Analyser / Returns Guard - browser only, no backend" -ForegroundColor DarkGray
Write-Host ("`nopen:  http://localhost:7860/demo/index.html" ) -ForegroundColor Cyan
Write-Host   "       (localhost avoids the Chrome insecure-origin flag entirely)`n"
