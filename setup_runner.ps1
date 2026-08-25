# =======================================================================
#  Mirror Fit - install the GitHub Actions runner as a Windows service
#
#  Usage (elevated PowerShell, on the GPU box):
#      powershell -ExecutionPolicy Bypass .\setup_runner.ps1 -Token <REG_TOKEN>
#
#  Get REG_TOKEN from:
#      github.com/indwar7/Mirror_Fit -> Settings -> Actions -> Runners
#      -> New self-hosted runner   (the token is valid for one hour)
#
#  Why this exists: .github/workflows/deploy.yml already deploys on every
#  push to main, but it needs a self-hosted runner on this box to do it.
#  When the runner is not registered the workflow does not fail - it sits
#  queued forever, which looks exactly like a push that did nothing, and
#  every deploy then has to be done by hand.
#
#  Installing it as a SERVICE, rather than running run.cmd in a window, is
#  the whole point: this instance gets stopped and started regularly, and a
#  runner living in someone's RDP session dies with that session. A service
#  set to Automatic comes back on its own.
# =======================================================================
param(
  [Parameter(Mandatory = $true)] [string]$Token,
  [string]$Url             = "https://github.com/indwar7/Mirror_Fit",
  [string]$Dir             = "C:\actions-runner",
  [string]$Name            = "g5-box",
  [string]$Labels          = "self-hosted,windows,gpu",
  # Blank runs the service as NT AUTHORITY\NETWORK SERVICE. Pass the account
  # you normally work as if a job needs that profile - the backends use
  # absolute paths (C:\miniconda3, C:\virtual-try-on, C:\logs) so the default
  # is usually fine, but anything reading %USERPROFILE% (the huggingface
  # cache, for one) will look somewhere different under a service account.
  [string]$ServiceAccount  = "",
  [string]$ServicePassword = ""
)

$ErrorActionPreference = "Stop"

$RUNNER_VERSION = "2.336.0"
$RUNNER_SHA256  = "d59123a43003e357b0805b5d0f611d0bd2f65ab67d51bd070dd4e7a0f685c162"

function Say($msg, $colour = "Gray") { Write-Host $msg -ForegroundColor $colour }

Say "`nMirror Fit - Actions runner setup" "Cyan"

# ── 1. Elevation ────────────────────────────────────────────────────────
$admin = ([Security.Principal.WindowsPrincipal] `
          [Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
  Say "  Run this from an elevated PowerShell - installing a service needs it." "Red"
  exit 1
}

# ── 2. Already there? ───────────────────────────────────────────────────
$existing = Get-Service "actions.runner.*" -ErrorAction SilentlyContinue
if ($existing) {
  Say ("  found existing service: {0} ({1})" -f $existing.Name, $existing.Status) "Yellow"
  Say  "  it will be reconfigured against this registration token." "DarkGray"
}

# ── 3. Download, pinned and verified ────────────────────────────────────
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
Set-Location $Dir

if (-not (Test-Path (Join-Path $Dir "config.cmd"))) {
  $zip = Join-Path $Dir "actions-runner.zip"
  $url = "https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/actions-runner-win-x64-$RUNNER_VERSION.zip"
  Say "  downloading runner $RUNNER_VERSION ..." "Cyan"
  # Some Windows builds still default to TLS 1.0, which github.com refuses.
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  $ProgressPreference = "SilentlyContinue"   # the progress bar makes this ~10x slower
  Invoke-WebRequest -Uri $url -OutFile $zip

  $got = (Get-FileHash -Path $zip -Algorithm SHA256).Hash.ToLower()
  if ($got -ne $RUNNER_SHA256) {
    Say "  checksum mismatch - refusing to install." "Red"
    Say ("    expected {0}" -f $RUNNER_SHA256) "DarkGray"
    Say ("    got      {0}" -f $got) "DarkGray"
    Remove-Item $zip -Force
    exit 1
  }
  Say "  checksum ok" "Green"
  Expand-Archive -Path $zip -DestinationPath $Dir -Force
  Remove-Item $zip -Force
} else {
  Say "  runner already unpacked in $Dir - reusing it" "DarkGray"
}

# ── 4. Register ─────────────────────────────────────────────────────────
# --replace takes over a stale registration under the same name instead of
# erroring, which is what you want after the box has been rebuilt.
$cfg = @(
  "--unattended", "--replace",
  "--url", $Url, "--token", $Token,
  "--name", $Name, "--labels", $Labels,
  "--work", "_work",
  "--runasservice"
)
if ($ServiceAccount) {
  $cfg += @("--windowslogonaccount", $ServiceAccount)
  if ($ServicePassword) { $cfg += @("--windowslogonpassword", $ServicePassword) }
  Say ("  registering as a service running under {0}" -f $ServiceAccount) "Cyan"
} else {
  Say "  registering as a service (NETWORK SERVICE)" "Cyan"
}

& (Join-Path $Dir "config.cmd") @cfg
if ($LASTEXITCODE -ne 0) {
  Say "`n  config.cmd failed. The usual cause is an expired registration token" "Red"
  Say "  - they last one hour. Generate a fresh one and re-run." "Red"
  exit 1
}

# ── 5. Survive reboots and crashes ──────────────────────────────────────
$svc = Get-Service "actions.runner.*" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $svc) {
  Say "  registered, but no service appeared - check $Dir\_diag" "Red"
  exit 1
}

Set-Service -Name $svc.Name -StartupType Automatic
# Restart on crash as well as on boot. Without this a runner that dies once
# stays dead until someone notices deploys have stopped landing.
& sc.exe failure $svc.Name reset= 86400 actions= restart/5000/restart/10000/restart/30000 | Out-Null

if ($svc.Status -ne "Running") { Start-Service -Name $svc.Name }
Start-Sleep -Seconds 4
$svc = Get-Service -Name $svc.Name

Say ("`n  service : {0}" -f $svc.Name)
Say ("  status  : {0}" -f $svc.Status) $(if ($svc.Status -eq "Running") { "Green" } else { "Red" })
Say ("  startup : {0}" -f (Get-Service -Name $svc.Name).StartType)

# ── 6. What the deploy will inherit ─────────────────────────────────────
# A service reads the machine environment once, at start. Setting HF_TOKEN
# after this point will not reach it until the service is restarted.
$hf = [Environment]::GetEnvironmentVariable("HF_TOKEN", "Machine")
if ($hf) {
  Say "`n  HF_TOKEN is set machine-wide - deploys will inherit it." "Green"
} else {
  Say "`n  HF_TOKEN is not set machine-wide. Try-On falls back to weaker" "Yellow"
  Say  "  masking without it. Set it, then restart this service:" "Yellow"
  Say  '    [Environment]::SetEnvironmentVariable("HF_TOKEN","hf_...","Machine")' "DarkGray"
  Say ("    Restart-Service {0}" -f $svc.Name) "DarkGray"
}

Say "`n  done. Push to main and it deploys itself." "Cyan"
Say  "  check:  github.com/indwar7/Mirror_Fit/settings/actions/runners" "DarkGray"
Say  "  logs :  $Dir\_diag`n" "DarkGray"
