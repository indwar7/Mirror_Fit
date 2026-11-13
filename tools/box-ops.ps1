# One-shot box maintenance, run as Administrator through an RDP alternate
# shell when the Actions runner is down. Its report goes to a file under
# demo/, which the face-swap backend serves on :7860, so it can be read from
# outside without a working runner. The report deletes itself after 15 min.
$ErrorActionPreference = 'Continue'
$root = 'C:\virtual-try-on'
$out  = Join-Path $root 'demo\ops-k3v9.txt'
$svcCache = 'C:\Windows\ServiceProfiles\NetworkService\.cache\huggingface'

function GB($d) {
  if (-not (Test-Path $d)) { return 'missing' }
  [math]::Round(((Get-ChildItem $d -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum) / 1GB, 2)
}

$r = @()
$r += 'ran at ' + (Get-Date -Format s) + ' as ' + (whoami)
$freeBefore = [math]::Round((Get-PSDrive C).Free / 1GB, 2)

# The deploy of 2026-11-13 07:20 ran try-on as NETWORK SERVICE, which has no
# model cache, so it re-downloaded the weights here until the disk was full.
# This folder holds nothing but that partial download.
$r += 'service-account HF cache GB: ' + (GB $svcCache)
Remove-Item $svcCache -Recurse -Force -ErrorAction SilentlyContinue
$r += 'C: free GB before ' + $freeBefore + ' after ' + [math]::Round((Get-PSDrive C).Free / 1GB, 2)

$r += '--- sessions ---';        $r += (& query session 2>&1 | Out-String)
$r += '--- disk ---';            $r += (Get-PSDrive C | Select-Object @{n='UsedGB';e={[math]::Round($_.Used/1GB,1)}},@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}} | Out-String)
$r += '--- runner service ---'
$r += (Get-CimInstance Win32_Service -Filter "Name like 'actions.runner%'" | Select-Object Name,StartName,State,StartMode | Out-String)
$r += '--- machine env ---'
foreach ($v in 'HF_HOME','HF_HUB_CACHE','HF_TOKEN') {
  $x = [Environment]::GetEnvironmentVariable($v, 'Machine')
  if ($v -eq 'HF_TOKEN') { $x = if ($x) { 'set' } else { '' } }
  $r += "$v=$x"
}
$r += '--- Administrator HF hub ---'
$r += ((Get-ChildItem 'C:\Users\Administrator\.cache\huggingface\hub' -ErrorAction SilentlyContinue | ForEach-Object Name) -join ', ')
$r += '--- folder sizes GB ---'
foreach ($d in 'C:\Users\Administrator\.cache\huggingface', $root, 'C:\miniconda3', 'C:\actions-runner', 'C:\logs',
               'C:\Users\Administrator\Downloads', 'C:\Windows\Temp', 'C:\Users\Administrator\AppData\Local\Temp') {
  $r += "$d : " + (GB $d)
}
$r += '--- python ---';          $r += (Get-Process python* -ErrorAction SilentlyContinue | Select-Object Id,StartTime,Path | Out-String)
$r += '--- listening ---'
$r += (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in 7860,7862,8000,7863 } | Select-Object LocalPort,OwningProcess | Out-String)
$r += '--- try-on err (service run, tail 5) ---'
$r += (Get-Content 'C:\logs\ec2amaz28ltj0q\try-on.err.log' -Tail 5 -ErrorAction SilentlyContinue | Out-String)
$r | Set-Content $out

# Bring the runner back so deploys and diag work again.
try { Restart-Service 'actions.runner*' -ErrorAction Stop; Add-Content $out 'runner restart: ok' }
catch { Add-Content $out ('runner restart: ' + $_.Exception.Message) }
Add-Content $out ('runner now: ' + ((Get-Service 'actions.runner*' -ErrorAction SilentlyContinue | ForEach-Object Status) -join ','))

# Start try-on as this account, whose cache already holds the weights.
Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -Command `"& $root\start_all.ps1 -Only tryon *> C:\logs\start_all_rdp.log`"" -WindowStyle Hidden
Add-Content $out 'try-on start requested; see C:\logs\start_all_rdp.log'

# The report is on a public port: remove it after a quarter of an hour.
Start-Process powershell -ArgumentList "-Command Start-Sleep 900; Remove-Item '$out','$root\demo\shell-ran.txt' -Force" -WindowStyle Hidden
