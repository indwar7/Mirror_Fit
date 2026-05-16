# LUCY LivePortrait backend - Windows Server setup
#
# Run this ONCE on the AWS g5.xlarge to:
#   1. Create a Python 3.11 conda env (separate from instantid env).
#   2. Install LivePortrait Python deps.
#   3. Clone the official KwaiVGI/LivePortrait repo.
#   4. Download the pretrained model weights (~2.2 GB).
#
# Total time ~10-15 min mostly model downloads.
# After this, run start.ps1 to launch the server on port 7862.

$ErrorActionPreference = "Stop"

Write-Host "[setup] creating conda env 'liveportrait' (Python 3.11)" -ForegroundColor Cyan
conda create -n liveportrait python=3.11 -y
conda activate liveportrait

Write-Host "[setup] installing Python deps" -ForegroundColor Cyan
pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host "[setup] cloning KwaiVGI/LivePortrait (upstream code)" -ForegroundColor Cyan
$lpRoot = Join-Path $PSScriptRoot "LivePortrait"
if (-not (Test-Path $lpRoot)) {
    git clone --depth 1 https://github.com/KwaiVGI/LivePortrait $lpRoot
}

Write-Host "[setup] downloading LivePortrait model weights (~2.2 GB)" -ForegroundColor Cyan
$weights = Join-Path $lpRoot "pretrained_weights"
# Use `hf` (new HuggingFace CLI). `huggingface-cli` is deprecated and
# returns 0 exit code with only a warning, causing silent download failure.
hf download KwaiVGI/LivePortrait --local-dir $weights
if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup] ERROR: HuggingFace download failed (exit $LASTEXITCODE)" -ForegroundColor Red
    exit 1
}
# Sanity check: verify the 4 critical .pth files actually exist.
$required = @(
    "liveportrait\base_models\appearance_feature_extractor.pth",
    "liveportrait\base_models\motion_extractor.pth",
    "liveportrait\base_models\warping_module.pth",
    "liveportrait\base_models\spade_generator.pth"
)
foreach ($f in $required) {
    $full = Join-Path $weights $f
    if (-not (Test-Path $full)) {
        Write-Host "[setup] ERROR: required file missing after download: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[setup] DONE. Run: powershell .\start.ps1" -ForegroundColor Green
