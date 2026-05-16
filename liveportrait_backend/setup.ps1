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
# NB: `huggingface-cli` is deprecated in huggingface_hub 0.26+ — it
# prints a warning and exits 0 WITHOUT downloading anything. The new
# entrypoint is `hf`. Using `hf download` so the setup actually fetches
# the weights instead of silently no-op'ing.
hf download KwaiVGI/LivePortrait --local-dir $weights

# Verify the appearance feature extractor actually landed on disk
$probe = Join-Path $weights "liveportrait\base_models\appearance_feature_extractor.pth"
if (-not (Test-Path $probe)) {
    Write-Host "[setup] ERROR: weights did not download to $probe" -ForegroundColor Red
    Write-Host "[setup] Try manually: hf download KwaiVGI/LivePortrait --local-dir $weights" -ForegroundColor Yellow
    exit 1
}

Write-Host "[setup] DONE. Run: powershell .\start.ps1" -ForegroundColor Green
