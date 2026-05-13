# LUCY InstantID backend - Windows Server setup
#
# Run this ONCE on the AWS g5.xlarge to:
#   1. Create a Python 3.11 conda env (PyTorch CUDA wheels exist for 3.11
#      but not 3.14 which the existing face_swap_backend uses).
#   2. Install all Python deps.
#   3. Download InstantID + SDXL Lightning + antelopev2 model files.
#
# Total time: ~20 min (mostly model downloads, ~12 GB).
# After this, run start.ps1 to launch the server on port 7861.

$ErrorActionPreference = "Stop"

Write-Host "[setup] creating conda env 'instantid' (Python 3.11)" -ForegroundColor Cyan
conda create -n instantid python=3.11 -y
conda activate instantid

Write-Host "[setup] installing Python deps" -ForegroundColor Cyan
pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host "[setup] downloading SDXL base + Lightning UNet + InstantID weights" -ForegroundColor Cyan
$models = Join-Path $PSScriptRoot "models"
New-Item -ItemType Directory -Force -Path $models | Out-Null

# 1. SDXL base 1.0 (will be cached by huggingface_hub on first pipeline load,
#    but pre-downloading avoids the cold start blocking the demo).
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 `
    --include "*.safetensors" "*.json" `
    --exclude "*sd_xl_base_1.0_0.9vae.safetensors"

# 2. SDXL Lightning 4-step UNet
huggingface-cli download ByteDance/SDXL-Lightning `
    sdxl_lightning_4step_unet.safetensors `
    --local-dir $models

# 3. InstantID (ControlNet + IP-Adapter)
$instantid = Join-Path $models "InstantID"
huggingface-cli download InstantX/InstantID `
    --local-dir $instantid `
    --include "ControlNetModel/*" "ip-adapter.bin"

# 4. antelopev2 - InsightFace bundle InstantID requires.
#    Official Google Drive link is broken; use the InstantID GH issue #61
#    mirror. If this URL stops working, search the issue thread for a fresh one.
$antelope = Join-Path $models "antelopev2"
if (-not (Test-Path $antelope)) {
    Write-Host "[setup] downloading antelopev2 (~250 MB)" -ForegroundColor Cyan
    $zip = Join-Path $models "antelopev2.zip"
    Invoke-WebRequest `
        -Uri "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip" `
        -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $models -Force
    Remove-Item $zip
}

Write-Host "[setup] DONE. Run: powershell .\start.ps1" -ForegroundColor Green
