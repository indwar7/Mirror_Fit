# LUCY LivePortrait backend - Windows Server setup
#
# Run this ONCE on the AWS g5.xlarge to:
#   1. Create a Python 3.11 conda env (separate from instantid env).
#   2. Install LivePortrait Python deps.
#   3. Clone the official KwaiVGI/LivePortrait repo.
#   4. Download the pretrained model weights (~2.2 GB).
#
# Total time ~10-15 min mostly model downloads. Safe to re-run: every step
# is skipped if it already completed.
# After this, run start.ps1 to launch the server on port 7862.

$ErrorActionPreference = "Stop"

# Every command below runs through `conda run -n liveportrait` rather than
# `conda activate`. Activation is a no-op in a non-interactive shell (which
# is exactly how CI and Start-Process invoke this), and the failure mode is
# silent: pip installs 2.2 GB of CUDA wheels into base and the env stays empty.
$envName = "liveportrait"

$exists = (conda env list) -match "^$envName\s"
if ($exists) {
    Write-Host "[setup] conda env '$envName' already exists - skipping create" -ForegroundColor DarkGray
} else {
    Write-Host "[setup] creating conda env '$envName' (Python 3.11)" -ForegroundColor Cyan
    conda create -n $envName python=3.11 -y
}

Write-Host "[setup] installing Python deps into '$envName'" -ForegroundColor Cyan
conda run -n $envName --no-capture-output pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Host "[setup] ERROR: pip install failed" -ForegroundColor Red; exit 1 }

Write-Host "[setup] verifying CUDA torch landed in the env" -ForegroundColor Cyan
conda run -n $envName --no-capture-output python -c "import torch; print('[setup] torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { Write-Host "[setup] ERROR: torch did not install correctly" -ForegroundColor Red; exit 1 }

Write-Host "[setup] cloning KwaiVGI/LivePortrait (upstream code)" -ForegroundColor Cyan
$lpRoot = Join-Path $PSScriptRoot "LivePortrait"
if (-not (Test-Path $lpRoot)) {
    git clone --depth 1 https://github.com/KwaiVGI/LivePortrait $lpRoot
}
if (-not (Test-Path (Join-Path $lpRoot "src"))) {
    Write-Host "[setup] ERROR: LivePortrait\src missing - the clone did not complete" -ForegroundColor Red
    exit 1
}

Write-Host "[setup] downloading LivePortrait model weights (~2.2 GB)" -ForegroundColor Cyan
$weights = Join-Path $lpRoot "pretrained_weights"
# NB: `huggingface-cli` is deprecated in huggingface_hub 0.26+ - it
# prints a warning and exits 0 WITHOUT downloading anything. The new
# entrypoint is `hf`. Using `hf download` so the setup actually fetches
# the weights instead of silently no-op'ing.
conda run -n $envName --no-capture-output hf download KwaiVGI/LivePortrait --local-dir $weights

# Verify the pieces the engine actually loads at startup are all on disk.
$probes = @(
    "liveportrait\base_models\appearance_feature_extractor.pth",
    "liveportrait\base_models\motion_extractor.pth",
    "liveportrait\base_models\warping_module.pth",
    "liveportrait\base_models\spade_generator.pth",
    "liveportrait\retargeting_models\stitching_retargeting_module.pth",
    "liveportrait\landmark.onnx",
    "insightface\models\buffalo_l\det_10g.onnx"
)
$missing = @()
foreach ($probe in $probes) {
    if (-not (Test-Path (Join-Path $weights $probe))) { $missing += $probe }
}
if ($missing.Count -gt 0) {
    Write-Host "[setup] ERROR: weights incomplete under $weights" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host "          missing: $m" -ForegroundColor Red }
    Write-Host "[setup] Retry: conda run -n $envName hf download KwaiVGI/LivePortrait --local-dir $weights" -ForegroundColor Yellow
    exit 1
}

Write-Host "[setup] running offline logic tests" -ForegroundColor Cyan
conda run -n $envName --no-capture-output python (Join-Path $PSScriptRoot "test_engine_logic.py")
if ($LASTEXITCODE -ne 0) { Write-Host "[setup] WARNING: logic tests failed - the server may still start" -ForegroundColor Yellow }

Write-Host "[setup] DONE. Run: powershell .\start.ps1" -ForegroundColor Green
