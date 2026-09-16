#!/usr/bin/env bash
# Phase 0: provision a Linux GPU box for the live pipeline.
#
# Ubuntu 24.04, NVIDIA driver, CUDA, Python env, StreamDiffusionV2 and
# the Wan 2.1 1.3B weights, then a smoke test that fails loudly rather
# than leaving a half-built box that breaks during the first session.
#
#   bash setup_gpu_box.sh
#
# Re-runnable: every step checks before it installs.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/virtual-try-on}"
VENV="${VENV:-$HOME/.venv/live}"
MODEL_ID="${LIVE_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "Checking the GPU"
if ! command -v nvidia-smi >/dev/null; then
  die "nvidia-smi not found. Install the NVIDIA driver first:
  sudo apt update && sudo apt install -y nvidia-driver-560
then reboot and re-run this script."
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 20000 ]; then
  die "Only ${VRAM}MB of VRAM. The 1.3B model needs ~20GB at 480p."
elif [ "$VRAM" -lt 30000 ]; then
  echo "WARNING: ${VRAM}MB VRAM. Below the 32GB the plan assumes; expect"
  echo "         to drop to 360p or fewer steps."
fi

say "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev build-essential \
  ffmpeg libgl1 libglib2.0-0 git curl

say "Python environment at $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel

say "PyTorch (CUDA 12.4)"
python -c "import torch" 2>/dev/null || \
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124

say "Pipeline requirements"
pip install -q -r "$(dirname "$0")/requirements.txt"
pip install -q diffusers transformers accelerate peft safetensors lpips

say "StreamDiffusionV2"
# Published on PyPI (chenfengxu714/StreamDiffusionV2 upstream). It caps
# Python at <3.13 and expects Linux + NVIDIA, so check before installing
# rather than letting pip fail three minutes in.
PYVER=$(python -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PYVER" in
  3.10|3.11|3.12) ;;
  *) die "streamdiffusionv2 needs Python 3.10-3.12, this env has $PYVER.
Create one:  conda create -n live python=3.11 -y && conda activate live" ;;
esac
python -c "import streamdiffusionv2" 2>/dev/null || pip install -q streamdiffusionv2

say "Model weights: $MODEL_ID"
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_ID")
print("weights at", p)
PY

say "Smoke test"
python - <<'PY'
import sys, torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
dev = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print(f"  {dev}: {free/2**30:.1f}GB free of {total/2**30:.1f}GB")

# A real allocation, not just a version check: a box that reports CUDA
# but cannot allocate is the failure that shows up mid-session.
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
import time
t0 = time.monotonic()
for _ in range(50):
    x = x @ x.T
    x = x / x.norm()
torch.cuda.synchronize()
dt = time.monotonic() - t0
print(f"  50 matmuls at 4096x4096 fp16: {dt*1000:.0f}ms")
if dt > 3.0:
    print("  WARNING: unexpectedly slow for a modern card")
PY

say "Done"
cat <<EOT

Start the server:
  source $VENV/bin/activate
  cd $REPO_DIR/live_pipeline
  python -m server.app

Then open http://<this-box>:8080/mirror.html

Ports 8080 (signaling) and the WebRTC UDP range must be open in the
firewall and, on a cloud box, in the security group as well.
EOT
