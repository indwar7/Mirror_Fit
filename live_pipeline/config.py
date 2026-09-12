"""Tunables for the live pipeline. One place, so the worker and the
server never disagree about frame size or session length."""
import os


def _int(name, default):
    return int(os.environ.get(name, default))


def _float(name, default):
    return float(os.environ.get(name, default))


# --- video ---
WIDTH = _int("LIVE_WIDTH", 854)          # 480p-class, matches the plan's target
HEIGHT = _int("LIVE_HEIGHT", 480)
TARGET_FPS = _int("LIVE_FPS", 12)        # plan: 12-16 fps on a 5090

# --- diffusion ---
# Low strength: the 3D composite already carries the garment, the model
# only has to make it look real. High strength repaints the whole frame.
STRENGTH = _float("LIVE_STRENGTH", 0.35)
STEPS = _int("LIVE_STEPS", 2)            # StreamDiffusionV2 runs 1-4 steps
PROMPT = os.environ.get(
    "LIVE_PROMPT",
    "a photo of a person wearing a jacket, natural fabric folds, studio light",
)
NEGATIVE_PROMPT = os.environ.get("LIVE_NEGATIVE_PROMPT", "blurry, distorted, extra limbs")

# --- session / queue ---
SESSION_SECONDS = _int("LIVE_SESSION_SECONDS", 30)   # plan: 30-second sessions
MAX_CONCURRENT = _int("LIVE_MAX_CONCURRENT", 1)      # one GPU, one live stream

# --- model ---
MODEL_ID = os.environ.get("LIVE_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B")
LORA_PATH = os.environ.get("LIVE_LORA_PATH", "")     # empty in Phase 1
DEVICE = os.environ.get("LIVE_DEVICE", "cuda")
