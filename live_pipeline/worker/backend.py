"""The model behind a live session.

Two implementations share one interface so the whole pipeline — WebRTC,
queue, client — can be built and tested on a laptop, then run unchanged
on the GPU box. `PassthroughBackend` is not a mock for tests only: it is
also the honest fallback when no GPU is free.
"""
from __future__ import annotations

import logging
import time

import numpy as np

import config

log = logging.getLogger(__name__)


class Backend:
    """Turns one incoming frame into one outgoing frame."""

    name = "base"

    def warmup(self) -> None:
        """Run once before the first session so the first frame is not slow."""

    def process(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PassthroughBackend(Backend):
    """Returns the 3D composite untouched.

    This is what a user sees while waiting for a GPU, and what runs when
    the pipeline is developed off-box. The garment is already in the
    frame — it just has not been made photoreal yet.
    """

    name = "passthrough"

    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame


class StreamDiffusionBackend(Backend):
    """StreamDiffusionV2 on Wan 2.1 1.3B, video-to-video.

    The 3D composite arrives with the garment already placed, so strength
    stays low: the model restyles the composite rather than inventing a
    jacket. Import is deferred to __init__ so that importing this module
    costs nothing on a machine with no CUDA.
    """

    name = "streamdiffusion"

    def __init__(self) -> None:
        from streamdiffusionv2 import StreamDiffusionV2  # noqa: PLC0415

        log.info("loading %s on %s", config.MODEL_ID, config.DEVICE)
        t0 = time.monotonic()
        self.pipe = StreamDiffusionV2.from_pretrained(
            config.MODEL_ID,
            device=config.DEVICE,
            dtype="float16",
        )
        if config.LORA_PATH:
            # Phase 4 output: the garment LoRA drops in here, nothing else changes.
            log.info("loading LoRA %s", config.LORA_PATH)
            self.pipe.load_lora(config.LORA_PATH)
        self.pipe.prepare(
            prompt=config.PROMPT,
            negative_prompt=config.NEGATIVE_PROMPT,
            num_inference_steps=config.STEPS,
            strength=config.STRENGTH,
        )
        log.info("model ready in %.1fs", time.monotonic() - t0)

    def warmup(self) -> None:
        blank = np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
        for _ in range(config.STEPS):
            self.pipe(blank)

    def process(self, frame: np.ndarray) -> np.ndarray:
        return self.pipe(frame)

    def close(self) -> None:
        self.pipe = None


def build_backend() -> Backend:
    """Pick a backend, preferring the real model but never crashing a
    session because the GPU is missing."""
    try:
        return StreamDiffusionBackend()
    except Exception as exc:  # ImportError, CUDA missing, OOM
        log.warning("falling back to passthrough: %s", exc)
        return PassthroughBackend()
