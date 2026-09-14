"""Render the 3D jacket onto a frame, using the app's own rig.

The plan says the training render must use "the same rig and maths as
the app". A second implementation in Python would drift from
mirror.html within a week and the LoRA would learn to correct a
composite that production never produces.

So this drives the real page in headless Chromium instead: same
three.js, same MediaPipe landmarks, same bone mapping, same calibration.
Slower than a reimplementation, and correct by construction.
"""
from __future__ import annotations

import base64
import logging
import pathlib

import numpy as np

log = logging.getLogger(__name__)

CLIENT_DIR = pathlib.Path(__file__).resolve().parent.parent / "client"

# Feed a still frame to the page in place of the webcam, let the rig
# settle on it, and read the composite back.
_DRIVE = """
async (dataUrl) => {
  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  // The page exposes these for exactly this purpose (see mirror.html).
  await window.__feedFrame(img);
  return window.__composite();
}
"""


class GuideRenderer:
    """Headless mirror.html. One instance per worker process."""

    def __init__(self, width: int, height: int, fit: float = 1.0) -> None:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",   # deterministic software GL
                "--enable-unsafe-swiftshader",
                "--disable-gpu-sandbox",
            ]
        )
        self.page = self.browser.new_page(viewport={"width": width, "height": height})
        self.page.goto((CLIENT_DIR / "mirror.html").as_uri() + "?headless=1")
        self.page.wait_for_function("window.__ready === true", timeout=120_000)
        self.page.evaluate("(f) => window.__setFit(f)", fit)
        log.info("guide renderer ready at %dx%d", width, height)

    def render(self, bgr: np.ndarray) -> np.ndarray | None:
        """Return the frame with the 3D jacket composited, or None if the
        pose was not found (person out of frame, motion blur)."""
        import cv2  # noqa: PLC0415

        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return None
        url = "data:image/png;base64," + base64.b64encode(buf).decode()
        out = self.page.evaluate(_DRIVE, url)
        if not out:
            return None
        raw = base64.b64decode(out.split(",", 1)[1])
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return arr

    def close(self) -> None:
        self.browser.close()
        self._pw.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
