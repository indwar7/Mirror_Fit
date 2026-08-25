"""
Real VTON backends: human parsing, DensePose, and the try-on model.

Separated from tryon.py for the same reason pose_backends.py is separated from
avatar_validation.py — the masking and paste-back rules carry the identity
guarantee and must be testable on a machine with none of these installed.

Loading never raises; it returns None and TryOnPipeline raises when a caller
actually tries to use a missing backend. The failure then names which model is
absent, at the point where it would have mattered.

These are thin adapters. Each expects its upstream project to be installed on
the box — SCHP for parsing, detectron2 for DensePose, CatVTON or IDM-VTON for
the try-on itself. None of them is vendored here.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class SCHPParser:
    """Self-Correction Human Parsing, ATR label set (18 classes)."""

    def __init__(self, model): self._model = model

    @classmethod
    def load(cls) -> Optional["SCHPParser"]:
        try:
            import torch  # noqa: F401
            from schp import SCHP  # type: ignore
            weights = os.environ.get("SCHP_WEIGHTS", "exp-schp-201908301523-atr.pth")
            return cls(SCHP(weights, dataset="atr"))
        except Exception as e:
            log.warning("[vton] SCHP parser unavailable (%s: %s)", type(e).__name__, e)
            return None

    def parse(self, image_bgr: np.ndarray) -> np.ndarray:
        return self._model.parse(image_bgr)


class DensePose:
    """detectron2 DensePose IUV map."""

    def __init__(self, predictor): self._p = predictor

    @classmethod
    def load(cls) -> Optional["DensePose"]:
        try:
            from densepose_wrapper import DensePosePredictor  # type: ignore
            return cls(DensePosePredictor())
        except Exception as e:
            log.warning("[vton] DensePose unavailable (%s: %s)", type(e).__name__, e)
            return None

    def densepose(self, image_bgr: np.ndarray) -> np.ndarray:
        return self._p.predict(image_bgr)


class CatVTON:
    """CatVTON, conditioned on the agnostic mask and DensePose.

    The tryon_backend service already loads CatVTON for live try-on. Point
    LUCY_CATVTON_URL at it to reuse that process instead of loading a second
    copy of the weights into this one — an A10G does not have room for both.
    """

    def __init__(self, model): self._model = model

    @classmethod
    def load(cls) -> Optional["CatVTON"]:
        try:
            from catvton_wrapper import CatVTONModel  # type: ignore
            return cls(CatVTONModel())
        except Exception as e:
            log.warning("[vton] CatVTON unavailable (%s: %s)", type(e).__name__, e)
            return None

    def generate(self, person_bgr, agnostic_mask, densepose, garment_bgr, category):
        return self._model.infer(
            person=person_bgr, mask=agnostic_mask,
            densepose=densepose, cloth=garment_bgr, category=category,
        )


def load_backends():
    """(parser, densepose, vton). Any of them may be None."""
    parser, dense, vton = SCHPParser.load(), DensePose.load(), CatVTON.load()
    missing = [n for n, b in (("parser", parser), ("densepose", dense), ("vton", vton))
               if b is None]
    if missing:
        log.error("[vton] try-on will fail closed — missing: %s", ", ".join(missing))
    return parser, dense, vton
