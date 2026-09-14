"""Blue-screen keying and garment masking.

Two jobs, both about removing things the model must not copy:

1. The blue screen becomes a neutral background, so the model does not
   learn "jacket implies blue room".
2. The *real* jacket is masked out of the input frame. Without this the
   model can read the answer straight off its own input and the LoRA
   learns nothing.
"""
from __future__ import annotations

import cv2
import numpy as np

# Blue screen in HSV. Generous on saturation/value, tight on hue, so
# shadowed folds of the screen still key out but skin never does.
BLUE_LO = np.array([95, 80, 60], dtype=np.uint8)
BLUE_HI = np.array([130, 255, 255], dtype=np.uint8)


def blue_mask(bgr: np.ndarray) -> np.ndarray:
    """1 where the pixel is blue screen, 0 where it is the subject."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    # Close pinholes in the screen, then open to drop blue speckle on the
    # subject (denim, a watch face) that would punch holes in the person.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return (m > 0).astype(np.uint8)


def despill(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove blue bounce light from the subject's edges.

    Without this, every training target has a blue rim and the model
    reproduces it faithfully on people who were never near a blue screen.
    """
    out = bgr.astype(np.float32)
    b, g, r = out[..., 0], out[..., 1], out[..., 2]
    # Where blue leads the other channels, pull it back to their max.
    cap = np.maximum(g, r)
    spill = (b > cap) & (mask == 0)
    out[..., 0] = np.where(spill, cap, b)
    return np.clip(out, 0, 255).astype(np.uint8)


def replace_background(bgr: np.ndarray, mask: np.ndarray, colour=(128, 128, 128)) -> np.ndarray:
    """Flat neutral grey behind the subject."""
    out = despill(bgr, mask)
    out[mask == 1] = colour
    return out


def garment_mask(bgr: np.ndarray, person: np.ndarray, hue_lo: int, hue_hi: int) -> np.ndarray:
    """1 where the real jacket is.

    The jacket is a solid non-blue colour by the recording protocol, so a
    hue window inside the person's silhouette finds it. `person` keeps
    the window from matching a same-coloured wall.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (h >= hue_lo) & (h <= hue_hi) & (s > 60) & (v > 40) & (person == 1)
    m = m.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    # Keep only the largest blob: stray same-hue pixels (a logo, a shoe)
    # would otherwise be erased from the input for no reason.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == biggest).astype(np.uint8)
    return m


def erase_garment(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint the real jacket away so the model cannot copy it."""
    if mask.sum() == 0:
        return bgr
    # Dilate first: the garment's own edge pixels carry its colour, and
    # inpainting from them reconstructs a ghost of the jacket.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    wide = cv2.dilate(mask, k, iterations=2)
    return cv2.inpaint(bgr, wide, 7, cv2.INPAINT_TELEA)
