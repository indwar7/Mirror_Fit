"""
Skin-tone matching between a swapped face and the body it landed on.

The enrolled selfie and the base-body photograph were lit by different rooms.
Swap one onto the other and the join at the neck is the first thing anyone
notices — a face a shade lighter or warmer than the neck under it reads as
wrong long before anyone can say why.

Mean/std transfer in LAB, computed on skin pixels only. LAB because L carries
lightness separately from the a/b colour axes, so exposure and colour cast can
be corrected without dragging one into the other. Skin pixels only because
including hair, clothing or background pulls the statistics toward whatever
happened to be in frame.

Direction: the FACE is corrected toward the BODY, never the other way round.
The body is the larger, continuous region a viewer reads as the person's skin
tone; correcting it to match a small face patch would recolour the whole figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchStats:
    """What the correction did, so a caller can log or reject it."""
    face_mean: tuple[float, float, float]
    body_mean: tuple[float, float, float]
    shift: tuple[float, float, float]
    face_pixels: int
    body_pixels: int
    applied: bool
    reason: str = ""


# Below this there are not enough pixels for a mean to mean anything.
MIN_SKIN_PIXELS = 200

# A correction larger than this is not a lighting difference, it is a mistake —
# a mask that caught hair or background. Applying it would recolour the face
# wildly, so it is skipped and reported rather than trusted.
#
# UNITS: cv2's 8-bit LAB is NOT CIELAB's native range. L is scaled from 0-100
# to 0-255 (x2.55) and a/b are offset by +128 into 0-255. These limits are in
# cv2's units. Writing them as if L were 0-100 made the ceiling ~17 L*, which
# refused ordinary studio-vs-phone lighting differences as "not skin".
MAX_L_SHIFT = 90.0     # ~35 L* — a large but real difference in exposure
MAX_AB_SHIFT = 20.0    # a/b are unscaled, so this is ~20 CIELAB units


def _stats(lab: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    sel = lab[mask > 0]
    return sel.mean(axis=0), sel.std(axis=0), sel.shape[0]


def match_face_to_body(
    image_bgr: np.ndarray,
    face_mask: np.ndarray,
    body_skin_mask: np.ndarray,
    strength: float = 1.0,
) -> tuple[np.ndarray, MatchStats]:
    """Recolour the face region toward the body's skin statistics.

    `body_skin_mask` should be the neck and hands — continuous skin that
    belongs to the base body and was NOT replaced by the swap. Using the whole
    figure would include clothing.

    `strength` in [0, 1] scales the correction. 1.0 matches the means exactly;
    lower leaves some of the original face tone, which is safer when the two
    photographs were lit very differently and an exact match looks flat.

    Returns the corrected image and a MatchStats saying whether it was applied.
    Never raises for a small or empty mask — it declines, and says so.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    face_n = int((face_mask > 0).sum())
    body_n = int((body_skin_mask > 0).sum())
    if face_n < MIN_SKIN_PIXELS or body_n < MIN_SKIN_PIXELS:
        return image_bgr, MatchStats(
            (0, 0, 0), (0, 0, 0), (0, 0, 0), face_n, body_n, False,
            f"not enough skin pixels (face={face_n}, body={body_n}, "
            f"need {MIN_SKIN_PIXELS})",
        )

    f_mean, f_std, _ = _stats(lab, face_mask)
    b_mean, b_std, _ = _stats(lab, body_mask_or(body_skin_mask))
    shift = b_mean - f_mean

    if abs(shift[0]) > MAX_L_SHIFT or max(abs(shift[1]), abs(shift[2])) > MAX_AB_SHIFT:
        return image_bgr, MatchStats(
            tuple(f_mean), tuple(b_mean), tuple(shift), face_n, body_n, False,
            f"shift too large (L={shift[0]:.1f}, a={shift[1]:.1f}, b={shift[2]:.1f}) "
            f"— the masks probably caught something that is not skin",
        )

    strength = float(np.clip(strength, 0.0, 1.0))
    # Guard the std ratio: a nearly uniform face patch gives a tiny std, and
    # dividing by it amplifies noise into blotches.
    scale = np.where(f_std > 1.0, b_std / np.maximum(f_std, 1e-6), 1.0)
    scale = np.clip(scale, 0.6, 1.6)

    out = lab.copy()
    sel = face_mask > 0
    corrected = (lab[sel] - f_mean) * scale + f_mean + shift
    out[sel] = lab[sel] * (1.0 - strength) + corrected * strength

    out[:, :, 0] = np.clip(out[:, :, 0], 0, 255)
    out[:, :, 1:] = np.clip(out[:, :, 1:], 0, 255)
    converted = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Copy back only inside the face mask. Returning `converted` wholesale
    # would perturb every pixel in the image by a few levels — BGR->LAB->BGR
    # is lossy — so the body and background would drift even though nothing
    # asked them to. Same rule as the try-on paste-back: pixels nobody meant
    # to change must come through bit-identical.
    result = image_bgr.copy()
    result[sel] = converted[sel]

    return result, MatchStats(
        tuple(f_mean), tuple(b_mean), tuple(shift), face_n, body_n, True,
    )


def body_mask_or(mask: np.ndarray) -> np.ndarray:
    """Identity helper kept for readability at the call site above."""
    return mask


def skin_mask_from_parse(parse: np.ndarray, atr: dict) -> np.ndarray:
    """Neck and hands from an ATR parse — the body skin that stays put.

    ATR has no explicit neck label. Arms and legs are the reliable exposed
    skin; the neck falls between face and upper_clothes and is picked up by
    dilating the face region downward at the call site if wanted.
    """
    mask = np.zeros(parse.shape[:2], np.uint8)
    for label in ("left_arm", "right_arm", "left_leg", "right_leg"):
        mask[parse == atr[label]] = 255
    return mask


def face_region_mask(shape: tuple[int, int], bbox) -> np.ndarray:
    """Soft-edged ellipse over a face bbox, for when no parse is available."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = (int(v) for v in bbox[:4])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx, ry = max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return mask


def optional_match(
    image_bgr: np.ndarray,
    face_mask: Optional[np.ndarray],
    body_skin_mask: Optional[np.ndarray],
    strength: float = 0.85,
) -> tuple[np.ndarray, MatchStats]:
    """Match when both masks exist; otherwise return the image and say why.

    Declining is not the same as failing: a swap without tone matching is still
    a usable avatar, just a less well blended one. The caller logs the reason
    rather than the whole operation erroring.
    """
    if face_mask is None or body_skin_mask is None:
        return image_bgr, MatchStats(
            (0, 0, 0), (0, 0, 0), (0, 0, 0), 0, 0, False,
            "no parse available for skin masks",
        )
    return match_face_to_body(image_bgr, face_mask, body_skin_mask, strength)
