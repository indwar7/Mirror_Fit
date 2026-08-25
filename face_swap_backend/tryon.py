"""
Garment try-on for an avatar.

A purpose-built VTON model (CatVTON / IDM-VTON) conditioned on a human parse
and DensePose, not a general text-to-image prompt. The distinction matters: a
prompt-driven model redraws the whole frame and there is no guarantee the face
survives it. A VTON model plus an explicit paste-back means identity is not
merely *likely* to be preserved — it is arithmetically preserved, because the
pixels outside the garment mask are copied, not regenerated.

    THE TRY-ON MUST NEVER ALTER IDENTITY.

`composite()` enforces that, and its test asserts bit-identical equality
outside the mask rather than a perceptual similarity score.

Backends are injected
---------------------
The parser, DensePose and VTON models live on the GPU box. They are passed in,
so the masking and compositing rules — the parts with the identity guarantee in
them — are unit-testable on a machine with none of them installed. Missing
backends raise; nothing here degrades to "return the input unchanged", which
would look like a working try-on that silently did nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

CATEGORIES = ("upper", "lower", "dress")


class TryOnUnavailable(RuntimeError):
    """A required model is not loaded. Never swallowed, never downgraded."""


# ── Human parsing ────────────────────────────────────────────────────────────
# SCHP/ATR label ids. Only the ones the masks below need are named; the parser
# returns a full label map and the rest are left alone by construction.
ATR = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
    "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9,
    "right_shoe": 10, "face": 11, "left_leg": 12, "right_leg": 13,
    "left_arm": 14, "right_arm": 15, "bag": 16, "scarf": 17,
}

# What each category is allowed to replace. Arms are included for upper and
# dress because a sleeve legitimately changes how much arm is covered; hands
# are not, and neither is anything on the head.
_REPLACEABLE = {
    "upper": ("upper_clothes", "dress", "left_arm", "right_arm"),
    "lower": ("pants", "skirt"),
    "dress": ("upper_clothes", "dress", "skirt", "left_arm", "right_arm"),
}

# Never replaced, for any category. Face and hair are identity; shoes and bag
# are not what was asked for.
_PROTECTED = ("face", "hair", "hat", "sunglasses", "left_shoe", "right_shoe", "bag")


class HumanParser(Protocol):
    def parse(self, image_bgr: np.ndarray) -> np.ndarray:
        """HxW label map of ATR ids."""


class DensePoseBackend(Protocol):
    def densepose(self, image_bgr: np.ndarray) -> np.ndarray: ...


class VTONBackend(Protocol):
    def generate(
        self,
        person_bgr: np.ndarray,
        agnostic_mask: np.ndarray,
        densepose: np.ndarray,
        garment_bgr: np.ndarray,
        category: str,
    ) -> np.ndarray: ...


@dataclass
class AvatarMaps:
    """Parse + DensePose for one avatar. Stable across garments, so cached."""
    parse: np.ndarray
    densepose: np.ndarray


def cloth_agnostic_mask(parse: np.ndarray, category: str) -> np.ndarray:
    """Where the VTON model may paint, as a uint8 0/255 mask.

    Built by union of the replaceable labels then subtraction of the protected
    ones, in that order. The subtraction is not redundant: parsers routinely
    bleed `upper_clothes` a few pixels into the jaw and hairline, and those
    pixels are exactly the ones a viewer reads as the face changing.
    """
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}")

    mask = np.zeros(parse.shape[:2], np.uint8)
    for label in _REPLACEABLE[category]:
        mask[parse == ATR[label]] = 255
    for label in _PROTECTED:
        mask[parse == ATR[label]] = 0
    return mask


def composite(original_bgr: np.ndarray, generated_bgr: np.ndarray,
              mask: np.ndarray) -> np.ndarray:
    """Generated pixels inside the mask, original pixels everywhere else.

    A hard binary copy, not a blend. A feathered edge would leave the face and
    background *nearly* unchanged, and "nearly" is not a guarantee anybody can
    check. The seam is handled by the mask being drawn a little inside the
    garment boundary, not by softening it here.

    Raises on a shape mismatch rather than resizing: silently resampling the
    original would break the very property this function exists to provide.
    """
    if original_bgr.shape != generated_bgr.shape:
        raise ValueError(
            f"shape mismatch: original {original_bgr.shape} vs generated "
            f"{generated_bgr.shape} — refusing to resample the original"
        )
    if mask.shape[:2] != original_bgr.shape[:2]:
        raise ValueError(
            f"mask {mask.shape[:2]} does not match image {original_bgr.shape[:2]}"
        )

    inside = (mask > 127)[:, :, None]
    return np.where(inside, generated_bgr, original_bgr)


def identity_preserved(original_bgr: np.ndarray, result_bgr: np.ndarray,
                       mask: np.ndarray) -> bool:
    """True when every pixel outside the mask is bit-identical.

    Exposed so the endpoint can assert it on real output rather than trusting
    composite() to have been the last thing that touched the image.
    """
    outside = (mask <= 127)
    return bool(np.array_equal(original_bgr[outside], result_bgr[outside]))


class TryOnPipeline:
    """Parse → agnostic mask → VTON → paste back.

    `maps_cache` is keyed by avatar id: the parse and DensePose depend only on
    the avatar, so trying ten garments runs them once.
    """

    def __init__(
        self,
        parser: Optional[HumanParser],
        densepose: Optional[DensePoseBackend],
        vton: Optional[VTONBackend],
    ):
        self._parser = parser
        self._densepose = densepose
        self._vton = vton
        self._maps_cache: dict[str, AvatarMaps] = {}

    @property
    def available(self) -> bool:
        return all(b is not None for b in (self._parser, self._densepose, self._vton))

    def _require(self) -> None:
        missing = [
            name for name, backend in (
                ("human parser (SCHP/ATR)", self._parser),
                ("DensePose", self._densepose),
                ("VTON model (CatVTON/IDM-VTON)", self._vton),
            ) if backend is None
        ]
        if missing:
            raise TryOnUnavailable(
                "Try-on unavailable — not loaded: " + ", ".join(missing) +
                ". Refusing rather than returning the avatar unchanged, which "
                "would look like a try-on that had worked."
            )

    def maps_for(self, avatar_id: str, avatar_bgr: np.ndarray) -> AvatarMaps:
        self._require()
        cached = self._maps_cache.get(avatar_id)
        if cached is not None:
            return cached
        maps = AvatarMaps(
            parse=self._parser.parse(avatar_bgr),
            densepose=self._densepose.densepose(avatar_bgr),
        )
        self._maps_cache[avatar_id] = maps
        return maps

    def invalidate(self, avatar_id: str) -> None:
        """Drop cached maps — the avatar image changed underneath them."""
        self._maps_cache.pop(avatar_id, None)

    def tryon(self, avatar_id: str, avatar_bgr: np.ndarray,
              garment_bgr: np.ndarray, category: str) -> np.ndarray:
        self._require()
        maps = self.maps_for(avatar_id, avatar_bgr)
        mask = cloth_agnostic_mask(maps.parse, category)

        if not mask.any():
            raise ValueError(
                f"Nothing to replace for category '{category}' — the parser "
                f"found no {category} region on this avatar."
            )

        generated = self._vton.generate(
            person_bgr=avatar_bgr, agnostic_mask=mask,
            densepose=maps.densepose, garment_bgr=garment_bgr, category=category,
        )
        result = composite(avatar_bgr, generated, mask)

        # Belt and braces on the one property that must hold. Cheap next to a
        # diffusion pass, and it fails loudly if composite() is ever changed.
        if not identity_preserved(avatar_bgr, result, mask):
            raise RuntimeError(
                "try-on altered pixels outside the garment mask — refusing to "
                "return an image whose identity may have changed"
            )
        return result
