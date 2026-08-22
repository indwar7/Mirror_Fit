"""
Body measurements → which pre-rendered body template to use.

Why templates instead of generating a body per user
---------------------------------------------------
Two hard constraints decide this:

  1. face_swap_backend cannot run Stable Diffusion at all. Its env is Python
     3.14, which has no PyTorch CUDA wheels (this is the documented reason
     instantid_backend exists as a separate 3.11 service). torch/diffusers are
     not in its requirements and cannot be.
  2. Even where SD does run, it does not obey numbers. "waist 82 cm" is not a
     thing you can prompt for — you can only describe a build in words, and
     words land you in a bucket anyway.

So bucketing loses almost nothing over per-user generation, and buys instant
response, a body library you can actually look at and approve, and selection
logic that is plain testable code rather than a GPU round-trip.

The axes
--------
Two axes, because two are what the measurements actually support:

  size   — overall girth, from chest/bust circumference
  taper  — chest-to-waist ratio, i.e. how much the torso narrows

Shoulder is collected too, but it is a *width* in cm while chest and waist are
*circumferences*, so it is not directly comparable to them. It is used only to
nudge the taper axis when shoulders are unusually broad or narrow for the
chest — see `_shoulder_nudge`.

Thresholds
----------
Size cutoffs follow common Indian ready-to-wear chest/bust sizing (S/M/L/XL)
rather than being invented: menswear S≈91cm, M≈97-102, L≈107, XL≈112;
womenswear S≈81-86, M≈91, L≈97, XL≈102.

Taper cutoffs use chest-to-waist ratio, the standard drop measurement tailors
work from. They are deliberately wide bands — the point is to pick one of three
renders, not to grade a physique.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ── Axes ─────────────────────────────────────────────────────────────────────
SIZES = ("slim", "average", "broad")
TAPERS = ("tapered", "regular", "straight")

# Chest/bust circumference in cm → size band. Upper bound of each band.
_SIZE_CUTOFFS_CM = {
    "male": (94.0, 106.0),    # <94 slim | 94-106 average | >106 broad
    "female": (86.0, 97.0),   # <86 slim | 86-97  average | >97  broad
}

# Chest-to-waist ratio → taper band. Lower bound of each band.
#   >= 1.22  strong narrowing at the waist  -> "tapered"
#   >= 1.08  ordinary drop                  -> "regular"
#   <  1.08  little to no narrowing         -> "straight"
_TAPER_TAPERED = 1.22
_TAPER_REGULAR = 1.08

# Shoulder width as a fraction of chest/bust circumference, as (narrow, broad).
#
# These MUST be per-gender. Typical shoulder widths are ~45-48cm for men over a
# ~95-105cm chest (≈0.46), but ~36-40cm for women over an ~85-95cm bust (≈0.42).
# A single shared threshold read every ordinary female frame as narrow-
# shouldered and cancelled out a genuine taper — an hourglass build with a
# chest-to-waist ratio of 1.31 was being filed as "regular".
_SHOULDER_RATIOS = {
    "male": (0.43, 0.49),
    "female": (0.39, 0.45),
}

DEFAULT_GENDER = "male"


@dataclass(frozen=True)
class Measurements:
    """What the user types in. Circumferences in cm, shoulder is a width."""

    chest_cm: float
    waist_cm: float
    shoulder_cm: Optional[float] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "chest_cm": self.chest_cm,
            "waist_cm": self.waist_cm,
            "shoulder_cm": self.shoulder_cm,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
        }


class MeasurementError(ValueError):
    """A measurement is missing or outside any plausible human range."""


# Generous bounds — the job here is to reject typos and unit mix-ups (inches
# entered as cm, a waist of 3), not to police body size.
_PLAUSIBLE = {
    "chest_cm": (50.0, 200.0),
    "waist_cm": (40.0, 200.0),
    "shoulder_cm": (25.0, 70.0),
    "height_cm": (100.0, 230.0),
    "weight_kg": (20.0, 300.0),
}


def parse_measurements(raw: dict) -> Measurements:
    """Validate and coerce a measurements dict. Raises MeasurementError."""
    def num(key: str, required: bool) -> Optional[float]:
        value = raw.get(key)
        if value is None or value == "":
            if required:
                raise MeasurementError(f"{key} is required")
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            raise MeasurementError(f"{key} must be a number")
        lo, hi = _PLAUSIBLE[key]
        if not (lo <= out <= hi):
            # Naming the range makes the common failure — inches typed into a
            # cm field — obvious from the error alone.
            raise MeasurementError(
                f"{key}={out} is outside the plausible range {lo}-{hi} cm/kg. "
                f"Measurements are in centimetres, not inches."
            )
        return out

    chest = num("chest_cm", True)
    waist = num("waist_cm", True)
    assert chest is not None and waist is not None  # num() raises otherwise

    if waist > chest * 1.6:
        raise MeasurementError(
            "waist_cm is implausibly large relative to chest_cm — check the two "
            "are not swapped."
        )

    return Measurements(
        chest_cm=chest,
        waist_cm=waist,
        shoulder_cm=num("shoulder_cm", False),
        height_cm=num("height_cm", False),
        weight_kg=num("weight_kg", False),
    )


def _size_band(chest_cm: float, gender: str) -> str:
    small, large = _SIZE_CUTOFFS_CM.get(gender, _SIZE_CUTOFFS_CM[DEFAULT_GENDER])
    if chest_cm < small:
        return "slim"
    if chest_cm <= large:
        return "average"
    return "broad"


def _shoulder_nudge(m: Measurements, gender: str) -> int:
    """+1 toward tapered, -1 toward straight, 0 when shoulders are unremarkable.

    Shoulder width is not comparable to a circumference, so it is used only as
    a tie-breaker on the silhouette rather than as a measurement in its own
    right.
    """
    if not m.shoulder_cm or m.chest_cm <= 0:
        return 0
    narrow, broad = _SHOULDER_RATIOS.get(gender, _SHOULDER_RATIOS[DEFAULT_GENDER])
    ratio = m.shoulder_cm / m.chest_cm
    if ratio >= broad:
        return 1
    if ratio <= narrow:
        return -1
    return 0


def _taper_band(m: Measurements, gender: str = DEFAULT_GENDER) -> str:
    if m.waist_cm <= 0:
        return "regular"
    ratio = m.chest_cm / m.waist_cm

    if ratio >= _TAPER_TAPERED:
        index = 0  # tapered
    elif ratio >= _TAPER_REGULAR:
        index = 1  # regular
    else:
        index = 2  # straight

    # Broad shoulders read as more tapered, narrow as less. Clamped so the
    # nudge can only move one band, never invert the ratio's verdict.
    index -= _shoulder_nudge(m, gender)
    index = max(0, min(len(TAPERS) - 1, index))
    return TAPERS[index]


def normalise_gender(gender: Optional[str]) -> str:
    g = (gender or "").strip().lower()
    return g if g in _SIZE_CUTOFFS_CM else DEFAULT_GENDER


def body_id(m: Measurements, gender: Optional[str]) -> str:
    """The template id for this body: e.g. `body_m_average_tapered`."""
    g = normalise_gender(gender)
    return f"body_{g[0]}_{_size_band(m.chest_cm, g)}_{_taper_band(m, g)}"


def describe(m: Measurements, gender: Optional[str]) -> dict:
    """Selection plus the reasoning behind it, so the UI (and a bug report) can
    show why a given body was chosen."""
    g = normalise_gender(gender)
    return {
        "body_id": body_id(m, gender),
        "gender": g,
        "size": _size_band(m.chest_cm, g),
        "taper": _taper_band(m, g),
        "chest_to_waist": round(m.chest_cm / m.waist_cm, 3) if m.waist_cm else None,
        "shoulder_to_chest": (
            round(m.shoulder_cm / m.chest_cm, 3) if m.shoulder_cm and m.chest_cm else None
        ),
    }


def all_body_ids() -> list[str]:
    """Every template the library must contain — what generate_bodies.py renders."""
    return [
        f"body_{g[0]}_{size}_{taper}"
        for g in ("male", "female")
        for size in SIZES
        for taper in TAPERS
    ]
