"""
Measurements → which curated base body to use.

This is a pure function. It does no I/O, loads no model, and never creates an
image. Given a gender and optional measurements it returns the id of one of the
eight photographs in assets/base_bodies/, and that is all it does.

Why it is only a lookup
-----------------------
The previous design generated a body per bin with a text-to-image model at
build time. Generation is stochastic: some renders came out with two torsos,
and because the gate in front of them only asked "is a face detectable?" — a
question a two-torso image answers "yes" — eighteen broken figures shipped
twice. Swapping SD 1.5 for SDXL lowered the rate; it did not make the output
something you could rely on without a human looking at every frame.

So the bodies are now a fixed, human-approved asset set, and this module's only
job is to choose between them. There is deliberately no fallback that
synthesises anything: if a bin has no photograph, the caller returns 501 naming
the missing id rather than inventing a stand-in.

The bins
--------
Four builds per gender, chosen because they are what the measurements can
actually separate and what apparel sizing already thinks in:

    slim      small girth
    average   the middle
    athletic  ordinary girth, pronounced chest-to-waist drop
    plus      large girth

Cutoffs follow common Indian ready-to-wear chest/bust bands rather than being
invented. Drop (chest ÷ waist) is the tailor's own measure of taper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

GENDERS = ("male", "female")
BUILDS = ("slim", "average", "athletic", "plus")
DEFAULT_GENDER = "male"
DEFAULT_BUILD = "average"

# Chest/bust circumference in cm. (slim_below, plus_above).
# Menswear S≈91, M≈97-102, L≈107, XL≈112; womenswear S≈81-86, M≈91, L≈97, XL≈102.
_GIRTH_CUTOFFS = {
    "male":   (94.0, 108.0),
    "female": (86.0, 100.0),
}

# Waist alone can also put someone in `plus` — a large waist with an ordinary
# chest is a shape the chest cutoff would otherwise miss entirely.
_WAIST_PLUS = {"male": 100.0, "female": 94.0}

# Chest ÷ waist at or above which the silhouette reads as tapered.
_ATHLETIC_DROP = {"male": 1.22, "female": 1.28}


class MeasurementError(ValueError):
    """A measurement is missing, non-numeric, or outside any plausible range."""


# Generous. The job is to catch typos and unit mix-ups — inches entered as cm,
# a waist of 3 — not to police body size.
_PLAUSIBLE = {
    "height_cm": (100.0, 230.0),
    "chest_cm":  (50.0, 200.0),
    "waist_cm":  (40.0, 200.0),
    "hips_cm":   (50.0, 200.0),
}


@dataclass(frozen=True)
class Measurements:
    """Circumferences in cm. Every field optional — see `bin_for`."""
    height_cm: Optional[float] = None
    chest_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hips_cm: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "height_cm": self.height_cm,
            "chest_cm": self.chest_cm,
            "waist_cm": self.waist_cm,
            "hips_cm": self.hips_cm,
        }

    @property
    def has_torso(self) -> bool:
        """True when chest AND waist are both present.

        Both or neither: one alone cannot produce a drop ratio, and using it
        for girth while silently ignoring the missing half would look like the
        measurement had been taken into account.
        """
        return self.chest_cm is not None and self.waist_cm is not None


def parse_measurements(raw: dict) -> Measurements:
    """Validate and coerce. Absent keys stay None; bad values raise."""
    def num(key: str) -> Optional[float]:
        value = raw.get(key)
        if value is None or value == "":
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            raise MeasurementError(f"{key} must be a number")
        lo, hi = _PLAUSIBLE[key]
        if not (lo <= out <= hi):
            # Naming the range makes the common failure — inches typed into a
            # cm field — obvious from the message alone.
            raise MeasurementError(
                f"{key}={out} is outside the plausible range {lo}-{hi}. "
                f"Measurements are in centimetres, not inches."
            )
        return out

    m = Measurements(
        height_cm=num("height_cm"), chest_cm=num("chest_cm"),
        waist_cm=num("waist_cm"), hips_cm=num("hips_cm"),
    )
    if m.chest_cm is not None and m.waist_cm is not None and m.waist_cm > m.chest_cm * 1.6:
        raise MeasurementError(
            "waist_cm is implausibly large relative to chest_cm — check the two "
            "are not swapped."
        )
    return m


def normalise_gender(gender: Optional[str]) -> str:
    g = (gender or "").strip().lower()
    return g if g in GENDERS else DEFAULT_GENDER


def bin_for(gender: Optional[str], m: Optional[Measurements]) -> str:
    """The build bin. Returns 'average' when there is nothing to go on.

    Order is load-bearing and is checked by the tests: plus is decided before
    athletic, so a large tapered frame reads as plus rather than athletic —
    the garment has to fit the girth first, and taper is the finer distinction.
    """
    g = normalise_gender(gender)
    if m is None or not m.has_torso:
        return DEFAULT_BUILD

    slim_below, plus_above = _GIRTH_CUTOFFS[g]
    chest, waist = m.chest_cm, m.waist_cm

    if chest > plus_above or waist > _WAIST_PLUS[g]:
        return "plus"
    if chest < slim_below and waist < _WAIST_PLUS[g]:
        return "slim"
    if waist > 0 and (chest / waist) >= _ATHLETIC_DROP[g]:
        return "athletic"
    return DEFAULT_BUILD


def base_body_id(gender: Optional[str], m: Optional[Measurements] = None) -> str:
    """e.g. `body_male_athletic`. Names an asset; does not guarantee it exists."""
    return f"body_{normalise_gender(gender)}_{bin_for(gender, m)}"


def describe(gender: Optional[str], m: Optional[Measurements] = None) -> dict:
    """The choice plus the reasoning, so a UI (or a bug report) can show why."""
    g = normalise_gender(gender)
    build = bin_for(gender, m)
    drop = None
    if m is not None and m.has_torso and m.waist_cm:
        drop = round(m.chest_cm / m.waist_cm, 3)
    return {
        "base_body_id": base_body_id(gender, m),
        "gender": g,
        "build": build,
        "chest_to_waist": drop,
        # True when nothing was supplied and the middle bin was assumed, so the
        # client can say "standard build" rather than implying it was measured.
        "from_defaults": m is None or not m.has_torso,
    }


def all_base_body_ids() -> list[str]:
    """Every id the asset set must contain — what the manifest is checked against."""
    return [f"body_{g}_{b}" for g in GENDERS for b in BUILDS]
