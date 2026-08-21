"""Fit assessment — domain definitions.

What the model actually measures
--------------------------------
Fit is geometry, not opinion. A garment has four measurements that decide
whether it fits: chest, shoulder, body length and sleeve length. A wearer
has the corresponding body measurements. Fit is the *ratio* between them.

Ratios rather than absolute sizes, for two reasons:
  1. Scale invariance. The try-on measures in pixels, and pixel scale
     changes every time the shopper moves closer to or further from the
     camera. A ratio does not.
  2. Sizing is relative anyway. A 42" chest garment is oversized on one
     body and undersized on another; only the relationship is meaningful.

The tolerance bands below are apparel "ease" allowances — the industry's
own numbers for how much room a garment leaves over the body. They are
not invented for this model:

    chest ease      slim     5-10 cm   -> ratio ~1.05-1.11
                    regular 10-16 cm   -> ratio ~1.11-1.18
                    relaxed 18-26 cm   -> ratio ~1.20-1.29
                    oversized  30 cm+  -> ratio  1.32+
                    too tight  <4 cm   -> ratio  <1.04

Ease differs by garment: a jacket is worn over other layers, so its
"correct" ease is larger than a t-shirt's. That is why IDEAL_EASE is keyed
by garment type rather than being one global constant.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Classes ─────────────────────────────────────────────────────────────
# Ordered from best to worst so the label index is itself meaningful.
CLASSES = (
    "MADE_FOR_YOU",   # every measurement inside the tailored band
    "GOOD_FIT",       # within normal ready-to-wear tolerance
    "OVERSIZED",      # consistently too large
    "UNDERSIZED",     # consistently too small
    "POOR_FIT",       # measurements disagree — fits nowhere
)

CLASS_COPY = {
    "MADE_FOR_YOU": (
        "Made for your body",
        "Every measurement sits in the tailored band. This is as close as "
        "ready-to-wear gets to made-to-measure.",
    ),
    "GOOD_FIT": (
        "Good fit",
        "Within normal tolerance across the board. Nothing a shopper would "
        "send back on fit alone.",
    ),
    "OVERSIZED": (
        "Oversized",
        "Consistently larger than this body. Fine if the shopper wants it "
        "loose, a return risk if they do not.",
    ),
    "UNDERSIZED": (
        "Undersized",
        "Consistently tighter than this body needs. The most common cause "
        "of a fit return.",
    ),
    "POOR_FIT": (
        "Poor fit",
        "The measurements disagree with each other — the right size in one "
        "place and wrong in another. No single size will fix this cut.",
    ),
}

# ── Ideal ease by garment, as garment/body ratio ────────────────────────
# (chest, shoulder, length, sleeve). Length and sleeve are ratios against
# the wearer's torso and arm rather than against a garment spec.
@dataclass(frozen=True)
class EaseProfile:
    chest: float
    shoulder: float
    length: float
    sleeve: float


IDEAL_EASE = {
    "tshirt":   EaseProfile(chest=1.14, shoulder=1.03, length=1.00, sleeve=1.00),
    "shirt":    EaseProfile(chest=1.16, shoulder=1.04, length=1.06, sleeve=1.00),
    "jacket":   EaseProfile(chest=1.24, shoulder=1.07, length=1.04, sleeve=1.02),
    "dress":    EaseProfile(chest=1.12, shoulder=1.02, length=1.00, sleeve=1.00),
    "trousers": EaseProfile(chest=1.10, shoulder=1.00, length=1.00, sleeve=1.00),
}

GARMENT_TYPES = tuple(IDEAL_EASE.keys())

# How far a ratio may drift from ideal before it stops being "tailored",
# and before it stops being wearable at all. Expressed as a multiplier on
# the ideal, so it scales with the garment's own ease profile.
TAILORED_TOLERANCE = 0.045   # +-4.5% of ideal  -> MADE_FOR_YOU
WEARABLE_TOLERANCE = 0.115   # +-11.5% of ideal -> still GOOD_FIT

RAW_FEATURES = [
    "chest_ratio",
    "shoulder_ratio",
    "length_ratio",
    "sleeve_ratio",
    "drape_slack",       # how much fabric hangs unsupported (0 = clinging)
    "shoulder_drop_cm",  # garment shoulder seam past the wearer's shoulder point
]

# Derived features — see derive() for why these are necessary rather than
# convenient.
DERIVED_FEATURES = [
    "dev_chest",
    "dev_shoulder",
    "dev_length",
    "dev_sleeve",
    "worst_abs_dev",
    "dev_spread",
    "core_dev_mean",
]

FEATURES = RAW_FEATURES + DERIVED_FEATURES
CATEGORICAL = ["garment_type"]


def deviations(row: dict) -> dict[str, float]:
    """Signed deviation of each measurement from its ideal, as a fraction.

    Positive means the garment is larger than ideal there. This is the
    quantity a human tailor reasons about, and it is what the score and
    the explanation are both built from.
    """
    ease = IDEAL_EASE.get(row.get("garment_type", "tshirt"), IDEAL_EASE["tshirt"])
    return {
        "chest": row["chest_ratio"] / ease.chest - 1.0,
        "shoulder": row["shoulder_ratio"] / ease.shoulder - 1.0,
        "length": row["length_ratio"] / ease.length - 1.0,
        "sleeve": row["sleeve_ratio"] / ease.sleeve - 1.0,
    }


def derive(row: dict) -> dict[str, float]:
    """Turn four raw ratios into the quantities that decide fit.

    This is not cosmetic feature engineering — it is what makes the problem
    linearly separable at all.

    A garment is "made for you" when *every* measurement is close to ideal.
    In raw-ratio space that is a bounded box around the origin, and a linear
    model cannot describe a bounded region: hyperplanes only ever cut space
    in half. Trained on the raw ratios, logistic regression scored 0.00
    recall on MADE_FOR_YOU — not badly, but never once.

    `worst_abs_dev` (how wrong is the worst measurement) and `dev_spread`
    (do the measurements disagree with each other) turn those boxes into
    half-spaces, which a linear model handles exactly. `core_dev_mean`
    carries the sign, separating "too big everywhere" from "too small
    everywhere".

    These are also the three questions a tailor asks, which is the reason
    they work rather than a coincidence.
    """
    d = deviations(row)
    every = [d["chest"], d["shoulder"], d["length"], d["sleeve"]]
    return {
        "dev_chest": d["chest"],
        "dev_shoulder": d["shoulder"],
        "dev_length": d["length"],
        "dev_sleeve": d["sleeve"],
        # How wrong is the worst measurement?
        "worst_abs_dev": max(abs(v) for v in every),
        # Do the measurements disagree with each other?
        "dev_spread": max(every) - min(every),
        # Which direction, on the two measurements that matter most?
        "core_dev_mean": (d["chest"] + d["shoulder"]) / 2.0,
    }


def fit_score(row: dict) -> float:
    """0-100. How close this garment is to being cut for this body.

    A weighted RMS of the deviations, mapped so that a perfectly tailored
    garment scores 100 and one at the edge of wearability scores ~55.

    Chest and shoulder carry the most weight because they are the hardest
    to alter and the most visible when wrong. Length is adjustable by a
    tailor, sleeve more so, hence the lower weights.
    """
    d = deviations(row)
    weights = {"chest": 0.38, "shoulder": 0.32, "length": 0.18, "sleeve": 0.12}
    penalty = sum(w * (d[k] ** 2) for k, w in weights.items()) ** 0.5
    # 0.115 deviation (edge of wearable) should land near 55.
    score = 100.0 * (1.0 - (penalty / WEARABLE_TOLERANCE) * 0.45)
    return float(max(0.0, min(100.0, score)))


def mismatch_from_score(score: float) -> float:
    """Convert a 0-100 fit score into the 0-1 `fit_mismatch_score` that the
    returns model consumes. This is the join between the two systems: what
    the mirror measured at purchase is what the returns desk sees later."""
    return float(max(0.0, min(1.0, (100.0 - score) / 100.0)))
