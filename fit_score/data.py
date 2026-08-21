"""SYNTHETIC fit dataset.

    ⚠️  Generated, not observed. There is no measured garment/body corpus
        in this repository. Metrics describe this generative process only.
        Replace `load_fits()` with real measurements — garment spec sheets
        joined to fitted-customer measurements — before quoting accuracy.

The generative process starts from the physical situation rather than from
the labels, which is what keeps it honest:

  1. A body is drawn (chest, shoulder, torso, arm) with realistic
     correlations — broad shoulders come with a broad chest.
  2. A garment is drawn as a *size pick*, because that is the real
     mechanism: a shopper chooses S/M/L/XL, and the garment's measurements
     follow from that choice plus the brand's grading, not from the body.
  3. Ratios fall out of the two. Labels are then assigned from the ratios
     by the same tolerance rules the spec defines.

So the label is a consequence of geometry, not an independent draw. A
model that learns the geometry will do well; one that memorises noise will
not, because measurement noise and brand grading variance are injected on
top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spec import (
    CATEGORICAL,
    CLASSES,
    DERIVED_FEATURES,
    FEATURES,
    derive,
    GARMENT_TYPES,
    IDEAL_EASE,
    TAILORED_TOLERANCE,
    WEARABLE_TOLERANCE,
    deviations,
)

TARGET = "fit_class"

# Brand size grading: how each size scales the garment relative to the
# body it is cut for. Real grading steps are ~4-5 cm chest per size.
SIZE_STEP = {"XS": -0.12, "S": -0.06, "M": 0.0, "L": 0.07, "XL": 0.15, "XXL": 0.24}
SIZES = tuple(SIZE_STEP.keys())


def load_fits(n: int = 14_000, seed: int = 20260821) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Body ────────────────────────────────────────────────────────────
    # One latent frame-size drives every body measurement, so proportions
    # stay physically plausible instead of each column being independent.
    frame = rng.normal(0.0, 1.0, size=n)
    chest_cm = 96.0 + 9.5 * frame + rng.normal(0, 2.6, n)
    shoulder_cm = 44.0 + 3.1 * frame + rng.normal(0, 1.1, n)
    torso_cm = 68.0 + 3.4 * frame + rng.normal(0, 2.1, n)
    arm_cm = 60.0 + 2.7 * frame + rng.normal(0, 1.6, n)

    garment_type = rng.choice(GARMENT_TYPES, size=n, p=[0.30, 0.24, 0.16, 0.16, 0.14])

    # ── The shopper's size choice ───────────────────────────────────────
    # Most people pick correctly; a meaningful minority size up (comfort,
    # or a brand that runs small) or size down (vanity sizing, or wanting
    # a slim look). That mis-picking is the main source of bad fit.
    pick = rng.choice(
        np.arange(len(SIZES)),
        size=n,
        p=[0.06, 0.20, 0.32, 0.24, 0.13, 0.05],
    )
    # Nudge the pick toward the size that actually suits the body, so the
    # choice is correlated with the frame rather than random.
    ideal_pick = np.clip(np.round(2 + frame * 1.15).astype(int), 0, len(SIZES) - 1)
    follows_advice = rng.random(n) < 0.55
    pick = np.where(follows_advice, ideal_pick, pick)
    size_label = np.array(SIZES)[pick]
    grade = np.array([SIZE_STEP[s] for s in size_label])

    # ── Garment measurements ────────────────────────────────────────────
    # Cut for a reference body of the chosen size, plus the garment's own
    # ease profile, plus brand-to-brand grading variance.
    ease_chest = np.array([IDEAL_EASE[g].chest for g in garment_type])
    ease_shoulder = np.array([IDEAL_EASE[g].shoulder for g in garment_type])
    ease_length = np.array([IDEAL_EASE[g].length for g in garment_type])
    ease_sleeve = np.array([IDEAL_EASE[g].sleeve for g in garment_type])

    reference_chest = 96.0 + grade * 96.0
    reference_shoulder = 44.0 + grade * 44.0
    reference_torso = 68.0 + grade * 68.0
    reference_arm = 60.0 + grade * 60.0

    brand_var = rng.normal(0, 0.035, n)          # brand grading inconsistency
    cut_var = rng.normal(0, 0.030, n)            # per-garment cut variance

    g_chest = reference_chest * ease_chest * (1 + brand_var + rng.normal(0, 0.022, n))
    g_shoulder = reference_shoulder * ease_shoulder * (1 + brand_var + rng.normal(0, 0.020, n))
    g_length = reference_torso * ease_length * (1 + cut_var + rng.normal(0, 0.028, n))
    g_sleeve = reference_arm * ease_sleeve * (1 + cut_var + rng.normal(0, 0.030, n))

    # A minority of garments are badly graded — right in one dimension,
    # wrong in another. These are the POOR_FIT cases, and they are the
    # reason a single "size" label cannot express fit.
    inconsistent = rng.random(n) < 0.10
    g_shoulder = np.where(
        inconsistent, g_shoulder * (1 + rng.normal(0, 0.115, n)), g_shoulder
    )
    g_length = np.where(
        inconsistent, g_length * (1 + rng.normal(0, 0.130, n)), g_length
    )

    # ── Ratios: what the mirror can actually measure ────────────────────
    # Camera measurement noise — the try-on estimates these from pose
    # landmarks and a garment silhouette, not from a tape measure.
    def noisy(x, sd=0.014):
        return x * (1 + rng.normal(0, sd, n))

    chest_ratio = noisy(g_chest / chest_cm)
    shoulder_ratio = noisy(g_shoulder / shoulder_cm)
    length_ratio = noisy(g_length / torso_cm)
    sleeve_ratio = noisy(g_sleeve / arm_cm, 0.018)

    # Drape slack: unsupported fabric. Follows from chest ease — a roomier
    # garment hangs more — and is what a viewer perceives as "baggy".
    drape_slack = np.clip(
        (chest_ratio / ease_chest - 1.0) * 2.4 + rng.normal(0, 0.05, n), -0.35, 1.0
    )
    # Shoulder seam drop past the shoulder point, in cm.
    shoulder_drop_cm = (shoulder_ratio / ease_shoulder - 1.0) * shoulder_cm + rng.normal(0, 0.4, n)

    frame_df = pd.DataFrame({
        "chest_ratio": chest_ratio,
        "shoulder_ratio": shoulder_ratio,
        "length_ratio": length_ratio,
        "sleeve_ratio": sleeve_ratio,
        "drape_slack": drape_slack,
        "shoulder_drop_cm": shoulder_drop_cm,
        "garment_type": garment_type,
        "size_label": size_label,
        # Raw measurements kept as metadata, not model features. The model
        # reasons in ratios (scale-invariant); humans reason in centimetres,
        # so the UI needs both.
        "body_chest_cm": chest_cm.round(1),
        "body_shoulder_cm": shoulder_cm.round(1),
        "body_torso_cm": torso_cm.round(1),
        "body_arm_cm": arm_cm.round(1),
        "garment_chest_cm": g_chest.round(1),
        "garment_shoulder_cm": g_shoulder.round(1),
        "garment_length_cm": g_length.round(1),
        "garment_sleeve_cm": g_sleeve.round(1),
    })

    # Derived features, computed the same way the browser will compute
    # them at scoring time (see spec.derive).
    derived = [derive(frame_df.iloc[i]) for i in range(len(frame_df))]
    for key in DERIVED_FEATURES:
        frame_df[key] = [d[key] for d in derived]

    frame_df[TARGET] = [
        _classify(frame_df.iloc[i]) for i in range(len(frame_df))
    ]
    return frame_df


def _classify(row) -> str:
    """Assign a fit class from the deviations, by the spec's own rules.

    Deliberately rule-based: the label *is* the geometry. The learned model
    then has to recover these boundaries from noisy measurements, which is
    the realistic task — in production the labels come from returns
    outcomes and tailor assessments, which are noisier still.
    """
    d = deviations(row)
    core = [d["chest"], d["shoulder"]]
    every = [d["chest"], d["shoulder"], d["length"], d["sleeve"]]

    worst = max(abs(v) for v in every)
    spread = max(every) - min(every)

    # Measurements pulling in opposite directions: no size fixes this.
    if spread > 0.20 and worst > WEARABLE_TOLERANCE:
        return "POOR_FIT"
    if worst <= TAILORED_TOLERANCE:
        return "MADE_FOR_YOU"
    if worst <= WEARABLE_TOLERANCE:
        return "GOOD_FIT"
    # Consistently off in one direction — that is a sizing problem.
    if np.mean(core) > 0:
        return "OVERSIZED"
    if np.mean(core) < 0:
        return "UNDERSIZED"
    return "POOR_FIT"


def split_frame(frame: pd.DataFrame, seed: int, train=0.6, valid=0.2):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    a, b = int(len(frame) * train), int(len(frame) * (train + valid))
    return (
        frame.iloc[order[:a]].reset_index(drop=True),
        frame.iloc[order[a:b]].reset_index(drop=True),
        frame.iloc[order[b:]].reset_index(drop=True),
    )
