"""
Anatomy validation for anything that becomes an avatar.

Why this module exists
----------------------
Two batches of base bodies shipped with a second torso stacked below the first,
and both times the check in front of them reported success. The check asked
"is a face detectable?", and a two-torso figure has exactly one detectable
face. The lesson is not "add more checks" — it is that the check must look at
the thing that was wrong: the body.

The second failure was worse than the first. The check was written to degrade
politely when its model was missing: it printed a warning, fell back to face
detection alone, and eighteen broken templates were written by a run that
ended with "all templates usable". So:

    THIS VALIDATOR FAILS CLOSED.

If the pose model cannot be loaded, every call raises ValidatorUnavailable.
There is no flag in this module to skip it and no path that returns "ok"
without having actually looked. A caller that wants to bypass validation has
to not call it, visibly, in code review.

Backends are injected
---------------------
The real pose and segmentation models (RTMPose / MediaPipe / YOLOv8-pose, and
a human parser) only exist on the GPU box. They are passed in rather than
imported here, which keeps the anatomy rules — the part that was actually
wrong — unit-testable anywhere, including the fail-closed behaviour itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np


class ValidatorUnavailable(RuntimeError):
    """The models this validator needs are not loadable.

    Raised — never swallowed, never downgraded to a warning. See the module
    docstring for why this is an exception rather than a fallback.
    """


# ── Keypoint schema ──────────────────────────────────────────────────────────
# A single internal name set, so MediaPipe (33 landmarks), COCO-17 (RTMPose,
# YOLOv8-pose) and anything else are adapted at the backend boundary rather
# than leaking two conventions into the rules below.
REQUIRED_KEYPOINTS = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

# Below this a keypoint is treated as absent. Deliberately not low: a body
# whose ankles are only 30% confident is a body whose feet are probably not
# in the frame, which is exactly the "cropped figure" case.
MIN_KEYPOINT_CONFIDENCE = 0.60


@dataclass(frozen=True)
class Keypoint:
    x: float          # pixels
    y: float          # pixels, growing downward
    confidence: float


@dataclass(frozen=True)
class Person:
    """One detected person's keypoints, keyed by the names above."""
    keypoints: dict[str, Keypoint]

    def get(self, name: str) -> Optional[Keypoint]:
        kp = self.keypoints.get(name)
        if kp is None or kp.confidence < MIN_KEYPOINT_CONFIDENCE:
            return None
        return kp


class PoseBackend(Protocol):
    """Returns one Person per human found. Must return [] for none."""
    def detect(self, image_bgr: np.ndarray) -> Sequence[Person]: ...


class SegmentationBackend(Protocol):
    """Returns a boolean person mask, or None when segmentation is unavailable
    for this image (not for a missing model — that is a load-time failure)."""
    def person_mask(self, image_bgr: np.ndarray) -> Optional[np.ndarray]: ...


# ── Anthropometric bands ─────────────────────────────────────────────────────
# Segment proportions from Drillis & Contini's body-segment tables, the
# standard reference used in biomechanics. As a fraction of standing height:
#
#     shoulder (acromion)  0.818      biacromial width  0.259
#     hip (trochanter)     0.530      bi-iliac width    0.191
#     knee                 0.285
#     ankle                0.039
#     nose                ~0.936
#
# The measurable span in an image is nose-to-ankle, so the ratios below are
# expressed against that span (0.936 - 0.039 = 0.897 of height) rather than
# against a head-top nobody can locate reliably.
_SPAN_OF_HEIGHT = 0.897

def _band(nominal_of_height: float, tolerance: float) -> tuple[float, float]:
    nominal = nominal_of_height / _SPAN_OF_HEIGHT
    return nominal * (1 - tolerance), nominal * (1 + tolerance)

# Tolerances are wide. The job is to reject a figure that is not a human
# shape, not to grade posture: clothing, camera perspective and pose-model
# noise all move these by more than a little.
SEGMENT_BANDS = {
    "shoulder_to_hip": _band(0.818 - 0.530, 0.45),   # torso length
    "hip_to_knee":     _band(0.530 - 0.285, 0.45),   # thigh
    "knee_to_ankle":   _band(0.285 - 0.039, 0.45),   # shank
    "shoulder_width":  _band(0.259, 0.55),           # widest natural variation
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationResult:
    """Per-check outcomes.

    Deliberately not a bare bool. Reporting "18 templates passed" is what let
    two rounds of broken assets through; a caller should be able to say which
    check failed on which image.
    """
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def reason(self) -> str:
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failures)

    def report(self) -> str:
        return "\n".join(
            f"  [{'PASS' if c.name and c.passed else 'FAIL'}] {c.name}"
            + (f" — {c.detail}" if c.detail else "")
            for c in self.checks
        )


class AvatarValidator:
    """Anatomy gate for every image that becomes, or is built from, an avatar."""

    def __init__(
        self,
        pose: Optional[PoseBackend],
        segmentation: Optional[SegmentationBackend] = None,
    ):
        # Constructing with pose=None is allowed so a caller can hold a
        # validator whose backend failed to load; every validate() then raises.
        # This is the shape that makes "models missing" impossible to ignore.
        self._pose = pose
        self._seg = segmentation

    @property
    def available(self) -> bool:
        return self._pose is not None

    def validate(self, image_bgr: np.ndarray) -> ValidationResult:
        """Raises ValidatorUnavailable if the pose model is not loaded."""
        if self._pose is None:
            raise ValidatorUnavailable(
                "Pose model unavailable — refusing to validate. A figure with "
                "two torsos passes a face-only check, which is how broken base "
                "bodies shipped twice. Install the pose backend; there is no "
                "skip flag."
            )

        result = ValidationResult()
        people = list(self._pose.detect(image_bgr))

        # 1. Exactly one person.
        if len(people) != 1:
            result.checks.append(Check(
                "single_person", False,
                f"{len(people)} people detected, need exactly 1",
            ))
            # Everything downstream reasons about "the" person; without one,
            # further checks would be meaningless rather than merely failing.
            return result
        result.checks.append(Check("single_person", True, "1 person"))

        person = people[0]

        # 2. Every required keypoint present and confident.
        missing = [n for n in REQUIRED_KEYPOINTS if person.get(n) is None]
        result.checks.append(Check(
            "complete_skeleton", not missing,
            "all keypoints present" if not missing
            else f"missing/low-confidence: {', '.join(missing)}",
        ))
        if missing:
            return result

        kp = {n: person.get(n) for n in REQUIRED_KEYPOINTS}
        # "neck" is not a landmark in either convention; it is the midpoint of
        # the shoulders, which is what the ordering check below actually needs.
        y = {
            "nose": kp["nose"].y,
            "neck": (kp["left_shoulder"].y + kp["right_shoulder"].y) / 2,
            "hips": (kp["left_hip"].y + kp["right_hip"].y) / 2,
            "knees": (kp["left_knee"].y + kp["right_knee"].y) / 2,
            "ankles": (kp["left_ankle"].y + kp["right_ankle"].y) / 2,
        }

        # 3. Vertical ordering. This is the check a stacked double-figure fails.
        order = ["nose", "neck", "hips", "knees", "ankles"]
        bad_pair = next(
            ((a, b) for a, b in zip(order, order[1:]) if not y[a] < y[b]), None
        )
        result.checks.append(Check(
            "vertical_order", bad_pair is None,
            "nose < shoulders < hips < knees < ankles" if bad_pair is None
            else f"{bad_pair[0]} (y={y[bad_pair[0]]:.0f}) not above "
                 f"{bad_pair[1]} (y={y[bad_pair[1]]:.0f})",
        ))
        if bad_pair is not None:
            return result

        # 4. Segment proportions against the nose-to-ankle span.
        span = y["ankles"] - y["nose"]
        if span <= 0:
            result.checks.append(Check("proportions", False, "zero-height figure"))
            return result

        measured = {
            "shoulder_to_hip": (y["hips"] - y["neck"]) / span,
            "hip_to_knee":     (y["knees"] - y["hips"]) / span,
            "knee_to_ankle":   (y["ankles"] - y["knees"]) / span,
            "shoulder_width":  abs(kp["left_shoulder"].x - kp["right_shoulder"].x) / span,
        }
        out_of_band = {
            name: (value, SEGMENT_BANDS[name])
            for name, value in measured.items()
            if not (SEGMENT_BANDS[name][0] <= value <= SEGMENT_BANDS[name][1])
        }
        result.checks.append(Check(
            "proportions", not out_of_band,
            "within human range" if not out_of_band
            else "; ".join(
                f"{n}={v:.3f} outside {lo:.3f}-{hi:.3f}"
                for n, (v, (lo, hi)) in out_of_band.items()
            ),
        ))
        if out_of_band:
            return result

        # 5. One connected person mask. Catches a second figure that the pose
        #    model missed but the segmenter sees.
        if self._seg is not None:
            mask = self._seg.person_mask(image_bgr)
            if mask is None:
                result.checks.append(Check(
                    "single_blob", False, "segmentation returned no mask"))
                return result
            blobs = _count_significant_blobs(mask)
            result.checks.append(Check(
                "single_blob", blobs == 1,
                "one connected figure" if blobs == 1
                else f"{blobs} separate figures in the person mask",
            ))

        return result


def _count_significant_blobs(mask: np.ndarray, min_fraction: float = 0.02) -> int:
    """Connected components in a person mask, ignoring specks.

    min_fraction discards antialiasing crumbs and stray pixels; a real second
    figure is nowhere near 2% of the frame.
    """
    import cv2

    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total = binary.size
    significant = 0
    for i in range(1, count):          # 0 is background
        if stats[i, cv2.CC_STAT_AREA] / total >= min_fraction:
            significant += 1
    return significant
