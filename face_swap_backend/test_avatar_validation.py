"""
Tests for the anatomy gate.

Runs on stdlib unittest so it executes anywhere, including a machine with no
pose model installed — which is the point: the most important test here is
that a missing model makes the call FAIL rather than pass.

    python -m unittest test_avatar_validation -v
    pytest test_avatar_validation.py          # also works
"""
from __future__ import annotations

import unittest

import numpy as np

from avatar_validation import (
    AvatarValidator,
    Keypoint,
    Person,
    ValidatorUnavailable,
    MIN_KEYPOINT_CONFIDENCE,
    _count_significant_blobs,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────
def human(height_px: float = 1000.0, cx: float = 500.0, conf: float = 0.99) -> Person:
    """A skeleton with textbook Drillis & Contini proportions.

    Fractions of standing height: nose .936, shoulders .818, hips .530,
    knees .285, ankles .039, biacromial width .259. Anything derived from
    this should sit dead centre of every band.
    """
    def y_at(frac: float) -> float:
        return height_px * (1.0 - frac)

    half_shoulder = height_px * 0.259 / 2
    half_hip = height_px * 0.191 / 2
    return Person({
        "nose":           Keypoint(cx, y_at(0.936), conf),
        "left_shoulder":  Keypoint(cx - half_shoulder, y_at(0.818), conf),
        "right_shoulder": Keypoint(cx + half_shoulder, y_at(0.818), conf),
        "left_hip":       Keypoint(cx - half_hip, y_at(0.530), conf),
        "right_hip":      Keypoint(cx + half_hip, y_at(0.530), conf),
        "left_knee":      Keypoint(cx - half_hip, y_at(0.285), conf),
        "right_knee":     Keypoint(cx + half_hip, y_at(0.285), conf),
        "left_ankle":     Keypoint(cx - half_hip, y_at(0.039), conf),
        "right_ankle":    Keypoint(cx + half_hip, y_at(0.039), conf),
    })


class FakePose:
    def __init__(self, people): self._people = people
    def detect(self, image_bgr): return self._people


class FakeSeg:
    def __init__(self, mask): self._mask = mask
    def person_mask(self, image_bgr): return self._mask


BLANK = np.zeros((64, 64, 3), np.uint8)


def failed_check_names(result):
    return [c.name for c in result.failures]


# ── The test the spec asks for by name ───────────────────────────────────────
class TestFailsClosed(unittest.TestCase):
    """A missing model must break the call, not quietly weaken it.

    This is the regression for the actual production incident: the previous
    check degraded to face-only when mediapipe was absent, and 18 two-torso
    templates were written by a run that reported success.
    """

    def test_missing_pose_model_raises(self):
        validator = AvatarValidator(pose=None)
        with self.assertRaises(ValidatorUnavailable):
            validator.validate(BLANK)

    def test_missing_pose_model_is_not_merely_reported(self):
        # The failure must be an exception, not a ValidationResult saying
        # "ok: False" that a caller might log and move past.
        validator = AvatarValidator(pose=None)
        try:
            validator.validate(BLANK)
        except ValidatorUnavailable as e:
            self.assertIn("refusing", str(e).lower())
        else:
            self.fail("validate() returned instead of raising")

    def test_availability_is_visible_without_calling(self):
        self.assertFalse(AvatarValidator(pose=None).available)
        self.assertTrue(AvatarValidator(pose=FakePose([human()])).available)

    def test_no_skip_flag_exists_in_the_module(self):
        # Guards against someone reintroducing the escape hatch that caused
        # the incident. If a bypass is ever wanted it must be at the call
        # site, visibly, not hidden in here.
        import avatar_validation, inspect
        source = inspect.getsource(avatar_validation)
        for hatch in ("SKIP_VALIDATION", "skip_validation", "allow_unvalidated"):
            self.assertNotIn(hatch, source)


# ── Two torsos: the failure that shipped twice ───────────────────────────────
class TestTwoTorsoRejection(unittest.TestCase):
    """The duplicated-figure case, from three angles.

    NOTE: these are synthesised, not the captured production images — those
    live on the GPU box and are not reachable from here. The blob test below
    is real pixels through real cv2; the skeleton tests drive the rules with
    the keypoint patterns a doubled figure produces.
    """

    def test_two_detected_people_rejected(self):
        # What a segmenter/pose model usually reports for a stacked figure.
        validator = AvatarValidator(pose=FakePose([human(), human(cx=520)]))
        result = validator.validate(BLANK)
        self.assertFalse(result.ok)
        self.assertIn("single_person", failed_check_names(result))
        self.assertIn("2 people", result.reason())

    def test_zero_people_rejected(self):
        result = AvatarValidator(pose=FakePose([])).validate(BLANK)
        self.assertFalse(result.ok)
        self.assertIn("single_person", failed_check_names(result))

    def test_stacked_skeleton_breaks_vertical_order(self):
        # A skeleton fitted across BOTH torsos: the second torso's hips end up
        # above the first torso's knees, inverting the ordering.
        p = human()
        kps = dict(p.keypoints)
        kps["left_hip"] = Keypoint(kps["left_hip"].x, 100.0, 0.99)
        kps["right_hip"] = Keypoint(kps["right_hip"].x, 100.0, 0.99)
        result = AvatarValidator(pose=FakePose([Person(kps)])).validate(BLANK)
        self.assertFalse(result.ok)
        self.assertIn("vertical_order", failed_check_names(result))

    def test_two_blobs_in_person_mask_rejected(self):
        # Real image data through the real connected-components code: two
        # separated figures in the person mask.
        mask = np.zeros((400, 200), np.uint8)
        mask[20:170, 60:140] = 255     # upper figure
        mask[220:380, 60:140] = 255    # lower figure, not touching
        validator = AvatarValidator(pose=FakePose([human()]), segmentation=FakeSeg(mask))
        result = validator.validate(BLANK)
        self.assertFalse(result.ok)
        self.assertIn("single_blob", failed_check_names(result))
        self.assertIn("2 separate figures", result.reason())

    def test_blob_counter_on_real_masks(self):
        one = np.zeros((400, 200), np.uint8)
        one[20:380, 60:140] = 255
        self.assertEqual(_count_significant_blobs(one), 1)

        two = np.zeros((400, 200), np.uint8)
        two[20:170, 60:140] = 255
        two[220:380, 60:140] = 255
        self.assertEqual(_count_significant_blobs(two), 2)

        # A speck must not read as a second person.
        speck = one.copy()
        speck[0:3, 0:3] = 255
        self.assertEqual(_count_significant_blobs(speck), 1)


# ── The rules themselves ─────────────────────────────────────────────────────
class TestSkeletonCompleteness(unittest.TestCase):
    def test_textbook_human_passes_every_check(self):
        mask = np.zeros((400, 200), np.uint8)
        mask[20:380, 60:140] = 255
        result = AvatarValidator(
            pose=FakePose([human()]), segmentation=FakeSeg(mask)
        ).validate(BLANK)
        self.assertTrue(result.ok, result.reason())
        self.assertEqual(
            [c.name for c in result.checks],
            ["single_person", "complete_skeleton", "vertical_order",
             "proportions", "single_blob"],
        )

    def test_missing_ankles_rejected_and_named(self):
        p = human()
        kps = {k: v for k, v in p.keypoints.items() if k != "left_ankle"}
        result = AvatarValidator(pose=FakePose([Person(kps)])).validate(BLANK)
        self.assertFalse(result.ok)
        self.assertIn("complete_skeleton", failed_check_names(result))
        self.assertIn("left_ankle", result.reason())

    def test_low_confidence_counts_as_missing(self):
        p = human()
        kps = dict(p.keypoints)
        weak = MIN_KEYPOINT_CONFIDENCE - 0.01
        kps["right_knee"] = Keypoint(kps["right_knee"].x, kps["right_knee"].y, weak)
        result = AvatarValidator(pose=FakePose([Person(kps)])).validate(BLANK)
        self.assertFalse(result.ok)
        self.assertIn("right_knee", result.reason())


class TestProportions(unittest.TestCase):
    def test_absurd_torso_rejected(self):
        # Hips pushed almost to the knees: torso far too long for the span.
        p = human()
        kps = dict(p.keypoints)
        for side in ("left_hip", "right_hip"):
            kps[side] = Keypoint(kps[side].x, 700.0, 0.99)
        result = AvatarValidator(pose=FakePose([Person(kps)])).validate(BLANK)
        self.assertFalse(result.ok)
        self.assertIn("proportions", failed_check_names(result))

    def test_impossibly_wide_shoulders_rejected(self):
        p = human()
        kps = dict(p.keypoints)
        kps["left_shoulder"] = Keypoint(0.0, kps["left_shoulder"].y, 0.99)
        kps["right_shoulder"] = Keypoint(1400.0, kps["right_shoulder"].y, 0.99)
        result = AvatarValidator(pose=FakePose([Person(kps)])).validate(BLANK)
        self.assertFalse(result.ok)
        self.assertIn("shoulder_width", result.reason())

    def test_real_human_variation_still_passes(self):
        # Bands must not be so tight that ordinary people fail. A range of
        # heights and a shoulder width 25% off nominal both have to pass.
        for h in (600.0, 1000.0, 1800.0):
            with self.subTest(height=h):
                r = AvatarValidator(pose=FakePose([human(height_px=h)])).validate(BLANK)
                self.assertTrue(r.ok, f"h={h}: {r.reason()}")

        p = human()
        kps = dict(p.keypoints)
        cx, nominal_half = 500.0, 1000 * 0.259 / 2
        kps["left_shoulder"] = Keypoint(cx - nominal_half * 1.25, kps["left_shoulder"].y, 0.99)
        kps["right_shoulder"] = Keypoint(cx + nominal_half * 1.25, kps["right_shoulder"].y, 0.99)
        r = AvatarValidator(pose=FakePose([Person(kps)])).validate(BLANK)
        self.assertTrue(r.ok, r.reason())


class TestReporting(unittest.TestCase):
    def test_result_reports_per_check_not_a_bare_bool(self):
        # Aggregate reporting is what let "all 18 templates usable" be printed
        # over 18 broken files.
        result = AvatarValidator(pose=FakePose([])).validate(BLANK)
        self.assertTrue(hasattr(result, "checks"))
        self.assertTrue(result.report().strip().startswith("["))
        self.assertEqual(len(result.failures), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
