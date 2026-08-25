"""
Real pose / segmentation backends for AvatarValidator.

Kept out of avatar_validation.py on purpose: the anatomy rules are the part
that was wrong and they must be testable on a laptop with no models. This file
is the only place that imports them.

Loading never raises. It returns None, and AvatarValidator raises when a caller
tries to validate with a None backend — so a missing model surfaces at the
point of use with a message about what it was about to let through, rather than
as an import error somewhere unrelated.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from avatar_validation import Keypoint, Person

log = logging.getLogger(__name__)

# MediaPipe Pose landmark indices → the validator's names. MediaPipe emits 33
# landmarks; only these nine carry the structure the anatomy rules reason
# about. COCO-17 backends (RTMPose, YOLOv8-pose) map to the same nine names.
_MEDIAPIPE_INDICES = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


class MediaPipePose:
    """MediaPipe Pose adapter.

    Caveat worth knowing: MediaPipe Pose detects at most ONE person. It cannot
    report the "2 people" case on its own, which is exactly the two-torso
    signature — so a MediaPipe-only validator leans on the vertical-ordering,
    proportion and blob checks for that. Pair it with a segmentation backend,
    or prefer a multi-person model (YOLOv8-pose, RTMPose) where available.
    """

    def __init__(self, pose):
        self._pose = pose

    @classmethod
    def load(cls) -> Optional["MediaPipePose"]:
        try:
            import mediapipe as mp
            pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5,
            )
            return cls(pose)
        except Exception as e:
            log.warning("[pose] MediaPipe unavailable (%s: %s)", type(e).__name__, e)
            return None

    def detect(self, image_bgr: np.ndarray) -> Sequence[Person]:
        import cv2
        res = self._pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return []
        h, w = image_bgr.shape[:2]
        lm = res.pose_landmarks.landmark
        return [Person({
            name: Keypoint(lm[i].x * w, lm[i].y * h, lm[i].visibility)
            for name, i in _MEDIAPIPE_INDICES.items()
        })]


class YoloPose:
    """YOLOv8-pose adapter — multi-person, so it can actually report 2 people.

    COCO-17 has no nose-to-ankle gaps for our nine names, so the mapping is
    direct. Confidence comes from the per-keypoint score.
    """

    _COCO = {
        "nose": 0,
        "left_shoulder": 5, "right_shoulder": 6,
        "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14,
        "left_ankle": 15, "right_ankle": 16,
    }

    def __init__(self, model):
        self._model = model

    @classmethod
    def load(cls, weights: str = "yolov8n-pose.pt") -> Optional["YoloPose"]:
        try:
            from ultralytics import YOLO
            return cls(YOLO(weights))
        except Exception as e:
            log.warning("[pose] YOLOv8-pose unavailable (%s: %s)", type(e).__name__, e)
            return None

    def detect(self, image_bgr: np.ndarray) -> Sequence[Person]:
        results = self._model(image_bgr, verbose=False)
        people: list[Person] = []
        for r in results:
            if r.keypoints is None:
                continue
            xy = r.keypoints.xy.cpu().numpy()          # (n, 17, 2)
            conf = r.keypoints.conf
            conf = conf.cpu().numpy() if conf is not None else np.ones(xy.shape[:2])
            for person_i in range(xy.shape[0]):
                people.append(Person({
                    name: Keypoint(
                        float(xy[person_i, idx, 0]),
                        float(xy[person_i, idx, 1]),
                        float(conf[person_i, idx]),
                    )
                    for name, idx in cls._COCO.items()
                }))
        return people


class SelfieSegmentation:
    """MediaPipe selfie segmentation, for the connected-blob check."""

    def __init__(self, segmenter):
        self._seg = segmenter

    @classmethod
    def load(cls) -> Optional["SelfieSegmentation"]:
        try:
            import mediapipe as mp
            seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            return cls(seg)
        except Exception as e:
            log.warning("[pose] segmentation unavailable (%s: %s)", type(e).__name__, e)
            return None

    def person_mask(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        import cv2
        res = self._seg.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if res.segmentation_mask is None:
            return None
        return (res.segmentation_mask > 0.5).astype(np.uint8) * 255


def load_validator():
    """The validator this service should use.

    Prefers a multi-person pose model, because "exactly one person" is the
    cheapest and most direct two-torso check and MediaPipe cannot express it.
    Falls back to MediaPipe with segmentation attached so the blob check covers
    what the single-person model cannot see.

    Returns an AvatarValidator either way — possibly one with no backend, which
    raises on use. That is deliberate: the failure belongs at the write that
    would have shipped a bad asset, not at import.
    """
    from avatar_validation import AvatarValidator

    pose = YoloPose.load() or MediaPipePose.load()
    if pose is None:
        log.error("[pose] NO POSE MODEL — avatar writes will fail closed")
    return AvatarValidator(pose=pose, segmentation=SelfieSegmentation.load())
