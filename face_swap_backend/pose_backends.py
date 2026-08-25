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
import pathlib
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


# Google's hosted Tasks models. tryon_backend already fetches from the same
# place; the download helper mirrors its .part-then-rename so a killed process
# cannot leave a truncated model that then fails to load forever.
_MP_DIR = pathlib.Path(__file__).parent / "models" / "mediapipe"
_MP_POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker.task"
)
_MP_SEG_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)


def _mp_download(url: str, dest: pathlib.Path) -> pathlib.Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    log.info("[pose] downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


class MediaPipePose:
    """MediaPipe Pose via the TASKS API.

    Not `mp.solutions`. That legacy API is absent from the mediapipe builds on
    this project's own boxes — `module 'mediapipe' has no attribute 'solutions'`
    — and tryon_backend/model.py already says so in a comment: the Tasks API is
    the one that works on Windows, Linux and Mac. Writing this against
    `solutions` meant the validator could not load at all, which fail-closed
    then correctly turned into "refusing to validate".

    num_poses=2 matters: it lets MediaPipe report the two-people case that is
    the two-torso signature. The legacy solutions API could only ever find one.
    """

    def __init__(self, landmarker, mp_module):
        self._landmarker = landmarker
        self._mp = mp_module

    @classmethod
    def load(cls) -> Optional["MediaPipePose"]:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker, PoseLandmarkerOptions, RunningMode,
            )
            path = _mp_download(_MP_POSE_URL, _MP_DIR / "pose_landmarker.task")
            landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                running_mode=RunningMode.IMAGE,
                # Two, so a duplicated figure can be reported as two people
                # rather than silently resolving to one.
                num_poses=2,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            ))
            return cls(landmarker, mp)
        except Exception as e:
            log.warning("[pose] MediaPipe Tasks unavailable (%s: %s)",
                        type(e).__name__, e)
            return None

    def detect(self, image_bgr: np.ndarray) -> Sequence[Person]:
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(mp_image)
        if not res.pose_landmarks:
            return []
        h, w = image_bgr.shape[:2]
        people = []
        for landmarks in res.pose_landmarks:
            people.append(Person({
                name: Keypoint(landmarks[i].x * w, landmarks[i].y * h,
                               getattr(landmarks[i], "visibility", 1.0))
                for name, i in _MEDIAPIPE_INDICES.items()
            }))
        return people


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
    """Person mask for the connected-blob check, via the Tasks API."""

    def __init__(self, segmenter, mp_module):
        self._seg = segmenter
        self._mp = mp_module

    @classmethod
    def load(cls) -> Optional["SelfieSegmentation"]:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                ImageSegmenter, ImageSegmenterOptions, RunningMode,
            )
            path = _mp_download(_MP_SEG_URL, _MP_DIR / "selfie_segmenter.tflite")
            seg = ImageSegmenter.create_from_options(ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                running_mode=RunningMode.IMAGE,
                output_category_mask=False,
                output_confidence_masks=True,
            ))
            return cls(seg, mp)
        except Exception as e:
            log.warning("[pose] segmentation unavailable (%s: %s)",
                        type(e).__name__, e)
            return None

    def person_mask(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._seg.segment(mp_image)
        masks = getattr(res, "confidence_masks", None)
        if not masks:
            return None
        return (masks[0].numpy_view() > 0.5).astype(np.uint8) * 255


def load_validator():
    """The validator this service should use.

    Either backend can report the two-people case that is the two-torso
    signature: YOLOv8-pose is multi-person by nature, and the MediaPipe Tasks
    landmarker is configured with num_poses=2. YOLO is preferred only because
    it needs no per-image model download; MediaPipe is a complete substitute,
    not a degraded one.

    Returns an AvatarValidator either way — possibly one with no backend, which
    raises on use. That is deliberate: the failure belongs at the write that
    would have shipped a bad asset, not at import.
    """
    from avatar_validation import AvatarValidator

    pose = YoloPose.load() or MediaPipePose.load()
    if pose is None:
        log.error(
            "[pose] NO POSE MODEL — avatar writes and curation will fail "
            "closed. Install one: `pip install mediapipe` (Tasks API, "
            "downloads its model on first use) or `pip install ultralytics`."
        )
    return AvatarValidator(pose=pose, segmentation=SelfieSegmentation.load())
