"""
LivePortrait runner for LUCY — driving-frame expression transfer.

Per-frame contract:
    prepare_source(session_id, source_bgr) -> bool  (one-time per session)
    drive(session_id, driving_bgr) -> ndarray | None

Why this exists separately from face_swap_backend and instantid_backend:
LivePortrait is built specifically to carry the DRIVING frame's expression
(blinks, smiles, brow movement) onto a SOURCE identity. InstantID/PuLID
generate from prompt+ID embedding and have no driving channel, so they
produced "static" swap faces. LivePortrait is the correct architectural
choice for live mirror-style face animation.

Heavy lifting (Cropper, LivePortraitWrapper) comes from the upstream
KwaiVGI/LivePortrait repo, cloned during setup.ps1 into
liveportrait_backend/LivePortrait/. We add it to sys.path at import.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_REPO = _HERE / "LivePortrait"


def _lazy_imports():
    """Defer the repo-path check and all heavy imports until the engine
    is constructed. Keeps the module importable for FastAPI startup
    logging even when LivePortrait/ has not been cloned yet."""
    if not _REPO.exists():
        raise FileNotFoundError(
            f"LivePortrait repo not found at {_REPO}. Run setup.ps1 first."
        )
    # Add upstream repo to sys.path the first time we actually need it.
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    import torch
    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.utils.cropper import Cropper
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.utils.camera import get_rotation_matrix
    return (
        torch,
        LivePortraitWrapper,
        Cropper,
        InferenceConfig,
        CropConfig,
        get_rotation_matrix,
    )


class LivePortraitEngine:
    """One process-wide pipeline + many per-session source caches.

    Thread-safety: per-frame work serialized under one lock because the
    wrapper holds GPU state that cannot be entered concurrently.

    Session lifecycle:
        prepare_source(sid, src_bgr) -> caches appearance feature, kp_source,
            initial rotation matrix, paste-back transform, neutral
            driving info (filled on first drive() call).
        drive(sid, drv_bgr) -> returns animated source frame as BGR ndarray.
        drop_session(sid) -> releases cached tensors.
    """

    def __init__(self):
        (
            torch,
            LivePortraitWrapper,
            Cropper,
            InferenceConfig,
            CropConfig,
            get_rotation_matrix,
        ) = _lazy_imports()

        self._torch = torch
        self._get_rotation_matrix = get_rotation_matrix

        cfg = InferenceConfig(
            flag_use_half_precision=True,
            flag_do_crop=True,
            flag_stitching=True,
            flag_relative_motion=True,
            flag_pasteback=True,
            flag_eye_retargeting=False,   # off for live driving
            flag_lip_retargeting=False,   # off for live driving
            flag_normalize_lip=True,
            flag_do_torch_compile=False,  # Windows + Triton flaky as of 2026
        )
        self._cfg = cfg

        log.info("[LP] loading LivePortraitWrapper (motion, warp, generator)")
        self.wrap = LivePortraitWrapper(inference_cfg=cfg)
        log.info("[LP] loading Cropper (insightface buffalo_l + landmark.onnx)")
        self.cropper = Cropper(crop_cfg=CropConfig())

        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()
        log.info("[LP] engine ready")

    # ── Source registration (one-time per WebSocket session) ───────────────
    def prepare_source(self, session_id: str, source_bgr: np.ndarray) -> bool:
        """Crop the source face, extract appearance features and source
        keypoints. Caches everything needed for the per-frame drive call.
        Returns True on success, False if no face detected."""
        try:
            crop = self.cropper.crop_source_image(source_bgr, self.cropper.crop_cfg)
        except Exception as e:
            log.warning("[LP] source crop failed: %s", e)
            return False
        if crop is None or "img_crop_256x256" not in crop:
            log.warning("[LP] no face in source")
            return False

        I_s = self.wrap.prepare_source(crop["img_crop_256x256"])
        x_s_info = self.wrap.get_kp_info(I_s)
        f_s      = self.wrap.extract_feature_3d(I_s)
        x_s      = self.wrap.transform_keypoint(x_s_info)
        R_s = self._get_rotation_matrix(
            x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]
        )

        self._sessions[session_id] = {
            "I_s":        I_s,
            "x_s_info":   x_s_info,
            "f_s":        f_s,
            "x_s":        x_s,
            "R_s":        R_s,
            "M_c2o":      crop["M_c2o"],
            "src_full":   source_bgr,
            "mask_crop":  crop.get("mask_crop"),
            # Filled on the first drive() call. Without a neutral driving
            # baseline relative-motion produces a garbage frame.
            "x_d_0_info": None,
            "R_d_0":      None,
            # Last-known driving crop bbox for fast re-crop (avoid running
            # the full Cropper every frame — re-detect only every Nth frame).
            "last_bbox":  None,
            "frame_n":    0,
        }
        log.info("[LP] source prepared for session %s", session_id)
        return True

    def drop_session(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess is not None:
            with self._torch.cuda.device("cuda"):
                self._torch.cuda.empty_cache()

    # ── Per-frame driving ───────────────────────────────────────────────────
    def drive(self, session_id: str, driving_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Animate the cached source by the driving frame's expression
        and head pose. Returns the result as BGR ndarray (same shape as
        the source's full frame so the demo client can overlay 1-to-1).
        """
        sess = self._sessions.get(session_id)
        if sess is None:
            log.warning("[LP] unknown session: %s", session_id)
            return None

        with self._lock:
            return self._drive_locked(sess, driving_bgr)

    def _drive_locked(self, sess: dict, driving_bgr: np.ndarray) -> Optional[np.ndarray]:
        torch = self._torch

        # Crop the driving frame. To save ~10 ms, only re-run the full
        # Cropper every 10th frame; in between, reuse the previous bbox.
        sess["frame_n"] += 1
        crop_d = None
        if sess["frame_n"] % 10 == 1 or sess["last_bbox"] is None:
            try:
                crop_d = self.cropper.crop_driving_image(driving_bgr)
            except Exception:
                crop_d = None
            if crop_d is not None:
                sess["last_bbox"] = crop_d.get("bbox")
        if crop_d is None and sess["last_bbox"] is not None:
            # Fast path: cv2 crop using last bbox, then resize to 256
            x1, y1, x2, y2 = (int(v) for v in sess["last_bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(driving_bgr.shape[1], x2), min(driving_bgr.shape[0], y2)
            if x2 - x1 < 16 or y2 - y1 < 16:
                return None
            patch = driving_bgr[y1:y2, x1:x2]
            patch = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_LINEAR)
            crop_d = {"img_crop_256x256": patch}
        if crop_d is None:
            return None

        I_d = self.wrap.prepare_source(crop_d["img_crop_256x256"])
        x_d_info = self.wrap.get_kp_info(I_d)
        R_d = self._get_rotation_matrix(
            x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"]
        )

        # First driving frame becomes the neutral baseline.
        if sess["x_d_0_info"] is None:
            sess["x_d_0_info"] = {
                "kp":    x_d_info["kp"].clone(),
                "exp":   x_d_info["exp"].clone(),
                "t":     x_d_info["t"].clone(),
                "scale": x_d_info["scale"].clone(),
            }
            sess["R_d_0"] = R_d.clone()

        x_d_0 = sess["x_d_0_info"]
        R_d_0 = sess["R_d_0"]

        # Relative-motion driving: drive by delta from neutral.
        R_new = (R_d @ R_d_0.permute(0, 2, 1)) @ sess["R_s"]
        delta_exp = x_d_info["exp"] - x_d_0["exp"] + sess["x_s_info"]["exp"]
        scale_new = sess["x_s_info"]["scale"] * (x_d_info["scale"] / x_d_0["scale"])
        t_new = sess["x_s_info"]["t"] + (x_d_info["t"] - x_d_0["t"])
        t_new[..., 2] = 0  # zero out z translation — convention

        x_d_new = scale_new * (sess["x_s_info"]["kp"] @ R_new + delta_exp) + t_new
        if self._cfg.flag_stitching:
            x_d_new = self.wrap.stitching(sess["x_s"], x_d_new)

        out = self.wrap.warp_decode(sess["f_s"], sess["x_s"], x_d_new)
        I_p = self.wrap.parse_output(out["out"])   # HxWx3 RGB uint8

        if self._cfg.flag_pasteback and sess["mask_crop"] is not None:
            try:
                from src.utils.crop import paste_back
                I_p = paste_back(I_p, sess["M_c2o"], sess["src_full"], sess["mask_crop"])
            except Exception as e:
                log.warning("[LP] paste_back failed: %s", e)

        return cv2.cvtColor(I_p, cv2.COLOR_RGB2BGR)
