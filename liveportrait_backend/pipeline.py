"""
LivePortrait runner for LUCY — driving-frame expression transfer.

Per-frame contract:
    prepare_source(session_id, source_bgr) -> bool  (one-time per session)
    drive(session_id, driving_bgr) -> DriveResult

Why this exists separately from face_swap_backend and instantid_backend:
LivePortrait is built specifically to carry the DRIVING frame's expression
(blinks, smiles, brow movement) onto a SOURCE identity. InstantID/PuLID
generate from prompt+ID embedding and have no driving channel, so they
produced "static" swap faces. LivePortrait is the correct architectural
choice for live mirror-style face animation.

Heavy lifting (Cropper, LivePortraitWrapper) comes from the upstream
KwaiVGI/LivePortrait repo, cloned during setup.ps1 into
liveportrait_backend/LivePortrait/. We add it to sys.path at import.

Driving-frame alignment
-----------------------
The motion extractor is only meaningful when every driving frame is
cropped the same way the source was: face-aligned, same scale, same
vertical offset. We therefore follow upstream's video path exactly —
detect once, then track landmarks frame-to-frame with landmark.onnx
(~2 ms) and run `crop_image` with the driving crop params. Re-detection
happens on a slow cadence purely to correct tracker drift.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_REPO = _HERE / "LivePortrait"

# ── Tunables ──────────────────────────────────────────────────────────────
# All of these are per-session and can be changed at runtime by the client
# (WS `config` message) so the deployed box can be tuned without a redeploy.
DEFAULT_PARAMS: dict[str, float] = {
    # Expression amplification on the delta from the neutral baseline.
    # With a properly averaged baseline, mild amplification is enough;
    # 1.35+ produced distorted geometry when baseline drift added up.
    "exp_amp": 1.1,
    # How much of the driver's head rotation is carried onto the portrait.
    # 0.0 = head locked to the source pose (only the face animates), which
    # is the safe default: large relative yaw stretches a still portrait.
    # 0.6-0.8 gives a natural "mirror" feel on near-frontal avatars.
    "pose_gain": 0.0,
    # EMA weight on the *new* frame's expression delta. 0.7 → ~110 ms tau:
    # catches single-frame outliers, passes blinks through near full
    # amplitude. Lower = smoother but laggier.
    "smooth": 0.7,
}

# Frames averaged into the neutral baseline before live driving starts.
BASELINE_FRAMES = 10
# Face re-detection cadence (frames). Landmarks are tracked every frame;
# detection only corrects tracker drift, so it is cheap to amortise — at
# 20 fps this is a detector pass every ~0.75 s, which bounds how long a
# diverged tracker can keep animating on nonsense.
DETECT_EVERY = 15
# Longest run of tracking-only frames tolerated before a forced re-detect.
MAX_TRACK_MISS = 3
# Source frames are downscaled to this max dimension. Paste-back and JPEG
# encode both scale with it, and anything above 720 is invisible in a
# webcam-sized canvas while costing real milliseconds per frame.
SOURCE_MAX_DIM = 720


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
    from src.utils.crop import crop_image, prepare_paste_back, paste_back
    from src.utils.io import contiguous, resize_to_limit

    return dict(
        torch=torch,
        LivePortraitWrapper=LivePortraitWrapper,
        Cropper=Cropper,
        InferenceConfig=InferenceConfig,
        CropConfig=CropConfig,
        get_rotation_matrix=get_rotation_matrix,
        crop_image=crop_image,
        prepare_paste_back=prepare_paste_back,
        paste_back=paste_back,
        contiguous=contiguous,
        resize_to_limit=resize_to_limit,
    )


@dataclass
class DriveResult:
    """One frame's outcome. `state` lets the client tell the difference
    between "no face in shot" and "still measuring your neutral face",
    which used to both surface as a silent no-op."""

    state: str                          # "live" | "calibrating" | "no_face"
    image: Optional[np.ndarray] = None  # BGR, full source frame size
    infer_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.image is not None


@dataclass
class _Session:
    # Source (static for the life of the session)
    I_s: Any = None
    x_s_info: Any = None
    f_s: Any = None
    x_s: Any = None
    R_s: Any = None
    M_c2o: Any = None
    src_full: Any = None      # RGB — paste_back composites onto RGB
    mask_ori: Any = None      # float mask in source-frame space
    # Driving-frame landmark tracking (original driving-frame space)
    lmk_track: Optional[np.ndarray] = None
    track_center: Optional[tuple] = None
    track_miss: int = 0
    force_detect: bool = False
    # Neutral baseline, averaged over the first BASELINE_FRAMES frames
    base: Optional[dict] = None
    _accum: Optional[dict] = None
    # Rolling state
    exp_smooth: Any = None
    frame_n: int = 0
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    last_seen: float = 0.0
    infer_ema: float = 0.0
    drives: int = 0


class LivePortraitEngine:
    """One process-wide pipeline + many per-session source caches.

    Thread-safety: every public method takes `self._lock`. The wrapper
    holds GPU state that cannot be entered concurrently, and the session
    map is mutated from FastAPI's executor threads.

    Session lifecycle:
        prepare_source(sid, src_bgr) -> caches appearance feature, kp_source,
            source rotation, paste-back transform + mask.
        drive(sid, drv_bgr) -> DriveResult with the animated frame.
        drop_session(sid) -> releases cached tensors.
    """

    def __init__(self):
        u = _lazy_imports()
        self._torch = u["torch"]
        self._get_rotation_matrix = u["get_rotation_matrix"]
        self._crop_image = u["crop_image"]
        self._prepare_paste_back = u["prepare_paste_back"]
        self._paste_back = u["paste_back"]
        self._contiguous = u["contiguous"]
        self._resize_to_limit = u["resize_to_limit"]

        torch = self._torch
        # Fixed 256x256 input shape every frame — let cuDNN autotune once.
        torch.backends.cudnn.benchmark = True

        cfg = u["InferenceConfig"](
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
        self.wrap = u["LivePortraitWrapper"](inference_cfg=cfg)
        log.info("[LP] loading Cropper (insightface buffalo_l + landmark.onnx)")
        self.cropper = u["Cropper"](crop_cfg=u["CropConfig"]())
        self._crop_cfg = self.cropper.crop_cfg
        # Upstream renamed this attribute (landmark_runner → human_landmark_runner)
        # and setup.ps1 clones HEAD, so accept either.
        self._lmk_runner = getattr(
            self.cropper, "human_landmark_runner", None
        ) or getattr(self.cropper, "landmark_runner")

        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()
        self._started = time.time()
        self._warmup()
        log.info("[LP] engine ready")

    # ── Startup warmup ──────────────────────────────────────────────────────
    def _warmup(self) -> None:
        """Run one synthetic frame through the network so cuDNN autotuning
        and lazy CUDA context creation happen before the first user frame
        instead of adding ~1 s to it. No face needed — the cropper is not
        involved and the output is discarded."""
        try:
            dummy = np.full((256, 256, 3), 127, dtype=np.uint8)
            I = self.wrap.prepare_source(dummy)
            info = self.wrap.get_kp_info(I)
            f = self.wrap.extract_feature_3d(I)
            x = self.wrap.transform_keypoint(info)
            self.wrap.warp_decode(f, x, self.wrap.stitching(x, x))
            log.info("[LP] warmup pass complete")
        except Exception as e:  # never block startup on a warmup failure
            log.warning("[LP] warmup skipped: %s", e)

    # ── Source registration (one-time per WebSocket session) ───────────────
    def prepare_source(
        self, session_id: str, source_bgr: np.ndarray, **params: float
    ) -> bool:
        """Crop the source face, extract appearance features and source
        keypoints. Caches everything needed for the per-frame drive call.
        Returns True on success, False if no face detected.

        Upstream cropper takes RGB (it converts internally). Passing BGR
        produces double-conversion and the model outputs scrambled colour
        noise — which is exactly what we saw before this fix.
        """
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        # Cap the working resolution: every paste-back and JPEG encode for
        # the rest of the session is proportional to it. `division=2` keeps
        # the warp transform on whole pixels.
        source_rgb = self._resize_to_limit(source_rgb, SOURCE_MAX_DIM, 2)

        with self._lock:
            try:
                crop = self.cropper.crop_source_image(source_rgb, self._crop_cfg)
            except Exception as e:
                log.warning("[LP] source crop failed: %s", e)
                return False
            if crop is None or "img_crop_256x256" not in crop:
                log.warning("[LP] no face in source")
                return False

            I_s = self.wrap.prepare_source(crop["img_crop_256x256"])
            x_s_info = self.wrap.get_kp_info(I_s)
            f_s = self.wrap.extract_feature_3d(I_s)
            x_s = self.wrap.transform_keypoint(x_s_info)
            R_s = self._get_rotation_matrix(
                x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]
            )

            # Paste-back needs the mask warped into source-frame space. The
            # source never moves, so this is computed once per session —
            # previously `crop.get("mask_crop")` was read, but the cropper
            # returns no such key, so paste-back silently never ran and the
            # client got a bare 512x512 crop instead of the full portrait.
            mask_ori = None
            if self._cfg.flag_pasteback:
                try:
                    mask_ori = self._prepare_paste_back(
                        self._cfg.mask_crop,
                        crop["M_c2o"],
                        dsize=(source_rgb.shape[1], source_rgb.shape[0]),
                    )
                except Exception as e:
                    log.warning("[LP] paste-back mask unavailable: %s", e)

            sess = _Session(
                I_s=I_s, x_s_info=x_s_info, f_s=f_s, x_s=x_s, R_s=R_s,
                M_c2o=crop["M_c2o"], src_full=source_rgb, mask_ori=mask_ori,
                last_seen=time.time(),
            )
            for k, v in params.items():
                if k in DEFAULT_PARAMS:
                    try:
                        sess.params[k] = float(v)
                    except (TypeError, ValueError):
                        log.warning("[LP] ignoring bad %s=%r", k, v)
            self._sessions[session_id] = sess

        log.info(
            "[LP] source prepared for session %s (%dx%d, pasteback=%s)",
            session_id, source_rgb.shape[1], source_rgb.shape[0],
            mask_ori is not None,
        )
        return True

    def configure(self, session_id: str, **params: float) -> dict:
        """Live-tune expression amp / pose follow / smoothing without
        restarting the session. Unknown keys are ignored."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return {}
            for k, v in params.items():
                if k in DEFAULT_PARAMS:
                    try:
                        sess.params[k] = float(v)
                    except (TypeError, ValueError):
                        continue
            return dict(sess.params)

    def recalibrate(self, session_id: str) -> bool:
        """Drop the neutral baseline so the next frames re-measure it.
        Used when the user changes seat/lighting and the portrait locks
        into a skewed resting expression."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return False
            sess.base = None
            sess._accum = None
            sess.exp_smooth = None
            return True

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            sess = self._sessions.pop(session_id, None)
            if sess is None:
                return
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()

    def sweep(self, max_idle: float = 300.0) -> int:
        """Release sessions whose client vanished without a clean close
        (mobile Safari backgrounding, laptop lid, dropped LTE). Without
        this their appearance features sit in VRAM until restart."""
        now = time.time()
        with self._lock:
            stale = [
                sid for sid, s in self._sessions.items()
                if now - s.last_seen > max_idle
            ]
            for sid in stale:
                self._sessions.pop(sid, None)
            if stale and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        if stale:
            log.info("[LP] swept %d idle session(s): %s", len(stale), ", ".join(stale))
        return len(stale)

    def stats(self) -> dict:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self._started, 1),
                "sessions": len(self._sessions),
                "detail": [
                    {
                        "id": sid,
                        "frames": s.drives,
                        "infer_ms": round(s.infer_ema, 1),
                        "calibrated": s.base is not None,
                        "params": s.params,
                        "idle_s": round(time.time() - s.last_seen, 1),
                    }
                    for sid, s in self._sessions.items()
                ],
            }

    # ── Driving-frame crop (detect once, then track) ────────────────────────
    def _crop_driving(self, sess: _Session, driving_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return a 256x256 RGB face-aligned crop of the driving frame, or
        None when there is no face to work with.

        Alignment must match how the source was cropped or the expression
        coefficients are noise. Upstream does detect-then-track for video;
        so do we. The previous implementation re-used `lmk_crop_lst`
        (which is in *crop* space) as a bbox into the *original* frame,
        so 9 of every 10 frames were driven by a misaligned, near-full-frame
        crop — that is what produced the weak expressions and the visible
        pulse every 10th frame.
        """
        cfg = self._crop_cfg
        # Re-detect on a slow cadence to correct drift, and on every frame
        # while the detector is missing — a user who walked out of shot must
        # surface as no_face within a few frames, not at the next cadence
        # boundary.
        force_detect = (
            sess.lmk_track is None
            or sess.force_detect
            or sess.frame_n % DETECT_EVERY == 1
            or sess.track_miss > 0
        )
        sess.force_detect = False

        lmk = None
        if force_detect:
            try:
                faces = self.cropper.face_analysis_wrapper.get(
                    self._contiguous(driving_rgb[..., ::-1]),  # BGR for insightface
                    flag_do_landmark_2d_106=True,
                    direction=cfg.direction,
                )
            except Exception as e:
                if sess.frame_n % 60 == 1:
                    log.warning("[LP] face detect failed: %s", e)
                faces = []
            if faces:
                lmk = self._lmk_runner.run(driving_rgb, faces[0].landmark_2d_106)
                sess.track_miss = 0
            elif sess.lmk_track is None:
                return None                      # nothing to fall back to
            else:
                # Detector blinked (motion blur, backlight). Keep tracking
                # from the last landmarks, but give up after MAX_TRACK_MISS
                # consecutive misses so a user who left the frame reports
                # no_face instead of animating on garbage.
                sess.track_miss += 1
                if sess.track_miss > MAX_TRACK_MISS:
                    sess.lmk_track = None
                    sess.track_center = None
                    sess.track_miss = 0
                    return None

        if lmk is None:
            lmk = self._lmk_runner.run(driving_rgb, sess.lmk_track)

        # Guard against a diverged tracker feeding a degenerate crop. The
        # landmark runner will happily lock onto a wall once the face is
        # gone, so its output is sanity-checked rather than trusted.
        if not np.all(np.isfinite(lmk)):
            sess.lmk_track = None
            sess.track_center = None
            return None
        span = float(max(np.ptp(lmk[:, 0]), np.ptp(lmk[:, 1])))
        if span < 16.0:
            sess.lmk_track = None
            sess.track_center = None
            return None

        h, w = driving_rgb.shape[:2]
        cx, cy = float(lmk[:, 0].mean()), float(lmk[:, 1].mean())
        if not (0 <= cx < w and 0 <= cy < h):
            sess.lmk_track = None            # tracker walked off the image
            sess.track_center = None
            return None
        if sess.track_center is not None:
            dx, dy = abs(cx - sess.track_center[0]), abs(cy - sess.track_center[1])
            if dx > 0.4 * w or dy > 0.4 * h:
                # A face cannot cross half the frame in one frame; make the
                # detector confirm it before we keep driving off this track.
                sess.force_detect = True
        sess.track_center = (cx, cy)

        sess.lmk_track = lmk
        ret = self._crop_image(
            driving_rgb,
            lmk,
            dsize=cfg.dsize,
            scale=cfg.scale_crop_driving_video,
            vx_ratio=cfg.vx_ratio_crop_driving_video,
            vy_ratio=cfg.vy_ratio_crop_driving_video,
            flag_do_rot=False,   # keep head roll in the signal, as upstream does
        )
        return cv2.resize(ret["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)

    # ── Per-frame driving ───────────────────────────────────────────────────
    def drive(self, session_id: str, driving_bgr: np.ndarray) -> DriveResult:
        """Animate the cached source by the driving frame's expression
        (and optionally head pose). Returns the result as a BGR ndarray
        the same size as the source frame, so the client can overlay 1-to-1.
        """
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                log.warning("[LP] unknown session: %s", session_id)
                return DriveResult(state="no_face")
            t0 = time.perf_counter()
            res = self._drive_locked(sess, driving_bgr)
            dt = (time.perf_counter() - t0) * 1000
            sess.last_seen = time.time()
            sess.drives += 1
            sess.infer_ema = dt if sess.infer_ema == 0 else 0.1 * dt + 0.9 * sess.infer_ema
            res.infer_ms = dt
            return res

    def _drive_locked(self, sess: _Session, driving_bgr: np.ndarray) -> DriveResult:
        sess.frame_n += 1
        driving_rgb = cv2.cvtColor(driving_bgr, cv2.COLOR_BGR2RGB)

        crop256 = self._crop_driving(sess, driving_rgb)
        if crop256 is None:
            return DriveResult(state="no_face")

        I_d = self.wrap.prepare_source(crop256)
        x_d_info = self.wrap.get_kp_info(I_d)

        # ── Neutral baseline ────────────────────────────────────────────────
        # Averaged over the first N frames so a non-neutral first capture
        # (mid-blink, slight smile, off-centre head) doesn't permanently
        # miscalibrate every subsequent delta. During collection the source
        # is returned unchanged — ~500 ms of still portrait, then live.
        if sess.base is None:
            if not self._accumulate_baseline(sess, x_d_info):
                return DriveResult(
                    state="calibrating",
                    image=cv2.cvtColor(sess.src_full, cv2.COLOR_RGB2BGR),
                )

        base = sess.base
        p = sess.params

        # ── Expression ──────────────────────────────────────────────────────
        raw_delta = x_d_info["exp"] - base["exp"]
        if sess.exp_smooth is None:
            sess.exp_smooth = raw_delta.clone()
        else:
            a = p["smooth"]
            sess.exp_smooth = a * raw_delta + (1.0 - a) * sess.exp_smooth
        delta_exp = p["exp_amp"] * sess.exp_smooth + sess.x_s_info["exp"]

        # ── Head pose ───────────────────────────────────────────────────────
        # pose_gain 0 keeps the head locked to the source (stable for still
        # portraits); >0 carries a scaled share of the driver's rotation
        # delta. Working in Euler space rather than composing rotation
        # matrices is what makes the gain well-defined — and the baseline
        # angles average correctly, which averaged rotation matrices do not.
        gain = p["pose_gain"]
        if gain > 0.0:
            R_new = self._get_rotation_matrix(
                sess.x_s_info["pitch"] + gain * (x_d_info["pitch"] - base["pitch"]),
                sess.x_s_info["yaw"] + gain * (x_d_info["yaw"] - base["yaw"]),
                sess.x_s_info["roll"] + gain * (x_d_info["roll"] - base["roll"]),
            )
        else:
            R_new = sess.R_s

        # Scale and translation stay locked to the source: carrying them
        # through moved the whole head around the frame and broke the
        # paste-back seam.
        scale_new = sess.x_s_info["scale"]
        t_new = sess.x_s_info["t"].clone()
        t_new[..., 2] = 0

        x_d_new = scale_new * (sess.x_s_info["kp"] @ R_new + delta_exp) + t_new
        if self._cfg.flag_stitching:
            x_d_new = self.wrap.stitching(sess.x_s, x_d_new)

        out = self.wrap.warp_decode(sess.f_s, sess.x_s, x_d_new)
        # parse_output returns 1xHxWx3 (batch-prefixed) per upstream
        # docstring. Squeeze the batch dim so I_p is HxWx3 RGB uint8.
        I_p = self.wrap.parse_output(out["out"])
        if I_p.ndim == 4:
            I_p = I_p[0]

        if sess.mask_ori is not None:
            try:
                I_p = self._paste_back(I_p, sess.M_c2o, sess.src_full, sess.mask_ori)
            except Exception as e:
                log.warning("[LP] paste_back failed: %s", e)

        return DriveResult(state="live", image=cv2.cvtColor(I_p, cv2.COLOR_RGB2BGR))

    def _accumulate_baseline(self, sess: _Session, x_d_info: dict) -> bool:
        """Fold one frame into the neutral baseline. Returns True once the
        baseline is complete and live driving can start."""
        keys = ("exp", "pitch", "yaw", "roll")
        acc = sess._accum
        if acc is None:
            sess._accum = {k: x_d_info[k].clone() for k in keys}
            sess._accum["n"] = 1
            return False

        for k in keys:
            acc[k] = acc[k] + x_d_info[k]
        acc["n"] += 1
        if acc["n"] < BASELINE_FRAMES:
            return False

        n = acc["n"]
        sess.base = {k: acc[k] / n for k in keys}
        sess._accum = None
        return True
