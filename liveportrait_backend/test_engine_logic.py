"""
Offline tests for the LivePortrait session logic — no GPU, no weights, no
network. The upstream repo (torch, the wrapper, the cropper) is replaced by
stand-ins that record how they were called, so this exercises the parts we
actually wrote: driving-frame crop scheduling, neutral-baseline averaging,
pose gain, paste-back wiring, session lifecycle, and the server's input
validation.

Run:  python test_engine_logic.py
"""
from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        FAILURES.append(f"{name}{' — ' + detail if detail else ''}")
        print(f"  FAIL  {name}{' — ' + detail if detail else ''}")


# ── Stand-ins for the upstream repo ────────────────────────────────────────
class T(np.ndarray):
    """ndarray that answers .clone() like a torch tensor."""

    def clone(self):
        return np.array(self, copy=True).view(T)


def t(x) -> T:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32)).view(T)


class Calls:
    """Call counters shared with the tests."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.detect = 0
        self.track = 0
        self.rotation = 0
        self.paste_back = 0
        self.crop_image = 0
        self.detect_returns_face = True
        self.source_has_face = True


CALLS = Calls()


class _FakeWrapper:
    def __init__(self, inference_cfg=None, **kw):
        self.cfg = inference_cfg

    def prepare_source(self, img):
        assert img.shape == (256, 256, 3), f"network input must be 256x256, got {img.shape}"
        assert img.dtype == np.uint8
        return t(img.astype(np.float32).mean() / 255.0)

    def get_kp_info(self, I):
        seed = float(np.asarray(I).mean())
        return {
            "kp": t(np.full((1, 21, 3), seed)),
            "exp": t(np.full((1, 21, 3), seed)),
            "pitch": t([[seed * 10]]),
            "yaw": t([[seed * 5]]),
            "roll": t([[seed * 2]]),
            "scale": t([[1.0]]),
            "t": t([[0.0, 0.0, 0.0]]),
        }

    def extract_feature_3d(self, I):
        return t(np.zeros((1, 32, 16, 16, 16)))

    def transform_keypoint(self, info):
        return t(np.zeros((1, 21, 3)))

    def stitching(self, x_s, x_d):
        return x_d

    def warp_decode(self, f_s, x_s, x_d):
        return {"out": t(np.zeros((1, 3, 512, 512)))}

    def parse_output(self, out):
        return np.zeros((1, 512, 512, 3), dtype=np.uint8)


class _FakeCropCfg:
    dsize = 512
    scale = 2.3
    vx_ratio = 0.0
    vy_ratio = -0.125
    flag_do_rot = True
    direction = "large-small"
    scale_crop_driving_video = 2.2
    vx_ratio_crop_driving_video = 0.0
    vy_ratio_crop_driving_video = -0.1


class _FakeFace:
    landmark_2d_106 = np.tile(np.linspace(50, 200, 106)[:, None], (1, 2)).astype(np.float32)


class _FakeFaceAnalysis:
    def get(self, img_bgr, **kw):
        CALLS.detect += 1
        return [_FakeFace()] if CALLS.detect_returns_face else []


class _FakeLandmarkRunner:
    def run(self, img_rgb, lmk):
        CALLS.track += 1
        return np.asarray(lmk, dtype=np.float32) + 0.5


class _FakeCropper:
    def __init__(self, crop_cfg=None, **kw):
        self.crop_cfg = crop_cfg or _FakeCropCfg()
        self.face_analysis_wrapper = _FakeFaceAnalysis()
        self.human_landmark_runner = _FakeLandmarkRunner()

    def crop_source_image(self, img_rgb, crop_cfg):
        if not CALLS.source_has_face:
            return None
        return {
            "img_crop_256x256": np.zeros((256, 256, 3), dtype=np.uint8),
            "M_c2o": np.eye(3, dtype=np.float32),
        }


class _FakeInferenceConfig:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.mask_crop = np.full((512, 512, 3), 255, dtype=np.uint8)
        self.input_shape = (256, 256)


def _fake_get_rotation_matrix(pitch, yaw, roll):
    CALLS.rotation += 1
    return t(np.tile(np.eye(3), (1, 1, 1)))


def _fake_crop_image(img, pts, **kw):
    CALLS.crop_image += 1
    dsize = kw.get("dsize", 512)
    return {"img_crop": np.zeros((dsize, dsize, 3), dtype=np.uint8),
            "M_c2o": np.eye(3, dtype=np.float32)}


def _fake_prepare_paste_back(mask_crop, M_c2o, dsize):
    return np.ones((dsize[1], dsize[0], 3), dtype=np.float32)


def _fake_paste_back(img_crop, M_c2o, img_ori, mask_ori):
    CALLS.paste_back += 1
    return np.zeros_like(img_ori)


def _fake_contiguous(x):
    return np.ascontiguousarray(x)


def _fake_resize_to_limit(img, max_dim=1920, division=2):
    """Faithful copy of upstream's behaviour — the assertions about source
    downscaling are only meaningful if this matches."""
    import cv2

    h, w = img.shape[:2]
    if max_dim > 0 and max(h, w) > max_dim:
        s = max_dim / max(h, w)
        h, w = int(h * s), int(w * s)
        img = cv2.resize(img, (w, h))
    if division > 1:
        h, w = h - (h % division), w - (w % division)
        img = img[:h, :w]
    return img


def install_fakes() -> None:
    """Register stand-in modules so pipeline._lazy_imports() resolves."""
    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False))
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = torch

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    mod("src")
    mod("src.utils")
    mod("src.config")
    mod("src.live_portrait_wrapper", LivePortraitWrapper=_FakeWrapper)
    mod("src.utils.cropper", Cropper=_FakeCropper)
    mod("src.config.inference_config", InferenceConfig=_FakeInferenceConfig)
    mod("src.config.crop_config", CropConfig=_FakeCropCfg)
    mod("src.utils.camera", get_rotation_matrix=_fake_get_rotation_matrix)
    mod("src.utils.crop", crop_image=_fake_crop_image,
        prepare_paste_back=_fake_prepare_paste_back, paste_back=_fake_paste_back)
    mod("src.utils.io", contiguous=_fake_contiguous, resize_to_limit=_fake_resize_to_limit)


def frame(w=384, h=384) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ── Tests ──────────────────────────────────────────────────────────────────
def test_pipeline() -> None:
    import pipeline

    pipeline._REPO = _HERE          # skip the "clone the repo first" guard
    eng = pipeline.LivePortraitEngine()

    print("\nsource preparation")
    src = np.zeros((1500, 2000, 3), dtype=np.uint8)
    check("prepare_source succeeds", eng.prepare_source("s1", src) is True)
    sess = eng._sessions["s1"]
    check("source downscaled to SOURCE_MAX_DIM",
          max(sess.src_full.shape[:2]) == pipeline.SOURCE_MAX_DIM,
          f"got {sess.src_full.shape}")
    check("paste-back mask prepared (the old code never built one)",
          sess.mask_ori is not None and sess.mask_ori.shape[:2] == sess.src_full.shape[:2])

    CALLS.source_has_face = False
    check("faceless source rejected", eng.prepare_source("bad", src) is False)
    CALLS.source_has_face = True

    print("\nneutral-baseline calibration")
    states = [eng.drive("s1", frame()).state for _ in range(pipeline.BASELINE_FRAMES + 2)]
    check("first frames report calibrating",
          all(s == "calibrating" for s in states[: pipeline.BASELINE_FRAMES - 1]),
          str(states))
    check("live once the baseline is averaged",
          all(s == "live" for s in states[pipeline.BASELINE_FRAMES - 1:]), str(states))
    check("baseline averaged over BASELINE_FRAMES frames",
          eng._sessions["s1"].base is not None)

    print("\noutput geometry")
    res = eng.drive("s1", frame())
    check("result is the full source frame, not a bare crop",
          res.image.shape == sess.src_full.shape, str(res.image.shape))
    check("paste_back actually ran", CALLS.paste_back > 0)

    print("\ndriving-crop scheduling")
    CALLS.reset()
    eng.prepare_source("s2", np.zeros((720, 720, 3), dtype=np.uint8))
    n = 40
    for _ in range(n):
        eng.drive("s2", frame())
    expected_detects = 1 + (n - 1) // pipeline.DETECT_EVERY
    check(f"face detection runs {expected_detects}x in {n} frames, not every frame",
          CALLS.detect == expected_detects, f"got {CALLS.detect}")
    check("landmarks tracked on every frame", CALLS.track == n, f"got {CALLS.track}")
    check("every driving frame is face-aligned via crop_image",
          CALLS.crop_image == n, f"got {CALLS.crop_image}")

    print("\nno-face handling")
    CALLS.reset()
    CALLS.detect_returns_face = False
    eng.prepare_source("s3", np.zeros((720, 720, 3), dtype=np.uint8))
    check("no face at all -> no_face", eng.drive("s3", frame()).state == "no_face")

    CALLS.detect_returns_face = True
    eng.prepare_source("s4", np.zeros((720, 720, 3), dtype=np.uint8))
    for _ in range(pipeline.BASELINE_FRAMES + 1):
        eng.drive("s4", frame())
    CALLS.detect_returns_face = False
    budget = pipeline.DETECT_EVERY + pipeline.MAX_TRACK_MISS + 2
    seen = [eng.drive("s4", frame()).state for _ in range(budget)]
    check("brief detector miss keeps animating from the tracker",
          seen[0] == "live", str(seen))
    check(f"sustained miss degrades to no_face within {budget} frames",
          seen[-1] == "no_face", str(seen))
    CALLS.detect_returns_face = True

    print("\npose gain")
    CALLS.reset()
    eng.prepare_source("s5", np.zeros((720, 720, 3), dtype=np.uint8))
    for _ in range(pipeline.BASELINE_FRAMES + 4):
        eng.drive("s5", frame())
    check("pose_gain=0 locks the head to the source pose (no per-frame rotation)",
          CALLS.rotation == 1, f"rotation built {CALLS.rotation}x")

    CALLS.reset()
    eng.prepare_source("s6", np.zeros((720, 720, 3), dtype=np.uint8), pose_gain=0.7)
    check("init params reach the session", eng._sessions["s6"].params["pose_gain"] == 0.7)
    for _ in range(pipeline.BASELINE_FRAMES + 4):
        eng.drive("s6", frame())
    check("pose_gain>0 rebuilds the rotation each live frame",
          CALLS.rotation > 1, f"rotation built {CALLS.rotation}x")

    print("\nlive tuning + lifecycle")
    p = eng.configure("s6", exp_amp=1.4, pose_gain=0.3, bogus=9, smooth="nope")
    check("known params updated", p["exp_amp"] == 1.4 and p["pose_gain"] == 0.3)
    check("unknown params ignored", "bogus" not in p)
    check("unparseable values ignored", p["smooth"] == pipeline.DEFAULT_PARAMS["smooth"])
    check("configure on a dead session is a no-op", eng.configure("nope", exp_amp=2.0) == {})

    check("recalibrate clears the baseline", eng.recalibrate("s6") is True
          and eng._sessions["s6"].base is None)
    check("post-recalibrate frames report calibrating",
          eng.drive("s6", frame()).state == "calibrating")

    check("drive on an unknown session is safe", eng.drive("ghost", frame()).state == "no_face")

    stats = eng.stats()
    check("stats report live sessions", stats["sessions"] == len(eng._sessions))
    check("stats carry per-session detail", all(
        {"id", "frames", "infer_ms", "calibrated", "params"} <= set(d) for d in stats["detail"]))

    eng._sessions["s1"].last_seen = 0.0
    check("sweep drops abandoned sessions", eng.sweep(max_idle=1.0) >= 1)
    check("swept session is gone", "s1" not in eng._sessions)
    eng.drop_session("s2")
    check("drop_session releases the session", "s2" not in eng._sessions)
    eng.drop_session("s2")  # must not raise
    check("dropping twice is safe", True)


def test_server_validation() -> None:
    import server

    print("\nserver input validation")
    check("path traversal in avatar_id rejected",
          server._load_source({"avatar_id": "../../../etc/passwd"})[1] is not None)
    check("absolute-ish avatar_id rejected",
          server._load_source({"avatar_id": "a/b"})[1] is not None)
    check("empty init rejected", server._load_source({})[1] == "missing avatar_id or source_image")
    check("non-base64 source rejected",
          server._load_source({"source_image": "!!!not base64!!!"})[1] is not None)
    check("oversized source rejected",
          "too large" in (server._load_source(
              {"source_image": base64.b64encode(b"x" * (server._MAX_SOURCE_BYTES + 1)).decode()}
          )[1] or ""))
    check("undecodable image rejected",
          server._load_source({"source_image": base64.b64encode(b"not a jpeg").decode()})[1]
          is not None)

    import cv2
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    img, err = server._load_source({"source_image": base64.b64encode(buf).decode()})
    check("valid base64 JPEG accepted", err is None and img is not None and img.shape[:2] == (64, 64))

    p = server._params_from({"exp_amp": "1.4", "pose_gain": 0.5, "smooth": None, "junk": 1})
    check("params parsed from strings and floats", p == {"exp_amp": 1.4, "pose_gain": 0.5}, str(p))

    enc = server._encode_jpeg(np.zeros((32, 32, 3), dtype=np.uint8))
    check("jpeg encoder returns bytes", isinstance(enc, bytes) and enc[:2] == b"\xff\xd8")


if __name__ == "__main__":
    install_fakes()
    test_pipeline()
    test_server_validation()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("all checks passed")
