"""
LUCY AI — Face Swap Backend
FastAPI server: static face swap, live WebSocket swap, avatar catalogue,
and voice transform endpoint.
"""

import asyncio
import base64
import io
import json
import os
import contextlib
import pathlib
from typing import Optional

import cv2
import httpx
import numpy as np
import soundfile as sf
import librosa

from insightface.app import FaceAnalysis

from fastapi import (
    FastAPI, File, Form, HTTPException, Response,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────
_HERE          = pathlib.Path(__file__).parent
_AVATAR_CACHE  = _HERE / "avatars_cache"
_AVATAR_CACHE.mkdir(exist_ok=True)

_INSWAPPER_PATH = str(_HERE / "models" / "models" / "inswapper_128.onnx")
_CASCADE_PATH   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ─────────────────────────────────────────────────────────────────────────────
#  Avatar catalogue (18 entries)
# ─────────────────────────────────────────────────────────────────────────────
_AVATARS = [
    # ── Celebrity ────────────────────────────────────────────────────────────
    {"id": "cel_f_01", "name": "Emma",      "category": "Celebrity",
     "url": "https://randomuser.me/api/portraits/women/1.jpg"},
    {"id": "cel_f_02", "name": "Sophia",    "category": "Celebrity",
     "url": "https://randomuser.me/api/portraits/women/2.jpg"},
    {"id": "cel_m_01", "name": "Ethan",     "category": "Celebrity",
     "url": "https://randomuser.me/api/portraits/men/1.jpg"},
    {"id": "cel_m_02", "name": "Liam",      "category": "Celebrity",
     "url": "https://randomuser.me/api/portraits/men/2.jpg"},
    # ── AI Avatar ────────────────────────────────────────────────────────────
    {"id": "ai_f_01",  "name": "Nova",      "category": "AI Avatar",
     "url": "https://randomuser.me/api/portraits/women/10.jpg"},
    {"id": "ai_f_02",  "name": "Luna",      "category": "AI Avatar",
     "url": "https://randomuser.me/api/portraits/women/11.jpg"},
    {"id": "ai_m_01",  "name": "Atlas",     "category": "AI Avatar",
     "url": "https://randomuser.me/api/portraits/men/10.jpg"},
    {"id": "ai_m_02",  "name": "Orion",     "category": "AI Avatar",
     "url": "https://randomuser.me/api/portraits/men/11.jpg"},
    # ── Anime ────────────────────────────────────────────────────────────────
    {"id": "ani_f_01", "name": "Sakura",    "category": "Anime",
     "url": "https://randomuser.me/api/portraits/women/20.jpg"},
    {"id": "ani_f_02", "name": "Hana",      "category": "Anime",
     "url": "https://randomuser.me/api/portraits/women/21.jpg"},
    {"id": "ani_m_01", "name": "Ryu",       "category": "Anime",
     "url": "https://randomuser.me/api/portraits/men/20.jpg"},
    {"id": "ani_m_02", "name": "Kenji",     "category": "Anime",
     "url": "https://randomuser.me/api/portraits/men/21.jpg"},
    # ── Fantasy ──────────────────────────────────────────────────────────────
    {"id": "fan_f_01", "name": "Seraphina", "category": "Fantasy",
     "url": "https://randomuser.me/api/portraits/women/30.jpg"},
    {"id": "fan_f_02", "name": "Lyra",      "category": "Fantasy",
     "url": "https://randomuser.me/api/portraits/women/31.jpg"},
    {"id": "fan_m_01", "name": "Draven",    "category": "Fantasy",
     "url": "https://randomuser.me/api/portraits/men/30.jpg"},
    {"id": "fan_m_02", "name": "Caspian",   "category": "Fantasy",
     "url": "https://randomuser.me/api/portraits/men/31.jpg"},
    {"id": "fan_f_03", "name": "Elara",     "category": "Fantasy",
     "url": "https://randomuser.me/api/portraits/women/32.jpg"},
    {"id": "fan_m_03", "name": "Aldric",    "category": "Fantasy",
     "url": "https://randomuser.me/api/portraits/men/32.jpg"},
]
_AVATAR_MAP = {a["id"]: a for a in _AVATARS}

# ─────────────────────────────────────────────────────────────────────────────
#  Voice FX map
# ─────────────────────────────────────────────────────────────────────────────
# Key = prefix that the avatar_id starts with
_VOICE_FX: dict[str, dict] = {
    "cel_f": {"semitones":  2, "effect": None},
    "cel_m": {"semitones": -2, "effect": None},
    "ai_f":  {"semitones":  0, "effect": "robot"},
    "ai_m":  {"semitones": -1, "effect": "robot"},
    "ani_f": {"semitones":  8, "effect": None},
    "ani_m": {"semitones":  5, "effect": None},
    "fan_f": {"semitones":  3, "effect": "echo"},
    "fan_m": {"semitones": -5, "effect": None},
}

# ─────────────────────────────────────────────────────────────────────────────
#  Global model references
# ─────────────────────────────────────────────────────────────────────────────
_face_app:   Optional[FaceAnalysis] = None
_inswapper                          = None
_haar_cascade                       = None


def _load_models() -> None:
    """Load InsightFace FaceAnalysis, inswapper, and Haar cascade."""
    global _face_app, _inswapper, _haar_cascade

    # FaceAnalysis — buffalo_l with lower detection threshold for dark/angled faces
    _face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    _face_app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))

    # inswapper_128 ONNX
    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _inswapper = ort.InferenceSession(
        _INSWAPPER_PATH,
        sess_options=sess_opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Haar cascade (fallback detector)
    _haar_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

    print("[LUCY] Models loaded: FaceAnalysis buffalo_l + inswapper_128 + Haar cascade")


# ─────────────────────────────────────────────────────────────────────────────
#  Face detection helpers
# ─────────────────────────────────────────────────────────────────────────────
def _largest_face(faces):
    """Return the face with the largest bounding box area, or None."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _haar_detect(img_bgr: np.ndarray):
    """Haar cascade fallback — returns (x, y, w, h) or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # Largest by area
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    return faces[0]  # (x, y, w, h)


def _detect(img_bgr: np.ndarray):
    """Detect faces — InsightFace first, Haar fallback."""
    faces = _face_app.get(img_bgr)
    if faces:
        return _largest_face(faces), "insightface"
    haar = _haar_detect(img_bgr)
    return haar, "haar"


# ─────────────────────────────────────────────────────────────────────────────
#  Head / hair mask
# ─────────────────────────────────────────────────────────────────────────────
def _head_fg_mask(img: np.ndarray, face_bbox) -> np.ndarray:
    """
    Build a binary mask covering the head + hair region.

    1. Guaranteed oval base — never returns an all-zero mask.
    2. 2-pass GrabCut enhancement when the face bbox is reliable.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # ── Step 1: guaranteed oval base ────────────────────────────────────────
    x1, y1, x2, y2 = (int(v) for v in face_bbox[:4])
    fx, fy = (x1 + x2) // 2, (y1 + y2) // 2   # face centre
    fw, fh = x2 - x1, y2 - y1                    # face dimensions

    # Oval that covers face + hair above + slight width padding
    ell_cx, ell_cy = fx, fy - int(fh * 0.15)
    ell_rx = int(fw * 0.62)
    ell_ry = int(fh * 0.88)
    ell_cx = np.clip(ell_cx, ell_rx, w - ell_rx)
    ell_cy = np.clip(ell_cy, ell_ry, h - ell_ry)
    cv2.ellipse(mask, (ell_cx, ell_cy), (ell_rx, ell_ry), 0, 0, 360, 255, -1)

    # ── Step 2: GrabCut enhancement ─────────────────────────────────────────
    try:
        pad_x = int(fw * 1.2)
        pad_y = int(fh * 1.6)
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(w, x2 + pad_x)
        ry2 = min(h, y2 + int(fh * 0.4))
        roi_w, roi_h = rx2 - rx1, ry2 - ry1
        if roi_w > 10 and roi_h > 10:
            gc_mask = np.zeros((h, w), np.uint8)
            rect = (rx1, ry1, roi_w, roi_h)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(img, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            # Combine: keep oval base union GrabCut result
            mask = cv2.bitwise_or(mask, fg)
    except Exception:
        pass  # GrabCut can fail on degenerate inputs — oval base is sufficient

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
#  Style helpers
# ─────────────────────────────────────────────────────────────────────────────
def _is_cartoon(img: np.ndarray) -> bool:
    """Heuristic: cartoon/anime images have lower edge density."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    density = np.count_nonzero(edges) / edges.size
    return density < 0.04


def _color_transfer(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Transfer mean+std colour statistics from src to tgt in LAB space."""
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        sm, ss = src_lab[:, :, c].mean(), src_lab[:, :, c].std()
        tm, ts = tgt_lab[:, :, c].mean(), tgt_lab[:, :, c].std()
        if ts > 1e-6:
            tgt_lab[:, :, c] = (tgt_lab[:, :, c] - tm) * (ss / ts) + sm
    tgt_lab = np.clip(tgt_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(tgt_lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
#  Swap implementations
# ─────────────────────────────────────────────────────────────────────────────
def _run_inswapper(src_face, tgt_face, tgt_img: np.ndarray) -> np.ndarray:
    """Single inswapper pass — returns the modified target image."""
    # Build input dict from inswapper ONNX input names
    inp = _inswapper.get_inputs()
    # Standard inswapper_128 interface
    blob = tgt_face.normed_embedding.reshape(1, -1).astype(np.float32)
    # Warp target face patch to 128×128
    kps  = tgt_face.kps
    M    = cv2.estimateAffinePartial2D(kps, _get_arcface_template(128))[0]
    face_patch = cv2.warpAffine(tgt_img, M, (128, 128), flags=cv2.INTER_LINEAR)
    face_patch_f = (face_patch.astype(np.float32) / 255.0)
    face_patch_f = face_patch_f[:, :, ::-1]  # BGR→RGB
    face_patch_f = face_patch_f.transpose(2, 0, 1)[np.newaxis]

    result_patch = _inswapper.run(
        [inp[0].name],
        {inp[0].name: face_patch_f, inp[1].name: blob},
    )[0][0]

    result_patch = result_patch.transpose(1, 2, 0)[:, :, ::-1]  # RGB→BGR
    result_patch = (np.clip(result_patch, 0, 1) * 255).astype(np.uint8)

    # Warp result back to original image space
    M_inv = cv2.invertAffineTransform(M)
    swapped = tgt_img.copy()
    patch_back = cv2.warpAffine(result_patch, M_inv, (tgt_img.shape[1], tgt_img.shape[0]),
                                flags=cv2.INTER_LINEAR)
    # Build a tight elliptical blend mask centred on the face
    x1, y1, x2, y2 = (int(v) for v in tgt_face.bbox)
    blend_mask = np.zeros(tgt_img.shape[:2], np.uint8)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx, ry = max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)
    cv2.ellipse(blend_mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    blend_mask = cv2.GaussianBlur(blend_mask, (31, 31), 0)
    alpha = blend_mask[:, :, np.newaxis].astype(np.float32) / 255.0
    swapped = (patch_back.astype(np.float32) * alpha +
               swapped.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return swapped


_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _get_arcface_template(size: int) -> np.ndarray:
    scale = size / 112.0
    return _ARCFACE_DST * scale


def _swap_real(src_img: np.ndarray, tgt_img: np.ndarray) -> np.ndarray:
    """
    Real-face swap:
      1. inswapper on face region
      2. SeamlessClone head+hair oval back onto original target
    """
    src_faces = _face_app.get(src_img)
    tgt_faces = _face_app.get(tgt_img)
    src_face  = _largest_face(src_faces)
    tgt_face  = _largest_face(tgt_faces)
    if src_face is None or tgt_face is None:
        raise ValueError("Could not detect face in source or target image")

    # Step 1 — inswapper
    swapped = _run_inswapper(src_face, tgt_face, tgt_img)

    # Step 2 — hair oval seamlessClone to bring hair from swapped→tgt
    hair_mask = _head_fg_mask(swapped, tgt_face.bbox)
    # Centre of mass of mask for seamlessClone
    M = cv2.moments(hair_mask)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        x1, y1, x2, y2 = (int(v) for v in tgt_face.bbox)
        cX, cY = (x1 + x2) // 2, (y1 + y2) // 2
    try:
        result = cv2.seamlessClone(swapped, tgt_img, hair_mask, (cX, cY), cv2.NORMAL_CLONE)
    except cv2.error:
        result = swapped  # fallback: seamlessClone failed, keep inswapper result
    return result


def _swap_geometric(src_img: np.ndarray, tgt_img: np.ndarray) -> np.ndarray:
    """
    Cartoon/anime swap:
      - Strip background corners to avoid artefacts
      - Colour transfer for style adaptation
      - Single seamlessClone with full head+hair mask
    """
    src_faces = _face_app.get(src_img)
    tgt_faces = _face_app.get(tgt_img)
    src_face  = _largest_face(src_faces)
    tgt_face  = _largest_face(tgt_faces)
    if src_face is None or tgt_face is None:
        raise ValueError("Could not detect face in source or target image")

    # Colour-match source to target palette
    src_adapted = _color_transfer(tgt_img, src_img)

    # Resize source to match target face scale
    sx1, sy1, sx2, sy2 = (int(v) for v in src_face.bbox)
    tx1, ty1, tx2, ty2 = (int(v) for v in tgt_face.bbox)
    scale = max((tx2 - tx1) / max(sx2 - sx1, 1), (ty2 - ty1) / max(sy2 - sy1, 1))
    new_w = int(src_img.shape[1] * scale)
    new_h = int(src_img.shape[0] * scale)
    src_resized = cv2.resize(src_adapted, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Detect face in resized source
    src_r_faces = _face_app.get(src_resized)
    src_r_face  = _largest_face(src_r_faces)
    if src_r_face is None:
        src_r_face = src_face  # fallback

    # Build composite: paste resized source face region onto target
    canvas = tgt_img.copy()
    sr_x1, sr_y1, sr_x2, sr_y2 = (int(v) for v in src_r_face.bbox)
    patch_w, patch_h = sr_x2 - sr_x1, sr_y2 - sr_y1
    tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
    dst_x1 = max(0, tcx - patch_w // 2)
    dst_y1 = max(0, tcy - patch_h // 2)
    dst_x2 = min(canvas.shape[1], dst_x1 + patch_w)
    dst_y2 = min(canvas.shape[0], dst_y1 + patch_h)
    act_w, act_h = dst_x2 - dst_x1, dst_y2 - dst_y1
    patch = src_resized[sr_y1:sr_y1 + act_h, sr_x1:sr_x1 + act_w]
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = patch

    # Head+hair mask for seamlessClone
    face_bbox_for_mask = [dst_x1, dst_y1, dst_x2, dst_y2]
    head_mask = _head_fg_mask(canvas, face_bbox_for_mask)
    M = cv2.moments(head_mask)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = tcx, tcy
    try:
        result = cv2.seamlessClone(canvas, tgt_img, head_mask, (cX, cY), cv2.MIXED_CLONE)
    except cv2.error:
        result = canvas
    return result


def _swap(src_img: np.ndarray, tgt_img: np.ndarray) -> np.ndarray:
    """Dispatcher: cartoon → _swap_geometric, real → _swap_real."""
    if _is_cartoon(src_img):
        return _swap_geometric(src_img, tgt_img)
    return _swap_real(src_img, tgt_img)


def _swap_live(src_img: np.ndarray, tgt_img: np.ndarray) -> np.ndarray:
    """Dispatcher for live (lower latency) swap."""
    if _is_cartoon(src_img):
        return _swap_geometric(src_img, tgt_img)
    return _swap_real(src_img, tgt_img)


def _do_swap(src_img: np.ndarray, frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Fast swap for WebSocket live path.
    For real faces: inswapper only (no seamlessClone hair pass) for speed.
    For cartoon: full geometric swap.
    """
    if _is_cartoon(src_img):
        try:
            return _swap_geometric(src_img, frame)
        except Exception:
            return None

    src_faces = _face_app.get(src_img)
    tgt_faces = _face_app.get(frame)
    src_face  = _largest_face(src_faces)
    tgt_face  = _largest_face(tgt_faces)
    if src_face is None or tgt_face is None:
        return None
    try:
        return _run_inswapper(src_face, tgt_face, frame)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Voice transform helpers
# ─────────────────────────────────────────────────────────────────────────────
def _apply_robot(y: np.ndarray, sr: int) -> np.ndarray:
    """Amplitude modulation at 80 Hz — classic robot effect."""
    t = np.arange(len(y)) / sr
    carrier = np.sin(2 * np.pi * 80 * t)
    return (y * carrier).astype(np.float32)


def _apply_echo(y: np.ndarray, sr: int) -> np.ndarray:
    """Two-tap echo: 80 ms delay at 0.45 gain + 160 ms at 0.20 gain."""
    delay1 = int(0.080 * sr)  # 80 ms
    delay2 = int(0.160 * sr)  # 160 ms
    out = y.copy()
    if delay1 < len(y):
        out[delay1:] += 0.45 * y[: len(y) - delay1]
    if delay2 < len(y):
        out[delay2:] += 0.20 * y[: len(y) - delay2]
    # Normalise to avoid clipping
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out /= peak
    return out.astype(np.float32)


def _transform_voice(audio_bytes: bytes, avatar_id: str) -> bytes:
    """
    Load audio → pitch shift → optional effect → encode WAV → return bytes.
    Matches avatar_id prefix to _VOICE_FX table.
    """
    # Determine FX params — match longest prefix first
    fx_params = {"semitones": 0, "effect": None}
    for prefix in sorted(_VOICE_FX.keys(), key=len, reverse=True):
        if avatar_id.startswith(prefix):
            fx_params = _VOICE_FX[prefix]
            break

    # Decode audio from bytes (handles webm, ogg, mp3, wav, etc.)
    buf = io.BytesIO(audio_bytes)
    y, sr = librosa.load(buf, sr=None, mono=True)

    # Pitch shift
    semitones = fx_params["semitones"]
    if semitones != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

    # Effect
    effect = fx_params.get("effect")
    if effect == "robot":
        y = _apply_robot(y, sr)
    elif effect == "echo":
        y = _apply_echo(y, sr)

    # Encode to WAV in memory
    out_buf = io.BytesIO()
    sf.write(out_buf, y, sr, format="WAV", subtype="PCM_16")
    out_buf.seek(0)
    return out_buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI app — lifespan
# ─────────────────────────────────────────────────────────────────────────────
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup (runs once in the main process)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_models)
    yield
    # Cleanup (nothing to do for now)


app = FastAPI(title="LUCY AI Face Swap Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Utility: decode uploaded image bytes
# ─────────────────────────────────────────────────────────────────────────────
def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


def _encode_jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": _inswapper is not None}


@app.post("/face-swap")
async def face_swap(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
):
    """Static face swap — returns JPEG image."""
    src_bytes = await source.read()
    tgt_bytes = await target.read()
    src_img = _decode_image(src_bytes)
    tgt_img = _decode_image(tgt_bytes)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _swap, src_img, tgt_img)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Swap failed: {e}")

    return Response(content=_encode_jpeg(result), media_type="image/jpeg")


@app.get("/avatars")
async def list_avatars():
    """Return avatar catalogue."""
    avatars_out = [
        {
            "id":        a["id"],
            "name":      a["name"],
            "category":  a["category"],
            "image_url": f"/avatars/{a['id']}/image",
        }
        for a in _AVATARS
    ]
    return {"avatars": avatars_out}


@app.get("/avatars/{avatar_id}/image")
async def get_avatar_image(avatar_id: str):
    """Serve avatar image — cached on disk after first fetch."""
    if avatar_id not in _AVATAR_MAP:
        raise HTTPException(status_code=404, detail="Avatar not found")

    cache_path = _AVATAR_CACHE / f"{avatar_id}.jpg"
    if cache_path.exists():
        return FileResponse(str(cache_path), media_type="image/jpeg")

    url = _AVATAR_MAP[avatar_id]["url"]
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(url)
            r.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch avatar image: {e}")

    cache_path.write_bytes(r.content)
    return Response(content=r.content, media_type="image/jpeg")


@app.post("/face-swap/with-avatar")
async def face_swap_with_avatar(
    target:    UploadFile = File(...),
    avatar_id: str        = Form(...),
):
    """Swap target face with a catalogue avatar."""
    if avatar_id not in _AVATAR_MAP:
        raise HTTPException(status_code=404, detail="Avatar not found")

    cache_path = _AVATAR_CACHE / f"{avatar_id}.jpg"
    if not cache_path.exists():
        url = _AVATAR_MAP[avatar_id]["url"]
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
        cache_path.write_bytes(r.content)

    src_img = _decode_image(cache_path.read_bytes())
    tgt_img = _decode_image(await target.read())

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _swap, src_img, tgt_img)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Swap failed: {e}")

    return Response(content=_encode_jpeg(result), media_type="image/jpeg")


# ─────────────────────────────────────────────────────────────────────────────
#  Voice transform endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/voice/transform")
async def voice_transform(
    audio:     UploadFile = File(...),
    avatar_id: str        = Form(...),
):
    """
    Accept a browser audio recording (webm/ogg) and an avatar_id.
    Apply pitch shift + optional effect, return transformed WAV.
    """
    if avatar_id not in _AVATAR_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown avatar_id: {avatar_id}")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    loop = asyncio.get_event_loop()
    try:
        wav_bytes = await loop.run_in_executor(None, _transform_voice, audio_bytes, avatar_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice transform failed: {e}")

    return Response(content=wav_bytes, media_type="audio/wav")


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket — live face swap
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/live-swap")
async def ws_live_swap(ws: WebSocket):
    await ws.accept()

    src_img: Optional[np.ndarray] = None
    loop = asyncio.get_event_loop()

    try:
        async for raw in ws.iter_text():
            msg = json.loads(raw)

            if msg["type"] == "init":
                if "avatar_id" in msg:
                    # Load avatar from server cache — no image upload needed
                    av_id      = msg["avatar_id"]
                    cache_path = _AVATAR_CACHE / f"{av_id}.jpg"
                    if not cache_path.exists():
                        av = _AVATAR_MAP.get(av_id)
                        if av is None:
                            await ws.send_text(json.dumps({"type":"error","message":"Unknown avatar"}))
                            continue
                        async with httpx.AsyncClient(timeout=10) as client:
                            r = await client.get(av["url"])
                            cache_path.write_bytes(r.content)
                    src_img = await loop.run_in_executor(None, _decode_image, cache_path.read_bytes())
                else:
                    b64       = msg["source_image"]
                    img_bytes = base64.b64decode(b64)
                    src_img   = await loop.run_in_executor(None, _decode_image, img_bytes)
                await ws.send_text(json.dumps({"type": "ready"}))

            elif msg["type"] == "frame":
                if src_img is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "Send init first"}))
                    continue

                b64 = msg["image"]
                frame_bytes = base64.b64decode(b64)
                frame = await loop.run_in_executor(None, _decode_image, frame_bytes)

                result = await loop.run_in_executor(None, _do_swap, src_img, frame)
                if result is None:
                    await ws.send_text(json.dumps({"type": "no_face"}))
                else:
                    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    b64_out = base64.b64encode(buf).decode("ascii")
                    await ws.send_text(json.dumps({"type": "result", "image": b64_out}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        with contextlib.suppress(Exception):
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))


# ─────────────────────────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        workers=1,          # GPU models must stay in one process
    )
