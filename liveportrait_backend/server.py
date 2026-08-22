"""
LUCY LivePortrait server — port 7862.

Runs in its own conda env (Python 3.11). Shares the same WebSocket
protocol as instantid_backend so the demo client can switch backends
by changing one constant.

WS protocol (/ws/liveportrait-swap):
    Client -> server (text JSON):
        {"type": "init", "avatar_id": "gen_f_01", "session_id": "...",
         "pose_gain": 0.0, "exp_amp": 1.1, "smooth": 0.7}
        {"type": "init", "source_image": "<base64 JPEG/PNG>", ...}
        {"type": "frame", "image": "<base64 JPEG>"}
        {"type": "config", "pose_gain": 0.7}      # live tuning, no restart
        {"type": "recalibrate"}                   # re-measure neutral face
        {"type": "ping"} / {"type": "pong"}
    Client -> server (binary):
        raw JPEG bytes — same as a `frame` message with a third less
        bandwidth than base64, and the reply comes back binary too.
    Server -> client:
        {"type": "ready", "session_id": ..., "params": {...}}
        {"type": "result", "image": "<base64 JPEG>", "ms": 41}
        binary JPEG bytes                         # reply to a binary frame
        {"type": "calibrating"}                   # measuring neutral face
        {"type": "no_face"} / {"type": "dropped"} / {"type": "error"}
        {"type": "ping"}                          # server keepalive

Avatars read from face_swap_backend/avatars_cache/<id>.jpg so all
three backends (face_swap, instantid, liveportrait) share the same set.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_AVATAR_CACHE = _HERE.parent / "face_swap_backend" / "avatars_cache"

# Ids arrive from the browser and are used to build a filesystem path, so
# they are whitelisted rather than escaped.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Cap on a base64 source image (8 MB decoded) — a stray upload should not
# take the process down.
_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_FRAME_BYTES = 4 * 1024 * 1024
# Per-frame logging is a debug aid, not steady-state behaviour: at 20 fps
# it writes ~72k lines an hour and computes min/max over the full frame.
_VERBOSE = os.environ.get("LP_VERBOSE", "").lower() in ("1", "true", "yes")
_JPEG_QUALITY = int(os.environ.get("LP_JPEG_QUALITY", "85"))
# Sessions whose client vanished without closing the socket.
_SWEEP_EVERY = 60.0
_SESSION_TTL = 300.0

_engine = None
_engine_error: Optional[str] = None


def _get_engine():
    global _engine, _engine_error
    if _engine is None:
        from pipeline import LivePortraitEngine
        log.info("[LP] cold-loading engine (this takes ~15-30 s)")
        try:
            _engine = LivePortraitEngine()
        except Exception as e:
            _engine_error = str(e)
            raise
        _engine_error = None
        log.info("[LP] engine ready for traffic")
    return _engine


async def _sweeper():
    """Drop abandoned sessions so their appearance features don't camp in
    VRAM until the next restart."""
    while True:
        await asyncio.sleep(_SWEEP_EVERY)
        if _engine is None:
            continue
        with contextlib.suppress(Exception):
            await asyncio.get_running_loop().run_in_executor(
                None, _engine.sweep, _SESSION_TTL
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _get_engine)
    except Exception as e:
        # Stay up so /health can report *why* the backend is dead instead
        # of the operator seeing a refused connection with no explanation.
        log.exception("[LP] engine load failed: %s", e)
    sweeper = asyncio.create_task(_sweeper())
    try:
        yield
    finally:
        sweeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper


app = FastAPI(title="LUCY LivePortrait Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health():
    return {
        "status": "ok" if _engine is not None else "loading",
        "engine_loaded": _engine is not None,
        "error": _engine_error,
        **({"stats": _engine.stats()} if _engine is not None else {}),
    }


@app.get("/avatars")
def avatars():
    """Which preset ids `init` will accept — saves the operator guessing
    when an avatar_id 404s."""
    if not _AVATAR_CACHE.exists():
        return {"avatars": [], "cache": str(_AVATAR_CACHE), "exists": False}
    ids = sorted(p.stem for p in _AVATAR_CACHE.glob("*.jpg"))
    return {"avatars": ids, "cache": str(_AVATAR_CACHE), "exists": True}


async def _ws_keepalive(ws: WebSocket, every: float = 20.0):
    try:
        while True:
            await asyncio.sleep(every)
            try:
                await ws.send_text(json.dumps({"type": "ping", "t": int(time.time() * 1000)}))
            except Exception:
                return
    except asyncio.CancelledError:
        return


def _decode_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_jpeg(img: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _params_from(msg: dict) -> dict:
    """Pull the tuning knobs out of an init/config message."""
    out = {}
    for k in ("exp_amp", "pose_gain", "smooth"):
        if k in msg:
            try:
                out[k] = float(msg[k])
            except (TypeError, ValueError):
                continue
    return out


def _load_source(msg: dict) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Resolve an init message to a BGR source image. Returns (image, error)."""
    avatar_id = msg.get("avatar_id")
    source_b64 = msg.get("source_image")

    if avatar_id:
        if not _ID_RE.match(str(avatar_id)):
            return None, f"invalid avatar_id: {avatar_id!r}"
        path = _AVATAR_CACHE / f"{avatar_id}.jpg"
        if not path.exists():
            return None, f"avatar not generated: {avatar_id}"
        img = cv2.imread(str(path))
        return (img, None) if img is not None else (None, "avatar image unreadable")

    if source_b64:
        # Decode in memory. The previous version wrote a temp file named
        # from the client-supplied session_id, which put a client string
        # straight into a filesystem path.
        try:
            raw = base64.b64decode(source_b64, validate=False)
        except (binascii.Error, ValueError) as e:
            return None, f"bad source_image: {e}"
        if len(raw) > _MAX_SOURCE_BYTES:
            return None, "source image too large (max 8 MB)"
        img = _decode_image(raw)
        return (img, None) if img is not None else (None, "source image not decodable")

    return None, "missing avatar_id or source_image"


@app.websocket("/ws/liveportrait-swap")
async def ws_liveportrait_swap(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    # The engine key is per-connection, never the client's session_id:
    # two browsers picking the same avatar used to collide on one cache
    # entry, and whichever disconnected first freed the other's source.
    engine_key = uuid.uuid4().hex
    client_label = "?"
    source_ready = False
    processing = False
    stats = {"frames": 0, "dropped": 0, "no_face": 0, "ms": 0.0, "t0": time.time()}

    async def send(obj: dict):
        await ws.send_text(json.dumps(obj))

    async def handle_frame(frame_bytes: bytes, binary: bool):
        """Shared by the JSON and binary transports."""
        nonlocal processing
        if not source_ready:
            await send({"type": "error", "message": "send init first"})
            return
        if processing:
            # Frame-drop gate so the client can outpace us without queuing.
            stats["dropped"] += 1
            await send({"type": "dropped"})
            return
        if len(frame_bytes) > _MAX_FRAME_BYTES:
            await send({"type": "error", "message": "frame too large"})
            return

        processing = True
        try:
            frame = await loop.run_in_executor(None, _decode_image, frame_bytes)
            if frame is None:
                await send({"type": "error", "message": "bad jpeg"})
                return

            engine = _get_engine()
            res = await loop.run_in_executor(None, engine.drive, engine_key, frame)

            if res.state == "no_face":
                stats["no_face"] += 1
                await send({"type": "no_face"})
                return

            jpeg = await loop.run_in_executor(None, _encode_jpeg, res.image)
            if jpeg is None:
                await send({"type": "error", "message": "encode failed"})
                return

            stats["frames"] += 1
            stats["ms"] += res.infer_ms
            if res.state == "calibrating":
                # Still measuring the neutral face — the frame is the
                # untouched portrait. Told apart from `live` so the client
                # can show "hold still" instead of silently looking frozen.
                await send({
                    "type": "calibrating",
                    "image": base64.b64encode(jpeg).decode(),
                })
                return

            if binary:
                await ws.send_bytes(jpeg)
            else:
                await send({
                    "type": "result",
                    "image": base64.b64encode(jpeg).decode(),
                    "ms": round(res.infer_ms),
                })

            if _VERBOSE:
                log.info("[LP] drive ok (%.0f ms) %s", res.infer_ms, res.image.shape)
            elif stats["frames"] % 100 == 0:
                elapsed = max(time.time() - stats["t0"], 1e-6)
                log.info(
                    "[LP] %s: %d frames, %.1f fps out, %.0f ms/frame infer, "
                    "%d dropped, %d no_face",
                    client_label, stats["frames"], stats["frames"] / elapsed,
                    stats["ms"] / stats["frames"], stats["dropped"], stats["no_face"],
                )
        finally:
            processing = False

    keepalive_task = asyncio.create_task(_ws_keepalive(ws))

    try:
        while True:
            packet = await ws.receive()
            if packet["type"] == "websocket.disconnect":
                break

            # ── Binary transport: the payload *is* the JPEG ─────────────
            if packet.get("bytes") is not None:
                await handle_frame(packet["bytes"], binary=True)
                continue

            raw = packet.get("text")
            if raw is None:
                continue
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                await send({"type": "error", "message": "bad json"})
                continue
            if not isinstance(msg, dict):
                await send({"type": "error", "message": "expected a json object"})
                continue

            mtype = msg.get("type")

            if mtype in ("ping", "pong"):
                if mtype == "ping":
                    with contextlib.suppress(Exception):
                        await send({"type": "pong", "t": msg.get("t")})
                continue

            if mtype == "init":
                client_label = str(msg.get("session_id") or msg.get("avatar_id") or "anon")[:64]
                src_bgr, err = _load_source(msg)
                if err:
                    await send({"type": "error", "message": err})
                    continue
                try:
                    engine = _get_engine()
                except Exception as e:
                    await send({"type": "error", "message": f"engine unavailable: {e}"})
                    continue
                try:
                    ok = await loop.run_in_executor(
                        None,
                        lambda: engine.prepare_source(engine_key, src_bgr, **_params_from(msg)),
                    )
                except Exception as e:
                    log.exception("[LP] init failed")
                    await send({"type": "error", "message": str(e)})
                    continue
                if not ok:
                    await send({"type": "error", "message": "no face in source image"})
                    continue
                source_ready = True
                stats.update(frames=0, dropped=0, no_face=0, ms=0.0, t0=time.time())
                log.info("[LP] session %s ready (client=%s)", engine_key[:8], client_label)
                await send({
                    "type": "ready",
                    "session_id": msg.get("session_id") or client_label,
                    "params": engine.configure(engine_key),
                })
                continue

            if mtype == "frame":
                image_b64 = msg.get("image")
                if not isinstance(image_b64, str):
                    await send({"type": "error", "message": "frame missing image"})
                    continue
                try:
                    frame_bytes = base64.b64decode(image_b64, validate=False)
                except (binascii.Error, ValueError):
                    await send({"type": "error", "message": "bad base64"})
                    continue
                await handle_frame(frame_bytes, binary=False)
                continue

            if mtype == "config":
                if not source_ready:
                    await send({"type": "error", "message": "send init first"})
                    continue
                params = _get_engine().configure(engine_key, **_params_from(msg))
                await send({"type": "config", "params": params})
                continue

            if mtype == "recalibrate":
                if not source_ready:
                    await send({"type": "error", "message": "send init first"})
                    continue
                _get_engine().recalibrate(engine_key)
                await send({"type": "recalibrating"})
                continue

            await send({"type": "error", "message": f"unknown message type: {mtype!r}"})

    except WebSocketDisconnect:
        pass
    except RuntimeError:
        # Raised by receive() once the socket is already closed.
        pass
    except Exception as e:
        log.exception("[LP] ws error")
        with contextlib.suppress(Exception):
            await send({"type": "error", "message": str(e)})
    finally:
        keepalive_task.cancel()
        with contextlib.suppress(Exception):
            await keepalive_task
        if _engine is not None:
            _engine.drop_session(engine_key)
        if stats["frames"]:
            elapsed = max(time.time() - stats["t0"], 1e-6)
            log.info(
                "[LP] session %s closed (client=%s): %d frames, %.1f fps, "
                "%.0f ms/frame, %d dropped, %d no_face",
                engine_key[:8], client_label, stats["frames"],
                stats["frames"] / elapsed, stats["ms"] / stats["frames"],
                stats["dropped"], stats["no_face"],
            )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("LP_PORT", "7862")),
        reload=False,
        workers=1,
    )
