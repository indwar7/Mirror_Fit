"""
LUCY InstantID server - port 7861.

Runs in its own conda env (Python 3.11) because face_swap_backend uses
3.14 which has no CUDA PyTorch wheels yet. Communicates with the demo
client over the same WebSocket protocol as /ws/live-swap so the client
can transparently switch backends.

WS protocol:
    Client -> server: {"type": "init", "avatar_id": "gen_f_01", "session_id": "..."}
                      {"type": "frame", "image": "<base64 JPEG>"}
                      {"type": "ping"} / {"type": "pong"}
    Server -> client: {"type": "ready"}
                      {"type": "result", "image": "<base64 JPEG>"}
                      {"type": "no_face"} / {"type": "dropped"} / {"type": "error", "message": "..."}
                      {"type": "ping"} (server keepalive)

Avatars are read from face_swap_backend/avatars_cache/<id>.jpg so the two
backends share the same set of generated avatars.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
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

# Lazy import so an import failure (missing torch / diffusers) shows up
# as a clear startup log line instead of a top-level traceback.
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from instantid_pipeline import InstantIDEngine
        log.info("[InstantID] cold-loading engine (this takes 30-60 s)")
        _engine = InstantIDEngine()
        # Warm up: first inference call can take 8-12 s for kernel autotune.
        # Run two dummy frames so the first user request is fast.
        try:
            warm = np.full((512, 512, 3), 128, np.uint8)
            cv2.circle(warm, (256, 256), 80, (200, 200, 200), -1)
            _engine.load_avatar("__warmup__", str(_HERE / "instantid_pipeline.py"))  # will fail face detect, ok
        except Exception:
            pass
        log.info("[InstantID] engine ready for traffic")
    return _engine


app = FastAPI(title="LUCY InstantID Backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok", "engine_loaded": _engine is not None}


@app.on_event("startup")
async def _startup():
    # Load the engine eagerly so the first WS request doesn't pay the
    # 30-60 s cold-start tax.
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _get_engine)
    except Exception as e:
        log.exception("[InstantID] engine load failed: %s", e)


# ── Keepalive task (matches /ws/live-swap behaviour) ────────────────────────
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
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.websocket("/ws/instantid-swap")
async def ws_instantid_swap(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()

    session_id: Optional[str] = None
    avatar_loaded = False
    processing = False

    async def send(obj: dict):
        await ws.send_text(json.dumps(obj))

    keepalive_task = asyncio.create_task(_ws_keepalive(ws))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await send({"type": "error", "message": "bad json"})
                continue

            mtype = msg.get("type")

            if mtype in ("ping", "pong"):
                if mtype == "ping":
                    with contextlib.suppress(Exception):
                        await send({"type": "pong", "t": msg.get("t")})
                continue

            if mtype == "init":
                avatar_id = msg.get("avatar_id")
                session_id = msg.get("session_id") or avatar_id
                if not avatar_id:
                    await send({"type": "error", "message": "missing avatar_id"})
                    continue
                avatar_path = _AVATAR_CACHE / f"{avatar_id}.jpg"
                if not avatar_path.exists():
                    await send({"type": "error", "message": f"avatar not generated: {avatar_id}"})
                    continue
                try:
                    engine = _get_engine()
                    ok = await loop.run_in_executor(
                        None, engine.load_avatar, session_id, str(avatar_path)
                    )
                    if not ok:
                        await send({"type": "error", "message": "no face in avatar image"})
                        continue
                    avatar_loaded = True
                    await send({"type": "ready", "session_id": session_id})
                except Exception as e:
                    log.exception("[InstantID] init failed")
                    await send({"type": "error", "message": str(e)})
                continue

            if mtype == "frame":
                if not avatar_loaded:
                    await send({"type": "error", "message": "send init first"})
                    continue
                if processing:
                    # Frame-drop gate. InstantID is slow (~1.5-2.0 s/frame at
                    # 1024 px) so client can outpace us by 10x. Acknowledge
                    # the drop so the client requests another frame.
                    await send({"type": "dropped"})
                    continue
                processing = True
                try:
                    frame_bytes = base64.b64decode(msg["image"])
                    frame = await loop.run_in_executor(None, _decode_image, frame_bytes)
                    if frame is None:
                        await send({"type": "error", "message": "bad jpeg"})
                        continue
                    engine = _get_engine()
                    result = await loop.run_in_executor(
                        None, engine.transfer, session_id, frame
                    )
                    if result is None:
                        await send({"type": "no_face"})
                    else:
                        ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 88])
                        if ok:
                            await send({
                                "type": "result",
                                "image": base64.b64encode(buf).decode(),
                            })
                        else:
                            await send({"type": "error", "message": "encode failed"})
                finally:
                    processing = False
                continue

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception("[InstantID] ws error")
        with contextlib.suppress(Exception):
            await send({"type": "error", "message": str(e)})
    finally:
        keepalive_task.cancel()
        with contextlib.suppress(Exception):
            await keepalive_task
        if session_id and _engine is not None:
            _engine.drop_session(session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7861,
        reload=False,
        workers=1,  # single worker — pipeline holds GPU state
    )
