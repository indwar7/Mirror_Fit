"""
LUCY LivePortrait server — port 7862.

Runs in its own conda env (Python 3.11). Shares the same WebSocket
protocol as instantid_backend so the demo client can switch backends
by changing one constant.

WS protocol (matches /ws/instantid-swap):
    Client -> server: {"type": "init", "avatar_id": "gen_f_01", "session_id": "..."}
                      {"type": "frame", "image": "<base64 JPEG>"}
                      {"type": "ping"} / {"type": "pong"}
    Server -> client: {"type": "ready"} (after source preparation)
                      {"type": "result", "image": "<base64 JPEG>"}
                      {"type": "no_face"} / {"type": "dropped"} / {"type": "error"}
                      {"type": "ping"} (server keepalive)

Avatars read from face_swap_backend/avatars_cache/<id>.jpg so all
three backends (face_swap, instantid, liveportrait) share the same set.
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

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from pipeline import LivePortraitEngine
        log.info("[LP] cold-loading engine (this takes ~15-30 s)")
        _engine = LivePortraitEngine()
        log.info("[LP] engine ready for traffic")
    return _engine


app = FastAPI(title="LUCY LivePortrait Backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok", "engine_loaded": _engine is not None}


@app.on_event("startup")
async def _startup():
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _get_engine)
    except Exception as e:
        log.exception("[LP] engine load failed: %s", e)


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


@app.websocket("/ws/liveportrait-swap")
async def ws_liveportrait_swap(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()

    session_id: Optional[str] = None
    source_ready = False
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
                source_image_b64 = msg.get("source_image")
                session_id = msg.get("session_id") or avatar_id or "upload"

                source_path: Optional[str] = None
                tmp_path: Optional[Path] = None

                if avatar_id:
                    p = _AVATAR_CACHE / f"{avatar_id}.jpg"
                    if not p.exists():
                        await send({"type": "error",
                                    "message": f"avatar not generated: {avatar_id}"})
                        continue
                    source_path = str(p)
                elif source_image_b64:
                    try:
                        img_bytes = base64.b64decode(source_image_b64)
                        tmp_path = _HERE / f"_upload_{session_id}.jpg"
                        tmp_path.write_bytes(img_bytes)
                        source_path = str(tmp_path)
                    except Exception as e:
                        await send({"type": "error",
                                    "message": f"bad source_image: {e}"})
                        continue
                else:
                    await send({"type": "error",
                                "message": "missing avatar_id or source_image"})
                    continue

                try:
                    engine = _get_engine()
                    src_bgr = cv2.imread(source_path)
                    if src_bgr is None:
                        await send({"type": "error",
                                    "message": "source image not readable"})
                        continue
                    ok = await loop.run_in_executor(
                        None, engine.prepare_source, session_id, src_bgr
                    )
                    if not ok:
                        await send({"type": "error",
                                    "message": "no face in source image"})
                        continue
                    source_ready = True
                    await send({"type": "ready", "session_id": session_id})
                except Exception as e:
                    log.exception("[LP] init failed")
                    await send({"type": "error", "message": str(e)})
                finally:
                    if tmp_path is not None and tmp_path.exists():
                        with contextlib.suppress(Exception):
                            tmp_path.unlink()
                continue

            if mtype == "frame":
                if not source_ready:
                    await send({"type": "error", "message": "send init first"})
                    continue
                if processing:
                    # Frame-drop gate so client can outpace us without queuing.
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
                        None, engine.drive, session_id, frame
                    )
                    if result is None:
                        await send({"type": "no_face"})
                    else:
                        ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
        log.exception("[LP] ws error")
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
        port=7862,
        reload=False,
        workers=1,
    )
