"""WebRTC signaling and session lifecycle.

The browser POSTs an SDP offer to /offer, gets an answer back, and the
media flows over UDP from there. Everything else here exists to make
sure a session takes a GPU slot for at most SESSION_SECONDS and always
gives it back.
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import uuid

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

import config
from .queue_manager import SessionQueue
from .track import TryOnTrack

log = logging.getLogger(__name__)

CLIENT_DIR = pathlib.Path(__file__).resolve().parent.parent / "client"

# One backend per process: the model is loaded once and shared, because
# only one session runs at a time anyway.
_backend = None
_queue = SessionQueue()
_peers: dict[str, RTCPeerConnection] = {}
_tracks: dict[str, TryOnTrack] = {}


def backend():
    global _backend
    if _backend is None:
        from worker.backend import build_backend  # deferred: keeps imports cheap

        _backend = build_backend()
        _backend.warmup()
    return _backend


async def offer(request: web.Request) -> web.Response:
    body = await request.json()
    sid = body.get("sid") or uuid.uuid4().hex[:12]

    pc = RTCPeerConnection()
    _peers[sid] = pc

    @pc.on("connectionstatechange")
    async def on_state() -> None:
        log.info("session %s connection %s", sid, pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await close_session(sid)

    @pc.on("track")
    def on_track(incoming) -> None:
        if incoming.kind != "video":
            return
        log.info("session %s video track received", sid)
        session = _sessions[sid]
        out = TryOnTrack(incoming, backend(), session)
        _tracks[sid] = out
        pc.addTrack(out)

    # Wait for a GPU before answering. The browser is showing its local
    # 3D view and polling /status while this blocks.
    session = await _queue.acquire(sid)
    _sessions[sid] = session

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=body["sdp"], type=body["type"])
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Hand the slot back when the session runs out, even if the browser
    # never closes the connection.
    asyncio.create_task(_expire(sid))

    return web.json_response(
        {
            "sid": sid,
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_seconds": config.SESSION_SECONDS,
        }
    )


_sessions: dict = {}


async def _expire(sid: str) -> None:
    session = _sessions.get(sid)
    if session is None:
        return
    await asyncio.sleep(session.seconds_left)
    log.info("session %s reached its time limit", sid)
    await close_session(sid)


async def close_session(sid: str) -> None:
    pc = _peers.pop(sid, None)
    _tracks.pop(sid, None)
    _sessions.pop(sid, None)
    if pc is not None:
        await pc.close()
    await _queue.release(sid)


async def status(request: web.Request) -> web.Response:
    sid = request.query.get("sid", "")
    payload = _queue.status(sid)
    track = _tracks.get(sid)
    if track is not None:
        payload["stats"] = track.stats()
    session = _sessions.get(sid)
    if session is not None:
        payload["seconds_left"] = round(session.seconds_left, 1)
    return web.json_response(payload)


async def hangup(request: web.Request) -> web.Response:
    body = await request.json()
    await close_session(body.get("sid", ""))
    return web.json_response({"ok": True})


async def on_shutdown(app: web.Application) -> None:
    await asyncio.gather(*(close_session(sid) for sid in list(_peers)))
    if _backend is not None:
        _backend.close()


def build_app() -> web.Application:
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    app.router.add_post("/hangup", hangup)
    app.router.add_get("/status", status)
    app.router.add_static("/", CLIENT_DIR, show_index=True)
    return app


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    web.run_app(build_app(), host="0.0.0.0", port=8080)
