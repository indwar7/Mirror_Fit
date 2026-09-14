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
from product import accounts, api, sessions
from product.accounts import AuthError, InsufficientCredits
from product.db import init as init_db
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

    # A live session costs a credit, so it needs an account. The credit
    # itself is taken later, when the GPU is actually granted.
    try:
        user = accounts.user_for_token(
            request.headers.get("Authorization", "")[7:] or request.cookies.get("token", "")
        )
    except AuthError as exc:
        raise web.HTTPUnauthorized(text=str(exc))
    if accounts.balance(user["id"]) < sessions.SESSION_COST:
        raise web.HTTPPaymentRequired(text="out of credits")

    # aiortc accepts a malformed offer without complaint and produces an
    # answer with no media at all. That connects, charges a credit, and
    # then streams nothing — so reject it here instead.
    sdp = body.get("sdp", "")
    if "m=video" not in sdp:
        raise web.HTTPBadRequest(text="offer carries no video track")

    sid = sessions.open_session(user["id"], body.get("garment"), body.get("fabric"))

    pc = RTCPeerConnection()
    _peers[sid] = pc

    @pc.on("connectionstatechange")
    async def on_state() -> None:
        log.info("session %s connection %s", sid, pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await close_session(sid, reason=pc.connectionState)

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

    # Charge only now. Someone who queued and left is not billed for a
    # stream they never saw. If the charge fails, hand the slot straight
    # back rather than holding a GPU for a session that cannot run.
    try:
        sessions.start_session(sid)
    except InsufficientCredits as exc:
        await close_session(sid)
        raise web.HTTPPaymentRequired(text=str(exc))

    # From here the credit is already spent, so any failure must give it
    # back rather than leave the user paying for a session that never ran.
    try:
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=body["sdp"], type=body["type"])
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception as exc:
        log.warning("session %s failed to negotiate: %s", sid, exc)
        sessions.refund(sid, reason="negotiation failed")
        await close_session(sid, reason="negotiation failed")
        raise web.HTTPBadRequest(text="could not negotiate a connection")

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
    await close_session(sid, reason="time limit")


async def close_session(sid: str, reason: str = "completed") -> None:
    pc = _peers.pop(sid, None)
    track = _tracks.pop(sid, None)
    started = _sessions.pop(sid, None)
    if pc is not None:
        await pc.close()
    if started is not None:
        st = track.stats() if track is not None else {}
        sessions.end_session(
            sid, frames=st.get("frames_out", 0), fps=st.get("fps"),
            backend=st.get("backend"), reason=reason,
        )
        # A session that produced nothing is our failure, not the user's.
        if st.get("frames_out", 0) == 0:
            sessions.refund(sid, reason="no frames delivered")
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
    init_db()
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    api.add_routes(app)
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
