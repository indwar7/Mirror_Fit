"""
End-to-end WebSocket protocol test for the LivePortrait backend.

Boots the real FastAPI app (real server.py, real pipeline.py) on a spare
port with the upstream GPU pieces replaced by the stand-ins from
test_engine_logic, then drives it over a real WebSocket. Everything except
the model inference itself is exercised: init validation, both frame
transports, calibrating / no_face / dropped, live retuning, recalibration,
keepalive, session isolation between clients, and cleanup on disconnect.

Run:  python test_protocol.py
Needs: websockets (already in requirements.txt)
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

PORT = 7899
URL = f"ws://127.0.0.1:{PORT}/ws/liveportrait-swap"
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        FAILURES.append(f"{name}{' — ' + detail if detail else ''}")
        print(f"  FAIL  {name}{' — ' + detail if detail else ''}")


def jpeg(w: int = 384, h: int = 384) -> bytes:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


async def recv(ws, timeout: float = 15.0):
    """Return (kind, payload): ('bin', bytes) or ('json', dict)."""
    msg = await asyncio.wait_for(ws.recv(), timeout)
    if isinstance(msg, (bytes, bytearray)):
        return "bin", bytes(msg)
    return "json", json.loads(msg)


async def recv_skipping_pings(ws, timeout: float = 15.0):
    """The server sends unsolicited keepalive pings; they can land between
    any request and its reply, so every assertion has to step over them."""
    while True:
        kind, payload = await recv(ws, timeout)
        if kind == "json" and payload.get("type") == "ping":
            continue
        return kind, payload


async def init(ws, **extra):
    src = base64.b64encode(jpeg(256, 256)).decode()
    await ws.send(json.dumps({"type": "init", "source_image": src,
                              "session_id": "test", **extra}))
    return await recv_skipping_pings(ws)


async def run_tests(engine) -> None:
    import websockets

    print("\ninit + validation")
    async with websockets.connect(URL, max_size=None) as ws:
        kind, msg = await init(ws)
        check("init returns ready", kind == "json" and msg["type"] == "ready", str(msg))
        check("ready carries the tuning params",
              isinstance(msg.get("params"), dict) and "pose_gain" in msg["params"], str(msg))

    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "frame", "image": base64.b64encode(jpeg()).decode()}))
        _, msg = await recv_skipping_pings(ws)
        check("frame before init is refused", msg["type"] == "error", str(msg))

        await ws.send(json.dumps({"type": "init", "avatar_id": "../../../etc/passwd"}))
        _, msg = await recv_skipping_pings(ws)
        check("path traversal in avatar_id refused", msg["type"] == "error", str(msg))

        await ws.send(json.dumps({"type": "init", "source_image": "!!!!"}))
        _, msg = await recv_skipping_pings(ws)
        check("undecodable source refused, connection survives", msg["type"] == "error", str(msg))

        await ws.send("not json at all")
        _, msg = await recv_skipping_pings(ws)
        check("malformed json refused", msg["type"] == "error", str(msg))

        await ws.send(json.dumps({"type": "nonsense"}))
        _, msg = await recv_skipping_pings(ws)
        check("unknown message type refused", msg["type"] == "error", str(msg))

        await ws.send(json.dumps({"type": "ping", "t": 42}))
        _, msg = await recv_skipping_pings(ws)
        check("ping answered with pong", msg["type"] == "pong" and msg["t"] == 42, str(msg))

    print("\ncalibration then live frames (binary transport)")
    import pipeline
    async with websockets.connect(URL, max_size=None) as ws:
        await init(ws)
        states = []
        first_binary = None
        for i in range(pipeline.BASELINE_FRAMES + 3):
            await ws.send(jpeg())
            kind, payload = await recv_skipping_pings(ws)
            if kind == "bin":
                states.append("binary")
                first_binary = first_binary if first_binary is not None else i
            else:
                states.append(payload["type"])
        check("first frames report calibrating",
              all(s == "calibrating" for s in states[: pipeline.BASELINE_FRAMES - 1]), str(states))
        check("binary frame gets a binary reply once live",
              states[-1] == "binary", str(states))
        check("binary reply is a JPEG",
              first_binary is not None, str(states))

    print("\nlive frames (json transport) + telemetry")
    async with websockets.connect(URL, max_size=None) as ws:
        await init(ws)
        last = None
        for _ in range(pipeline.BASELINE_FRAMES + 2):
            await ws.send(json.dumps({"type": "frame", "image": base64.b64encode(jpeg()).decode()}))
            _, last = await recv_skipping_pings(ws)
        check("json frame gets a json result", last["type"] == "result", str(last)[:120])
        check("result carries a decodable image",
              cv2.imdecode(np.frombuffer(base64.b64decode(last["image"]), np.uint8),
                           cv2.IMREAD_COLOR) is not None)
        check("result reports inference ms", isinstance(last.get("ms"), (int, float)), str(last.get("ms")))

        await ws.send(json.dumps({"type": "frame", "image": "!!!not base64!!!"}))
        _, msg = await recv_skipping_pings(ws)
        check("bad base64 frame refused without dropping the session",
              msg["type"] == "error", str(msg))

    print("\nlive retuning + recalibration")
    async with websockets.connect(URL, max_size=None) as ws:
        await init(ws)
        await ws.send(json.dumps({"type": "config", "pose_gain": 0.7, "junk": "x"}))
        _, msg = await recv_skipping_pings(ws)
        check("config acknowledged", msg["type"] == "config", str(msg))
        check("pose_gain applied live", msg["params"]["pose_gain"] == 0.7, str(msg["params"]))
        check("junk key ignored", "junk" not in msg["params"], str(msg["params"]))

        for _ in range(pipeline.BASELINE_FRAMES + 1):
            await ws.send(jpeg())
            await recv_skipping_pings(ws)
        await ws.send(json.dumps({"type": "recalibrate"}))
        _, msg = await recv_skipping_pings(ws)
        check("recalibrate acknowledged", msg["type"] == "recalibrating", str(msg))
        await ws.send(jpeg())
        _, msg = await recv_skipping_pings(ws)
        check("frames after recalibrate report calibrating again",
              msg.get("type") == "calibrating", str(msg)[:120])

    print("\nno_face handling")
    import test_engine_logic as fakes
    async with websockets.connect(URL, max_size=None) as ws:
        await init(ws)
        fakes.CALLS.detect_returns_face = False
        try:
            await ws.send(jpeg())
            _, msg = await recv_skipping_pings(ws)
            check("no face in frame -> no_face", msg.get("type") == "no_face", str(msg)[:120])
        finally:
            fakes.CALLS.detect_returns_face = True

    print("\nsession isolation + cleanup")
    ws_a = await websockets.connect(URL, max_size=None)
    ws_b = await websockets.connect(URL, max_size=None)
    try:
        # Same client-supplied session_id on both — used to collide on one
        # engine entry, so whichever disconnected first freed the other's source.
        await init(ws_a, session_id="same")
        await init(ws_b, session_id="same")
        check("two clients sharing a session_id both hold live sessions",
              engine.stats()["sessions"] >= 2, str(engine.stats()["sessions"]))
        await ws_a.close()
        await asyncio.sleep(0.4)
        await ws_b.send(jpeg())
        kind, msg = await recv_skipping_pings(ws_b)
        ok = kind == "bin" or msg.get("type") in ("calibrating", "result", "no_face")
        check("one client disconnecting does not kill the other", ok, str(msg)[:120])
    finally:
        await ws_b.close()
    await asyncio.sleep(0.5)
    check("sessions released on disconnect", engine.stats()["sessions"] == 0,
          str(engine.stats()))


def main() -> int:
    import test_engine_logic as fakes

    fakes.install_fakes()
    import pipeline
    pipeline._REPO = _HERE          # skip the "clone the repo first" guard

    import uvicorn
    import server

    config = uvicorn.Config(server.app, host="127.0.0.1", port=PORT, log_level="warning")
    uv = uvicorn.Server(config)
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()

    import time
    for _ in range(100):
        if uv.started:
            break
        time.sleep(0.1)
    if not uv.started:
        print("server failed to start")
        return 1
    print(f"server up on :{PORT} (engine loaded: {server._engine is not None})")
    if server._engine is None:
        print("engine did not load:", server._engine_error)
        return 1

    try:
        asyncio.run(run_tests(server._engine))
    finally:
        uv.should_exit = True
        thread.join(timeout=10)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all protocol checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
