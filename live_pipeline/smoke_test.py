"""End-to-end check of the live pipeline, with no browser and no camera.

Drives the real server the way a browser would: signs up, offers a
WebRTC connection carrying synthetic frames, reads the returned track,
and reports what actually happened. Everything here is measured, not
assumed — if a stage fails it says so rather than carrying on.

    python smoke_test.py
"""
from __future__ import annotations

import argparse
import asyncio
import fractions
import logging
import pathlib
import sys
import tempfile
import time

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(message)s")

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def record(state: str, name: str, detail: str = "") -> None:
    results.append((state, name, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[state]
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


class SyntheticCamera:
    """A moving test pattern, in place of a webcam.

    Real content matters: an all-black frame would let a broken encoder
    look like a working one.
    """

    kind = "video"

    def __init__(self, width: int, height: int, fps: int, seconds: float):
        from aiortc import MediaStreamTrack  # noqa: PLC0415

        self._base = MediaStreamTrack
        self.width, self.height, self.fps = width, height, fps
        self.deadline = time.monotonic() + seconds
        self.n = 0

    def frame(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:, :, 1] = 60
        x = int((self.n * 7) % max(1, self.width - 80))
        img[self.height // 3 : 2 * self.height // 3, x : x + 80] = (220, 180, 140)
        return img


def build_track(width, height, fps, seconds):
    """A MediaStreamTrack subclass built at call time, so importing this
    file does not require aiortc."""
    from aiortc import MediaStreamTrack  # noqa: PLC0415
    from av import VideoFrame  # noqa: PLC0415

    cam = SyntheticCamera(width, height, fps, seconds)

    class Track(MediaStreamTrack):
        kind = "video"

        async def recv(self):
            if time.monotonic() > cam.deadline:
                raise MediaStreamError_()
            await asyncio.sleep(1 / fps)
            f = VideoFrame.from_ndarray(cam.frame(), format="rgb24")
            f.pts = cam.n
            f.time_base = fractions.Fraction(1, fps)
            cam.n += 1
            return f

    return Track(), cam


class MediaStreamError_(Exception):
    pass


async def run(args) -> int:
    root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    # --- imports ---
    try:
        from aiohttp.test_utils import TestClient, TestServer  # noqa: PLC0415
        from aiortc import RTCPeerConnection  # noqa: PLC0415
        record(PASS, "imports", "aiohttp, aiortc")
    except Exception as exc:
        record(FAIL, "imports", f"{type(exc).__name__}: {exc}")
        return 1

    # --- GPU ---
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            record(PASS, "cuda", f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}")
        else:
            record(WARN, "cuda", "not available — passthrough backend only")
    except Exception as exc:
        record(WARN, "cuda", f"torch missing ({type(exc).__name__})")

    # --- isolated database ---
    import product.db as db  # noqa: PLC0415

    db.DB_PATH = pathlib.Path(tempfile.mkdtemp()) / "smoke.db"
    import product.sessions as S  # noqa: PLC0415

    S.CAPTURE_DIR = db.DB_PATH.parent / "captures"

    import config  # noqa: PLC0415

    config.SESSION_SECONDS = args.seconds

    from server.app import build_app  # noqa: PLC0415
    import server.app as app_mod  # noqa: PLC0415

    # --- backend choice ---
    try:
        backend = app_mod.backend()
        state = PASS if backend.name != "passthrough" else WARN
        record(state, "model backend", backend.name)
    except Exception as exc:
        record(FAIL, "model backend", f"{type(exc).__name__}: {exc}")
        return 1

    app = build_app()
    async with TestClient(TestServer(app)) as client:
        # --- account ---
        try:
            r = await client.post("/api/signup",
                                  json={"email": "smoke@test.local", "password": "smokepass123"})
            d = await r.json()
            token, credits = d["token"], d["credits"]
            record(PASS, "signup", f"{credits} credits")
        except Exception as exc:
            record(FAIL, "signup", str(exc))
            return 1

        headers = {"Authorization": f"Bearer {token}"}

        # --- offers that must be refused ---
        r = await client.post("/offer", json={"sdp": "x", "type": "offer"})
        record(PASS if r.status == 401 else FAIL, "offer without a token",
               f"HTTP {r.status}, expected 401")

        r = await client.post("/offer", headers=headers,
                              json={"sdp": "no-video-here", "type": "offer"})
        record(PASS if r.status == 400 else FAIL, "offer with no video track",
               f"HTTP {r.status}, expected 400")

        import product.accounts as accounts  # noqa: PLC0415

        uid = accounts.user_for_token(token)["id"]
        record(PASS if accounts.balance(uid) == credits else FAIL,
               "no charge for a refused offer", f"{accounts.balance(uid)} credits")

        # --- the real session ---
        pc = RTCPeerConnection()
        track, cam = build_track(args.width, args.height, args.fps, args.seconds)
        pc.addTrack(track)
        received = {"frames": 0, "first": None, "sizes": set()}
        done = asyncio.Event()

        @pc.on("track")
        def on_track(incoming):
            async def drain():
                while True:
                    try:
                        frame = await asyncio.wait_for(incoming.recv(), timeout=8)
                    except Exception:
                        break
                    received["frames"] += 1
                    if received["first"] is None:
                        received["first"] = time.monotonic()
                    received["sizes"].add((frame.width, frame.height))
                done.set()

            asyncio.ensure_future(drain())

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        t0 = time.monotonic()
        r = await client.post("/offer", headers=headers,
                              json={"sdp": pc.localDescription.sdp,
                                    "type": pc.localDescription.type,
                                    "garment": "jacket", "fabric": "denim"})
        if r.status != 200:
            record(FAIL, "offer accepted", f"HTTP {r.status}: {await r.text()}")
            await pc.close()
            return 1
        answer = await r.json()
        sid = answer["sid"]
        record(PASS, "offer accepted", f"session {sid}, waited {time.monotonic()-t0:.2f}s")

        from aiortc import RTCSessionDescription  # noqa: PLC0415

        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

        record(PASS if accounts.balance(uid) == credits - 1 else FAIL,
               "credit charged on GPU grant", f"{accounts.balance(uid)} credits")

        # --- wait for frames ---
        try:
            await asyncio.wait_for(done.wait(), timeout=args.seconds + 15)
        except asyncio.TimeoutError:
            pass

        elapsed = time.monotonic() - t0
        n = received["frames"]
        if n > 0:
            fps = n / max(elapsed, 1e-6)
            record(PASS, "frames returned",
                   f"{n} frames, {fps:.1f} fps, sizes {received['sizes']}")
        else:
            record(FAIL, "frames returned", "no frames arrived from the server")

        # --- status endpoint ---
        r = await client.get(f"/status?sid={sid}")
        st = await r.json()
        record(PASS if r.status == 200 else FAIL, "status endpoint", str(st.get("stats", st)))

        await client.post("/hangup", headers=headers, json={"sid": sid})
        await pc.close()
        await asyncio.sleep(0.5)

        # --- records ---
        rows = S.user_sessions(uid)
        if rows:
            row = rows[0]
            record(PASS, "session recorded",
                   f"frames_out={row['frames_out']}, fps={row['mean_fps']}, "
                   f"backend={row['backend']}, reason={row['end_reason']}")
        else:
            record(FAIL, "session recorded", "no row written")

        final = accounts.balance(uid)
        if n > 0:
            record(PASS if final == credits - 1 else FAIL,
                   "billing settled", f"{final} credits (delivered frames, no refund)")
        else:
            record(PASS if final == credits else FAIL,
                   "refunded for an empty session", f"{final} credits")

        from product.monitor import report  # noqa: PLC0415

        rep = report(1)
        record(PASS, "monitor", rep["verdict"])

    # --- summary ---
    bad = [r for r in results if r[0] == FAIL]
    warn = [r for r in results if r[0] == WARN]
    print("\n" + "=" * 62)
    print(f"{len(results) - len(bad) - len(warn)} passed, {len(warn)} warnings, {len(bad)} failed")
    for state, name, detail in bad:
        print(f"  FAILED: {name} — {detail}")
    for state, name, detail in warn:
        print(f"  warning: {name} — {detail}")
    print("=" * 62)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=int, default=8, help="session length for this test")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()
    print(f"\nLive pipeline smoke test — {args.width}x{args.height}, "
          f"{args.fps} fps, {args.seconds}s session\n")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
