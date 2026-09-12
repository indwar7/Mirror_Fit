"""The video track that sits between the camera and the browser.

Frames arrive from the client's WebRTC track, go through the model, and
leave as a new track the browser plays. The model runs in a thread: it
is synchronous and GPU-bound, and blocking the event loop here would
stall the WebRTC connection itself.
"""
from __future__ import annotations

import asyncio
import logging
import time

from av import VideoFrame
from aiortc import MediaStreamTrack

import config

log = logging.getLogger(__name__)


class TryOnTrack(MediaStreamTrack):
    """Reads `source`, returns model output at the same timestamps."""

    kind = "video"

    def __init__(self, source: MediaStreamTrack, backend, session) -> None:
        super().__init__()
        self.source = source
        self.backend = backend
        self.session = session
        self.frames_in = 0
        self.frames_out = 0
        self.dropped = 0
        self._busy = False
        self._last: VideoFrame | None = None
        self._latencies: list[float] = []

    async def recv(self) -> VideoFrame:
        frame = await self.source.recv()
        self.frames_in += 1

        if self.session.expired:
            # Session is over. Keep the track alive with the last good
            # frame; the server closes the connection separately.
            return self._last or frame

        # One frame in the model at a time. A 5090 at 12-16 fps still
        # takes ~70ms per frame, and the camera may push faster than
        # that — so drop rather than queue, which would only add latency.
        if self._busy:
            self.dropped += 1
            return self._last or frame

        self._busy = True
        t0 = time.monotonic()
        try:
            img = frame.to_ndarray(format="rgb24")
            out = await asyncio.to_thread(self.backend.process, img)
            new = VideoFrame.from_ndarray(out, format="rgb24")
        except Exception:
            log.exception("frame %d failed, passing the composite through", self.frames_in)
            return self._last or frame
        finally:
            self._busy = False

        # Reusing the source timing keeps A/V sync and pacing intact.
        new.pts, new.time_base = frame.pts, frame.time_base
        self._last = new
        self.frames_out += 1
        self._latencies.append(time.monotonic() - t0)
        return new

    def stats(self) -> dict:
        lat = self._latencies[-config.TARGET_FPS * 5 :] or [0.0]
        mean = sum(lat) / len(lat)
        return {
            "frames_in": self.frames_in,
            "frames_out": self.frames_out,
            "dropped": self.dropped,
            "mean_latency_ms": round(mean * 1000, 1),
            "fps": round(1 / mean, 1) if mean else 0.0,
            "backend": self.backend.name,
        }
