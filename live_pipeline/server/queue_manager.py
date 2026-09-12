"""Who gets the GPU, and for how long.

The plan is strict about this: one GPU serves one live session at a
time. Everyone else waits in line, sees their position, and watches the
local 3D view meanwhile — never a blank screen.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import config

log = logging.getLogger(__name__)


@dataclass
class Session:
    sid: str
    queued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None

    @property
    def seconds_left(self) -> float:
        if self.started_at is None:
            return float(config.SESSION_SECONDS)
        spent = time.monotonic() - self.started_at
        return max(0.0, config.SESSION_SECONDS - spent)

    @property
    def expired(self) -> bool:
        return self.started_at is not None and self.seconds_left <= 0


class SessionQueue:
    """FIFO over a fixed number of GPU slots."""

    def __init__(self, slots: int = config.MAX_CONCURRENT) -> None:
        self._slots = asyncio.Semaphore(slots)
        self._waiting: list[Session] = []
        self._active: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, sid: str) -> Session:
        """Wait for a GPU slot. Returns once the session may start."""
        session = Session(sid=sid)
        async with self._lock:
            self._waiting.append(session)
        log.info("session %s queued at position %d", sid, self.position(sid))

        await self._slots.acquire()

        async with self._lock:
            if session in self._waiting:
                self._waiting.remove(session)
            session.started_at = time.monotonic()
            self._active[sid] = session
        log.info("session %s started (waited %.1fs)", sid, session.started_at - session.queued_at)
        return session

    async def release(self, sid: str) -> None:
        async with self._lock:
            session = self._active.pop(sid, None)
            dropped = [s for s in self._waiting if s.sid == sid]
            for s in dropped:
                self._waiting.remove(s)
        if session is not None:
            self._slots.release()
            log.info("session %s ended", sid)
        elif dropped:
            log.info("session %s left the queue before starting", sid)

    def position(self, sid: str) -> int:
        """1-based place in line. 0 means the session is live."""
        if sid in self._active:
            return 0
        for i, s in enumerate(self._waiting, start=1):
            if s.sid == sid:
                return i
        return -1

    def estimated_wait(self, sid: str) -> float:
        """Seconds until this session is likely to start."""
        pos = self.position(sid)
        if pos <= 0:
            return 0.0
        # Everyone ahead runs a full session; those already live finish early.
        soonest = min((s.seconds_left for s in self._active.values()), default=0.0)
        return soonest + (pos - 1) * config.SESSION_SECONDS

    def status(self, sid: str) -> dict:
        return {
            "position": self.position(sid),
            "waiting": len(self._waiting),
            "active": len(self._active),
            "estimated_wait": round(self.estimated_wait(sid), 1),
            "session_seconds": config.SESSION_SECONDS,
        }
