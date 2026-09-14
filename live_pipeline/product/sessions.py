"""Session records, saved captures and usage rollups.

A session costs one credit, taken when the GPU is actually granted —
not when the user joins the queue. Someone who waits and then leaves
should not be charged for a stream they never saw.
"""
from __future__ import annotations

import pathlib
import sqlite3
import uuid

from . import accounts
from .db import connect, now, tx

CAPTURE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "captures"
SESSION_COST = 1


def open_session(user_id: str, garment: str | None, fabric: str | None) -> str:
    """Record a queue entry. No credit is taken yet."""
    sid = uuid.uuid4().hex[:16]
    with tx() as conn:
        conn.execute(
            "INSERT INTO sessions(id,user_id,garment,fabric,queued_at)"
            " VALUES(?,?,?,?,?)",
            (sid, user_id, garment, fabric, now()),
        )
    return sid


def start_session(sid: str) -> None:
    """The GPU is now this session's. Charge for it.

    Raises InsufficientCredits, which the caller must turn into a
    released slot — otherwise the GPU sits idle behind a user who
    cannot pay.
    """
    row = connect().execute("SELECT user_id FROM sessions WHERE id=?", (sid,)).fetchone()
    if row is None:
        raise KeyError(sid)
    accounts.spend(row["user_id"], SESSION_COST, "live session", ref=f"session:{sid}")
    with tx() as conn:
        conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (now(), sid))


def end_session(sid: str, *, frames: int = 0, fps: float | None = None,
                backend: str | None = None, reason: str = "completed") -> None:
    with tx() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at=?,frames_out=?,mean_fps=?,backend=?,end_reason=?"
            " WHERE id=?",
            (now(), frames, fps, backend, reason, sid),
        )


def refund(sid: str, reason: str) -> None:
    """Give the credit back when we failed, not the user.

    The ref is per-session, so a double refund is impossible even if
    this is called twice by two error paths.
    """
    row = connect().execute("SELECT user_id FROM sessions WHERE id=?", (sid,)).fetchone()
    if row is None:
        return
    accounts.grant(row["user_id"], SESSION_COST, f"refund: {reason}", ref=f"refund:{sid}")


def save_capture(sid: str, user_id: str, png: bytes) -> str:
    cid = uuid.uuid4().hex[:16]
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = CAPTURE_DIR / f"{cid}.png"
    path.write_bytes(png)
    with tx() as conn:
        conn.execute(
            "INSERT INTO captures(id,session_id,user_id,path,created_at) VALUES(?,?,?,?,?)",
            (cid, sid, user_id, str(path), now()),
        )
    return cid


def user_sessions(user_id: str, limit: int = 25) -> list[sqlite3.Row]:
    return connect().execute(
        "SELECT s.*, (SELECT COUNT(*) FROM captures c WHERE c.session_id=s.id) AS captures"
        " FROM sessions s WHERE s.user_id=? ORDER BY s.queued_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()


def user_captures(user_id: str, limit: int = 100) -> list[sqlite3.Row]:
    return connect().execute(
        "SELECT id,session_id,created_at FROM captures WHERE user_id=?"
        " ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()


def usage_summary(hours: int = 24) -> dict:
    """What the admin page and the monitor both read.

    Wait time is the number to watch: the plan adds a second card when it
    exceeds a few seconds at peak.
    """
    since = now() - hours * 3600
    conn = connect()
    row = conn.execute(
        """
        SELECT COUNT(*) AS sessions,
               COUNT(DISTINCT user_id) AS users,
               AVG(CASE WHEN started_at IS NOT NULL
                        THEN started_at - queued_at END) AS mean_wait,
               MAX(CASE WHEN started_at IS NOT NULL
                        THEN started_at - queued_at END) AS max_wait,
               AVG(CASE WHEN ended_at IS NOT NULL AND started_at IS NOT NULL
                        THEN ended_at - started_at END) AS mean_length,
               AVG(mean_fps) AS mean_fps,
               SUM(CASE WHEN started_at IS NULL THEN 1 ELSE 0 END) AS abandoned
        FROM sessions WHERE queued_at > ?
        """,
        (since,),
    ).fetchone()
    gpu_seconds = conn.execute(
        "SELECT COALESCE(SUM(ended_at - started_at),0) AS s FROM sessions"
        " WHERE started_at IS NOT NULL AND ended_at IS NOT NULL AND queued_at > ?",
        (since,),
    ).fetchone()["s"]
    out = {k: row[k] for k in row.keys()}
    out["gpu_seconds"] = round(float(gpu_seconds), 1)
    out["gpu_utilisation"] = round(float(gpu_seconds) / (hours * 3600), 4)
    out["window_hours"] = hours
    for k in ("mean_wait", "max_wait", "mean_length", "mean_fps"):
        if out[k] is not None:
            out[k] = round(float(out[k]), 2)
    return out
