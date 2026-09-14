"""SQLite store for accounts, credits and sessions.

SQLite because one GPU serves one session at a time — the write rate is
a handful of rows per minute, and a managed Postgres would be one more
thing to keep alive for no gain. WAL mode keeps the admin page readable
while a session is being written.

Money-adjacent state lives in `credit_ledger`, which is append-only. A
balance is the sum of its rows, never a column that can drift.
"""
from __future__ import annotations

import contextlib
import pathlib
import sqlite3
import threading
import time

DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "product.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id            TEXT PRIMARY KEY,
  email         TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  is_admin      INTEGER NOT NULL DEFAULT 0,
  created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_tokens (
  token      TEXT PRIMARY KEY,
  user_id    TEXT NOT NULL REFERENCES users(id),
  created_at REAL NOT NULL,
  expires_at REAL NOT NULL
);

-- Append-only. Positive rows are grants, negative rows are spends.
-- `ref` makes a grant idempotent: the same payment or promo can be
-- replayed without paying out twice.
CREATE TABLE IF NOT EXISTS credit_ledger (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id    TEXT NOT NULL REFERENCES users(id),
  delta      INTEGER NOT NULL,
  reason     TEXT NOT NULL,
  ref        TEXT,
  created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS credit_ref ON credit_ledger(ref) WHERE ref IS NOT NULL;
CREATE INDEX IF NOT EXISTS credit_user ON credit_ledger(user_id);

CREATE TABLE IF NOT EXISTS sessions (
  id          TEXT PRIMARY KEY,
  user_id     TEXT NOT NULL REFERENCES users(id),
  garment     TEXT,
  fabric      TEXT,
  queued_at   REAL NOT NULL,
  started_at  REAL,
  ended_at    REAL,
  frames_out  INTEGER DEFAULT 0,
  mean_fps    REAL,
  backend     TEXT,
  end_reason  TEXT
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id, queued_at DESC);

CREATE TABLE IF NOT EXISTS captures (
  id         TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id),
  user_id    TEXT NOT NULL REFERENCES users(id),
  path       TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS captures_user ON captures(user_id, created_at DESC);
"""

_local = threading.local()


def connect() -> sqlite3.Connection:
    """One connection per thread. aiohttp handlers and the session
    recorder run on different threads and must not share one."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: no implicit BEGIN, so tx() controls it.
        conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        _local.conn = conn
    return conn


def init() -> None:
    connect().executescript(SCHEMA)
    connect().commit()


@contextlib.contextmanager
def tx(immediate: bool = False):
    """A transaction that rolls back on any exception.

    `immediate` takes the write lock up front. Python's sqlite3 defers
    BEGIN until the first write, so a read-then-write sequence (check a
    balance, then debit it) would otherwise let two callers both read
    the old value and both pass the check.
    """
    conn = connect()
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def now() -> float:
    return time.time()
