"""Accounts, tokens and the credit ledger."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import uuid

from .db import connect, now, tx

TOKEN_TTL = 30 * 24 * 3600          # 30 days
SIGNUP_BONUS = int(os.environ.get("LIVE_SIGNUP_BONUS", "3"))
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

PBKDF_ROUNDS = 200_000


class AuthError(Exception):
    """Wrong credentials, or a token that is no longer good."""


class InsufficientCredits(Exception):
    pass


# ---------- passwords ----------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF_ROUNDS)
    return f"pbkdf2${PBKDF_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, rounds, salt_hex, want = stored.split("$")
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds)
        )
    except (ValueError, TypeError):
        return False
    # Constant time: a timing difference here leaks whether a prefix matched.
    return hmac.compare_digest(dk.hex(), want)


# ---------- users ----------

def create_user(email: str, password: str, is_admin: bool = False) -> str:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise ValueError("that does not look like an email address")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")

    uid = uuid.uuid4().hex
    try:
        with tx() as conn:
            conn.execute(
                "INSERT INTO users(id,email,password_hash,is_admin,created_at)"
                " VALUES(?,?,?,?,?)",
                (uid, email, hash_password(password), int(is_admin), now()),
            )
            if SIGNUP_BONUS:
                conn.execute(
                    "INSERT INTO credit_ledger(user_id,delta,reason,ref,created_at)"
                    " VALUES(?,?,?,?,?)",
                    (uid, SIGNUP_BONUS, "signup bonus", f"signup:{uid}", now()),
                )
    except sqlite3.IntegrityError as exc:
        raise ValueError("that email is already registered") from exc
    return uid


def login(email: str, password: str) -> str:
    row = connect().execute(
        "SELECT id,password_hash FROM users WHERE email=?", (email.strip().lower(),)
    ).fetchone()
    # Hash even when the user is absent, so a missing account and a wrong
    # password take the same time and cannot be told apart.
    stored = row["password_hash"] if row else hash_password("x")
    if not verify_password(password, stored) or row is None:
        raise AuthError("email or password is wrong")
    return issue_token(row["id"])


def issue_token(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    with tx() as conn:
        conn.execute(
            "INSERT INTO auth_tokens(token,user_id,created_at,expires_at) VALUES(?,?,?,?)",
            (token, user_id, now(), now() + TOKEN_TTL),
        )
    return token


def user_for_token(token: str) -> sqlite3.Row:
    row = connect().execute(
        "SELECT u.* FROM auth_tokens t JOIN users u ON u.id=t.user_id"
        " WHERE t.token=? AND t.expires_at > ?",
        (token, now()),
    ).fetchone()
    if row is None:
        raise AuthError("please sign in again")
    return row


def logout(token: str) -> None:
    with tx() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token=?", (token,))


# ---------- credits ----------

def balance(user_id: str) -> int:
    row = connect().execute(
        "SELECT COALESCE(SUM(delta),0) AS b FROM credit_ledger WHERE user_id=?",
        (user_id,),
    ).fetchone()
    return int(row["b"])


def grant(user_id: str, amount: int, reason: str, ref: str | None = None) -> int:
    """Add credits. A repeated `ref` is ignored rather than paid twice —
    payment webhooks retry, and a double grant is a refund to chase."""
    if amount <= 0:
        raise ValueError("grant amount must be positive")
    try:
        with tx() as conn:
            conn.execute(
                "INSERT INTO credit_ledger(user_id,delta,reason,ref,created_at)"
                " VALUES(?,?,?,?,?)",
                (user_id, amount, reason, ref, now()),
            )
    except sqlite3.IntegrityError:
        pass  # same ref already applied
    return balance(user_id)


def spend(user_id: str, amount: int, reason: str, ref: str | None = None) -> int:
    """Take credits, refusing to go negative.

    The check and the write share one transaction: two sessions starting
    at once would otherwise both read the same balance and both pass.
    """
    if amount <= 0:
        raise ValueError("spend amount must be positive")
    with tx(immediate=True) as conn:
        cur = conn.execute(
            "SELECT COALESCE(SUM(delta),0) AS b FROM credit_ledger WHERE user_id=?",
            (user_id,),
        ).fetchone()
        if int(cur["b"]) < amount:
            raise InsufficientCredits(f"needs {amount} credits, has {int(cur['b'])}")
        conn.execute(
            "INSERT INTO credit_ledger(user_id,delta,reason,ref,created_at)"
            " VALUES(?,?,?,?,?)",
            (user_id, -amount, reason, ref, now()),
        )
    return balance(user_id)


def history(user_id: str, limit: int = 50) -> list[sqlite3.Row]:
    return connect().execute(
        "SELECT delta,reason,created_at FROM credit_ledger"
        " WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
