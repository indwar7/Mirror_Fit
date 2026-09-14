"""HTTP routes for accounts, credits, session history and admin.

Mounted by server/app.py alongside the WebRTC signaling, so one process
serves the whole product.
"""
from __future__ import annotations

import base64
import functools
import logging

from aiohttp import web

from . import accounts, sessions
from .accounts import AuthError, InsufficientCredits

log = logging.getLogger(__name__)


def _token(request: web.Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.cookies.get("token", "")


def authenticated(handler):
    @functools.wraps(handler)
    async def wrapper(request: web.Request):
        try:
            request["user"] = accounts.user_for_token(_token(request))
        except AuthError as exc:
            raise web.HTTPUnauthorized(text=str(exc))
        return await handler(request)

    return wrapper


def admin_only(handler):
    @functools.wraps(handler)
    @authenticated
    async def wrapper(request: web.Request):
        if not request["user"]["is_admin"]:
            # 404, not 403: an admin page that announces itself invites
            # someone to go looking for the credentials.
            raise web.HTTPNotFound()
        return await handler(request)

    return wrapper


# ---------- auth ----------

async def signup(request: web.Request) -> web.Response:
    body = await request.json()
    try:
        uid = accounts.create_user(body.get("email", ""), body.get("password", ""))
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    token = accounts.issue_token(uid)
    return web.json_response({"token": token, "credits": accounts.balance(uid)})


async def signin(request: web.Request) -> web.Response:
    body = await request.json()
    try:
        token = accounts.login(body.get("email", ""), body.get("password", ""))
    except AuthError as exc:
        raise web.HTTPUnauthorized(text=str(exc))
    user = accounts.user_for_token(token)
    return web.json_response(
        {"token": token, "credits": accounts.balance(user["id"]),
         "is_admin": bool(user["is_admin"])}
    )


@authenticated
async def signout(request: web.Request) -> web.Response:
    accounts.logout(_token(request))
    return web.json_response({"ok": True})


@authenticated
async def me(request: web.Request) -> web.Response:
    u = request["user"]
    return web.json_response(
        {
            "email": u["email"],
            "credits": accounts.balance(u["id"]),
            "is_admin": bool(u["is_admin"]),
            "session_cost": sessions.SESSION_COST,
        }
    )


@authenticated
async def credit_history(request: web.Request) -> web.Response:
    rows = accounts.history(request["user"]["id"])
    return web.json_response(
        {"balance": accounts.balance(request["user"]["id"]),
         "entries": [dict(r) for r in rows]}
    )


# ---------- sessions ----------

@authenticated
async def my_sessions(request: web.Request) -> web.Response:
    rows = sessions.user_sessions(request["user"]["id"])
    return web.json_response({"sessions": [dict(r) for r in rows]})


@authenticated
async def save_capture(request: web.Request) -> web.Response:
    body = await request.json()
    data = body.get("png", "")
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        png = base64.b64decode(data, validate=True)
    except Exception:
        raise web.HTTPBadRequest(text="png must be base64")
    if len(png) > 8 * 1024 * 1024:
        raise web.HTTPRequestEntityTooLarge(max_size=8 * 1024 * 1024, actual_size=len(png))
    cid = sessions.save_capture(body.get("sid", ""), request["user"]["id"], png)
    return web.json_response({"id": cid})


@authenticated
async def my_captures(request: web.Request) -> web.Response:
    rows = sessions.user_captures(request["user"]["id"])
    return web.json_response({"captures": [dict(r) for r in rows]})


@authenticated
async def capture_file(request: web.Request) -> web.Response:
    cid = request.match_info["cid"]
    from .db import connect  # noqa: PLC0415

    row = connect().execute(
        "SELECT path FROM captures WHERE id=? AND user_id=?",
        (cid, request["user"]["id"]),
    ).fetchone()
    if row is None:
        raise web.HTTPNotFound()
    return web.FileResponse(row["path"])


# ---------- admin ----------

@admin_only
async def admin_usage(request: web.Request) -> web.Response:
    hours = int(request.query.get("hours", 24))
    return web.json_response(sessions.usage_summary(hours))


@admin_only
async def admin_grant(request: web.Request) -> web.Response:
    """Manual credit top-up. Stands in for payments until a provider is
    wired in; the ledger and its idempotency are already here."""
    body = await request.json()
    from .db import connect  # noqa: PLC0415

    row = connect().execute(
        "SELECT id FROM users WHERE email=?", (body.get("email", "").strip().lower(),)
    ).fetchone()
    if row is None:
        raise web.HTTPNotFound(text="no such user")
    try:
        bal = accounts.grant(
            row["id"], int(body.get("amount", 0)),
            body.get("reason", "admin grant"), ref=body.get("ref"),
        )
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    return web.json_response({"balance": bal})


def add_routes(app: web.Application) -> None:
    r = app.router
    r.add_post("/api/signup", signup)
    r.add_post("/api/signin", signin)
    r.add_post("/api/signout", signout)
    r.add_get("/api/me", me)
    r.add_get("/api/credits", credit_history)
    r.add_get("/api/sessions", my_sessions)
    r.add_post("/api/captures", save_capture)
    r.add_get("/api/captures", my_captures)
    r.add_get("/api/captures/{cid}", capture_file)
    r.add_get("/api/admin/usage", admin_usage)
    r.add_post("/api/admin/grant", admin_grant)
