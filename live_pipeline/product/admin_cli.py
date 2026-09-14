"""Admin tasks that need no browser.

The first admin has to be made from a shell — there is no privileged
signup route, because one would be a permanent way in.

    python -m product.admin_cli make-admin --email you@example.com
    python -m product.admin_cli grant --email a@b.com --amount 20
    python -m product.admin_cli usage --hours 24
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys

from . import accounts
from .db import connect, init, tx
from .monitor import report


def cmd_make_admin(args) -> int:
    email = args.email.strip().lower()
    row = connect().execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if row is None:
        pw = args.password or getpass.getpass("password for new admin: ")
        try:
            uid = accounts.create_user(email, pw, is_admin=True)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"created admin {email} ({uid})")
        return 0
    with tx() as conn:
        conn.execute("UPDATE users SET is_admin=1 WHERE id=?", (row["id"],))
    print(f"{email} is now an admin")
    return 0


def cmd_grant(args) -> int:
    row = connect().execute(
        "SELECT id FROM users WHERE email=?", (args.email.strip().lower(),)
    ).fetchone()
    if row is None:
        print("no such user", file=sys.stderr)
        return 1
    bal = accounts.grant(row["id"], args.amount, args.reason, ref=args.ref)
    print(f"{args.email}: {bal} credits")
    return 0


def cmd_users(args) -> int:
    rows = connect().execute(
        "SELECT u.email, u.is_admin, u.created_at,"
        " (SELECT COALESCE(SUM(delta),0) FROM credit_ledger c WHERE c.user_id=u.id) AS credits,"
        " (SELECT COUNT(*) FROM sessions s WHERE s.user_id=u.id) AS sessions"
        " FROM users u ORDER BY u.created_at DESC LIMIT ?",
        (args.limit,),
    ).fetchall()
    if not rows:
        print("no users yet")
        return 0
    print(f"{'email':<32}{'credits':>8}{'sessions':>10}  admin")
    for r in rows:
        print(f"{r['email']:<32}{r['credits']:>8}{r['sessions']:>10}  {'yes' if r['is_admin'] else ''}")
    return 0


def cmd_usage(args) -> int:
    print(json.dumps(report(args.hours), indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("make-admin", help="create or promote an admin")
    a.add_argument("--email", required=True)
    a.add_argument("--password", help="omit to be prompted")
    a.set_defaults(fn=cmd_make_admin)

    g = sub.add_parser("grant", help="add credits to an account")
    g.add_argument("--email", required=True)
    g.add_argument("--amount", type=int, required=True)
    g.add_argument("--reason", default="manual grant")
    g.add_argument("--ref", help="idempotency key; repeats are ignored")
    g.set_defaults(fn=cmd_grant)

    u = sub.add_parser("users", help="list accounts")
    u.add_argument("--limit", type=int, default=50)
    u.set_defaults(fn=cmd_users)

    s = sub.add_parser("usage", help="usage and queue report as JSON")
    s.add_argument("--hours", type=int, default=24)
    s.set_defaults(fn=cmd_usage)

    args = ap.parse_args(argv)
    init()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
