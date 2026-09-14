"""Phase 6: what to watch, and when to buy another card.

The plan's rule is explicit — add a second GPU when waits exceed a few
seconds at peak. That decision needs peak wait, not mean wait, so this
reports both and says plainly which side of the line the box is on.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

from .db import connect, init
from .sessions import usage_summary

# The plan's threshold: "a few seconds at peak".
WAIT_BUDGET_S = 5.0


def gpu_snapshot() -> dict | None:
    """Live GPU state, if nvidia-smi is on this box."""
    if not shutil.which("nvidia-smi"):
        return None
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
    except Exception:
        return None
    gpus = []
    for line in out:
        name, util, used, total, temp = (x.strip() for x in line.split(","))
        gpus.append({"name": name, "util_pct": int(util), "vram_used_mb": int(used),
                     "vram_total_mb": int(total), "temp_c": int(temp)})
    return {"gpus": gpus}


def peak_wait(hours: int) -> dict:
    """The worst hour in the window, not the average of all of them.

    A mean over a day hides the evening rush, which is exactly when the
    queue is the product's weakest point.
    """
    since = __import__("time").time() - hours * 3600
    rows = connect().execute(
        """
        SELECT CAST((queued_at - ?) / 3600 AS INT) AS hour,
               COUNT(*) AS n,
               MAX(started_at - queued_at) AS worst,
               AVG(started_at - queued_at) AS mean
        FROM sessions
        WHERE queued_at > ? AND started_at IS NOT NULL
        GROUP BY hour ORDER BY worst DESC LIMIT 1
        """,
        (since, since),
    ).fetchone()
    if row_is_empty(rows):
        return {"sessions": 0}
    return {"sessions": int(rows["n"]),
            "worst_wait_s": round(float(rows["worst"]), 2),
            "mean_wait_s": round(float(rows["mean"]), 2)}


def row_is_empty(row) -> bool:
    return row is None or row["n"] is None or row["worst"] is None


def report(hours: int = 24) -> dict:
    usage = usage_summary(hours)
    peak = peak_wait(hours)
    gpu = gpu_snapshot()

    verdict = "not enough data"
    if peak.get("sessions"):
        worst = peak["worst_wait_s"]
        if worst > WAIT_BUDGET_S * 3:
            verdict = f"add a card now — peak wait {worst:.0f}s"
        elif worst > WAIT_BUDGET_S:
            verdict = f"watch closely — peak wait {worst:.0f}s, budget {WAIT_BUDGET_S:.0f}s"
        else:
            verdict = f"one card is enough — peak wait {worst:.1f}s"

    return {"window_hours": hours, "usage": usage, "peak": peak,
            "gpu": gpu, "verdict": verdict}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    init()
    r = report(args.hours)
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    u, p = r["usage"], r["peak"]
    print(f"\nLast {r['window_hours']}h")
    print(f"  sessions        {u['sessions']}  ({u['users']} users, {u['abandoned']} abandoned)")
    print(f"  GPU busy        {u['gpu_seconds']:.0f}s  ({u['gpu_utilisation']*100:.1f}% of the window)")
    if u["mean_fps"]:
        print(f"  mean fps        {u['mean_fps']}")
    if u["mean_wait"] is not None:
        print(f"  wait            mean {u['mean_wait']}s, max {u['max_wait']}s")
    if p.get("sessions"):
        print(f"  busiest hour    {p['sessions']} sessions, worst wait {p['worst_wait_s']}s")
    if r["gpu"]:
        for g in r["gpu"]["gpus"]:
            print(f"  {g['name']}  {g['util_pct']}% util, "
                  f"{g['vram_used_mb']}/{g['vram_total_mb']}MB, {g['temp_c']}C")
    print(f"\n  {r['verdict']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
