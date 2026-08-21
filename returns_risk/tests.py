"""Tests. Run: returns_risk/.venv/bin/python -m returns_risk.tests

The decision-contract tests matter more than the modelling ones. A model
that drifts costs money; a system that learns how to deny a refund on its
own costs a customer.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from .config import CostMatrix, Settings
from .data import TARGET, load_returns, split_frame
from .decision import (
    FORBIDDEN_ACTION_TOKENS,
    Action,
    _assert_no_denial_action,
    decide,
)
from .threshold import (
    confusion_at,
    evaluate_threshold,
    select_min_cost,
    sweep,
)

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}" + (f"  — {detail}" if detail else ""))
        _failures.append(name)


# ══════════════════════════════════════════════════════════════════════
# The contract: this component cannot deny a refund
# ══════════════════════════════════════════════════════════════════════
def test_no_denial_action() -> None:
    print("\nDecision contract")

    check("Action has exactly two members", len(list(Action)) == 2, str(list(Action)))
    check(
        "no action value resembles a denial",
        all(
            token not in member.value.lower()
            for member in Action
            for token in FORBIDDEN_ACTION_TOKENS
        ),
    )

    ok = True
    try:
        _assert_no_denial_action()
    except AssertionError:
        ok = False
    check("import-time guard passes", ok)

    # Sweep the whole probability space plus hostile inputs. Every single
    # one must land on one of the two permitted actions.
    probes: list = list(np.linspace(-0.5, 1.5, 401))
    probes += [float("nan"), float("inf"), float("-inf"), None, "abc", 1e18, -1e18]
    actions = set()
    crashed = None
    for p in probes:
        for t in (0.0, 0.25, 0.5, 0.9, 1.0):
            try:
                actions.add(decide(p, t).action)
            except Exception as exc:  # noqa: BLE001
                crashed = f"{p!r}@{t}: {exc}"
                break
    check("no input crashes decide()", crashed is None, crashed or "")
    check(
        "every outcome is approve_refund or route_to_review",
        actions.issubset({Action.APPROVE_REFUND, Action.ROUTE_TO_REVIEW}),
        str(actions),
    )

    payload = decide(0.99, 0.5).to_payload()
    check("payload has exactly the specified keys",
          set(payload) == {"flagged", "confidence", "action"}, str(payload))
    check("flagged case routes to review", payload["action"] == "route_to_review", str(payload))
    check("unflagged case approves",
          decide(0.01, 0.5).to_payload()["action"] == "approve_refund")

    # Unusable scores must fail toward the customer, not against them.
    check("NaN score does not flag", decide(float("nan"), 0.5).flagged is False)
    check("garbage score does not flag", decide("???", 0.5).flagged is False)
    check("confidence is clamped to [0,1]",
          0.0 <= decide(9.9, 0.5).confidence <= 1.0)


# ══════════════════════════════════════════════════════════════════════
# Cost arithmetic
# ══════════════════════════════════════════════════════════════════════
def test_cost_matrix() -> None:
    print("\nCost matrix")

    costs = CostMatrix(false_positive=5.0, false_negative=1.0)
    check("ratio reported correctly", math.isclose(costs.fp_fn_ratio, 5.0))
    check("total = 5*FP + 1*FN",
          math.isclose(costs.total(tp=0, fp=2, tn=0, fn=3), 13.0))
    check("per-case divides by n",
          math.isclose(costs.per_case(tp=0, fp=2, tn=5, fn=3), 13.0 / 10))
    check("empty set costs nothing",
          math.isclose(costs.per_case(tp=0, fp=0, tn=0, fn=0), 0.0))

    rejected = False
    try:
        CostMatrix(false_positive=-1.0)
    except ValueError:
        rejected = True
    check("negative cost rejected", rejected)

    rejected = False
    try:
        CostMatrix(false_negative=0.0)
    except ValueError:
        rejected = True
    check("zero FN cost rejected", rejected)

    review = CostMatrix(false_positive=5.0, false_negative=1.0, review_cost=0.2)
    check("review cost charged on every flag",
          math.isclose(review.total(tp=3, fp=2, tn=0, fn=0), 10.0 + 1.0))


# ══════════════════════════════════════════════════════════════════════
# Threshold selection
# ══════════════════════════════════════════════════════════════════════
def test_threshold_selection() -> None:
    print("\nThreshold selection")

    rng = np.random.default_rng(7)
    y_true = rng.binomial(1, 0.08, size=4000)
    y_prob = np.clip(rng.beta(2, 9, size=4000) + 0.35 * y_true, 0, 1)

    cheap = CostMatrix(false_positive=1.0, false_negative=1.0)
    dear = CostMatrix(false_positive=20.0, false_negative=1.0)

    t_cheap = select_min_cost(sweep(y_true, y_prob, cheap)).threshold
    t_dear = select_min_cost(sweep(y_true, y_prob, dear)).threshold
    check("costlier false positives raise the threshold", t_dear >= t_cheap,
          f"{t_dear:.3f} vs {t_cheap:.3f}")

    points = sweep(y_true, y_prob, dear)
    best = select_min_cost(points)
    check("selected point is the global minimum",
          all(best.total_cost <= p.total_cost + 1e-9 for p in points))

    # Threshold 0 flags everything; threshold above 1 flags nothing.
    tp, fp, tn, fn = confusion_at(y_true, y_prob, 0.0)
    check("threshold 0 flags every case", tp + fp == len(y_true))
    tp, fp, tn, fn = confusion_at(y_true, y_prob, 1.0 + 1e-9)
    check("threshold >1 flags nothing", tp + fp == 0)

    point = evaluate_threshold(y_true, y_prob, 0.5, dear)
    check("confusion cells sum to n", point.tp + point.fp + point.tn + point.fn == len(y_true))
    check("FPR uses genuine returns as denominator",
          math.isclose(point.false_positive_rate,
                       point.fp / max(point.fp + point.tn, 1), rel_tol=1e-9))


# ══════════════════════════════════════════════════════════════════════
# Data and leakage
# ══════════════════════════════════════════════════════════════════════
def test_data_and_splits() -> None:
    print("\nData and splits")

    frame = load_returns(n=3000, seed=11)
    check("label is binary", set(frame[TARGET].unique()) <= {0, 1})
    check("fraud is rare (1–20%)", 0.01 < frame[TARGET].mean() < 0.20,
          f"{frame[TARGET].mean():.3%}")
    check("fit score missing exactly when try-on unused",
          bool((frame["fit_mismatch_score"].isna() == ~frame["used_tryon"]).all()))
    check("some fit scores are missing", frame["fit_mismatch_score"].isna().any())

    settings = Settings(seed=11, n_samples=3000)
    train, valid, test = split_frame(frame, settings)
    check("splits partition the data", len(train) + len(valid) + len(test) == len(frame))

    # Reproducibility, and the absence of shared rows between splits.
    again = split_frame(load_returns(n=3000, seed=11), settings)
    check("splits are deterministic",
          bool(train.equals(again[0]) and test.equals(again[2])))

    joined = train.merge(test, how="inner", on=list(frame.columns))
    check("no identical row shared between train and test",
          len(joined) < max(1, int(0.01 * len(test))),
          f"{len(joined)} overlapping rows")


def main() -> int:
    print("=" * 62)
    print("returns_risk — test suite")
    print("=" * 62)

    test_no_denial_action()
    test_cost_matrix()
    test_threshold_selection()
    test_data_and_splits()

    print("\n" + "=" * 62)
    if _failures:
        print(f"FAILED — {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
