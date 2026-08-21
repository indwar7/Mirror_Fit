"""Cost-sensitive threshold selection.

The default 0.5 cut-point answers the question "is fraud more likely than
not?". That is the wrong question. The question we actually face is "does
flagging this return cost less than not flagging it?", and once false
positives and false negatives carry different costs, the answer moves away
from 0.5.

We sweep every distinct operating point on the precision-recall curve and
pick the one minimising total weighted cost. We do this on the *validation*
split, then report on test, so the headline number is not the number we
optimised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostMatrix

# A return is flagged when P(fraud) >= threshold.
FLAG_IF_GREATER_EQUAL = True


@dataclass(frozen=True)
class OperatingPoint:
    """Everything needed to judge one candidate threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    total_cost: float
    cost_per_case: float
    label: str = ""

    @property
    def precision(self) -> float:
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        actual = self.tp + self.fn
        return self.tp / actual if actual else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Share of *genuine* returns that get flagged. This is the number
        that matters to an innocent customer, and it is not the same as
        1 - precision."""
        negatives = self.fp + self.tn
        return self.fp / negatives if negatives else 0.0

    @property
    def flag_rate(self) -> float:
        """Share of all returns sent to manual review — the review team's
        workload."""
        n = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.fp) / n if n else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_row(self) -> dict:
        return {
            "label": self.label,
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "fpr": self.false_positive_rate,
            "flag_rate": self.flag_rate,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "total_cost": self.total_cost,
            "cost_per_case": self.cost_per_case,
        }


def break_even_probability(costs: CostMatrix) -> float:
    """The fraud probability above which flagging is the cheaper action.

    For one return with true fraud probability p:

        expected cost of flagging      = (1 - p) * C_FP      (we may be wrong)
        expected cost of not flagging  =      p  * C_FN      (it may be fraud)

    Flagging wins when (1 - p) * C_FP < p * C_FN, i.e.

        p > C_FP / (C_FP + C_FN)

    At the default 5:1 that is 0.833 — a deliberately high bar. It says: do
    not pull a customer into review unless the model is very sure, because
    being wrong is five times worse than letting one fraud through.

    This is the threshold a *perfectly calibrated* model should use. The
    empirical sweep below optimises on finite data instead, so the two
    should land close together. If they diverge sharply, the probabilities
    are miscalibrated and the cost argument is standing on sand.
    """
    denominator = costs.false_positive + costs.false_negative
    return costs.false_positive / denominator if denominator else 0.5


def confusion_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    flagged = y_prob >= threshold if FLAG_IF_GREATER_EQUAL else y_prob > threshold
    positive = y_true == 1
    tp = int(np.sum(flagged & positive))
    fp = int(np.sum(flagged & ~positive))
    fn = int(np.sum(~flagged & positive))
    tn = int(np.sum(~flagged & ~positive))
    return tp, fp, tn, fn


def evaluate_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, costs: CostMatrix, label: str = ""
) -> OperatingPoint:
    tp, fp, tn, fn = confusion_at(y_true, y_prob, threshold)
    return OperatingPoint(
        threshold=float(threshold),
        tp=tp, fp=fp, tn=tn, fn=fn,
        total_cost=costs.total(tp=tp, fp=fp, tn=tn, fn=fn),
        cost_per_case=costs.per_case(tp=tp, fp=fp, tn=tn, fn=fn),
        label=label,
    )


def candidate_thresholds(y_prob: np.ndarray) -> np.ndarray:
    """Every distinct operating point on the PR curve.

    Cost is piecewise-constant between observed probabilities, so evaluating
    at each distinct score plus the boundaries covers the space exactly —
    no arbitrary grid resolution to defend.
    """
    return np.unique(np.concatenate([[0.0], np.asarray(y_prob), [1.0 + 1e-9]]))


def sweep(y_true: np.ndarray, y_prob: np.ndarray, costs: CostMatrix) -> list[OperatingPoint]:
    return [
        evaluate_threshold(y_true, y_prob, t, costs)
        for t in candidate_thresholds(y_prob)
    ]


def select_min_cost(points: list[OperatingPoint]) -> OperatingPoint:
    """Lowest weighted cost wins.

    Ties are broken toward the *higher* threshold. Two thresholds with equal
    cost are not equally good in practice: the higher one flags fewer people,
    which means less customer friction and a smaller review queue for the
    same money. Preferring it is consistent with why we weighted FP heavily
    in the first place.
    """
    best = min(points, key=lambda p: (p.total_cost, -p.threshold))
    return OperatingPoint(**{**best.__dict__, "label": "cost-optimal"})


def select_max_f1(points: list[OperatingPoint]) -> OperatingPoint:
    best = max(points, key=lambda p: (p.f1, p.threshold))
    return OperatingPoint(**{**best.__dict__, "label": "max-F1"})


def select_at_precision(points: list[OperatingPoint], min_precision: float) -> OperatingPoint | None:
    """Lowest threshold that still clears a precision floor — a common
    policy-driven alternative to pure cost minimisation."""
    eligible = [p for p in points if p.precision >= min_precision and (p.tp + p.fp) > 0]
    if not eligible:
        return None
    best = min(eligible, key=lambda p: p.threshold)
    return OperatingPoint(**{**best.__dict__, "label": f"precision>={min_precision:g}"})


def candidate_table(
    y_true: np.ndarray, y_prob: np.ndarray, costs: CostMatrix, min_precision: float = 0.60
) -> pd.DataFrame:
    """Three or four defensible candidates, side by side.

    The point of the table is that the chosen threshold is visibly the
    result of a comparison, not a number someone liked.
    """
    points = sweep(y_true, y_prob, costs)

    rows = [
        evaluate_threshold(y_true, y_prob, 0.5, costs, label="default 0.5"),
        select_max_f1(points),
    ]
    at_precision = select_at_precision(points, min_precision)
    if at_precision is not None:
        rows.append(at_precision)
    rows.append(select_min_cost(points))

    frame = pd.DataFrame([r.as_row() for r in rows])
    return frame.drop_duplicates(subset=["threshold"], keep="last").reset_index(drop=True)


def ratio_sensitivity(
    y_true_valid: np.ndarray,
    y_prob_valid: np.ndarray,
    y_true_test: np.ndarray,
    y_prob_test: np.ndarray,
    ratios: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0),
) -> pd.DataFrame:
    """How much does the FP:FN ratio actually move the decision?

    If the chosen threshold barely moves across plausible ratios, arguing
    about the exact ratio is wasted effort. If it moves a lot, the ratio is
    the most important number in the system and deserves scrutiny from
    whoever owns the policy — not from whoever wrote the model.
    """
    rows = []
    for ratio in ratios:
        costs = CostMatrix(false_positive=ratio, false_negative=1.0)
        chosen = select_min_cost(sweep(y_true_valid, y_prob_valid, costs))
        on_test = evaluate_threshold(y_true_test, y_prob_test, chosen.threshold, costs)
        rows.append({
            "fp_fn_ratio": f"{ratio:g}:1",
            "threshold": chosen.threshold,
            "test_precision": on_test.precision,
            "test_recall": on_test.recall,
            "test_fpr": on_test.false_positive_rate,
            "test_flag_rate": on_test.flag_rate,
            "test_cost_per_case": on_test.cost_per_case,
        })
    return pd.DataFrame(rows)
