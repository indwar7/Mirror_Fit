"""Cost model and run settings for the return-fraud risk scorer.

Everything that encodes a *policy* judgement lives here, so it can be
changed without touching modelling code. The FP:FN ratio in particular is
a business stance, not a statistical fact — see README.md for the
derivation and why it is deliberately not hardcoded downstream.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostMatrix:
    """Cost of each confusion-matrix cell, in abstract 'loss units'.

    One unit == the merchandise loss absorbed when a single fraudulent
    return is approved. Everything else is expressed relative to that, so
    the ratio is what matters, not the scale.

    Defaults encode: a false positive is 5x as costly as a false negative.
    Correct decisions cost nothing in this model. `review_cost` is charged
    on *every* flagged case (true and false positives alike) because a
    human reviewer is paid either way; it defaults to 0 so the headline
    numbers isolate the FP/FN tradeoff, but it is available for operational
    planning.
    """

    false_positive: float = 5.0
    false_negative: float = 1.0
    true_positive: float = 0.0
    true_negative: float = 0.0
    review_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "false_positive",
            "false_negative",
            "true_positive",
            "true_negative",
            "review_cost",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.false_negative == 0:
            raise ValueError("false_negative must be > 0 to define a meaningful ratio")

    @property
    def fp_fn_ratio(self) -> float:
        return self.false_positive / self.false_negative

    def total(self, *, tp: int, fp: int, tn: int, fn: int) -> float:
        """Total weighted cost for a set of predictions."""
        return (
            tp * self.true_positive
            + fp * self.false_positive
            + tn * self.true_negative
            + fn * self.false_negative
            + (tp + fp) * self.review_cost
        )

    def per_case(self, *, tp: int, fp: int, tn: int, fn: int) -> float:
        """Weighted cost per return processed — the comparable figure across
        datasets of different sizes."""
        n = tp + fp + tn + fn
        if n == 0:
            return 0.0
        return self.total(tp=tp, fp=fp, tn=tn, fn=fn) / n

    def describe(self) -> str:
        return (
            f"FP={self.false_positive:g}  FN={self.false_negative:g}  "
            f"ratio={self.fp_fn_ratio:g}:1"
            + (f"  review={self.review_cost:g}" if self.review_cost else "")
        )


@dataclass(frozen=True)
class Settings:
    """Run configuration. Split sizes are fractions of the full dataset."""

    seed: int = 20260821
    n_samples: int = 12_000

    # Three-way split. The threshold is chosen on validation and reported on
    # test, so the reported cost is not the cost we optimised against.
    train_frac: float = 0.60
    valid_frac: float = 0.20
    # test_frac is the remainder

    model: str = "logreg"  # "logreg" | "gbt"

    def __post_init__(self) -> None:
        if not 0 < self.train_frac < 1:
            raise ValueError("train_frac must be in (0, 1)")
        if not 0 < self.valid_frac < 1:
            raise ValueError("valid_frac must be in (0, 1)")
        if self.train_frac + self.valid_frac >= 1:
            raise ValueError("train_frac + valid_frac must leave room for a test split")
        if self.model not in ("logreg", "gbt"):
            raise ValueError(f"unknown model {self.model!r}")

    @property
    def test_frac(self) -> float:
        return 1.0 - self.train_frac - self.valid_frac
