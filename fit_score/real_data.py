"""RentTheRunway fit dataset — real customer measurements and fit outcomes.

Source
------
Rishabh Misra, Mengting Wan, Julian McAuley, "Decomposing fit semantics for
product size recommendation in metric spaces", RecSys 2018.
https://cseweb.ucsd.edu/~jmcauley/datasets.html

192,544 rentals. Each row is a real customer who rented a real garment in a
real size and afterwards reported whether it ran **small**, **fit**, or
**large**. This is the only public dataset with genuine fit outcomes, which
is why the synthetic generator is retired in favour of it.

What the labels are, and are not
--------------------------------
The label is *self-reported* by the renter, so it carries real-world noise:
people disagree about what "fits", and someone who wanted a loose drape
reports differently from someone who wanted it fitted. That noise is a
property of the problem, not a defect in the data — a production system
would face exactly the same labels.

The dataset has three classes, not five. "Made for you" and "poor fit" are
not labelled here and could not be, since nobody measured the garments.
The five-tier presentation is derived from the model's own calibrated
probabilities downstream — see real_model.py — and is clearly marked as a
derived view rather than a learned class.

Leakage
-------
The strongest known signal in size recommendation is *relative* size: how
the chosen size compares to what this customer usually takes, and to what
this item usually goes out in. Those aggregates are computed from the
TRAIN split only and joined onto validation and test. Computing them over
the full frame would leak the test rows' own sizes into their features and
inflate every number in the report.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import re

import numpy as np
import pandas as pd

CACHE = pathlib.Path(__file__).resolve().parent / "data_cache"
RTR_GZ = CACHE / "renttherunway.json.gz"
RTR_URL = ("https://mcauleylab.ucsd.edu/public_datasets/data/"
           "renttherunway/renttherunway_final_data.json.gz")

TARGET = "fit"
CLASSES = ("small", "fit", "large")          # ordered: too tight -> too loose

NUMERIC = [
    "height_in",
    "weight_lb",
    "bmi",
    "bust_band",
    "bust_cup",
    "size",
    "age",
    "size_vs_user",     # chosen size minus this customer's usual size
    "size_vs_item",     # chosen size minus this item's usual size
    "user_size_std",    # how variable this customer's sizing is
    # Collaborative history. Body measurements alone turn out to be weak
    # predictors of fit; what an item did for *previous* renters is much
    # stronger, because cut and grading are properties of the garment.
    "item_small_rate",
    "item_large_rate",
    "item_n",           # how much history backs those rates
    "user_small_rate",  # some people report "small" about everything
    "user_large_rate",
    "user_n",
    # How this shopper compares to the people this item usually goes to.
    # This is the personalisation signal: an item cut for slighter renters
    # will run small on a heavier one, regardless of the number on the
    # label. Absolute body measurements cannot express that; deltas can.
    "bmi_vs_item",
    "weight_vs_item",
    "height_vs_item",
    "bust_vs_item",
    "item_size_std",    # how widely this item's sizes vary
]
CATEGORICAL = ["body_type", "category", "rented_for"]
FEATURES = NUMERIC + CATEGORICAL

_CUP = {"aa": 0, "a": 1, "b": 2, "c": 3, "d": 4, "dd": 5, "ddd": 6,
        "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10}


def _height_inches(value) -> float:
    """'5\\' 8\"' -> 68.0"""
    if not isinstance(value, str):
        return np.nan
    m = re.match(r"\s*(\d+)\s*'\s*(\d+)?", value)
    if not m:
        return np.nan
    feet = int(m.group(1))
    inches = int(m.group(2)) if m.group(2) else 0
    total = feet * 12 + inches
    return float(total) if 48 <= total <= 84 else np.nan


def _weight_lb(value) -> float:
    if not isinstance(value, str):
        return np.nan
    m = re.match(r"\s*(\d+)", value)
    if not m:
        return np.nan
    w = float(m.group(1))
    return w if 70 <= w <= 400 else np.nan


def _bust(value) -> tuple[float, float]:
    """'34d' -> (34, 4). Cup letters are ordinal, so they carry order."""
    if not isinstance(value, str):
        return np.nan, np.nan
    m = re.match(r"\s*(\d{2})\s*([a-zA-Z]+)", value)
    if not m:
        return np.nan, np.nan
    band = float(m.group(1))
    cup = _CUP.get(m.group(2).lower().strip(), np.nan)
    return (band if 28 <= band <= 48 else np.nan), cup


def _int(value, lo, hi) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    return v if lo <= v <= hi else np.nan


def download() -> None:
    if RTR_GZ.exists():
        return
    import urllib.request
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"downloading {RTR_URL} …")
    urllib.request.urlretrieve(RTR_URL, RTR_GZ)


def load_raw() -> pd.DataFrame:
    """Parse the raw JSONL into typed columns."""
    download()
    records = []
    with gzip.open(RTR_GZ, "rt", encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            band, cup = _bust(r.get("bust size"))
            records.append({
                "fit": r.get("fit"),
                "user_id": r.get("user_id"),
                "item_id": r.get("item_id"),
                "height_in": _height_inches(r.get("height")),
                "weight_lb": _weight_lb(r.get("weight")),
                "bust_band": band,
                "bust_cup": cup,
                "size": _int(r.get("size"), 0, 60),
                "age": _int(r.get("age"), 14, 95),
                "body_type": r.get("body type") or "unknown",
                "category": r.get("category") or "unknown",
                "rented_for": r.get("rented for") or "unknown",
            })
    frame = pd.DataFrame.from_records(records)
    frame = frame[frame["fit"].isin(CLASSES)].reset_index(drop=True)

    # BMI is the single most informative body summary and is well defined
    # wherever both height and weight parsed.
    frame["bmi"] = 703.0 * frame["weight_lb"] / (frame["height_in"] ** 2)
    return frame


def split_frame(frame: pd.DataFrame, seed: int = 20260821,
                train: float = 0.60, valid: float = 0.20):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    a, b = int(len(frame) * train), int(len(frame) * (train + valid))
    return (
        frame.iloc[order[:a]].reset_index(drop=True),
        frame.iloc[order[a:b]].reset_index(drop=True),
        frame.iloc[order[b:]].reset_index(drop=True),
    )


def add_relative_size(train: pd.DataFrame, *others: pd.DataFrame):
    """Attach train-only user/item size norms to every split.

    `size` alone is close to meaningless across a rental catalogue — a 12 in
    one label is an 8 in another. What predicts fit is the *deviation* from
    the size this customer normally takes and the size this item normally
    goes out in. Both aggregates come from the training split only.
    """
    user_mean = train.groupby("user_id")["size"].mean()
    user_std = train.groupby("user_id")["size"].std()
    item_mean = train.groupby("item_id")["size"].mean()

    # ── Historical fit rates, smoothed toward the global prior ──────────
    # An item seen twice must not look as certain as one seen 200 times, so
    # each rate is pulled toward the overall base rate with a pseudo-count.
    # ALPHA=10 means an item needs ~10 rentals before its own history
    # outweighs the prior.
    ALPHA = 10.0
    prior_small = float((train[TARGET] == "small").mean())
    prior_large = float((train[TARGET] == "large").mean())

    def rates(key: str):
        grouped = train.groupby(key)[TARGET]
        n = grouped.size()
        small = grouped.apply(lambda s: (s == "small").sum())
        large = grouped.apply(lambda s: (s == "large").sum())
        return (
            (small + ALPHA * prior_small) / (n + ALPHA),
            (large + ALPHA * prior_large) / (n + ALPHA),
            n,
        )

    def loo_rates(frame: pd.DataFrame, key: str):
        """Leave-one-out rates for the TRAINING rows.

        A rate built from train labels encodes a training row's own label:
        for an item seen once, `item_small_rate` is a direct readout of
        that row's outcome. The model then learns to trust the feature far
        more than it should, and collapses on unseen data — which is
        exactly what happened here (gradient boosting reached a log loss of
        1.66, worse than a uniform guess at 1.10).

        Excluding each row from its own group's statistic removes the leak.
        Validation and test rows are not in the training counts at all, so
        they use the plain rates.
        """
        counts_n = frame.groupby(key)[TARGET].transform("size")
        is_small = (frame[TARGET] == "small").astype(float)
        is_large = (frame[TARGET] == "large").astype(float)
        sum_small = frame.groupby(key)[TARGET].transform(
            lambda s: (s == "small").sum())
        sum_large = frame.groupby(key)[TARGET].transform(
            lambda s: (s == "large").sum())
        denom = (counts_n - 1) + ALPHA
        return (
            (sum_small - is_small + ALPHA * prior_small) / denom,
            (sum_large - is_large + ALPHA * prior_large) / denom,
            counts_n - 1,
        )

    item_small, item_large, item_n = rates("item_id")
    user_small, user_large, user_n = rates("user_id")

    # ── Who normally wears this item ────────────────────────────────────
    # Train-only body profile per item. The global medians are the
    # fallback for an item with no history, which makes the delta zero —
    # i.e. "no evidence either way" rather than a fabricated difference.
    body_cols = ["bmi", "weight_lb", "height_in", "bust_band"]
    item_body = train.groupby("item_id")[body_cols].mean()
    global_body = train[body_cols].median()
    item_size_std_map = train.groupby("item_id")["size"].std()

    def attach(frame: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        out = frame.copy()
        out["size_vs_user"] = out["size"] - out["user_id"].map(user_mean)
        out["size_vs_item"] = out["size"] - out["item_id"].map(item_mean)
        out["user_size_std"] = out["user_id"].map(user_std)

        if is_train:
            # Leave-one-out, so a row never sees its own label.
            i_s, i_l, i_n = loo_rates(out, "item_id")
            u_s, u_l, u_n = loo_rates(out, "user_id")
            out["item_small_rate"], out["item_large_rate"], out["item_n"] = i_s, i_l, i_n
            out["user_small_rate"], out["user_large_rate"], out["user_n"] = u_s, u_l, u_n
        else:
            # Unseen item or user falls back to the global prior — the
            # honest answer for a cold start is "no information", not zero.
            out["item_small_rate"] = out["item_id"].map(item_small).fillna(prior_small)
            out["item_large_rate"] = out["item_id"].map(item_large).fillna(prior_large)
            out["item_n"] = out["item_id"].map(item_n).fillna(0.0)
            out["user_small_rate"] = out["user_id"].map(user_small).fillna(prior_small)
            out["user_large_rate"] = out["user_id"].map(user_large).fillna(prior_large)
            out["user_n"] = out["user_id"].map(user_n).fillna(0.0)

        # Body deltas against the item's usual renter.
        for col, feat in (
            ("bmi", "bmi_vs_item"),
            ("weight_lb", "weight_vs_item"),
            ("height_in", "height_vs_item"),
            ("bust_band", "bust_vs_item"),
        ):
            typical = out["item_id"].map(item_body[col]).fillna(global_body[col])
            out[feat] = out[col] - typical
        out["item_size_std"] = out["item_id"].map(item_size_std_map)
        return out

    return (attach(train, is_train=True),
            *(attach(f, is_train=False) for f in others))


def coverage(train: pd.DataFrame, other: pd.DataFrame) -> dict:
    """How much of `other` is cold-start? Determines how far the
    collaborative features can carry the model at all."""
    return {
        "item_seen": float(other["item_id"].isin(set(train["item_id"])).mean()),
        "user_seen": float(other["user_id"].isin(set(train["user_id"])).mean()),
    }
