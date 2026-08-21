"""SYNTHETIC return-fraud dataset.

    ⚠️  This is generated data, not observed data. There is no real returns
        log in this repository. Every metric produced downstream describes
        how the pipeline behaves on *this generative process* and is NOT
        evidence of real-world performance. Swap `load_returns()` for a real
        extract before drawing any business conclusion.

The generative process is written to be honest about what makes the problem
hard, rather than to flatter the model:

  * Fraud is rare (~7%), so accuracy is a useless metric and the cost
    framing does real work.
  * The strongest signal is an *interaction*, not a main effect: claiming
    "doesn't fit" is innocuous on its own, and suspicious only when the
    try-on at purchase predicted a good fit. A model that cannot represent
    that interaction will underperform, which is why we compare a linear
    model against gradient boosting rather than assuming one wins.
  * `fit_mismatch_score` is missing whenever the shopper skipped try-on.
    That missingness is itself informative, so it is modelled explicitly
    rather than quietly imputed away.
  * Substantial label noise is injected, so the achievable AUC is bounded
    well below 1.0. A pipeline that reports 0.99 here is leaking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GARMENT_TYPES = ("tshirt", "shirt", "jacket", "dress", "trousers")
RETURN_REASONS = ("doesnt_fit", "not_as_described", "damaged", "changed_mind", "wrong_item")

TARGET = "is_fraud"

NUMERIC_FEATURES = [
    "fit_mismatch_score",
    "days_since_purchase",
    "order_value_inr",
    "prior_orders_12m",
    "prior_returns_12m",
    "prior_return_rate",
    "account_age_days",
    "discount_pct",
]
CATEGORICAL_FEATURES = ["garment_type", "return_reason"]
BOOLEAN_FEATURES = ["used_tryon"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_returns(n: int = 12_000, seed: int = 20260821) -> pd.DataFrame:
    """Generate a synthetic returns table with a `is_fraud` label."""
    rng = np.random.default_rng(seed)

    # ── Population segments ─────────────────────────────────────────────
    # A small group of serial return-abusers. This segment is what makes
    # the problem tractable at a punitive FP:FN ratio: flagging only ever
    # pays off when the model can reach very high confidence, which in turn
    # requires that *some* fraud leaves a strong, multi-signal trace. A
    # generator with only diffuse signal would correctly produce a
    # "flag nothing" policy — mathematically right, operationally useless.
    is_abuser = rng.random(n) < 0.040

    # ── Customer history ────────────────────────────────────────────────
    account_age_days = np.where(
        is_abuser,
        rng.gamma(shape=1.4, scale=90.0, size=n),
        rng.gamma(shape=2.0, scale=260.0, size=n),
    ).clip(1, 3000)
    prior_orders_12m = np.where(
        is_abuser, rng.poisson(lam=9.0, size=n), rng.poisson(lam=4.0, size=n)
    )
    # Return propensity is a stable customer trait — benign for most people,
    # extreme for abusers.
    return_propensity = np.where(
        is_abuser, rng.beta(a=7.0, b=2.0, size=n), rng.beta(a=1.6, b=5.0, size=n)
    )
    prior_returns_12m = rng.binomial(prior_orders_12m, return_propensity)
    prior_return_rate = np.where(
        prior_orders_12m > 0, prior_returns_12m / np.maximum(prior_orders_12m, 1), 0.0
    )

    # ── Order ───────────────────────────────────────────────────────────
    garment_type = rng.choice(GARMENT_TYPES, size=n, p=[0.30, 0.24, 0.14, 0.16, 0.16])
    base_value = {
        "tshirt": 900.0, "shirt": 1600.0, "jacket": 4200.0,
        "dress": 2600.0, "trousers": 1900.0,
    }
    order_value_inr = np.array(
        [base_value[g] for g in garment_type]
    ) * rng.lognormal(mean=0.0, sigma=0.42, size=n)
    order_value_inr = order_value_inr.round(0).clip(299, 40_000)
    discount_pct = (rng.beta(a=1.8, b=4.0, size=n) * 70).round(1)

    # ── Try-on usage and the fit signal ─────────────────────────────────
    # Higher-value and better-fitting categories drive try-on adoption.
    p_tryon = _sigmoid(
        -0.15
        + 0.45 * (np.log(order_value_inr) - np.log(2000.0))
        + 0.30 * np.isin(garment_type, ("dress", "jacket"))
    )
    used_tryon = rng.random(n) < p_tryon

    # fit_mismatch_score only exists when the shopper actually used try-on.
    latent_mismatch = rng.beta(a=2.0, b=4.2, size=n)
    fit_mismatch_score = np.where(used_tryon, latent_mismatch, np.nan)

    # ── Return timing ───────────────────────────────────────────────────
    days_since_purchase = rng.gamma(shape=2.2, scale=5.0, size=n).clip(1, 60).round(0)

    # ── Return reason ───────────────────────────────────────────────────
    # A genuine poor fit pushes the reason toward "doesn't fit".
    fit_push = np.where(used_tryon, latent_mismatch, 0.35)
    reason_logits = np.column_stack([
        0.55 + 2.4 * fit_push,                     # doesnt_fit
        0.35 + 0.30 * (discount_pct / 70.0) + 1.30 * is_abuser,     # not_as_described
        -0.25 + 0.20 * np.isin(garment_type, ("jacket", "dress"))
              + 1.20 * is_abuser,                  # damaged
        0.50 - 0.60 * fit_push,                    # changed_mind
        -0.70 * np.ones(n),                        # wrong_item
    ])
    reason_p = np.exp(reason_logits)
    reason_p /= reason_p.sum(axis=1, keepdims=True)
    reason_idx = np.array([rng.choice(len(RETURN_REASONS), p=row) for row in reason_p])
    return_reason = np.array(RETURN_REASONS)[reason_idx]

    # ── Fraud propensity ────────────────────────────────────────────────
    claims_fit = return_reason == "doesnt_fit"
    # The core interaction: "doesn't fit" is only suspicious when the
    # try-on at purchase said it *would* fit.
    inconsistent_fit_claim = claims_fit & used_tryon & (latent_mismatch < 0.25)
    # Or when they claim a fit problem having never checked the fit.
    unverified_fit_claim = claims_fit & ~used_tryon

    logit = (
        -3.30
        + 3.40 * is_abuser
        + 2.60 * (prior_return_rate - 0.25)
        + 0.85 * (return_reason == "not_as_described")
        + 0.80 * (return_reason == "damaged")
        + 1.80 * inconsistent_fit_claim
        + 0.80 * unverified_fit_claim
        # A high measured mismatch is exculpatory: the garment really was wrong.
        - 1.60 * np.nan_to_num(fit_mismatch_score, nan=0.0) * claims_fit
        + 0.55 * (np.log(order_value_inr) - np.log(2000.0))
        + 0.70 * (days_since_purchase > 28)
        - 0.50 * (account_age_days / 365.0).clip(0, 4)
        + 0.40 * (discount_pct > 45)
    )

    # Label noise: real fraud labels come from imperfect manual review, so
    # even a perfectly specified model cannot reach AUC 1.0 here.
    logit += rng.normal(0.0, 0.55, size=n)
    is_fraud = (rng.random(n) < _sigmoid(logit)).astype(int)

    frame = pd.DataFrame({
        "fit_mismatch_score": fit_mismatch_score,
        "garment_type": garment_type,
        "return_reason": return_reason,
        "days_since_purchase": days_since_purchase,
        "order_value_inr": order_value_inr,
        "prior_orders_12m": prior_orders_12m,
        "prior_returns_12m": prior_returns_12m,
        "prior_return_rate": prior_return_rate,
        "account_age_days": account_age_days.round(0),
        "discount_pct": discount_pct,
        "used_tryon": used_tryon,
        TARGET: is_fraud,
    })
    return frame


def split_frame(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic three-way split: train / validation / test.

    The validation split exists solely to choose the decision threshold. The
    test split is touched once, for reporting. Selecting the threshold on
    test would make the reported cost optimistically biased.
    """
    rng = np.random.default_rng(settings.seed)
    order = rng.permutation(len(frame))
    n_train = int(len(frame) * settings.train_frac)
    n_valid = int(len(frame) * settings.valid_frac)

    idx_train = order[:n_train]
    idx_valid = order[n_train:n_train + n_valid]
    idx_test = order[n_train + n_valid:]

    return (
        frame.iloc[idx_train].reset_index(drop=True),
        frame.iloc[idx_valid].reset_index(drop=True),
        frame.iloc[idx_test].reset_index(drop=True),
    )
