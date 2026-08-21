"""Feature preprocessing and model fitting.

Design note — why there is no `class_weight="balanced"` here:

    The asymmetry between false positives and false negatives is applied
    exactly once, at the decision threshold (see threshold.py). Re-weighting
    the training objective as well would apply it twice: the model would
    already be skewed toward predicting the minority class, and the
    threshold sweep would then skew it again, on top of probabilities that
    no longer mean what they claim to mean.

    Keeping the training objective unweighted leaves the predicted
    probabilities approximately calibrated, which is a precondition for the
    cost sweep to be meaningful at all. We report Brier score so that
    assumption is checked rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def build_preprocessor() -> ColumnTransformer:
    """Numeric imputation keeps a missingness indicator.

    `fit_mismatch_score` is absent precisely when the shopper skipped
    try-on, and that absence carries signal. Dropping the indicator would
    throw it away and quietly make an unverified fit claim look identical
    to a verified one.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None)

    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("bool", "passthrough", BOOLEAN_FEATURES),
    ])


def build_model(kind: str, seed: int) -> Pipeline:
    if kind == "logreg":
        estimator = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            random_state=seed,
        )
    elif kind == "gbt":
        estimator = HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=220,
            learning_rate=0.06,
            l2_regularization=1.0,
            min_samples_leaf=40,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown model {kind!r}")

    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])


@dataclass(frozen=True)
class ScoredSplit:
    """Predicted fraud probabilities alongside their true labels."""

    name: str
    y_true: np.ndarray
    y_prob: np.ndarray

    @property
    def n(self) -> int:
        return len(self.y_true)

    @property
    def base_rate(self) -> float:
        return float(self.y_true.mean())


def fit_and_score(
    kind: str,
    seed: int,
    train: pd.DataFrame,
    others: dict[str, pd.DataFrame],
) -> tuple[Pipeline, dict[str, ScoredSplit]]:
    """Fit on `train`, then score every frame in `others`."""
    model = build_model(kind, seed)
    model.fit(train[FEATURES], train[TARGET].to_numpy())

    scored: dict[str, ScoredSplit] = {}
    for name, frame in others.items():
        prob = model.predict_proba(frame[FEATURES])[:, 1]
        scored[name] = ScoredSplit(name=name, y_true=frame[TARGET].to_numpy(), y_prob=prob)
    return model, scored


def ranking_quality(split: ScoredSplit) -> dict[str, float]:
    """Threshold-free quality plus a calibration check.

    ROC AUC and average precision measure ranking. Brier score measures
    whether the probabilities themselves are trustworthy — which matters
    here because the whole cost argument is built on top of them.
    """
    return {
        "roc_auc": float(roc_auc_score(split.y_true, split.y_prob)),
        "avg_precision": float(average_precision_score(split.y_true, split.y_prob)),
        "brier": float(brier_score_loss(split.y_true, split.y_prob)),
        "base_rate": split.base_rate,
    }
