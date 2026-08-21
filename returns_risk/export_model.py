"""Export the trained model as a JSON artifact the browser can score.

The shipped model is logistic regression, so inference is a dot product and
a sigmoid — small enough to run client-side with no server round trip. That
matters for the demo: the returns screen keeps working even if the GPU box
is busy or down, and there is nothing to fail live on camera.

Exported: the fitted preprocessing constants (imputer medians, scaler
mean/scale, one-hot categories), the coefficients, the validation-selected
threshold, the candidate table, and four real test-set cases to load as
presets — including the worked false positive.

Run:
    returns_risk/.venv/bin/python -m returns_risk.export_model
Writes:
    demo/returns_model.json
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .config import CostMatrix, Settings
from .data import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_returns,
    split_frame,
)
from .pipeline import fit_and_score
from .threshold import (
    break_even_probability,
    candidate_table,
    evaluate_threshold,
    select_min_cost,
    sweep,
)

OUT = pathlib.Path(__file__).resolve().parent.parent / "demo" / "returns_model.json"


def _py(value):
    """numpy -> json-safe python."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_py(v) for v in value.tolist()]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main() -> int:
    settings = Settings()
    costs = CostMatrix(false_positive=5.0, false_negative=1.0)

    frame = load_returns(n=settings.n_samples, seed=settings.seed)
    train, valid, test = split_frame(frame, settings)

    model, scored = fit_and_score(
        settings.model, settings.seed, train, {"valid": valid, "test": test}
    )
    valid_split, test_split = scored["valid"], scored["test"]

    chosen = select_min_cost(sweep(valid_split.y_true, valid_split.y_prob, costs))
    on_test = evaluate_threshold(
        test_split.y_true, test_split.y_prob, chosen.threshold, costs
    )

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    num_pipe = prep.named_transformers_["num"]
    imputer = num_pipe.named_steps["impute"]
    scaler = num_pipe.named_steps["scale"]
    ohe = prep.named_transformers_["cat"]

    artifact = {
        "_comment": (
            "Trained on SYNTHETIC data — see returns_risk/README.md. Scores a "
            "return request and routes to human review; it can never deny a refund."
        ),
        "generated_from": {
            "seed": settings.seed,
            "model": settings.model,
            "n_train": len(train),
            "fp_fn_ratio": costs.fp_fn_ratio,
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "boolean": BOOLEAN_FEATURES,
        },
        # SimpleImputer(add_indicator=True) appends one indicator column per
        # feature that had missing values during fit. Only fit_mismatch_score
        # does, so the browser must append exactly that one column.
        "preprocess": {
            "impute_medians": _py(imputer.statistics_),
            "indicator_features": [
                NUMERIC_FEATURES[i] for i in _py(imputer.indicator_.features_)
            ],
            "scaler_mean": _py(scaler.mean_),
            "scaler_scale": _py(scaler.scale_),
            "categories": {
                col: _py(cats) for col, cats in zip(CATEGORICAL_FEATURES, ohe.categories_)
            },
        },
        "model": {
            "coef": _py(clf.coef_[0]),
            "intercept": float(clf.intercept_[0]),
        },
        "decision": {
            "threshold": float(chosen.threshold),
            "break_even": float(break_even_probability(costs)),
            "cost_per_launch": 10,
        },
        "test_metrics": {
            "n": int(test_split.n),
            "precision": on_test.precision,
            "recall": on_test.recall,
            "fpr": on_test.false_positive_rate,
            "flag_rate": on_test.flag_rate,
            "total_cost": on_test.total_cost,
            "tp": on_test.tp, "fp": on_test.fp,
            "fn": on_test.fn, "tn": on_test.tn,
        },
        "candidates": json.loads(
            candidate_table(valid_split.y_true, valid_split.y_prob, costs)
            .to_json(orient="records")
        ),
        "presets": _build_presets(test, test_split, chosen.threshold),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=1), encoding="utf-8")

    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    print(f"  threshold {chosen.threshold:.4f}   features {len(artifact['model']['coef'])}")
    print(f"  presets: {[p['label'] for p in artifact['presets']]}")
    return 0


def _build_presets(test, test_split, threshold: float) -> list[dict]:
    """Four real test-set rows worth demonstrating."""
    flagged = test_split.y_prob >= threshold
    truth = test_split.y_true == 1

    def pick(mask, *, highest: bool) -> int | None:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return None
        scores = test_split.y_prob[idx]
        return int(idx[np.argmax(scores) if highest else np.argmin(scores)])

    wanted = [
        ("False positive",
         "A GENUINE return the model wrongly flagged — its most confident mistake. "
         "Watch it route to review, not denial.",
         pick(flagged & ~truth, highest=True)),
        ("Caught fraud",
         "A fraudulent return the model flagged correctly.",
         pick(flagged & truth, highest=True)),
        ("Clean return",
         "An ordinary genuine return. Approved with no friction.",
         pick(~flagged & ~truth, highest=False)),
        ("Missed fraud",
         "Fraud that slipped through. Accepted deliberately: under a 5:1 cost "
         "matrix, flagging borderline cases costs more than absorbing this.",
         pick(~flagged & truth, highest=True)),
    ]

    presets = []
    for label, note, index in wanted:
        if index is None:
            continue
        row = test.iloc[index]
        presets.append({
            "label": label,
            "note": note,
            "truth": int(row[TARGET]),
            "expected_probability": float(test_split.y_prob[index]),
            "values": {
                name: _py(row[name])
                for name in NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
            },
        })
    return presets


if __name__ == "__main__":
    raise SystemExit(main())
