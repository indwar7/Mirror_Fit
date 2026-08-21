"""Fit classifier: train, evaluate, export.

Multinomial logistic regression over the six geometric ratios plus garment
type. Chosen over a tree ensemble for the same reason as the returns
model: the probabilities are used directly (they drive the confidence a
shopper sees), a linear model keeps them calibrated, and every coefficient
is inspectable. Gradient boosting is trained alongside so the choice stays
evidence-based rather than habitual.

Run:
    returns_risk/.venv/bin/python -m fit_score.model            # report
    returns_risk/.venv/bin/python -m fit_score.model --export   # + JSON
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import TARGET, load_fits, split_frame
from .spec import (
    CATEGORICAL,
    CLASSES,
    CLASS_COPY,
    FEATURES,
    IDEAL_EASE,
    TAILORED_TOLERANCE,
    WEARABLE_TOLERANCE,
    deviations,
    fit_score,
    mismatch_from_score,
)

OUT = pathlib.Path(__file__).resolve().parent.parent / "demo" / "fit_model.json"
SEED = 20260821


def build(kind: str) -> Pipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler(), FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    if kind == "logreg":
        clf = LogisticRegression(max_iter=3000, C=2.0, random_state=SEED)
    else:
        clf = HistGradientBoostingClassifier(
            max_depth=5, max_iter=260, learning_rate=0.07,
            min_samples_leaf=30, random_state=SEED,
        )
    return Pipeline([("prep", prep), ("clf", clf)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit classifier")
    ap.add_argument("--export", action="store_true", help="write demo/fit_model.json")
    ap.add_argument("--samples", type=int, default=14_000)
    args = ap.parse_args(argv)

    frame = load_fits(n=args.samples, seed=SEED)
    train, valid, test = split_frame(frame, SEED)

    rule = "─" * 74
    print(f"\n{rule}\nFIT CLASSIFIER — how well does this garment fit this body?\n{rule}")
    print(
        "\n⚠️  SYNTHETIC DATA. Generated from apparel ease allowances, not measured\n"
        "    from real garments or customers. Metrics describe the generative\n"
        "    process, not real-world accuracy.\n"
    )
    print(f"Split      : train {len(train):,} | valid {len(valid):,} | test {len(test):,}")
    counts = frame[TARGET].value_counts()
    print("Class mix  : " + "  ".join(
        f"{c} {counts.get(c,0)/len(frame):.1%}" for c in CLASSES))

    # ── Model comparison ────────────────────────────────────────────────
    print(f"\n{rule}\n1 · MODEL SELECTION (validation)\n{rule}\n")
    fitted = {}
    rows = []
    for kind in ("logreg", "gbt"):
        model = build(kind)
        model.fit(train[FEATURES + CATEGORICAL], train[TARGET])
        proba = model.predict_proba(valid[FEATURES + CATEGORICAL])
        pred = model.classes_[proba.argmax(1)]
        rows.append({
            "model": kind,
            "accuracy": accuracy_score(valid[TARGET], pred),
            "macro_f1": f1_score(valid[TARGET], pred, average="macro"),
            "log_loss": log_loss(valid[TARGET], proba, labels=list(model.classes_)),
        })
        fitted[kind] = model
    print(pd.DataFrame(rows).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\nLog loss is the calibration check — the shopper is shown a confidence,\n"
        "so the probabilities have to mean something, not just rank correctly."
    )

    model = fitted["logreg"]
    labels = list(model.classes_)

    # ── Held-out ────────────────────────────────────────────────────────
    print(f"\n{rule}\n2 · HELD-OUT TEST PERFORMANCE\n{rule}\n")
    proba = model.predict_proba(test[FEATURES + CATEGORICAL])
    pred = model.classes_[proba.argmax(1)]
    print(f"Accuracy  : {accuracy_score(test[TARGET], pred):.4f}")
    print(f"Macro F1  : {f1_score(test[TARGET], pred, average='macro'):.4f}")

    order = [c for c in CLASSES if c in labels]
    cm = confusion_matrix(test[TARGET], pred, labels=order)
    print("\nConfusion matrix (rows = truth, cols = predicted)\n")
    width = max(len(c) for c in order) + 2
    print(" " * width + "".join(f"{c[:9]:>11}" for c in order))
    for i, c in enumerate(order):
        print(f"{c:<{width}}" + "".join(f"{v:>11}" for v in cm[i]))

    print(
        "\nThe costly confusion is OVERSIZED <-> UNDERSIZED: telling a shopper to\n"
        "size the wrong way is worse than saying nothing. Check that off-diagonal."
    )
    if "OVERSIZED" in order and "UNDERSIZED" in order:
        i, j = order.index("OVERSIZED"), order.index("UNDERSIZED")
        wrong_way = int(cm[i][j] + cm[j][i])
        total = int(cm.sum())
        print(f"  wrong-direction sizing advice: {wrong_way} of {total} ({wrong_way/total:.2%})")

    # ── Per-class recall ────────────────────────────────────────────────
    print(f"\n{rule}\n3 · PER-CLASS BEHAVIOUR (test)\n{rule}\n")
    per = []
    for c in order:
        mask = test[TARGET].to_numpy() == c
        got = pred[mask]
        per.append({
            "class": c,
            "support": int(mask.sum()),
            "recall": float((got == c).mean()) if mask.any() else 0.0,
            "precision": float((test[TARGET].to_numpy()[pred == c] == c).mean())
                          if (pred == c).any() else 0.0,
        })
    print(pd.DataFrame(per).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ── Worked example ──────────────────────────────────────────────────
    print(f"\n{rule}\n4 · WORKED EXAMPLE\n{rule}\n")
    idx = int(np.argmax(proba.max(1) * (test[TARGET].to_numpy() == "UNDERSIZED")))
    row = test.iloc[idx]
    dev = deviations(row)
    score = fit_score(row)
    print(f"Garment    : {row['garment_type']}  size {row['size_label']}")
    for k in ("chest", "shoulder", "length", "sleeve"):
        print(f"  {k:<9} {dev[k]:+.1%} vs ideal ease")
    print(f"\nFit score  : {score:.0f}/100")
    print(f"Verdict    : {pred[idx]}  ({proba[idx].max():.1%} confidence)")
    print(f"Truth      : {row[TARGET]}")
    print(f"\nfit_mismatch_score handed to the returns model: {mismatch_from_score(score):.3f}")
    print(
        "  That is the join between the two systems — what the mirror measured at\n"
        "  purchase is what the returns desk sees weeks later if the item comes back."
    )

    if args.export:
        _export(model, test, proba, pred)
    print()
    return 0


def _export(model, test, proba, pred) -> None:
    """Dump weights + presets so the browser can score without a server."""
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    scaler = prep.named_transformers_["num"]
    ohe = prep.named_transformers_["cat"]

    presets = []
    for target in CLASSES:
        mask = (test[TARGET].to_numpy() == target) & (pred == target)
        if not mask.any():
            mask = test[TARGET].to_numpy() == target
        if not mask.any():
            continue
        i = int(np.flatnonzero(mask)[np.argmax(proba[mask].max(1))])
        row = test.iloc[i]
        presets.append({
            "label": CLASS_COPY[target][0],
            "cls": target,
            "truth": row[TARGET],
            "values": {k: float(row[k]) for k in FEATURES} | {
                "garment_type": str(row["garment_type"]),
                "size_label": str(row["size_label"]),
            },
            "cm": {k: float(row[k]) for k in (
                "body_chest_cm", "body_shoulder_cm", "body_torso_cm", "body_arm_cm",
                "garment_chest_cm", "garment_shoulder_cm",
                "garment_length_cm", "garment_sleeve_cm")},
        })

    artifact = {
        "_comment": (
            "Fit classifier trained on SYNTHETIC data derived from apparel ease "
            "allowances. Demonstrates the measurement architecture, not real accuracy."
        ),
        "classes": list(clf.classes_),
        "class_copy": {k: list(v) for k, v in CLASS_COPY.items()},
        "features": {"numeric": FEATURES, "categorical": CATEGORICAL},
        "preprocess": {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "categories": {c: list(v) for c, v in zip(CATEGORICAL, ohe.categories_)},
        },
        "model": {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()},
        "ease": {k: [v.chest, v.shoulder, v.length, v.sleeve] for k, v in IDEAL_EASE.items()},
        "tolerance": {"tailored": TAILORED_TOLERANCE, "wearable": WEARABLE_TOLERANCE},
        "test_metrics": {
            "n": int(len(test)),
            "accuracy": float(accuracy_score(test[TARGET], pred)),
            "macro_f1": float(f1_score(test[TARGET], pred, average="macro")),
        },
        "presets": presets,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    print(f"  classes {len(artifact['classes'])}  presets {len(presets)}")


if __name__ == "__main__":
    sys.exit(main())
