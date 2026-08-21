"""Fit classifier trained on the real RentTheRunway outcomes.

Run:
    returns_risk/.venv/bin/python -m fit_score.real_model
    returns_risk/.venv/bin/python -m fit_score.real_model --export

Read the macro F1, not the accuracy. 73.8% of this dataset is labelled
"fit", so a model that always answers "fit" scores 73.8% accuracy while
being completely useless — it would never once warn a shopper. Every
number here is reported against that baseline.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .real_data import (
    CATEGORICAL,
    coverage,
    CLASSES,
    FEATURES,
    NUMERIC,
    TARGET,
    add_relative_size,
    load_raw,
    split_frame,
)

OUT = pathlib.Path(__file__).resolve().parent.parent / "demo" / "fit_model_real.json"
SEED = 20260821


def build(kind: str, class_weight=None) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                              min_frequency=30), CATEGORICAL),
    ])
    if kind == "logreg":
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED,
                                 class_weight=class_weight)
    else:
        clf = HistGradientBoostingClassifier(
            max_depth=6, max_iter=400, learning_rate=0.07,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12,
            random_state=SEED, class_weight=class_weight,
        )
    return Pipeline([("prep", prep), ("clf", clf)])


def report(name, y_true, proba, classes) -> dict:
    pred = classes[proba.argmax(1)]
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "log_loss": log_loss(y_true, proba, labels=list(classes)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit classifier on real data")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args(argv)

    rule = "─" * 76
    print(f"\n{rule}\nFIT CLASSIFIER — trained on REAL RentTheRunway outcomes\n{rule}")

    frame = load_raw()
    train, valid, test = split_frame(frame, SEED)
    train, valid, test = add_relative_size(train, valid, test)

    counts = frame[TARGET].value_counts()
    majority = counts.max() / len(frame)
    print(f"\nSource     : Misra, Wan & McAuley, RecSys 2018 (UCSD)")
    print(f"Rentals    : {len(frame):,} real customer fit outcomes")
    print(f"Split      : train {len(train):,} | valid {len(valid):,} | test {len(test):,}")
    print("Labels     : " + "  ".join(
        f"{c} {counts.get(c,0):,} ({counts.get(c,0)/len(frame):.1%})" for c in CLASSES))
    print(f"\nMajority-class baseline: {majority:.3%} accuracy by always answering \"fit\".")
    print("That is the number to beat, and it is why macro F1 is the headline here —")
    print("a model that never warns anyone is worthless however accurate it looks.")

    cov = coverage(train, test)
    print(f"\nCold start in test: item seen in train {cov['item_seen']:.1%}, "
          f"user seen {cov['user_seen']:.1%}")
    print("Collaborative features can only help where there is history to read.")

    print(f"\nField coverage after parsing (missing values are imputed, with an indicator):")
    for col in ("height_in", "weight_lb", "bust_band", "bust_cup", "age", "bmi"):
        print(f"  {col:<12} {train[col].notna().mean():6.1%}")

    y_train = train[TARGET].to_numpy()
    y_valid = valid[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()

    # ── Model selection ─────────────────────────────────────────────────
    print(f"\n{rule}\n1 · MODEL SELECTION (validation)\n{rule}\n")
    rows, fitted = [], {}
    for name, kind, weight in (
        ("logreg", "logreg", None),
        ("logreg + balanced", "logreg", "balanced"),
        ("gbt", "gbt", None),
        ("gbt + balanced", "gbt", "balanced"),
    ):
        model = build(kind, weight)
        model.fit(train[FEATURES], y_train)
        proba = model.predict_proba(valid[FEATURES])
        rows.append(report(name, y_valid, proba, model.classes_))
        fitted[name] = model
    rows.append({"model": "always \"fit\"", "accuracy": (y_valid == "fit").mean(),
                 "macro_f1": f1_score(y_valid, np.full(len(y_valid), "fit"),
                                      average="macro"), "log_loss": np.nan})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\nclass_weight='balanced' trades accuracy for macro F1 — it stops the model\n"
        "hiding behind the majority class and makes it actually call small/large."
    )

    best = max(
        (r for r in rows if r["model"] != 'always "fit"'),
        key=lambda r: r["macro_f1"],
    )["model"]
    model = fitted[best]
    classes = model.classes_
    print(f"\nSelected: {best}  (highest macro F1 on validation)")

    # ── Held-out ────────────────────────────────────────────────────────
    print(f"\n{rule}\n2 · HELD-OUT TEST PERFORMANCE\n{rule}\n")
    proba = model.predict_proba(test[FEATURES])
    pred = classes[proba.argmax(1)]
    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    base_acc = (y_test == "fit").mean()
    base_f1 = f1_score(y_test, np.full(len(y_test), "fit"), average="macro")
    print(f"Accuracy  : {acc:.4f}   (baseline {base_acc:.4f}, {acc-base_acc:+.4f})")
    print(f"Macro F1  : {mf1:.4f}   (baseline {base_f1:.4f}, {mf1-base_f1:+.4f})")

    order = [c for c in CLASSES if c in classes]
    cm = confusion_matrix(y_test, pred, labels=order)
    width = max(len(c) for c in order) + 2
    print("\nConfusion matrix (rows = truth, cols = predicted)\n")
    print(" " * width + "".join(f"{c:>9}" for c in order))
    for i, c in enumerate(order):
        print(f"{c:<{width}}" + "".join(f"{v:>9,}" for v in cm[i]))

    print(f"\n{rule}\n3 · PER-CLASS BEHAVIOUR (test)\n{rule}\n")
    per = []
    for c in order:
        mask = y_test == c
        per.append({
            "class": c,
            "support": int(mask.sum()),
            "recall": float((pred[mask] == c).mean()) if mask.any() else 0.0,
            "precision": float((y_test[pred == c] == c).mean()) if (pred == c).any() else 0.0,
        })
    print(pd.DataFrame(per).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # The error that actually costs money.
    if "small" in order and "large" in order:
        i, j = order.index("small"), order.index("large")
        wrong = int(cm[i][j] + cm[j][i])
        print(f"\nWrong-direction advice (small<->large): {wrong:,} of {cm.sum():,} "
              f"({wrong/cm.sum():.2%})")
        print("Telling a shopper to size the wrong way is worse than staying quiet,")
        print("so it is tracked on its own rather than averaged into accuracy.")

    # ── What it learned ─────────────────────────────────────────────────
    if hasattr(model.named_steps["clf"], "coef_"):
        print(f"\n{rule}\n4 · WHAT THE MODEL LEARNED\n{rule}\n")
        names = _feature_names(model)
        coef = model.named_steps["clf"].coef_
        for ci, cls in enumerate(classes):
            top = sorted(zip(names, coef[ci]), key=lambda t: -abs(t[1]))[:5]
            print(f'  toward "{cls}":')
            for n, v in top:
                print(f"      {v:+.3f}  {n}")

    if args.export:
        _export(model, test, proba, pred, y_test, acc, mf1, base_acc)
    print()
    return 0


def _feature_names(model) -> list[str]:
    prep = model.named_steps["prep"]
    num = prep.named_transformers_["num"]
    names = list(NUMERIC)
    ind = getattr(num.named_steps["impute"], "indicator_", None)
    if ind is not None:
        names += [f"{NUMERIC[i]}__MISSING" for i in ind.features_]
    ohe = prep.named_transformers_["cat"]
    names += list(ohe.get_feature_names_out(CATEGORICAL))
    return names


def _export(model, test, proba, pred, y_test, acc, mf1, base_acc) -> None:
    """Export weights + real held-out presets for the browser."""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        print("\n[export] selected model is not linear; skipping browser export.")
        return

    prep = model.named_steps["prep"]
    num = prep.named_transformers_["num"]
    imputer = num.named_steps["impute"]
    scaler = num.named_steps["scale"]
    ohe = prep.named_transformers_["cat"]

    presets = []
    for cls in CLASSES:
        mask = (y_test == cls) & (pred == cls)
        if not mask.any():
            continue
        idx = int(np.flatnonzero(mask)[np.argmax(proba[mask].max(1))])
        row = test.iloc[idx]
        presets.append({
            "label": {"small": "Ran small", "fit": "True to size",
                      "large": "Ran large"}[cls],
            "cls": cls,
            "truth": cls,
            "values": {k: (None if pd.isna(row[k]) else float(row[k])) for k in NUMERIC}
                      | {k: str(row[k]) for k in CATEGORICAL},
        })

    artifact = {
        "_comment": (
            "Trained on REAL RentTheRunway fit outcomes (Misra/Wan/McAuley, RecSys "
            "2018). Labels are customer self-reported: small / fit / large."
        ),
        "source": {
            "name": "RentTheRunway fit dataset",
            "citation": "Misra, Wan & McAuley, RecSys 2018 (UCSD)",
            "url": "https://cseweb.ucsd.edu/~jmcauley/datasets.html",
            "rentals": int(len(test) * 5),
        },
        "classes": list(clf.classes_),
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
        "preprocess": {
            "impute_medians": [None if np.isnan(v) else float(v)
                               for v in imputer.statistics_],
            "indicator_features": [NUMERIC[i] for i in imputer.indicator_.features_],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "categories": {c: list(map(str, v))
                           for c, v in zip(CATEGORICAL, ohe.categories_)},
            "infrequent": {
                c: (list(map(str, arr)) if arr is not None else [])
                for c, arr in zip(CATEGORICAL, ohe.infrequent_categories_ or
                                  [None] * len(CATEGORICAL))
            },
        },
        "model": {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()},
        "test_metrics": {
            "n": int(len(test)),
            "accuracy": float(acc),
            "macro_f1": float(mf1),
            "baseline_accuracy": float(base_acc),
        },
        "presets": presets,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size:,} bytes)  presets {len(presets)}")


if __name__ == "__main__":
    sys.exit(main())
