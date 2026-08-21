"""End-to-end evaluation report.

Run:  returns_risk/.venv/bin/python -m returns_risk.evaluate
Flags: --ratio 5.0   --model logreg|gbt   --seed N
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .config import CostMatrix, Settings
from .data import FEATURES, TARGET, load_returns, split_frame
from .decision import Action, decide
from .pipeline import fit_and_score, ranking_quality
from .threshold import (
    break_even_probability,
    candidate_table,
    evaluate_threshold,
    ratio_sensitivity,
    select_min_cost,
    sweep,
)

RULE = "─" * 78


def _h1(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def _h2(text: str) -> None:
    print(f"\n{text}\n{'·' * len(text)}")


def _table(frame: pd.DataFrame, floats: dict[str, str] | None = None) -> str:
    out = frame.copy()
    for col, fmt in (floats or {}).items():
        if col in out.columns:
            out[col] = out[col].map(lambda v: format(v, fmt) if pd.notna(v) else "—")
    return out.to_string(index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cost-sensitive return-fraud threshold report")
    parser.add_argument("--ratio", type=float, default=5.0,
                        help="false-positive cost as a multiple of false-negative cost")
    parser.add_argument("--model", choices=("logreg", "gbt"), default="logreg")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--samples", type=int, default=12_000)
    args = parser.parse_args(argv)

    settings = Settings(seed=args.seed, n_samples=args.samples, model=args.model)
    costs = CostMatrix(false_positive=args.ratio, false_negative=1.0)

    # ── Data ────────────────────────────────────────────────────────────
    frame = load_returns(n=settings.n_samples, seed=settings.seed)
    train, valid, test = split_frame(frame, settings)

    _h1("RETURN-FRAUD RISK — COST-SENSITIVE THRESHOLD REPORT")
    print(
        "\n⚠️  SYNTHETIC DATA. There is no real returns log in this repository.\n"
        "    Every number below describes behaviour on a generated dataset and is\n"
        "    NOT evidence of real-world performance. Re-run against a real extract\n"
        "    before making any policy decision.\n"
    )
    print(f"Cost matrix     : {costs.describe()}")
    print(f"Model           : {settings.model}")
    print(f"Seed            : {settings.seed}")
    print(
        f"Split           : train {len(train):,} | valid {len(valid):,} | test {len(test):,}"
    )
    print(
        f"Fraud base rate : train {train[TARGET].mean():.3%} | "
        f"valid {valid[TARGET].mean():.3%} | test {test[TARGET].mean():.3%}"
    )
    print(
        "\nThe threshold is chosen on VALIDATION and reported on TEST. Choosing it\n"
        "on test would make the reported cost optimistically biased."
    )

    # ── Model comparison ────────────────────────────────────────────────
    _h1("1 · MODEL SELECTION (validation)")
    comparison = []
    fitted: dict[str, tuple] = {}
    for kind in ("logreg", "gbt"):
        model, scored = fit_and_score(kind, settings.seed, train,
                                      {"valid": valid, "test": test})
        fitted[kind] = (model, scored)
        q = ranking_quality(scored["valid"])
        comparison.append({"model": kind, **q})
    print()
    print(_table(pd.DataFrame(comparison), {
        "roc_auc": ".4f", "avg_precision": ".4f", "brier": ".4f", "base_rate": ".4f",
    }))
    print(
        "\nBrier score is the calibration check. The cost sweep assumes the predicted\n"
        "probabilities mean what they say; a poorly calibrated model would make the\n"
        "chosen threshold arbitrary even if its ranking (AUC) looked fine."
    )

    model, scored = fitted[settings.model]
    valid_split, test_split = scored["valid"], scored["test"]

    # ── Threshold candidates ────────────────────────────────────────────
    _h1(f"2 · THRESHOLD CANDIDATES (validation, FP:FN = {costs.fp_fn_ratio:g}:1)")
    table = candidate_table(valid_split.y_true, valid_split.y_prob, costs, min_precision=0.60)
    print()
    print(_table(table[[
        "label", "threshold", "precision", "recall", "fpr", "flag_rate",
        "f1", "fp", "fn", "cost_per_case",
    ]], {
        "threshold": ".3f", "precision": ".3f", "recall": ".3f", "fpr": ".4f",
        "flag_rate": ".3f", "f1": ".3f", "cost_per_case": ".4f",
    }))
    print(
        "\nRead the last column, not the F1 column. The max-F1 row is the point a\n"
        "conventional pipeline would ship; under this cost matrix it is not the\n"
        "cheapest, because F1 treats a wrongly-flagged customer and a missed fraud\n"
        "as equally bad and the business does not."
    )

    chosen = select_min_cost(sweep(valid_split.y_true, valid_split.y_prob, costs))
    break_even = break_even_probability(costs)

    print(
        f"\nTheoretical break-even : P(fraud) > {break_even:.3f}"
        f"   [ = FP / (FP + FN) = {costs.false_positive:g} / "
        f"{costs.false_positive + costs.false_negative:g} ]"
    )
    print(f"Empirical cost-optimal : P(fraud) > {chosen.threshold:.3f}   (swept on validation)")
    gap = abs(chosen.threshold - break_even)
    if gap <= 0.10:
        print("The two agree, which is what a well-calibrated model should produce.")
    else:
        print(
            f"These differ by {gap:.3f}. The empirical sweep is authoritative — it\n"
            "optimises the real cost on real scores — but a large gap is a\n"
            "calibration warning worth investigating before shipping."
        )

    # ── Held-out performance ────────────────────────────────────────────
    _h1("3 · HELD-OUT TEST PERFORMANCE AT THE CHOSEN THRESHOLD")
    on_test = evaluate_threshold(test_split.y_true, test_split.y_prob, chosen.threshold, costs)
    default = evaluate_threshold(test_split.y_true, test_split.y_prob, 0.5, costs)

    print(f"\nChosen threshold        : {chosen.threshold:.4f}  (selected on validation)")
    print(f"\n  Precision             : {on_test.precision:.3f}")
    print(f"  Recall                : {on_test.recall:.3f}")
    print(f"  False-positive rate   : {on_test.false_positive_rate:.4f}"
          f"   ({on_test.fp} of {on_test.fp + on_test.tn} genuine returns flagged)")
    print(f"  Flag rate             : {on_test.flag_rate:.3f}"
          f"   ({on_test.tp + on_test.fp} of {test_split.n} returns to review)")
    print(f"  Weighted cost (total) : {on_test.total_cost:.1f} loss units")
    print(f"  Weighted cost (/case) : {on_test.cost_per_case:.4f} loss units")
    print(f"\n  Confusion  TP={on_test.tp}  FP={on_test.fp}  FN={on_test.fn}  TN={on_test.tn}")

    delta = default.total_cost - on_test.total_cost
    pct = (delta / default.total_cost * 100) if default.total_cost else 0.0
    print(f"\n  vs default 0.5 threshold:")
    print(f"    cost {default.total_cost:.1f} -> {on_test.total_cost:.1f} "
          f"({delta:+.1f}, {pct:+.1f}%)")
    print(f"    false positives {default.fp} -> {on_test.fp}"
          f"   ({default.fp - on_test.fp} fewer genuine customers pulled into review)")
    print(f"    false negatives {default.fn} -> {on_test.fn}"
          f"   ({on_test.fn - default.fn:+d} more frauds absorbed — the accepted trade)")

    # ── Sensitivity ─────────────────────────────────────────────────────
    _h1("4 · SENSITIVITY TO THE COST RATIO")
    sens = ratio_sensitivity(
        valid_split.y_true, valid_split.y_prob,
        test_split.y_true, test_split.y_prob,
    )
    print()
    print(_table(sens, {
        "threshold": ".3f", "test_precision": ".3f", "test_recall": ".3f",
        "test_fpr": ".4f", "test_flag_rate": ".3f", "test_cost_per_case": ".4f",
    }))
    print(
        "\nThe ratio is a policy input, not a measurement. This table exists so the\n"
        "person who owns the policy can see what their choice actually changes\n"
        "before signing off on it."
    )

    # ── Worked false positive ───────────────────────────────────────────
    _h1("5 · WORKED EXAMPLE — A REAL FALSE POSITIVE FROM THE TEST SET")

    flagged = test_split.y_prob >= chosen.threshold
    fp_mask = flagged & (test_split.y_true == 0)

    if not fp_mask.any():
        print("\nNo false positives at this threshold — nothing to demonstrate.")
    else:
        # The worst case: the genuine return the model was most wrongly
        # confident about. If graceful failure holds here, it holds anywhere.
        fp_indices = np.flatnonzero(fp_mask)
        worst = int(fp_indices[np.argmax(test_split.y_prob[fp_indices])])
        row = test.iloc[worst]

        print(f"\nTest-set row {worst} — a GENUINE return (is_fraud = 0) that the model flagged.")
        print("Selected as the highest-confidence false positive, i.e. the worst case.\n")

        for name in FEATURES:
            value = row[name]
            if isinstance(value, float) and np.isnan(value):
                shown = "— (shopper skipped try-on)"
            elif isinstance(value, (float, np.floating)):
                shown = f"{value:,.3f}"
            elif isinstance(value, (bool, np.bool_)):
                shown = str(bool(value))
            else:
                shown = f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)
            print(f"    {name:<22} {shown}")

        decision = decide(
            probability=float(test_split.y_prob[worst]),
            threshold=chosen.threshold,
            row=row.to_dict(),
        )

        print(f"\n  Ground truth            : GENUINE (is_fraud = 0)")
        print(f"  Model P(fraud)          : {test_split.y_prob[worst]:.4f}")
        print(f"  Threshold               : {chosen.threshold:.4f}")
        print(f"\n  Returned payload        : {decision.to_payload()}")
        print(f"  Review-queue payload    : {decision.to_review_payload()}")

        assert decision.action is Action.ROUTE_TO_REVIEW
        assert decision.action.value != "auto_deny"

        print(
            "\n  What the customer experiences:\n"
            "    The refund is NOT denied and NOT blocked. The return joins a manual\n"
            "    review queue with the reason codes above attached, and a human makes\n"
            "    the call. The cost of this error is a delay and some reviewer time —\n"
            "    not a refused refund.\n"
            "\n  This is the graceful-failure requirement: the system's worst mistake\n"
            "  on its most confident wrong answer is still only a request for a\n"
            "  second opinion."
        )

    # ── Guarantees ──────────────────────────────────────────────────────
    _h1("6 · ENFORCED GUARANTEES")
    print(f"\n  Actions this system can emit : {[a.value for a in Action]}")
    print("  Automated denial             : unreachable — no such value exists in the")
    print("                                 Action enum, and decision.py fails at import")
    print("                                 if one is ever added (see tests.py).")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
