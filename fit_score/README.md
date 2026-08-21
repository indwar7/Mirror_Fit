# Fit Analyser

Grades a garment against a body: **made for you / good fit / oversized /
undersized / poor fit**, plus a 0–100 fit score.

```bash
returns_risk/.venv/bin/python -m fit_score.model            # report
returns_risk/.venv/bin/python -m fit_score.model --export   # + demo/fit_model.json
```

> ⚠️ **Synthetic data.** Generated from published apparel ease allowances, not
> measured from real garments or customers. Metrics describe the generative
> process, not real-world accuracy.

## What it measures

Fit is geometry. Four garment measurements against four body measurements,
expressed as **ratios** — scale-invariant, because the try-on measures in pixels
and pixel scale changes every time the shopper moves.

Tolerance bands are industry ease allowances, not invented constants:

| Ease | chest ratio |
|---|---|
| too tight | < 1.04 |
| slim | 1.05–1.11 |
| regular | 1.11–1.18 |
| relaxed | 1.20–1.29 |
| oversized | 1.32+ |

Ease differs by garment — a jacket is worn over layers — so `IDEAL_EASE` is keyed
by type rather than being one global number.

## The finding that shaped the model

Trained on the raw ratios, logistic regression scored **0.00 recall on
MADE_FOR_YOU** — it never once identified a tailored garment. Not a tuning
problem, a representational one: "every measurement close to ideal" is a
**bounded box** around the origin, and a linear model can only cut with
hyperplanes. It cannot enclose a region.

Three derived features fix it by turning those boxes into half-spaces:

- `worst_abs_dev` — how wrong is the worst measurement?
- `dev_spread` — do the measurements disagree with each other?
- `core_dev_mean` — which direction, on chest and shoulder?

They are also the three questions a tailor asks, which is why they work rather
than being a coincidence.

| | raw ratios | + derived |
|---|---|---|
| accuracy | 0.708 | **0.966** |
| macro F1 | 0.469 | **0.959** |
| MADE_FOR_YOU recall | **0.000** | **0.913** |

## Held-out results (test n=2,800)

| Class | Recall | Precision |
|---|---|---|
| MADE_FOR_YOU | 0.913 | 0.964 |
| GOOD_FIT | 0.986 | 0.971 |
| OVERSIZED | 0.963 | 0.974 |
| UNDERSIZED | 0.930 | 0.947 |
| POOR_FIT | 0.972 | 0.972 |

**Wrong-direction sizing advice — 0.46%.** This is the error that matters:
telling a shopper to size up when they should size down is worse than saying
nothing, so it is tracked separately rather than hidden inside an accuracy
figure.

## Model choice

Gradient boosting scores higher (0.998 vs 0.966) and is trained alongside for
comparison. Logistic regression ships because it exports to ~9 KB of JSON and
runs in the browser with no server, and because every coefficient is
inspectable. The 3-point gap is on synthetic data; the deployment property is
real.

## Where it connects

`fit_score` → `mismatch_from_score()` → **`fit_mismatch_score`**, which is an
input feature of `returns_risk`. What the mirror measured at purchase is what
the returns desk sees weeks later: a high mismatch makes a *"doesn't fit"* claim
**more** credible, not less, because the garment really was cut wrong for that
body.
