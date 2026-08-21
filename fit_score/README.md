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

---

# Trained on real data

The synthetic model above is retained for reference, but the shipped
Fit Analyser now trains on **real fit outcomes**.

```bash
returns_risk/.venv/bin/python -m fit_score.real_model --export
```

**Source:** Misra, Wan & McAuley, *"Decomposing fit semantics for product size
recommendation in metric spaces"*, RecSys 2018 — the RentTheRunway dataset
([UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets.html)). **192,544 real
rentals**, each with the customer's measurements, the size they took, and what
they reported afterwards: `small` / `fit` / `large`.

## The number that matters

**73.8% of the data is labelled "fit."** So a model that always answers "fit"
scores **73.8% accuracy** while never once warning a shopper. Accuracy is the
wrong headline; macro F1 is the honest one.

| | Accuracy | Macro F1 |
|---|---|---|
| always "fit" | **0.742** | 0.284 |
| shipped model | 0.665 | **0.471** |

The model **gives up accuracy on purpose** to actually call small and large.

## Honest performance

| Class | Recall | Precision | Support |
|---|---|---|---|
| small | 0.282 | 0.315 | 5,128 |
| fit | 0.788 | 0.783 | 28,563 |
| large | 0.341 | 0.318 | 4,818 |

**Precision on small/large is ~0.31 — roughly two in three warnings are false
alarms.** Wrong-direction advice (telling someone to size up when they should
size down) is **1.62%**.

That is the honest state of the art from body measurements alone, and it is
worth contrasting with the synthetic model's 0.966 accuracy. The synthetic
number was a property of the generator, not of the problem.

## What actually predicts fit

Body measurements are weak. **Collaborative history is strong** — adding
per-item and per-user historical fit rates moved macro F1 from 0.369 to 0.471
and cut wrong-direction advice from 5.12% to 1.62%.

The learned coefficients are readable and match retail intuition:

| Feature | Toward | Weight |
|---|---|---|
| `user_small_rate` | small | +1.47 |
| `user_large_rate` | large | +1.41 |
| `item_small_rate` | small | +0.76 |
| `category_coat` | large | +0.71 |
| `category_skirt` | small | +0.71 |

Some people report "small" about everything; some items just run small; coats
run large and skirts run small. All learned, none hand-coded.

Those rates are computed on the **training split only** and joined onto
validation and test. Computing them over the full frame would leak each test
row's own outcome into its features.

## Why only three classes

The real data has `small` / `fit` / `large`. **"Made for you" and "poor fit"
cannot be learned from it** — nobody measured the garments, so there is no
ground truth for tailored-versus-merely-acceptable. The five-tier version
exists only in the synthetic model, where the generator defined those bands.

Claiming five tiers from this data would be inventing labels.

## Limitations

- **Rental, not retail.** Renters choose differently from buyers, and the
  catalogue skews to dresses and gowns (73% of rows).
- **Self-reported labels.** "Fit" means different things to different people.
- **Cold start.** 52.8% of test users are unseen in training; for a genuinely
  new shopper the collaborative signal falls back to the global prior.
- Gradient boosting was tried and did *worse* here (macro F1 0.347) — it
  overfit the collaborative features badly (log loss 1.66 vs 0.96).

---

# Does it actually reduce returns?

Accuracy was the wrong thing to optimise. A fit warning **moves the shopper to a
different size**, so it is never free:

| Prediction | Truth | Outcome |
|---|---|---|
| "fit" | anything | no advice given — neutral |
| small/large, **right** | matches | **return prevented** |
| small/large, **wrong** | actually fit | **return caused** — we moved them off the size that worked |
| small/large, wrong way | opposite | still a return, plus lost trust |

**Net = prevented − caused.** Measured on 38,509 held-out rentals:

| | Warnings | Prevented | Caused | **Net** | Precision |
|---|---|---|---|---|---|
| warn on every prediction | 21,682 | 6,592 | 14,334 | **−7,742** | 0.304 |
| warn above 0.85 confidence | 2,593 | 1,597 | 963 | **+634** | 0.616 |

**A model that always speaks destroys value** — it causes more than twice the
returns it prevents. The confidence gate is what makes it deployable, and it is
the same lesson as the returns model: the hard part is knowing when *not* to act.

Every prevented return is also a refund that never reaches the payment rail.

## Two bugs this uncovered

**Target leakage.** `item_small_rate` was computed from train labels, so for a
training row it partly encoded that row's own outcome. Gradient boosting
exploited it and collapsed — log loss **1.66**, worse than a uniform guess
(1.10). Leave-one-out encoding (excluding each row from its own group's
statistic) fixed it:

| | before LOO | after LOO |
|---|---|---|
| GBT log loss | 1.66 | **0.70** |
| macro F1 | 0.471 | **0.489** |

**Dead sliders.** The four history sliders were not wired into `readForm`, so
they silently did nothing and the model saw imputed medians. Caught by diffing
the browser's feature vector against scikit-learn's, feature by feature.

## Verification

The browser reproduces scikit-learn exactly — all 89 features identical, and all
three presets match on class, confidence and `p_fit`. Missing values propagate
as missing rather than being replaced with plausible defaults, so the
missing-indicators fire the same way they did at training time.
