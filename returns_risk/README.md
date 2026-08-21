# Return-fraud risk: cost-sensitive threshold selection

A binary classifier over return requests that outputs a **probability**, and a
threshold chosen by **minimising weighted cost** rather than maximising accuracy
or F1.

The system can flag a return for human review. It cannot deny a refund. That
constraint is enforced in code, not by convention.

```bash
returns_risk/.venv/bin/python -m returns_risk.tests            # 32 checks
returns_risk/.venv/bin/python -m returns_risk.evaluate         # full report
returns_risk/.venv/bin/python -m returns_risk.evaluate --ratio 3 --model gbt
```

---

## ⚠️ The data is synthetic

There is no real returns log in this repository. `data.py` generates one. **Every
number below describes behaviour on a generative process I wrote, and is not
evidence of real-world performance.** The pipeline, the cost framing and the
threshold logic transfer; the metrics do not. Point `load_returns()` at a real
extract before anyone makes a policy decision from this.

---

## 1 · Why not 0.5

A 0.5 cut-point answers *"is fraud more likely than not?"*. That is the wrong
question. The real question is *"does flagging this return cost less than not
flagging it?"* — and once the two errors carry different costs, the answer moves.

For a return with fraud probability `p`:

```
expected cost of flagging      = (1 - p) · C_FP     # we might be wrong
expected cost of not flagging  =      p  · C_FN     # it might be fraud

flag when   (1 - p) · C_FP  <  p · C_FN
      i.e.  p > C_FP / (C_FP + C_FN)
```

At the default **5:1** that break-even sits at **p > 0.833**. A deliberately high
bar: don't pull someone into review unless the model is very sure.

The empirical sweep on validation independently landed on **0.787**. That it
agrees with the theoretical 0.833 is the evidence that the probabilities are
calibrated well enough for this whole argument to hold. If those two numbers had
diverged sharply, the cost reasoning would have been standing on sand.

---

## 2 · Justifying the 5:1 ratio

The brief specifies FP ≫ FN. It is worth being honest that **a strict
expected-value calculation does not obviously support 5:1**:

| | Estimate |
|---|---|
| **FN** — one fraudulent return absorbed | `L` (merchandise loss) — known precisely |
| **FP** — reviewer time | ~0.05–0.10 `L` |
| **FP** — delay → churn (5% × CLV ≈ 6 `L`) | ~0.30 `L` |
| **FP** — word-of-mouth amplification (~2×) | ~0.60 `L` total |

Point-estimate EV lands nearer **1:1**. So why ship 5:1?

**1. The two costs have different shapes.** The FN cost is a known scalar. The FP
cost is a wide distribution with a heavy right tail — most wrongly-flagged
customers shrug, a few churn loudly. Weighting above the point estimate is the
standard response to a heavy-tailed harm you cannot measure well.

**2. Errors are not spread evenly across people.** They concentrate on customers
with unusual-but-legitimate patterns: frequent returners, new accounts, people
who return often for size reasons. A 2% false-positive rate spread uniformly is
an annoyance; the same rate concentrated on one segment is systematic exclusion.
Expected value averages over exactly the structure that makes this harmful.

**3. The errors differ in recoverability.** A missed fraud is recoverable — the
pattern accumulates and the account surfaces later. A customer who quietly leaves
is not.

**4. On this data the argument is nearly moot.** Ratios of 3:1, 5:1 and 20:1 all
select the *same* threshold (0.787). Only 1:1 differs. So the aggressive stance
costs essentially nothing in recall versus a moderate one — worth knowing before
anyone spends a meeting arguing about the exact number.

**The counterweight is the routing decision.** Because a flag routes to review
rather than denial, the FP cost stays bounded at *friction*. If this component
auto-denied refunds, the FP cost would be far higher and 5:1 would not be nearly
conservative enough. The cost matrix and the action contract are load-bearing
for each other.

It is a **policy input, not a measurement** — which is exactly why it lives in
`config.py` as a parameter and why the report ships a sensitivity table.

---

## 3 · Results on held-out test (synthetic, seed 20260821, logreg, 5:1)

Threshold chosen on **validation**, reported on **test**. Selecting it on test
would make the reported cost optimistically biased.

### Candidate thresholds (validation)

| Threshold | Label | Precision | Recall | FPR | Flag rate | F1 | FP | FN | **Cost/case** |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.500 | default | 0.810 | 0.266 | 0.0049 | 0.024 | 0.400 | 11 | 130 | 0.0771 |
| 0.323 | max-F1 | 0.622 | 0.390 | 0.0189 | 0.046 | **0.479** | 42 | 108 | 0.1325 |
| 0.315 | precision ≥ 0.60 | 0.600 | 0.390 | 0.0207 | 0.048 | 0.473 | 46 | 108 | 0.1408 |
| **0.787** | **cost-optimal** | **1.000** | 0.147 | **0.0000** | 0.011 | 0.256 | **0** | 151 | **0.0629** |

The max-F1 row is what a conventional pipeline ships. Under this cost matrix it
is the **most expensive** option — more than double the cost-optimal — because F1
treats a wrongly-flagged customer and a missed fraud as equally bad, and the
business does not.

### At the chosen threshold (test set, n = 2,400)

| Metric | Value |
|---|---|
| Precision | **0.966** |
| Recall | **0.138** |
| False-positive rate | **0.0005** — 1 of 2,197 genuine returns |
| Flag rate | 0.012 — 29 returns to review |
| Weighted cost | **180.0** total · **0.0750** per case |
| Confusion | TP 28 · FP 1 · FN 175 · TN 2,196 |

Versus the default 0.5 threshold: cost **216.0 → 180.0 (−16.7%)**, false positives
**16 → 1**, at the price of **39 more frauds absorbed**. That trade is the entire
point — it is chosen deliberately, and priced.

### Sensitivity to the ratio

| FP:FN | Threshold | Precision | Recall | FPR | Cost/case |
|---|---:|---:|---:|---:|---:|
| 1:1 | 0.529 | 0.857 | 0.325 | 0.0050 | 0.0617 |
| 3:1 | 0.787 | 0.966 | 0.138 | 0.0005 | 0.0742 |
| 5:1 | 0.787 | 0.966 | 0.138 | 0.0005 | 0.0750 |
| 10:1 | 0.787 | 0.966 | 0.138 | 0.0005 | 0.0771 |

---

## 4 · Worked false positive (graceful failure)

Test row 2320 — the **highest-confidence false positive**, i.e. the model's worst
mistake. If graceful failure holds here it holds anywhere.

```
prior_return_rate    0.818      (9 returns from 11 orders)
return_reason        not_as_described
used_tryon           False      → fit_mismatch_score is absent
account_age_days     146
order_value_inr      1,091      discount 41.9%   returned after 10 days

Ground truth         GENUINE (is_fraud = 0)
Model P(fraud)       0.8291     (threshold 0.7871)
```

```json
{ "flagged": true, "confidence": 0.8291, "action": "route_to_review" }
```

The review queue additionally receives
`reason_codes: ["HIGH_PRIOR_RETURN_RATE", "REASON_NOT_AS_DESCRIBED"]`, because a
reviewer who cannot see *why* a case was flagged cannot competently overturn it.

**What this customer experiences:** a delay and a human looking at their case.
Not a denied refund, not a blocked return. The cost of the system's most
confident wrong answer is a request for a second opinion.

---

## 5 · How "never auto-deny" is enforced

Not by convention — structurally:

- `Action` is a closed `StrEnum` with exactly two members, `approve_refund` and
  `route_to_review`. There is no denial value available to return by mistake.
- `_assert_no_denial_action()` runs **at import** and raises if anyone later adds
  a member whose value contains `deny`, `reject`, `block`, `refuse` or `decline`.
- `tests.py` sweeps 400+ probabilities plus NaN, ±inf, `None`, strings and
  out-of-range floats across five thresholds, and asserts the action set never
  escapes those two values and never raises.
- Unusable scores (NaN, garbage) resolve to `0.0` — **failing toward the
  customer**, down the normal approval path, rather than flagging them.

---

## 6 · Design decisions worth knowing

**No `class_weight="balanced"`.** The FP/FN asymmetry is applied *once*, at the
threshold. Re-weighting the training objective too would apply it twice and would
distort the probabilities that the entire cost sweep depends on. Brier score is
reported so that calibration is checked rather than assumed.

**Three-way split.** Train fits the model, validation picks the threshold, test is
touched once for reporting.

**Missingness is modelled, not imputed away.** `fit_mismatch_score` is absent
exactly when the shopper skipped try-on, and that absence is informative — a fit
complaint from someone who never checked the fit reads differently from one that
contradicts a measured good fit. `SimpleImputer(add_indicator=True)` keeps it.

**Two model families are compared, not assumed.** Logistic regression (AUC 0.814,
AP 0.473) edges out gradient boosting (0.808 / 0.466) here, so the default is the
one that is also auditable and naturally calibrated — a real advantage for a
decision that affects customers.

---

## 7 · Limitations

**This is a triage tool, not a fraud-prevention system.** Recall at 5:1 is
**13.8%** — roughly six in seven fraudulent returns pass straight through. That is
not a defect, it is the direct consequence of the cost matrix: when a false
positive costs five times a false negative, the correct policy is to intervene
only on the most blatant cases. If the business wants meaningfully higher recall
it must either accept a lower ratio (with the customer-harm consequences priced
in) or obtain better features — not just move the threshold.

**Other gaps before production:**

- Synthetic data throughout; nothing here is validated against real returns.
- No fairness audit. Error concentration across customer segments is argument #2
  for the 5:1 ratio, yet it is not measured. Before shipping, false-positive rate
  should be sliced by segment — a uniform 0.05% that is 2% for one group is a
  different system than the numbers above suggest.
- No drift monitoring. Fraud adapts; a threshold fixed today decays.
- Costs are relative units. Calibrate `L` against real merchandise loss and real
  CLV before quoting savings in currency.
- The reviewer decision is not fed back as a label, so the model cannot learn
  from its own corrections — and reviewers only ever see flagged cases, which
  biases any naive retraining.
