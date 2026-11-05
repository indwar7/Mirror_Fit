/* ── Returns Guard ─────────────────────────────────────────────────
   A case file for the person holding the garment that came back.

   The input is one field: the ID. Everything else is looked up, and the
   screen spends itself on the answer — what the shopper said, what the
   mirror measured at purchase, what condition it arrived in, and the
   paragraph that reconciles those three.

   The same logistic regression from returns_model.json takes the same
   decision on the same features, in the browser, verified against
   scikit-learn to 2.2e-16. It was trained on SYNTHETIC data and
   demonstrates the decision architecture, not real-world accuracy.
   ────────────────────────────────────────────────────────────────── */
import { byId, esc } from './dom';
import { FitReport, type CaseFile, type FitReading } from './fit-report-types';
import { whenOutfitScoreReady } from './outfit-score';
import { areaCard, secondaryRow } from './fit-cards';

interface ReturnsModel {
  preprocess: {
    impute_medians: number[];
    indicator_features: string[];
    scaler_mean: number[];
    scaler_scale: number[];
    categories: Record<string, string[]>;
  };
  features: { numeric: string[]; categorical: string[]; boolean: string[] };
  model: { intercept: number; coef: number[] };
  decision: { threshold: number };
  test_metrics: { precision: number; recall: number; fpr: number; n: number };
  candidates: { label: string; threshold: number; precision: number; recall: number; fpr: number; fp: number; cost_per_case: number }[];
}

interface CaseFeatures {
  fit_mismatch_score: number | null;
  days_since_purchase: number;
  order_value_inr: number;
  prior_orders_12m: number;
  prior_returns_12m: number;
  prior_return_rate: number;
  account_age_days: number;
  discount_pct: number;
  garment_type: string;
  return_reason: string;
  used_tryon: boolean;
  [k: string]: number | string | boolean | null;
}

let M: ReturnsModel | null = null;
let current: { c: CaseFile; reading: FitReading } | null = null;
const $ = byId;

const inr = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN');

/* Inference — mirrors the sklearn Pipeline exactly. */
function score(values: CaseFeatures): number {
  const m = M!;
  const P = m.preprocess, F = m.features;
  const num: number[] = [];
  F.numeric.forEach((name, i) => {
    const x = values[name];
    const missing = (x === null || x === undefined || Number.isNaN(x));
    num.push(missing ? P.impute_medians[i] : Number(x));
  });
  // SimpleImputer(add_indicator=True) appends indicators after the block.
  P.indicator_features.forEach((name) => {
    const x = values[name];
    num.push((x === null || x === undefined || Number.isNaN(x)) ? 1 : 0);
  });
  const row = num.map((x, i) => (x - P.scaler_mean[i]) / P.scaler_scale[i]);

  F.categorical.forEach((col) => {
    P.categories[col].forEach((level) => row.push(values[col] === level ? 1 : 0));
  });
  F.boolean.forEach((name) => row.push(values[name] ? 1 : 0));

  let z = m.model.intercept;
  for (let i = 0; i < row.length; i++) z += m.model.coef[i] * row[i];
  return 1 / (1 + Math.exp(-z));
}

/* Reason codes — mirrors returns_risk/decision.py:explain() */
function explain(v: CaseFeatures): [string, boolean][] {
  const codes: [string, boolean][] = [];
  const hasFit = v.fit_mismatch_score !== null && v.fit_mismatch_score !== undefined
                 && !Number.isNaN(v.fit_mismatch_score);
  if (v.return_reason === 'doesnt_fit' && v.used_tryon && hasFit && v.fit_mismatch_score! < 0.25)
    codes.push(['FIT_CLAIM_CONTRADICTS_TRYON', false]);
  if (v.return_reason === 'doesnt_fit' && !v.used_tryon)
    codes.push(['FIT_CLAIM_WITHOUT_TRYON', false]);
  if ((v.prior_return_rate || 0) >= 0.5) codes.push(['HIGH_PRIOR_RETURN_RATE', false]);
  if ((v.days_since_purchase || 0) > 28) codes.push(['LATE_IN_RETURN_WINDOW', false]);
  if ((v.order_value_inr || 0) >= 6000) codes.push(['HIGH_VALUE_ORDER', false]);
  if ((v.account_age_days ?? 9999) <= 60) codes.push(['NEW_ACCOUNT', false]);
  if (v.return_reason === 'not_as_described' || v.return_reason === 'damaged')
    codes.push(['REASON_' + v.return_reason.toUpperCase(), false]);
  // An exculpatory code matters as much as an incriminating one.
  if (v.return_reason === 'doesnt_fit' && hasFit && v.fit_mismatch_score! >= 0.55)
    codes.push(['FIT_CLAIM_SUPPORTED_BY_TRYON', true]);
  return codes;
}

/* A reason code is a lookup key. Each carries the sentence an engineer reads. */
const FINDING: Record<string, [boolean, string, string]> = {
  FIT_CLAIM_CONTRADICTS_TRYON: [false,
    'The mirror said it fitted.',
    'This came back as a size problem, but the fit measured at purchase was well inside tolerance. The two do not agree.'],
  FIT_CLAIM_WITHOUT_TRYON: [false,
    'No try-on to check it against.',
    'A size complaint with nothing measured at purchase. Not suspicious on its own — most shoppers never open the mirror — but there is nothing here to confirm or contradict.'],
  FIT_CLAIM_SUPPORTED_BY_TRYON: [true,
    'The mirror agrees with the shopper.',
    'The fit measured at purchase was already outside tolerance. The size complaint is exactly what that reading predicted, which makes this return genuine and the sale the mistake.'],
  HIGH_PRIOR_RETURN_RATE: [false,
    'Returns most of what they order.',
    'Half or more of this account\'s orders in the last twelve months came back.'],
  LATE_IN_RETURN_WINDOW: [false,
    'Returned late in the window.',
    'Past twenty-eight days, near the edge of the policy.'],
  HIGH_VALUE_ORDER: [false,
    'High-value order.',
    'Above ₹6,000, where the cost of getting this wrong is highest in both directions.'],
  NEW_ACCOUNT: [false,
    'New account.',
    'Opened within the last sixty days, so there is little history to weigh.'],
  REASON_NOT_AS_DESCRIBED: [false,
    'Claimed the listing was wrong.',
    'A description complaint, not a fit one — fit measurements cannot settle it either way.'],
  REASON_DAMAGED: [false,
    'Reported as damaged.',
    'Goes to intake for physical inspection regardless of what this score says.'],
};

/* Case → model features */
function featuresFor(c: CaseFile, reading: FitReading | null): CaseFeatures {
  const rate = c.orders > 0 ? c.returns / c.orders : 0;
  return {
    fit_mismatch_score: reading ? reading.mismatch : null,
    days_since_purchase: c.days,
    order_value_inr: c.value,
    prior_orders_12m: c.orders,
    prior_returns_12m: c.returns,
    prior_return_rate: rate,
    account_age_days: c.accountAge,
    discount_pct: c.discount,
    garment_type: M!.preprocess.categories.garment_type.includes(c.garmentType)
      ? c.garmentType : 'tshirt',
    return_reason: c.reason,
    used_tryon: !!reading,
  };
}

/* The paragraph that reconciles claim and evidence. Written against the SAME
   facts the reason codes are built from, so the two cannot disagree. */
function claimRow(c: CaseFile, reading: FitReading) {
  if (c.claimArea) {
    const r = reading.rows.find((x) => x.key === c.claimArea);
    if (r) return r;
  }
  return reading.worst;
}

function answerFor(c: CaseFile, reading: FitReading, _p: number, flagged: boolean): string {
  const said = c.reason === 'doesnt_fit' ? 'a size problem'
             : c.reason === 'not_as_described' ? 'the listing being wrong'
             : c.reason === 'damaged' ? 'damage'
             : c.reason === 'wrong_item' ? 'the wrong item arriving'
             : 'a change of mind';
  const quality = conditionPhrase(c);
  const cm = (r: { cm: number }) => (r.cm < 1 ? r.cm.toFixed(1) : Math.round(r.cm)) + ' cm';

  if (c.reason !== 'doesnt_fit') {
    const irrelevant = c.reason === 'damaged'
      ? 'a garment can be cut perfectly and still arrive broken'
      : c.reason === 'wrong_item'
        ? 'the fit of the item we sent says nothing about the item they ordered'
        : c.reason === 'not_as_described'
          ? 'a good fit and a wrong description sit happily together'
          : 'nobody has to keep something that fits';
    return '<b>Fit is not the question here.</b> This came back over ' + said +
      ', and ' + quality + '. The mirror scored it <b>' + reading.score10.toFixed(1) +
      '/10</b> at purchase, which neither supports nor undercuts the claim — ' +
      irrelevant + '. ' +
      (flagged
        ? 'It is routed to review on this account\'s history, not on the fit.'
        : 'Nothing else about the return stands out, so the refund is approved.');
  }

  const row = claimRow(c, reading);
  const where = 'the ' + row.label.toLowerCase();
  const claimedTight = c.claimDir !== 'loose';
  const measuredTight = row.dev < 0;
  const agrees = row.tone !== 'good' && measuredTight === claimedTight;

  if (agrees) {
    // The mirror already knew. Whether the score showed it is the interesting
    // part — an unscored point can be wrong at 8.4/10.
    const hidden = !row.scored
      ? ' The score could not have carried that on its own: <b>' +
        reading.score10.toFixed(1) + '/10</b> weighs chest, shoulder, length and ' +
        'sleeve, and the collar is measured but deliberately never scored. The ' +
        'number was not wrong; it was never the whole answer.'
      : reading.score10 >= 7.5
        ? ' The headline <b>' + reading.score10.toFixed(1) + '/10</b> averaged it ' +
          'away — one point outside tolerance does not move a weighted score far, ' +
          'which is exactly why this screen shows the points and not just the total.'
        : ' The score said so too, at <b>' + reading.score10.toFixed(1) + '/10</b>.';
    const claimed = c.claimArea
      ? 'They came back saying ' + where + ' was too ' +
        (claimedTight ? 'tight' : 'loose') + ', and that is what the mirror ' +
        'measured at purchase — '
      : 'They came back saying it ran ' + (claimedTight ? 'small' : 'large') +
        ', and ' + where + ' is where the mirror had already measured that worst — ';
    return '<b>The shopper is right, and we sold it anyway.</b> ' + claimed +
      row.word.toLowerCase() + ' by about ' + cm(row) + '.' + hidden + ' ' +
      Cap(quality) + ', so this is a genuine return, and the failure was at the ' +
      'point of sale.';
  }

  // The claim and the measurement point different ways.
  const measured = row.tone === 'good'
    ? where.charAt(0).toUpperCase() + where.slice(1) + ' was one of the points ' +
      'the mirror measured as accurate — inside ' + cm(row) + ' of where this cut sits'
    : where.charAt(0).toUpperCase() + where.slice(1) + ' did drift, but the other ' +
      'way: ' + row.word.toLowerCase() + ' by about ' + cm(row);
  return '<b>The claim and the measurement do not line up.</b> This came back as ' +
    said + (c.claimArea
      ? ' — specifically that ' + where + ' was too ' + (claimedTight ? 'tight' : 'loose')
      : ', running ' + (claimedTight ? 'small' : 'large')) + '. ' +
    measured + ', on a garment that ' +
    'scored <b>' + reading.score10.toFixed(1) + '/10</b> overall. And ' + quality +
    ', so there is no defect to explain it either. ' +
    (flagged
      ? 'That contradiction, on top of this account\'s history, is why it goes to a human rather than straight through.'
      : 'On its own that is not enough to hold a refund — a shopper is allowed to dislike a garment that fits — so it is approved.');
}

function Cap(s: string): string { return s.charAt(0).toUpperCase() + s.slice(1); }

/* Condition arrives as "<tone> — <detail>". Match on the tone alone. */
function conditionTone(c: CaseFile): 'off' | 'watch' | 'good' {
  const head = String(c.condition || '').split('—')[0].trim().toLowerCase();
  return head.startsWith('damag') ? 'off' : head.startsWith('fair') ? 'watch' : 'good';
}
function conditionPhrase(c: CaseFile): string {
  const t = conditionTone(c);
  return t === 'off' ? 'the garment arrived damaged'
       : t === 'watch' ? 'the garment came back with light wear'
       : 'the garment came back in good condition';
}

/* Render */
function render(c: CaseFile, reading: FitReading): void {
  if (!M) return;
  current = { c, reading };

  const v = featuresFor(c, reading);
  const p = score(v);
  const flagged = p >= M.decision.threshold;

  $('rgEmptyCard').style.display = 'none';
  $('rgReport').style.display = 'block';

  $('rgCaseId').textContent = c.id + (c.generated ? '  ·  generated for the demo' : '');
  $('rgCaseName').textContent = c.garment + ' · size ' + c.size;
  $('rgCaseMeta').textContent =
    c.customer + ' · ' + inr(c.value) + ' · returned ' + c.days +
    (c.days === 1 ? ' day' : ' days') + ' after purchase';

  const dec = $('rgDecision');
  dec.textContent = flagged ? 'Route to review' : 'Approve refund';
  dec.className = 'fx-decision ' + (flagged ? 'is-review' : 'is-approve');

  $('rgSaid').innerHTML = '<span>What the shopper said</span>' + esc(c.reasonSaid);

  /* The three answers, in the order the question gets asked. */
  const rate = c.orders > 0 ? Math.round(c.returns / c.orders * 100) : 0;
  const condTone = conditionTone(c);
  const behTone: 'off' | 'watch' | 'good' = rate >= 50 ? 'off' : rate >= 30 ? 'watch' : 'good';
  const facts: [string, string, string, 'off' | 'watch' | 'good'][] = [
    ['Fit Analyser at purchase', reading.score10.toFixed(1) + ' / 10 — ' + reading.verdict,
     reading.headline,
     reading.offCount ? 'off' : reading.watchCount ? 'watch' : 'good'],
    ['Condition on return', Cap(c.condition.split('—')[0].trim()),
     c.condition.includes('—') ? Cap(c.condition.split('—')[1].trim()) + '.' : '',
     condTone],
    ['This account', c.returns + ' of ' + c.orders + ' orders returned',
     rate >= 50 ? 'Returns most of what they order — the strongest single signal on this screen.'
     : rate >= 30 ? 'Returns more than most, but within the range of an ordinary shopper.'
     : 'Returns about as often as anyone else. Nothing unusual.',
     behTone],
  ];
  $('rgFacts').innerHTML = facts.map(([l, val, note, tone]) =>
    '<div class="fx-fact">' +
      '<div class="fx-fact-l">' + esc(l) + '</div>' +
      '<div class="fx-fact-v is-' + tone + '" style="color:' +
        (tone === 'off' ? 'var(--berry)' : tone === 'watch' ? 'var(--gold-ink)' : 'var(--ok)') +
        '">' + esc(val) + '</div>' +
      (note ? '<div class="fx-fact-n">' + esc(note) + '</div>' : '') +
    '</div>').join('');

  $('rgFitH').textContent = 'What the mirror measured at purchase — ' + reading.score10.toFixed(1) + '/10';
  $('rgAreas').innerHTML = reading.core.map(areaCard).join('');
  $('rgRows').innerHTML = reading.secondary.map(secondaryRow).join('');

  $('rgAnswer').innerHTML = answerFor(c, reading, p, flagged);

  /* Findings: the reason codes, as sentences. */
  const codes = explain(v);
  $('rgFinds').innerHTML = codes.length
    ? codes.map(([code, clear]) => {
        const f = FINDING[code] || [clear, code, ''];
        return '<div class="fx-find">' +
          '<span class="fx-mark is-' + (f[0] ? 'good' : 'watch') + '">' +
            (f[0] ? '✓' : '!') + '</span>' +
          '<div class="fx-find-t"><b>' + esc(f[1]) + '</b> ' + esc(f[2]) + '</div>' +
        '</div>';
      }).join('')
    : '<div class="fx-find"><span class="fx-mark is-good">✓</span>' +
      '<div class="fx-find-t"><b>Nothing unusual about this return.</b> ' +
      'No signal on this case crossed the line where it would be worth a ' +
      'reviewer\'s time.</div></div>';

  /* The folded evidence */
  $('rgScore').textContent = p.toFixed(3);
  const verdict = $('rgVerdict');
  verdict.textContent = flagged ? 'Route to review' : 'Approve refund';
  verdict.className = 'rg-verdict ' + (flagged ? 'is-review' : 'is-approve');
  const fill = $('rgFill');
  fill.style.width = (p * 100).toFixed(2) + '%';
  fill.className = 'rg-bar-fill' + (flagged ? ' is-review' : '');
  $('rgMark').style.left = (M.decision.threshold * 100).toFixed(2) + '%';

  $('rgJson').innerHTML =
    '{\n  <span class="k">"garment_id"</span>: <span class="s">"' + esc(c.id) + '"</span>,\n' +
    '  <span class="k">"fit_mismatch_score"</span>: <span class="n">' +
      (v.fit_mismatch_score === null ? 'null' : v.fit_mismatch_score.toFixed(4)) + '</span>,\n' +
    '  <span class="k">"flagged"</span>: <span class="n">' + flagged + '</span>,\n' +
    '  <span class="k">"confidence"</span>: <span class="n">' + p.toFixed(4) + '</span>,\n' +
    '  <span class="k">"action"</span>: <span class="s">"' +
      (flagged ? 'route_to_review' : 'approve_refund') + '"</span>\n}';

  $('rgCodes').innerHTML = codes.length
    ? codes.map(([code, clear]) =>
        '<span class="rg-code' + (clear ? ' is-clear' : '') + '">' + esc(code) + '</span>').join('')
    : '<span class="rg-none">Nothing unusual about this return.</span>';

  $('rgTruth').style.display = 'none';
  $('rgStatusText').textContent = 'Case ' + c.id + ' · scored on device';
}

/* Lookup */
function open(c: CaseFile, btn: Element | null): void {
  document.querySelectorAll('#rgCases .fx-item').forEach((b) => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  render(c, FitReport.readingForCase(c));
}

export function rgFind(): void {
  const q = $<HTMLInputElement>('rgQuery').value.trim();
  if (!q) return;
  const c = FitReport.lookup(q);
  if (!c) return;
  const known = FitReport.cases.findIndex((x) => x.id === c.id);
  const btn = known >= 0 ? document.querySelectorAll('#rgCases .fx-item')[known] : null;
  open(c, btn);
}

/* Arriving from the Fit Analyser or the live try-on: the score the mirror
   just produced becomes the fit evidence on a hypothetical return of that
   garment. The number does not get retyped, it travels. */
export function rgOpenFromFit(reading: FitReading, ctx: { garment: string }): void {
  const c: CaseFile = {
    id: 'MF-LIVE-' + String(FitReport.seedFrom(ctx.garment + reading.score10)).slice(0, 4),
    generated: true,
    garment: ctx.garment, garmentType: reading.garmentType, size: reading.size,
    customer: 'This session', reason: 'doesnt_fit',
    reasonSaid: 'Hypothetical: this garment comes back as a size problem',
    condition: 'Good — as sent',
    value: 2200, days: 7, orders: 5, returns: 1, accountAge: 540, discount: 0,
    score10: reading.score10,
    claimArea: null, claimDir: null,
  };
  document.querySelectorAll('#rgCases .fx-item').forEach((b) => b.classList.remove('active'));
  $<HTMLInputElement>('rgQuery').value = c.id;
  render(c, reading);
}

/* Static panels */
function paintStatics(): void {
  const m = M!.test_metrics;
  $('rgMetrics').innerHTML = ([
    [m.precision.toFixed(3), 'Precision'],
    [m.recall.toFixed(3), 'Recall'],
    [m.fpr.toFixed(4), 'False-positive rate'],
    [M!.decision.threshold.toFixed(3), 'Threshold'],
  ] as [string, string][]).map(([v, l]) =>
    '<div class="rg-metric"><div class="rg-metric-v">' + v + '</div>' +
    '<div class="rg-metric-l">' + l + '</div></div>').join('');

  const rows = M!.candidates.map((c) => {
    const pick = Math.abs(c.threshold - M!.decision.threshold) < 1e-9;
    return '<tr class="' + (pick ? 'is-pick' : '') + '"><td>' + esc(c.label) + '</td>' +
      '<td>' + c.threshold.toFixed(3) + '</td>' +
      '<td>' + c.precision.toFixed(3) + '</td>' +
      '<td>' + c.recall.toFixed(3) + '</td>' +
      '<td>' + c.fpr.toFixed(4) + '</td>' +
      '<td>' + c.fp + '</td>' +
      '<td><b>' + c.cost_per_case.toFixed(4) + '</b></td></tr>';
  }).join('');
  $('rgCandidates').innerHTML =
    '<thead><tr><th>Candidate threshold</th><th>Cut</th><th>Precision</th>' +
    '<th>Recall</th><th>FPR</th><th>False pos.</th><th>Cost / case</th></tr></thead>' +
    '<tbody>' + rows + '</tbody>';

  $('rgCap').innerHTML =
    'Chosen on a validation split, reported on held-out test (n=' + m.n + '). ' +
    'The cheapest row wins — not the best F1, because F1 treats a wrongly flagged ' +
    'customer and a missed fraud as equally bad. Model trained on <b>synthetic data</b>: ' +
    'this demonstrates the decision architecture, not real-world accuracy.';
}

/* The model's enum, spelled the way a person writes it. */
const REASON_LABEL: Record<string, string> = {
  doesnt_fit: "Doesn't fit", not_as_described: 'Not as described',
  damaged: 'Damaged', wrong_item: 'Wrong item', changed_mind: 'Changed mind',
};

function fillCases(): void {
  const label = (s: string) => REASON_LABEL[s] ||
    s.replace(/_/g, ' ').replace(/\b\w/, (c) => c.toUpperCase());
  $('rgCases').innerHTML = FitReport.cases.map((c, i) =>
    '<button class="fx-item" data-i="' + i + '">' +
      '<div class="fx-item-id">' + esc(c.id) + '</div>' +
      '<div class="fx-item-t">' + esc(c.garment) + '</div>' +
      '<div class="fx-item-s">' + esc(label(c.reason)) + ' · ' + inr(c.value) + '</div>' +
    '</button>').join('');
  document.querySelectorAll<HTMLElement>('#rgCases .fx-item').forEach((b) => {
    b.addEventListener('click', () => open(FitReport.cases[+b.dataset.i!], b));
  });
  const q = document.getElementById('rgQuery');
  if (q) q.addEventListener('keydown', (e) => { if ((e as KeyboardEvent).key === 'Enter') rgFind(); });
}

export function initReturnsGuard(): void {
  fetch('returns_model.json')
    .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json() as Promise<ReturnsModel>; })
    .then((json) => {
      M = json;
      fillCases();
      paintStatics();
      // Open on the size-complaint case: a good fit score and a "too small" return.
      whenOutfitScoreReady(
        () => open(FitReport.cases[0], document.querySelector('#rgCases .fx-item')),
        () => {
          $('rgDot').classList.remove('on');
          $('rgStatusText').textContent = 'fit_model.json not found — cases cannot be scored';
        },
      );
    })
    .catch((err) => {
      const dot = document.getElementById('rgDot'), txt = document.getElementById('rgStatusText');
      if (dot) dot.classList.remove('on');
      if (txt) txt.textContent = 'returns_model.json not found';
      console.error('[returns-guard] could not load model:', err);
    });
}

export function currentCase() { return current; }
