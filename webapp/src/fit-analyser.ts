/* ── Fit Analyser ──────────────────────────────────────────────────
   Answers one question: you were given a score — why that score?

   The score arrives from the session the shopper just had, and the screen
   spends its whole surface on the reason: chest, shoulder, neck — accurate,
   or not, and by how much. Length and sleeve sit underneath.

   The real-outcome model (fit_model_real.json, 192,544 RentTheRunway
   rentals) is loaded only for the evidence panel at the bottom.
   ────────────────────────────────────────────────────────────────── */
import { byId, esc } from './dom';
import { FitReport, type FitReading, type CaseFile } from './fit-report-types';
import { whenOutfitScoreReady } from './outfit-score';
import { areaCard, secondaryRow } from './fit-cards';
import { fitSession } from './fit-session';
import { rgOpenFromFit } from './returns-guard';
import { switchTab } from './ui';

interface RealModel {
  test_metrics: { macro_f1: number };
  gate?: { impact?: { net?: number; warn_precision?: number }; ungated?: { net?: number } };
  source: { citation: string };
}

interface Context { label: string; garment: string }

const $ = byId;
let REAL: RealModel | null = null;        // fit_model_real.json — evidence panel only
let reading: FitReading | null = null;    // the current explanation
let context: Context | null = null;

/* The closing paragraph: what the shopper should actually do about it. */
function advice(r: FitReading): string {
  const worst = r.worst;
  if (worst.tone === 'good') {
    return '<b>Nothing to change.</b> Every point this cut is drawn around ' +
      'landed inside tolerance on you, so a different size would move you ' +
      'away from the fit, not towards it.';
  }
  const tight = worst.dev < 0;
  if (worst.key === 'neck') {
    return '<b>The size is right; the collar is not.</b> Neck sits outside ' +
      'the range this cut expects, and it is the one place a size change ' +
      'will not fix — the next size up moves the chest and shoulder too. ' +
      'Look for the same size in a cut with a ' + (tight ? 'wider' : 'closer') +
      ' collar.';
  }
  if (worst.tone === 'watch') {
    return '<b>Wearable as it is.</b> The ' + worst.label.toLowerCase() +
      ' is the only point that drifted, and not far enough to be worth ' +
      'a size change — moving up or down would trade this for a worse ' +
      'problem somewhere else.';
  }
  return '<b>Try one size ' + (tight ? 'up' : 'down') + '.</b> The ' +
    worst.label.toLowerCase() + ' is far enough outside the range that it ' +
    'will show, and it is the point a size change moves most.';
}

function show(r: FitReading | null, ctx: Context): void {
  reading = r; context = ctx;
  if (!r) return;

  $('faWhat').textContent = ctx.label;
  $('faVerdict').textContent = r.verdict;
  $('faHeadline').textContent = r.headline;

  $('faScore').textContent = r.score10.toFixed(1);
  const circ = 2 * Math.PI * 58;
  const frac = Math.max(0, Math.min(1, r.score10 / 10));
  $('faDialArc').setAttribute('stroke-dashoffset', String(circ * (1 - frac)));
  $('faDialArc').setAttribute('stroke', r.offCount ? '#A8194E' : r.watchCount ? '#8A6737' : '#1E7A52');

  $('faAreas').innerHTML = r.core.map(areaCard).join('');
  $('faRows').innerHTML = r.secondary.map(secondaryRow).join('');
  $('faAnswer').innerHTML = advice(r);

  $('faSend').style.display = 'block';
  $('faTech').style.display = 'block';
  paintTech(r, ctx);

  $('faStatusText').textContent = r.source === 'measured'
    ? 'Measured against the size chart'
    : 'Reconstructed from the session score';
}

/* Evidence panel: the per-point deviations, the payload the returns desk
   receives, and the honest read on how good this family of models is. */
function paintTech(r: FitReading, ctx: Context): void {
  $('faDevs').innerHTML = r.rows.map((row) => {
    const pct = Math.min(100, Math.abs(row.dev) / 0.20 * 50);
    const left = row.dev < 0 ? (50 - pct) : 50;
    return '<div class="fa-dev">' +
      '<div class="fa-dev-l">' + esc(row.label) + '</div>' +
      '<div class="fa-dev-track">' +
        '<div class="fa-dev-mid"></div>' +
        '<div class="fa-dev-bar' + (row.tone === 'good' ? '' : ' is-out') +
          '" style="left:' + left.toFixed(1) + '%;width:' + pct.toFixed(1) + '%"></div>' +
      '</div>' +
      '<div class="fa-dev-v' + (row.tone === 'good' ? '' : ' is-out') + '">' +
        (row.dev >= 0 ? '+' : '−') + (Math.abs(row.dev) * 100).toFixed(1) + '%' +
      '</div>' +
    '</div>';
  }).join('');

  $('faJson').innerHTML =
    '{\n  <span class="k">"garment"</span>: <span class="s">"' + esc(ctx.garment) + '"</span>,\n' +
    '  <span class="k">"size"</span>: <span class="s">"' + esc(r.size) + '"</span>,\n' +
    '  <span class="k">"fit_score"</span>: <span class="n">' + r.score10.toFixed(1) + '</span>,\n' +
    '  <span class="k">"fit_mismatch_score"</span>: <span class="n">' + r.mismatch.toFixed(3) + '</span>,\n' +
    '  <span class="k">"reading"</span>: <span class="s">"' + r.source + '"</span>,\n' +
    r.rows.map((row) => '  <span class="k">"dev_' + row.key + '"</span>: <span class="n">' +
      row.dev.toFixed(4) + '</span>').join(',\n') + '\n}';

  $('faHandoff').innerHTML = r.source === 'measured'
    ? '<b>Measured.</b> Chest and shoulder came from the session; length and ' +
      'sleeve are estimated from height, which is why they carry the smallest ' +
      'weight in the score. The collar is real geometry but sits outside the ' +
      'score — <code>fit_score</code> weights chest, shoulder, length and sleeve, ' +
      'and adding a fifth term would mean claiming a model nobody trained.'
    : '<b>Reconstructed, not measured.</b> No chest or shoulder was recorded for ' +
      'this session, so the per-point figures are the deviations that reproduce ' +
      'the stored score exactly through <code>fit_score()</code>. They are ' +
      'internally consistent with the number above and with each other, and they ' +
      'are the right shape for the reading — but they are an explanation of a ' +
      'score, not a measurement of a body. Feed the mirror a chest and shoulder ' +
      'and this same panel becomes measured.';

  if (!REAL) return;
  const m = REAL.test_metrics, g = REAL.gate || {}, i0 = g.impact || {}, u0 = g.ungated || {};
  $('faMetrics').innerHTML = ([
    ['+' + (i0.net || 0).toLocaleString(), 'Net returns prevented'],
    [((i0.warn_precision || 0) * 100).toFixed(0) + '%', 'Warning precision'],
    [(u0.net || 0).toLocaleString(), 'If it warned every time'],
    [m.macro_f1.toFixed(3), 'Macro F1 (base 0.284)'],
  ] as [string, string][]).map(([v, l]) => '<div class="fa-metric"><div class="fa-metric-v">' + v +
    '</div><div class="fa-metric-l">' + l + '</div></div>').join('');

  $('faCap').innerHTML =
    '<b>How good is fit prediction, honestly.</b> ' + esc(REAL.source.citation) +
    ' — 192,544 rentals with customer-reported outcomes. Read the macro F1, not ' +
    'the accuracy: 73.8% of that dataset is labelled &ldquo;fit&rdquo;, so always ' +
    'answering &ldquo;fit&rdquo; scores higher on accuracy while never warning ' +
    'anyone. Precision on the small and large calls is ~0.31 — roughly two in ' +
    'three warnings are false alarms. That is why this screen explains a fit ' +
    'rather than promising one, and why the advice above only moves you a size ' +
    'when a point is clearly outside the range.';
}

/* The live try-on session. The readout publishes fitSession.frReading the
   moment it settles, so both screens quote one number. */
export function faExplainLive(): void {
  const live = fitSession.frReading;
  if (!live) return;
  show(live.reading, { label: 'Your live try-on', garment: live.garment });
}

export function faNoteLiveSession(available: boolean): void {
  const b = document.getElementById('faUseLive') as HTMLButtonElement | null;
  if (!b) return;
  b.disabled = !available;
  $('faLiveHint').textContent = available
    ? 'Scored during your session. Opening it here does not re-run anything.'
    : 'Run a live try-on and the score it gives you opens here, broken down by where the garment actually sat.';
}

function explainCase(c: CaseFile, btn: Element | null): void {
  document.querySelectorAll('#faGarments .fx-item').forEach((b) => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  show(FitReport.readingForCase(c), { label: c.garment + ' · size ' + c.size, garment: c.garment });
}

/* Carry the measured mismatch to the returns desk. */
export function faSendToReturns(): void {
  if (!reading || !context) return;
  rgOpenFromFit(reading, context);
  switchTab('returns');
}

function boot(): void {
  $('faGarments').innerHTML = FitReport.cases.map((c, i) =>
    '<button class="fx-item" data-i="' + i + '">' +
      '<div class="fx-item-t">' + esc(c.garment) + '</div>' +
      '<div class="fx-item-s">Size ' + esc(c.size) + ' · scored ' + c.score10.toFixed(1) + '/10</div>' +
    '</button>').join('');
  document.querySelectorAll<HTMLElement>('#faGarments .fx-item').forEach((b) => {
    b.addEventListener('click', () => explainCase(FitReport.cases[+b.dataset.i!], b));
  });
  faNoteLiveSession(!!fitSession.frReading);
}

export function initFitAnalyser(): void {
  // OutfitScore.load() is what puts the size charts in reach. Bounded, not
  // open-ended: a missing fit_model.json surfaces as a named failure.
  whenOutfitScoreReady(
    () => {
      boot();
      explainCase(FitReport.cases[0], document.querySelector('#faGarments .fx-item'));
    },
    () => {
      const dot = document.getElementById('faDot'), txt = document.getElementById('faStatusText');
      if (dot) dot.classList.remove('on');
      if (txt) txt.textContent = 'fit_model.json not found — cannot size a garment';
      $('faHeadline').textContent =
        'The size charts this screen measures against could not be loaded, so ' +
        'there is nothing to explain a score with. Check that fit_model.json is ' +
        'being served next to this page.';
    },
  );

  fetch('fit_model_real.json')
    .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json() as Promise<RealModel>; })
    .then((json) => { REAL = json; if (reading && context) paintTech(reading, context); })
    .catch((err: Error) => console.warn('[fit-analyser] evidence panel unavailable —', err.message));
}
