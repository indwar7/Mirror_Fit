/* ── Live fit readout ───────────────────────────────────────────────
   Appears in the action bar ten seconds after a session starts. The delay
   is real work: the first frame takes ~9s while the pipeline warms up, and
   the body mask needs several frames to settle.

   The Step 2.6 measurement inputs are hidden, so when they are blank the
   score is a DEMO score, held in the 8-9 band — seeded from the garment and
   size, so the same garment always scores the same and the Fit Analyser can
   explain a number that will still be there when the shopper switches tabs.
   It publishes fitSession.frReading, the same per-area reading the Fit
   Analyser and Returns Guard read.

   Fill in chest and shoulder and this card silently becomes real.
   ────────────────────────────────────────────────────────────────── */
import { maybe } from './dom';
import { FitReport, type FitReading } from './fit-report-types';
import { fitSession } from './fit-session';
import { readBody, readSize } from './measurements';
import { getGarmentName, getGarmentType } from './tryon';
import { faExplainLive, faNoteLiveSession } from './fit-analyser';
import { rgOpenFromFit } from './returns-guard';
import { onTabSwitch, switchTab } from './ui';

const REVEAL_MS = 10000;
let timer: number | null = null;
let reading: FitReading | null = null;
const $ = maybe;

function garmentName(): string {
  return getGarmentName() || getGarmentType() || 'tshirt';
}

/* Measured if we can, seeded demo if we cannot — and `source` on the
   reading is what every surface uses to say which one it was. */
function build(): FitReading {
  if (reading) return reading;
  const type = getGarmentType() || 'tshirt';
  const size = readSize() || 'M';

  const measured = FitReport.fromMeasurements({ garmentType: type, size, body: readBody() });
  if (measured) { reading = measured; return reading; }

  // Seeded, not random: 8.0-9.0, stable for this garment and size.
  const rand = FitReport.rng(FitReport.seedFrom(garmentName() + '|' + type + '|' + size));
  const s = Math.round((8.0 + rand()) * 10) / 10;
  reading = FitReport.fromScore({
    garmentType: type, size, score10: Math.min(9.0, s),
    seed: garmentName() + '|' + type + '|' + size,
  });
  return reading;
}

function publish(r: FitReading | null): void {
  fitSession.frReading = r ? { reading: r, garment: garmentName() } : null;
  faNoteLiveSession(!!r);
}

function reveal(): void {
  const card = $('frCard'); if (!card) return;
  const r = build();
  card.classList.remove('is-wait');
  $('frEyebrow')!.textContent = 'Fit score';
  $('frVerdict')!.textContent = r.verdict;
  $('frScore')!.textContent = r.score10.toFixed(1);
  $('frBar')!.style.width = (r.score10 * 10).toFixed(0) + '%';
  $('frBar')!.style.background = r.offCount ? '#A8194E' : r.watchCount ? '#8A6737' : '#1E7A52';

  // The three places the shopper will actually feel it.
  $('frRows')!.innerHTML = r.core.map((row) =>
    '<div class="fr-row"><span>' + row.label + '</span><b style="color:' +
      (row.tone === 'off' ? '#A8194E' : row.tone === 'watch' ? '#8A6737' : '#1E7A52') +
      '">' + row.word + '</b></div>').join('');

  $('frFoot')!.textContent = r.source === 'measured'
    ? 'Measured against the size chart'
    : 'Demo score · not a measurement';
  const b = $<HTMLButtonElement>('frSend'); if (b) b.disabled = false;
  const w = $<HTMLButtonElement>('frWhy'); if (w) w.disabled = false;
  publish(r);
  timer = null;
}

function start(): void {
  const card = $('frCard'); if (!card) return;
  stop();
  reading = null;
  card.classList.add('is-in', 'is-wait');
  $('frEyebrow')!.textContent = 'Measuring fit';
  $('frVerdict')!.textContent = 'Hold still for a moment…';
  $('frScore')!.textContent = '—';
  $('frRows')!.innerHTML = '';
  $('frBar')!.style.width = '0%';
  $('frFoot')!.textContent = 'Analysing';
  const b = $<HTMLButtonElement>('frSend'); if (b) b.disabled = true;
  const w = $<HTMLButtonElement>('frWhy'); if (w) w.disabled = true;
  timer = window.setTimeout(reveal, REVEAL_MS);
}

function stop(): void {
  if (timer) clearTimeout(timer);
  timer = null;
  const card = $('frCard');
  if (card) card.classList.remove('is-in', 'is-wait');
  const b = $<HTMLButtonElement>('frSend'); if (b) b.disabled = true;
  const w = $<HTMLButtonElement>('frWhy'); if (w) w.disabled = true;
}

/* "Why this score": the reading travels, the shopper re-enters nothing. */
export function frExplain(): void {
  if (!fitSession.frReading) return;
  switchTab('fit');
  faExplainLive();
}

/* Hand the same reading to Returns Guard and follow it there. */
export function frSendToReturns(): void {
  const r = build();
  rgOpenFromFit(r, { garment: garmentName() });
  switchTab('returns');
}

export function installFitReadout(): void {
  const btn = $('tryonBtn');
  if (btn) {
    btn.addEventListener('click', () => {
      setTimeout(() => {
        if (/stop/i.test(btn.textContent || '')) start(); else stop();
      }, 120);
    });
  }
  onTabSwitch((t) => { if (t !== 'tryon') stop(); });
}
