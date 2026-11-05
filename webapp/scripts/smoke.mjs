// Headless boot check for the built app. Serves dist/ with `vite preview`,
// opens it in Chromium, and verifies that main.ts ran to the end and that the
// three browser-only screens (Fit Analyser, Returns Guard, the fit handoff)
// work without the GPU box. Connection-refused noise from the backend probes
// is expected and filtered; any other console error or page error fails.
//
//   node scripts/smoke.mjs
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';

const PORT = 4173;
const URL = `http://localhost:${PORT}/`;

const server = spawn('npx', ['vite', 'preview', '--port', String(PORT), '--strictPort'], {
  stdio: ['ignore', 'pipe', 'pipe'],
});
await new Promise((resolve, reject) => {
  const t = setTimeout(() => reject(new Error('preview server did not start')), 20000);
  server.stdout.on('data', (d) => { if (String(d).includes('localhost:' + PORT)) { clearTimeout(t); resolve(); } });
  server.stderr.on('data', (d) => process.stderr.write(d));
});

const failures = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? '  ok   ' : '  FAIL ') + name + (detail ? '  — ' + detail : ''));
  if (!ok) failures.push(name);
};

const browser = await chromium.launch();
const page = await browser.newPage();
const consoleErrors = [];
const pageErrors = [];
page.on('console', (m) => {
  if (m.type() !== 'error') return;
  const text = m.text();
  // Expected without the backend: the avatar list, base-bodies probe, etc.
  if (/ERR_CONNECTION_REFUSED|Failed to fetch|net::ERR/.test(text)) return;
  consoleErrors.push(text);
});
page.on('pageerror', (e) => pageErrors.push(String(e)));

try {
  await page.goto(URL, { waitUntil: 'load' });
  await page.waitForFunction(() => document.documentElement.getAttribute('data-build'), null, { timeout: 10000 });
  check('main.ts booted (build stamp set)', true, await page.evaluate(() => document.documentElement.getAttribute('data-build')));

  check('switchTab(tryon) ran', await page.evaluate(() =>
    document.getElementById('panel-tryon').classList.contains('active') &&
    getComputedStyle(document.getElementById('tryonWrap')).display === 'grid'));

  check('AR avatar grid populated', (await page.locator('#arAvatarGrid .av-card').count()) === 4);

  // Landing is showing; the credit meter paints the balance.
  check('credit balance painted', (await page.locator('[data-credit-balance]').first().textContent()) === '1000');

  // Fit Analyser: fit_model.json loaded and the first case explained.
  await page.evaluate(() => window.enterFeature('fit'));
  await page.waitForFunction(() => document.getElementById('faScore').textContent !== '—', null, { timeout: 10000 });
  check('Fit Analyser lists 6 cases', (await page.locator('#faGarments .fx-item').count()) === 6);
  check('Fit Analyser explains case 1', await page.evaluate(() => /Oxford shirt/.test(document.getElementById('faWhat').textContent)));
  check('Fit Analyser area cards rendered', (await page.locator('#faAreas .fx-area').count()) === 3);

  // Returns Guard: returns_model.json loaded, a lookup renders a decision.
  await page.evaluate(() => window.enterFeature('returns'));
  await page.waitForFunction(() => document.querySelectorAll('#rgCases .fx-item').length === 6, null, { timeout: 10000 });
  await page.fill('#rgQuery', 'MF-TEE-1180');
  await page.evaluate(() => window.rgFind());
  check('Returns Guard renders a decision', await page.evaluate(() =>
    /Route to review|Approve refund/.test(document.getElementById('rgDecision').textContent)));
  check('Returns Guard case id shown', await page.evaluate(() =>
    document.getElementById('rgCaseId').textContent.startsWith('MF-TEE-1180')));
  // 6.9/10 sits between the two fit thresholds, so this case fires no code.
  check('Returns Guard says nothing unusual for the tee', (await page.locator('#rgCodes .rg-none').count()) === 1);
  // 8.4/10 with a size complaint contradicts the try-on: a code must fire.
  await page.fill('#rgQuery', 'MF-SHR-4471');
  await page.evaluate(() => window.rgFind());
  check('Returns Guard reason codes present for the shirt', await page.evaluate(() =>
    /FIT_CLAIM_CONTRADICTS_TRYON/.test(document.getElementById('rgCodes').textContent)));

  // Fit Analyser → Returns Guard handoff carries the reading over.
  await page.evaluate(() => window.switchTab('fit'));
  await page.evaluate(() => window.faSendToReturns());
  check('Fit → Returns handoff opens a live case', await page.evaluate(() =>
    document.getElementById('panel-returns').classList.contains('active') &&
    document.getElementById('rgCaseId').textContent.startsWith('MF-LIVE-')));

  // Live readout → Returns handoff (demo score path, no camera needed).
  await page.evaluate(() => window.frSendToReturns());
  check('Readout → Returns handoff works', await page.evaluate(() =>
    document.getElementById('rgCaseId').textContent.startsWith('MF-LIVE-')));

  // Try-On tab: garment type buttons and preview toggle wire up.
  await page.evaluate(() => window.switchTab('tryon'));
  await page.evaluate(() => window.selectGarmentType('jacket', document.querySelector('.garment-type-btn[data-type="jacket"]')));
  check('garment type toggles', await page.evaluate(() =>
    document.querySelector('.garment-type-btn[data-type="jacket"]').classList.contains('active')));
  await page.evaluate(() => window.setPreview('tryon', true));
  check('preview toggle shows camera pane', await page.evaluate(() =>
    document.getElementById('tryonGrid').style.gridTemplateColumns === '1fr 1fr'));

  // Credits: a launch charges 10, reset restores.
  await page.evaluate(() => { window.goLanding(); window.enterFeature('tryon'); });
  check('launch charged 10 credits', (await page.locator('[data-credit-balance]').first().textContent()) === '990');
  await page.evaluate(() => window.mfResetCredits());
  check('reset restores balance', (await page.locator('[data-credit-balance]').first().textContent()) === '1000');
} catch (e) {
  check('smoke run completed', false, String(e));
} finally {
  await browser.close();
  server.kill();
}

check('no unexpected console errors', consoleErrors.length === 0, consoleErrors.join(' | ').slice(0, 400));
check('no page errors', pageErrors.length === 0, pageErrors.join(' | ').slice(0, 400));

console.log(failures.length ? `\n${failures.length} check(s) failed` : '\nall checks passed');
process.exit(failures.length ? 1 : 0);
