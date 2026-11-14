// Headless check of the built app. Serves dist/ with `vite preview`, opens it
// in Chromium with a fake camera, and verifies:
//   - main.ts boots and the tabs wire up
//   - the browser-only screen (Fit Analyser) works
//   - the WHOLE Live Try-On loop works against a mocked try-on WebSocket:
//     camera opens, garment enables Start, init carries the garment, frames
//     flow one-at-a-time, results are drawn, a garment-type change re-inits,
//     the fit readout starts, Stop tears it down
//   - the page fits a phone viewport with no horizontal overflow
//   - three.js / MediaPipe are not downloaded until a feature needs them
// Connection-refused noise from the backend probes is expected and filtered;
// any other console error or page error fails.
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
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
});
const context = await browser.newContext({ permissions: ['camera'] });
const consoleErrors = [];
const pageErrors = [];
const requested = [];
function watch(page) {
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    const text = m.text();
    // Expected without the backend: the avatar list, base-bodies probe, etc.
    if (/ERR_CONNECTION_REFUSED|Failed to fetch|net::ERR/.test(text)) return;
    consoleErrors.push(text);
  });
  page.on('pageerror', (e) => pageErrors.push(String(e)));
  page.on('request', (r) => requested.push(r.url()));
}

try {
  const page = await context.newPage();
  watch(page);

  // ── Boot ───────────────────────────────────────────────────────────────
  await page.goto(URL, { waitUntil: 'load' });
  await page.waitForFunction(() => document.documentElement.getAttribute('data-build'), null, { timeout: 10000 });
  check('main.ts booted (build stamp set)', true, await page.evaluate(() => document.documentElement.getAttribute('data-build')));
  check('switchTab(tryon) ran', await page.evaluate(() =>
    document.getElementById('panel-tryon').classList.contains('active') &&
    getComputedStyle(document.getElementById('tryonWrap')).display === 'grid'));
  check('AR avatar grid populated', (await page.locator('#arAvatarGrid .av-card').count()) === 4);
  check('credit balance painted', (await page.locator('[data-credit-balance]').first().textContent()) === '1000');
  check('three.js not downloaded at boot', !requested.some((u) => /\/assets\/three-/.test(u)));
  check('MediaPipe not downloaded at boot', !requested.some((u) => /\/assets\/mediapipe-/.test(u)));
  check('manifest served', (await page.request.get(URL + 'manifest.webmanifest')).status() === 200);
  check('icons served', (await page.request.get(URL + 'icons/icon-192.png')).status() === 200
    && (await page.request.get(URL + 'icon.svg')).status() === 200);

  // ── Live Try-On, end to end against a mocked backend ───────────────────
  const sent = [];
  const jpegB64 = await page.evaluate(() => {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const x = c.getContext('2d'); x.fillStyle = '#a8194e'; x.fillRect(0, 0, 64, 64);
    return c.toDataURL('image/jpeg', 0.9).split(',')[1];
  });
  await page.routeWebSocket(/\/ws\/tryon/, (ws) => {
    ws.onMessage((raw) => {
      const msg = JSON.parse(String(raw));
      sent.push(msg);
      if (msg.type === 'init')  ws.send(JSON.stringify({ type: 'ready' }));
      if (msg.type === 'frame') ws.send(JSON.stringify({ type: 'result', image: jpegB64, diag: { region: 'torso', face: 'front', collar_px: 12, torso_px: 200 } }));
    });
  });

  await page.evaluate(() => window.enterFeature('tryon'));
  check('Start disabled before camera + garment', await page.locator('#tryonBtn').isDisabled());

  await page.evaluate(() => window.startTryOnCamera());
  await page.waitForFunction(() => {
    const v = document.getElementById('tryonVideo');
    return !!v.srcObject && v.videoWidth > 0 && !v.paused;
  }, null, { timeout: 10000 });
  check('camera opened and playing', true, await page.evaluate(() => {
    const v = document.getElementById('tryonVideo'); return v.videoWidth + 'x' + v.videoHeight;
  }));
  check('camera button reflects state', (await page.locator('#tryonCamBtn').textContent()) === 'Camera on');
  check('try-on preview is NOT mirrored', await page.evaluate(() => document.getElementById('tryonVideo').style.transform === 'none'));
  check('status tells the user what is next', /upload a garment/i.test(await page.locator('#tryonStatus').textContent()));
  check('Start still disabled without garment', await page.locator('#tryonBtn').isDisabled());

  const pngB64 = await page.evaluate(() => {
    const c = document.createElement('canvas'); c.width = 120; c.height = 160;
    const x = c.getContext('2d'); x.fillStyle = '#1f3a5f'; x.fillRect(0, 0, 120, 160);
    return c.toDataURL('image/png').split(',')[1];
  });
  await page.setInputFiles('input[onchange*="setGarment"]', { name: 'shirt.png', mimeType: 'image/png', buffer: Buffer.from(pngB64, 'base64') });
  await page.waitForFunction(() => document.getElementById('gp').style.display === 'block', null, { timeout: 5000 });
  check('garment preview shown', true);
  check('Start enabled once camera + garment', !(await page.locator('#tryonBtn').isDisabled()));

  await page.click('#tryonBtn');
  await page.waitForFunction(() => document.getElementById('tryonStatus').textContent === 'Try-on active', null, { timeout: 10000 });
  const init = sent.find((m) => m.type === 'init');
  check('init carries the garment and type', !!init && typeof init.garment_image === 'string' && init.garment_image.length > 100 && init.garment_type === 'tshirt');
  check('init carries fabric/back as null when unset', !!init && init.fabric_image === null && init.garment_back_image === null);
  check('connected dot on', await page.evaluate(() => document.getElementById('toDot').classList.contains('on')));
  check('button says Stop', (await page.locator('#tryonBtn').textContent()) === 'Stop');

  await page.waitForFunction(() => /\d+ms$/.test(document.getElementById('toSpinnerMs').textContent), null, { timeout: 10000 });
  await sleep(1800);
  const frames = sent.filter((m) => m.type === 'frame');
  // Reply-driven with a 66 ms floor: an instant mock yields at most ~15 fps.
  check('frames flow, one in flight, capped at ~15 fps', frames.length >= 4 && frames.length <= 40 && frames.every((f) => typeof f.image === 'string' && f.image.length > 100), frames.length + ' frames in ~2 s');
  check('result drawn on the canvas', await page.evaluate(() => {
    const c = document.getElementById('tryonCanvas');
    if (!c.width) return false;
    const [r, g, b] = c.getContext('2d').getImageData(c.width >> 1, c.height >> 1, 1, 1).data;
    return Math.abs(r - 168) < 12 && Math.abs(g - 25) < 12 && Math.abs(b - 78) < 12;
  }));
  check('spinner hidden after first result', await page.evaluate(() => document.getElementById('toSpinner').style.display === 'none'));
  check('diag overlay absent without ?diag', (await page.locator('#toDiag').count()) === 0);
  check('fit readout is measuring', await page.evaluate(() =>
    document.getElementById('frCard').classList.contains('is-wait') &&
    document.getElementById('frEyebrow').textContent === 'Measuring fit'));

  await page.evaluate(() => window.selectGarmentType('jacket', document.querySelector('.garment-type-btn[data-type="jacket"]')));
  await page.waitForFunction(() => true, null, { timeout: 1000 });
  await sleep(500);
  const inits = sent.filter((m) => m.type === 'init');
  check('garment type change re-inits the session live', inits.length >= 2 && inits[inits.length - 1].garment_type === 'jacket');

  await page.click('#tryonBtn');
  await sleep(300);
  check('fit readout hidden after Stop', await page.evaluate(() => !document.getElementById('frCard').classList.contains('is-in')));
  check('Stop resets the button', (await page.locator('#tryonBtn').textContent()) === 'Start Live Try-On');
  check('Stop clears the dot', await page.evaluate(() => !document.getElementById('toDot').classList.contains('on')));
  const before = sent.length; await sleep(1200);
  check('no frames sent after Stop', sent.length === before);

  // ── Browser-only screens ───────────────────────────────────────────────
  await page.evaluate(() => window.enterFeature('fit'));
  await page.waitForFunction(() => document.getElementById('faScore').textContent !== '—', null, { timeout: 10000 });
  check('Fit Analyser lists 6 cases', (await page.locator('#faGarments .fx-item').count()) === 6);
  check('Fit Analyser explains case 1', await page.evaluate(() => /Oxford shirt/.test(document.getElementById('faWhat').textContent)));
  check('Fit Analyser area cards rendered', (await page.locator('#faAreas .fx-area').count()) === 3);
  check('Returns Guard is gone', await page.evaluate(() =>
    !document.getElementById('panel-returns') && !document.getElementById('returnsWrap') &&
    !document.getElementById('frSend') && !document.getElementById('faSend') && !window.rgFind));
  check('tab buttons match the tab list', (await page.locator('.tab').count()) === 7);

  await page.evaluate(() => window.switchTab('tryon'));
  await page.evaluate(() => window.setPreview('tryon', true));
  check('preview toggle shows camera pane', await page.evaluate(() =>
    document.getElementById('tryonGrid').style.gridTemplateColumns === '1fr 1fr'));

  await page.evaluate(() => { window.goLanding(); window.enterFeature('tryon'); });
  check('launch charged 10 credits', (await page.locator('[data-credit-balance]').first().textContent()) === '980');
  await page.evaluate(() => window.mfResetCredits());
  check('reset restores balance', (await page.locator('[data-credit-balance]').first().textContent()) === '1000');

  // ── Phone viewport ─────────────────────────────────────────────────────
  const phone = await context.newPage();
  watch(phone);
  await phone.setViewportSize({ width: 390, height: 844 });
  await phone.goto(URL, { waitUntil: 'load' });
  await phone.waitForFunction(() => document.documentElement.getAttribute('data-build'), null, { timeout: 10000 });
  const overflow = (p) => p.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  check('landing has no horizontal overflow at 390px', (await overflow(phone)) <= 1, (await overflow(phone)) + 'px');
  await phone.evaluate(() => window.enterFeature('tryon'));
  await sleep(200);
  check('try-on screen has no horizontal overflow at 390px', (await overflow(phone)) <= 1, (await overflow(phone)) + 'px');
  check('camera button is visible on the phone', await phone.locator('#tryonCamBtn').isVisible());
  check('Start button is reachable on the phone', await phone.locator('#tryonBtn').isVisible());
  await phone.evaluate(() => window.enterFeature('fit'));
  await sleep(200);
  check('fit screen has no horizontal overflow at 390px', (await overflow(phone)) <= 1, (await overflow(phone)) + 'px');
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
