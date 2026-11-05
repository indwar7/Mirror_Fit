// Live Try-On (port 8000) plus the outfit rating that sits over it.
import { TRYON_WS } from './config';
import { MIRROR, drawCam } from './camera';
import { byId, maybe } from './dom';
import { OutfitScore } from './outfit-score';
import { FitReport, type FitCall } from './fit-report-types';
import { readBody, readSize } from './measurements';
import { setDot, setStatus, switchTab } from './ui';
import { rgOpenFromFit } from './returns-guard';

let toStream: MediaStream | null = null;
let toWs: WebSocket | null = null;
let toGarment: File | null = null;
let toRunning = false;
let toLatestResult: HTMLImageElement | null = null;
let toRafId: number | null = null;
let toSendTimer: number | null = null;
// Default garment type — keeps "garment-agnostic" behaviour until the user
// explicitly clicks T-shirt / Shirt / Jacket.
let toGarmentType = 'tshirt';

const toCvs = document.createElement('canvas');
const toCtx = toCvs.getContext('2d')!;

interface TryOnMsg {
  type: string;
  image?: string;
  message?: string;
  diag?: { region: string; face: string; collar_px: number; torso_px: number };
}

// Read by the live fit readout.
export function getGarmentName(): string { return toGarment ? toGarment.name : ''; }
export function getGarmentType(): string { return toGarmentType; }

function checkTryOnReady(): void {
  // Enable Start only when BOTH garment uploaded AND camera on.
  const ready = !!(toGarment && toStream);
  byId<HTMLButtonElement>('tryonBtn').disabled = !ready;
}

export function setGarment(inp: HTMLInputElement): void {
  toGarment = (inp.files && inp.files[0]) || null;
  checkTryOnReady();
  // If a live try-on is already running, re-send init so the server picks up
  // the new garment without the user having to stop/start.
  resendTryOnInit();
  // The preview <img> is what the rating samples the colour from, and it has
  // not decoded yet at this point — wait for it rather than reading a blank.
  const gp = maybe<HTMLImageElement>('gp');
  if (gp) {
    if (gp.complete && gp.naturalWidth) renderOutfitScore();
    else gp.addEventListener('load', renderOutfitScore, { once: true });
  }
}

export function selectGarmentType(type: string, btn: HTMLElement): void {
  toGarmentType = type;
  document.querySelectorAll('.garment-type-btn').forEach((b) => {
    b.classList.toggle('active', b === btn);
  });
  // Live switch: push the new type to the server immediately.
  resendTryOnInit();
  renderOutfitScore();
}

// ── Outfit rating ────────────────────────────────────────────────────────
// The outfit rating panel that used to sit over the live result is hidden.
// Flip this back to false to bring the panel straight back.
const OS_PANEL_HIDDEN = true;

let osLastFit: FitCall | null = null;

export function renderOutfitScore(): void {
  const panel = maybe('osPanel');
  if (!panel) return;
  if (OS_PANEL_HIDDEN) { panel.style.display = 'none'; return; }

  // Only sample the garment preview once the browser has decoded it.
  const gp = maybe<HTMLImageElement>('gp');
  const garmentImage = (gp && gp.src && gp.complete && gp.naturalWidth) ? gp : null;

  const result = OutfitScore.rate({
    garmentImage,
    garmentType: toGarmentType || 'tshirt',
    size: readSize(),
    skinTone: byId<HTMLSelectElement>('osSkin').value,
    body: readBody(),
  });

  if (result.overall === null) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';

  byId('osEmoji').textContent = result.emoji || '';
  byId('osOverall').textContent = result.overall.toFixed(1);
  byId('osGrade').textContent = result.grade ? 'Grade ' + result.grade : '';

  const fitEl = byId('osFit');
  const fitNoteEl = byId('osFitNote');
  if (result.fit) {
    fitEl.textContent = result.fit.label;
    // Un-confident verdicts are shown greyed, with the model's own reasoning.
    fitEl.style.color = result.fit.confident ? 'var(--ink)' : '#8A8378';
    fitNoteEl.textContent = result.fit.detail;
  } else {
    fitEl.textContent = '';
    fitNoteEl.textContent = '';
  }

  const colourEl = byId('osColour');
  if (result.colour) {
    const { r, g, b } = result.colour.rgb;
    colourEl.innerHTML =
      '<span style="width:12px;height:12px;border-radius:3px;flex:0 0 auto;' +
      'border:1px solid rgba(15,23,42,.18);background:rgb(' + r + ',' + g + ',' + b + ')"></span>' +
      '<span></span>';
    colourEl.lastElementChild!.textContent = result.colour.verdict;
  } else {
    colourEl.innerHTML = '';
  }

  byId('osNotes').textContent = result.notes.join(' ');

  // The handoff only exists when there is a measured fit behind it.
  osLastFit = result.fit;
  byId('osToReturns').style.display = result.fit ? 'block' : 'none';
}

/// Live try-on → Returns Guard, the same join Fit Analyser uses.
export function sendTryOnToReturns(): void {
  if (!osLastFit) return;
  const reading = FitReport.fromMeasurements({
    garmentType: toGarmentType || 'tshirt',
    size: readSize(),
    body: readBody(),
  });
  if (!reading) return;
  rgOpenFromFit(reading, { garment: 'Live try-on garment' });
  switchTab('returns');
}

// ── Fabric / back view ───────────────────────────────────────────────────
let toFabric: File | null = null;
let toFabricB64: string | null = null;       // Cached base64 — survives across WS reconnects

export function setFabric(inp: HTMLInputElement): void {
  toFabric = (inp.files && inp.files[0]) || null;
  if (!toFabric) return;
  const r = new FileReader();
  r.onload = (e) => {
    toFabricB64 = (e.target!.result as string).split(',')[1];
    // If a session is already running, push fabric live.
    if (toWs && toWs.readyState === WebSocket.OPEN) {
      toWs.send(JSON.stringify({ type: 'fabric', fabric_image: toFabricB64 }));
    }
  };
  r.readAsDataURL(toFabric);
}

export function clearFabric(): void {
  toFabric    = null;
  toFabricB64 = null;
  const input = document.querySelector<HTMLInputElement>('input[onchange*="setFabric"]');
  if (input) input.value = '';
  byId('fp').style.display = 'none';
  byId('fph').style.display = '';
  if (toWs && toWs.readyState === WebSocket.OPEN) {
    toWs.send(JSON.stringify({ type: 'fabric_clear' }));
  }
}

let toGarmentBack: File | null = null;
let toGarmentBackB64: string | null = null;   // real back-view photo (optional)

export function setGarmentBack(inp: HTMLInputElement): void {
  toGarmentBack = (inp.files && inp.files[0]) || null;
  if (!toGarmentBack) return;
  const r = new FileReader();
  r.onload = (e) => {
    toGarmentBackB64 = (e.target!.result as string).split(',')[1];
    if (toWs && toWs.readyState === WebSocket.OPEN) {
      toWs.send(JSON.stringify({ type: 'garment_back', garment_back_image: toGarmentBackB64 }));
    }
  };
  r.readAsDataURL(toGarmentBack);
}

export function clearGarmentBack(): void {
  toGarmentBack    = null;
  toGarmentBackB64 = null;
  const input = document.querySelector<HTMLInputElement>('input[onchange*="setGarmentBack"]');
  if (input) input.value = '';
  byId('gbp').style.display = 'none';
  byId('gbph').style.display = '';
  if (toWs && toWs.readyState === WebSocket.OPEN) {
    toWs.send(JSON.stringify({ type: 'garment_back_clear' }));
  }
}

function initPayload(garmentB64: string): string {
  return JSON.stringify({
    type: 'init',
    garment_image:      garmentB64,
    garment_type:       toGarmentType,
    fabric_image:       toFabricB64,        // include fabric if uploaded
    garment_back_image: toGarmentBackB64,   // include back-view photo if uploaded
  });
}

function resendTryOnInit(): void {
  if (!(toWs && toWs.readyState === WebSocket.OPEN && toGarment)) return;
  const ws = toWs;
  const r = new FileReader();
  r.onload = (e) => { ws.send(initPayload((e.target!.result as string).split(',')[1])); };
  r.readAsDataURL(toGarment);
}

// ── Session ──────────────────────────────────────────────────────────────
export async function startTryOnCamera(): Promise<void> {
  try {
    toStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
    byId<HTMLVideoElement>('tryonVideo').srcObject = toStream;
    const btn = byId('tryonCamBtn');
    btn.textContent = 'Camera on'; btn.className = 'btn btn-active';
    checkTryOnReady();
  } catch (e) { alert('Camera error: ' + (e as Error).message); }
}

export function toggleTryOn(): void { if (toRunning) stopTryOn(); else startTryOn(); }

let toProcessing = false, toFrameSentAt = 0, toResultAt = 0;

function toSpinner(on: boolean): void {
  byId('toSpinner').style.display = on ? 'flex' : 'none';
}

export function startTryOn(): void {
  if (!toStream) { setStatus('tryonStatus', 'error', 'Start camera first'); return; }
  if (!toGarment) { setStatus('tryonStatus', 'error', 'Upload garment first'); return; }
  const garment = toGarment;
  toLatestResult = null; toProcessing = false; toResultAt = 0;
  const ws = new WebSocket(TRYON_WS);
  toWs = ws;
  ws.onopen = () => {
    setDot('to', true);
    const r = new FileReader();
    r.onload = (e) => ws.send(initPayload((e.target!.result as string).split(',')[1]));
    r.readAsDataURL(garment);
    const btn = byId('tryonBtn');
    btn.textContent = 'Stop';
    btn.className = 'btn btn-stop';
    toRunning = true;
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data as string) as TryOnMsg;
    if (msg.type === 'ready') {
      setStatus('tryonStatus', 'success', 'Try-on active');
      startToRender(); startToTimer(); sendToFrame();
    } else if (msg.type === 'result') {
      const ms = Date.now() - toFrameSentAt;
      byId('toSpinnerMs').textContent = ms + 'ms';
      toProcessing = false; toSpinner(false);
      const img = new Image();
      img.onload = () => { toLatestResult = img; toResultAt = Date.now(); };
      img.src = 'data:image/jpeg;base64,' + msg.image;
      // Mask diagnostic, shown only with ?diag=1 on the URL. A screenshot of
      // a bad result then already carries the reason.
      if (msg.diag && new URLSearchParams(location.search).has('diag')) {
        let d = maybe('toDiag');
        if (!d) {
          d = document.createElement('div');
          d.id = 'toDiag';
          d.style.cssText = 'position:absolute;right:10px;bottom:10px;z-index:9;'
            + 'font:11px ui-monospace,Menlo,monospace;padding:6px 9px;border-radius:8px;'
            + 'background:rgba(20,17,15,.82);color:#fff;letter-spacing:.02em';
          (maybe('tryonCanvas')?.parentElement || document.body).appendChild(d);
        }
        d.textContent = `region=${msg.diag.region} face=${msg.diag.face}`
          + ` collar_px=${msg.diag.collar_px} torso_px=${msg.diag.torso_px}`;
      }
      sendToFrame();
    } else if (msg.type === 'dropped') {
      toProcessing = false; toSpinner(false);
      sendToFrame();
    } else if (msg.type === 'error') {
      toProcessing = false; toSpinner(false);
      setStatus('tryonStatus', 'error', msg.message ?? 'error');
    }
  };
  ws.onclose = () => { setDot('to', false); stopToTimer(); stopToRender(); toRunning = false; toSpinner(false); };
  ws.onerror = () => { setStatus('tryonStatus', 'error', 'Connection error'); stopTryOn(); };
}

export function stopTryOn(): void {
  if (toWs) toWs.close();
  stopToTimer(); stopToRender(); toRunning = false;
  const btn = byId('tryonBtn');
  btn.textContent = 'Start Live Try-On';
  btn.className = 'btn btn-primary';
}

function startToTimer(): void { if (toSendTimer) clearInterval(toSendTimer); toSendTimer = window.setInterval(sendToFrame, 800); }
function stopToTimer(): void { if (toSendTimer) { clearInterval(toSendTimer); toSendTimer = null; } }

function startToRender(): void {
  if (toRafId) cancelAnimationFrame(toRafId);
  const canvas = byId<HTMLCanvasElement>('tryonCanvas');
  const video  = byId<HTMLVideoElement>('tryonVideo');
  const ctx    = canvas.getContext('2d')!;
  (function render() {
    const vw = video.videoWidth || 480, vh = video.videoHeight || 480;
    const sz = Math.min(vw, vh);
    if (canvas.width !== sz || canvas.height !== sz) { canvas.width = sz; canvas.height = sz; }
    // A result that stopped arriving must not sit frozen on screen. Past the
    // timeout, fall back to the live camera, which is at least honest.
    const toFresh = toLatestResult && (Date.now() - toResultAt) < TO_STALL_MS;
    if (toFresh) { ctx.drawImage(toLatestResult!, 0, 0, sz, sz); }
    else if (video.readyState >= 2) { drawCam(ctx, video, sz, MIRROR.to); }
    if (toRunning) toRafId = requestAnimationFrame(render);
  })();
}
function stopToRender(): void { if (toRafId) { cancelAnimationFrame(toRafId); toRafId = null; } }

// One frame on the wire at a time. The watchdog is what stops a reply that
// never comes from wedging the stream permanently.
const TO_STALL_MS = 6000;
function sendToFrame(): void {
  if (!toWs || toWs.readyState !== WebSocket.OPEN) return;
  if (toProcessing && (Date.now() - toFrameSentAt) < TO_STALL_MS) return;
  const video = byId<HTMLVideoElement>('tryonVideo');
  const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
  const sz = Math.min(vw, vh);
  if (toCvs.width !== sz) { toCvs.width = sz; toCvs.height = sz; }
  drawCam(toCtx, video, sz, MIRROR.to);
  toProcessing = true; toFrameSentAt = Date.now();
  // Only show the spinner before the first result, to avoid a flash on a fast GPU.
  if (!toLatestResult) toSpinner(true);
  toWs.send(JSON.stringify({ type: 'frame', image: toCvs.toDataURL('image/jpeg', 0.85).split(',')[1] }));
}

// Pause when the tab is hidden; resume the send timer when it comes back.
export function pauseForHiddenTab(): void {
  if (toSendTimer) { clearInterval(toSendTimer); toSendTimer = null; }
}
export function resumeIfRunning(): void {
  if (toRunning) startToTimer();
}

export function initTryOn(): void {
  // Rates what the shopper is wearing. See outfit-score.ts for why fit is
  // NOT read off the try-on render.
  OutfitScore.load()
    .then(renderOutfitScore)
    .catch((err: Error) => console.warn('[outfit] fit_model.json not loaded —', err.message));
}
