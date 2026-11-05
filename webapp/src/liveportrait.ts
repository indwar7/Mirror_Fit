// Live Portrait (expression-driven swap, port 7862).
// Completely independent of liveswap — own stream, own WebSocket, own canvas.
//
// Transport: frames go out as raw binary JPEG and animated frames come back
// the same way. Status messages (calibrating / no_face / dropped / error)
// stay JSON.
import { LIVEPORTRAIT_WS, LUCY_SESSION_ID } from './config';
import { MIRROR, drawCam } from './camera';
import { avatarState } from './avatar-state';
import { byId } from './dom';
import { setDot, setStatus } from './ui';
import { isVoiceActive, startStreamingVoice, stopStreamingVoice } from './voice';

type Frame = ImageBitmap | HTMLImageElement;
interface LpMsg { type: string; t?: number; image?: string; message?: string }

let lpStream: MediaStream | null = null;
let lpWs: WebSocket | null = null;
let lpSource: { src: string } | null = null;
let lpRunning = false;
let lpLatestResult: Frame | null = null;
let lpRafId: number | null = null;
let lpWatchdog: number | null = null;
// Exactly one frame may be in flight.
let lpInFlight = false, lpSentAt = 0, lpCapturing = false;
let lpFrames = 0, lpStatsAt = 0, lpMs = 0;

const lpCvs = document.createElement('canvas');
const lpCtx = lpCvs.getContext('2d')!;

function closeFrame(f: Frame | null): void {
  if (f && 'close' in f) f.close();
}

function checkLpReady(): void {
  if (lpSource && lpStream) byId<HTMLButtonElement>('lpStartBtn').disabled = false;
}

export async function startLpCamera(): Promise<void> {
  try {
    lpStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    byId<HTMLVideoElement>('lpVideo').srcObject = lpStream;
    const btn = byId('lpCamBtn');
    btn.textContent = 'Camera on';
    btn.className = 'btn btn-success';
    checkLpReady();
  } catch (e) { setStatus('lpStatus', 'error', 'Camera blocked: ' + (e as Error).message); }
}

export function setLpSourceFile(inp: HTMLInputElement): void {
  const f = inp.files && inp.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = (e) => { lpSource = { src: e.target!.result as string }; checkLpReady(); };
  r.readAsDataURL(f);
}

function lpSend(obj: unknown): void {
  if (lpWs && lpWs.readyState === WebSocket.OPEN) lpWs.send(JSON.stringify(obj));
}

// Head-follow and recalibrate are live controls: the backend re-tunes an
// existing session, so neither needs a stop/start round trip.
export function setLpFollowHead(on: boolean): void { lpSend({ type: 'config', pose_gain: on ? 0.7 : 0.0 }); }
export function lpRecalibrate(): void {
  lpSend({ type: 'recalibrate' });
  setStatus('lpStatus', 'info', 'Recalibrating — hold a neutral face…');
}
export function toggleLp(): void { if (lpRunning) stopLp(); else startLp(); }

export function startLp(): void {
  setStatus('lpStatus', 'info', 'Connecting…');
  lpLatestResult = null; lpInFlight = false; lpFrames = 0; lpMs = 0; lpStatsAt = performance.now();
  const ws = new WebSocket(LIVEPORTRAIT_WS);
  lpWs = ws;
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    setDot('lp', true); byId('lpStatusText').textContent = 'Connected';
    const src_b64 = lpSource!.src.split(',')[1];
    ws.send(JSON.stringify({ type: 'init', source_image: src_b64,
                             session_id: LUCY_SESSION_ID + '_lp',
                             pose_gain: byId<HTMLInputElement>('lpFollowHead').checked ? 0.7 : 0.0 }));
    const btn = byId('lpStartBtn');
    btn.textContent = 'Stop';
    btn.className = 'btn btn-stop';
    byId<HTMLButtonElement>('lpRecalBtn').disabled = false;
    lpRunning = true;

    // Auto-start streaming voice — same UX as Live Face Swap. The demo treats
    // the two sockets as a single feature.
    if (!isVoiceActive()) {
      // Pick a default voice profile so users uploading a custom portrait
      // still get a transformed voice.
      if (!avatarState.selected) avatarState.selected = 'gen_m_01';
      startStreamingVoice().catch((err) => console.warn('[lp+voice] auto-start failed', err));
    }
  };
  ws.onmessage = async (e) => {
    // Binary reply = an animated frame.
    if (e.data instanceof ArrayBuffer) {
      lpArrive(performance.now());
      try {
        const bmp = await createImageBitmap(new Blob([e.data], { type: 'image/jpeg' }));
        closeFrame(lpLatestResult);
        lpLatestResult = bmp;
      } catch { /* a dropped frame is not worth tearing the session down */ }
      sendLpFrame();
      return;
    }
    const msg = JSON.parse(e.data as string) as LpMsg;
    if (msg.type === 'ping') { lpSend({ type: 'pong', t: msg.t }); return; }
    if (msg.type === 'ready') {
      setStatus('lpStatus', 'info', 'Hold a neutral face for a moment…');
      startLpRender(); startLpWatchdog(); sendLpFrame();
      return;
    }
    if (msg.type === 'calibrating' || msg.type === 'result') {
      // calibrating: the untouched portrait while the baseline is collected.
      // result: the JSON fallback transport.
      lpArrive(performance.now());
      const img = new Image();
      img.onload = () => { closeFrame(lpLatestResult); lpLatestResult = img; };
      img.src = 'data:image/jpeg;base64,' + msg.image;
      sendLpFrame();
      return;
    }
    if (msg.type === 'dropped') { lpInFlight = false; sendLpFrame(); return; }
    if (msg.type === 'no_face') {
      lpInFlight = false;
      setStatus('lpStatus', 'info', 'No face detected — move into frame');
      sendLpFrame();
      return;
    }
    if (msg.type === 'recalibrating') { lpLatestResult = null; return; }
    if (msg.type === 'config') { return; }
    if (msg.type === 'error') { setStatus('lpStatus', 'error', msg.message ?? 'error'); lpInFlight = false; }
  };
  ws.onclose = () => { setDot('lp', false); stopLpWatchdog(); stopLpRender(); lpRunning = false; };
  ws.onerror = () => { setStatus('lpStatus', 'error', 'Connection error — is port 7862 running?'); stopLp(); };
}

export function stopLp(): void {
  if (lpWs) lpWs.close();
  stopLpWatchdog(); stopLpRender(); lpRunning = false; lpInFlight = false;
  const btn = byId('lpStartBtn');
  btn.textContent = 'Start Live Portrait';
  btn.className = 'btn btn-primary';
  byId<HTMLButtonElement>('lpRecalBtn').disabled = true;
  byId('lpTelemetry').textContent = '';
  if (isVoiceActive()) stopStreamingVoice();
}

// Round-trip bookkeeping: clears the in-flight gate and keeps a 1 s rolling
// fps / latency readout so a slow link is visible instead of just "laggy".
function lpArrive(now: number): void {
  lpInFlight = false;
  lpFrames++; lpMs += now - lpSentAt;
  if (now - lpStatsAt >= 1000) {
    const fps = (lpFrames * 1000 / (now - lpStatsAt)).toFixed(1);
    const ms  = Math.round(lpMs / Math.max(lpFrames, 1));
    const el  = document.getElementById('lpTelemetry');
    if (el) el.textContent = `${fps} fps · ${ms} ms`;
    if (lpFrames > 2) setStatus('lpStatus', 'success', 'Live portrait active');
    lpFrames = 0; lpMs = 0; lpStatsAt = now;
  }
}

// The watchdog only covers a lost reply (backend restart, dropped packet).
function startLpWatchdog(): void {
  if (lpWatchdog) clearInterval(lpWatchdog);
  lpWatchdog = window.setInterval(() => {
    if (!lpRunning) return;
    if (lpInFlight && performance.now() - lpSentAt > 1500) { lpInFlight = false; sendLpFrame(); }
    else if (!lpInFlight) sendLpFrame();
  }, 500);
}
function stopLpWatchdog(): void { if (lpWatchdog) { clearInterval(lpWatchdog); lpWatchdog = null; } }

function startLpRender(): void {
  const cv = byId<HTMLCanvasElement>('lpCanvas'), ctx = cv.getContext('2d')!;
  const video = byId<HTMLVideoElement>('lpVideo');
  (function render() {
    if (!lpRunning) return;
    if (video.videoWidth > 0) {
      const vw = video.videoWidth, vh = video.videoHeight, sz = Math.min(vw, vh);
      if (cv.width !== sz) { cv.width = sz; cv.height = sz; }
      drawCam(ctx, video, sz, MIRROR.lp);
    }
    if (lpLatestResult) {
      ctx.globalAlpha = 1.0;
      // The result carries the photo's aspect ratio — cover-fit rather than
      // stretch, which visibly squashed every non-square upload.
      drawCover(ctx, lpLatestResult, cv.width, cv.height);
    }
    if (lpRunning) lpRafId = requestAnimationFrame(render);
  })();
}

function drawCover(ctx: CanvasRenderingContext2D, img: Frame, W: number, H: number): void {
  const iw = img.width || (img as HTMLImageElement).naturalWidth;
  const ih = img.height || (img as HTMLImageElement).naturalHeight;
  if (!iw || !ih) return;
  const s = Math.max(W / iw, H / ih), dw = iw * s, dh = ih * s;
  ctx.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
}
function stopLpRender(): void { if (lpRafId) { cancelAnimationFrame(lpRafId); lpRafId = null; } }

function sendLpFrame(): void {
  if (!lpWs || lpWs.readyState !== WebSocket.OPEN) return;
  if (lpInFlight || lpCapturing) return;
  const video = byId<HTMLVideoElement>('lpVideo');
  if (!video.videoWidth) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  const sz = Math.min(vw, vh, 384);
  if (lpCvs.width !== sz) { lpCvs.width = sz; lpCvs.height = sz; }
  drawCam(lpCtx, video, sz, MIRROR.lp);
  lpCapturing = true;
  // toBlob keeps the JPEG as bytes instead of a base64 data URL: ~33% smaller
  // and no main-thread string round trip.
  lpCvs.toBlob((blob) => {
    lpCapturing = false;
    if (!blob || !lpWs || lpWs.readyState !== WebSocket.OPEN || lpInFlight) return;
    blob.arrayBuffer().then((buf) => {
      if (!lpWs || lpWs.readyState !== WebSocket.OPEN) return;
      lpInFlight = true; lpSentAt = performance.now();
      lpWs.send(buf);
    }).catch(() => { lpInFlight = false; });
  }, 'image/jpeg', 0.70);
}
