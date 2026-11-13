// Live Face Swap.
//
// Two clocks run here, and keeping them apart is the whole design.
//
//   • The SERVER clock, ~5-8 fps. inswapper + CodeFormer on an A10G will not
//     go faster. It decides what the swapped face LOOKS like.
//   • The HEAD clock, display refresh. MediaPipe finds the user's face locally
//     in ~4 ms. It decides where that face SITS.
//
// The reply is cut down to its face oval and re-projected onto wherever the
// head is right now, through the similarity transform between the landmarks
// measured when that frame was captured and the landmarks in the frame on
// screen. Head motion tracks at display rate; only a change of expression
// waits for the server. Everywhere outside the oval is the live camera.
import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import { LUCY_SESSION_ID, selectedSwapWs } from './config';
import { MIRROR, attachStream, cameraErrorText, cameraUnavailableReason, drawCam } from './camera';
import { avatarState } from './avatar-state';
import { byId } from './dom';
import { createFaceLandmarker } from './mediapipe';
import { setDot, setStatus } from './ui';
import { isVoiceActive, startStreamingVoice, stopStreamingVoice } from './voice';

type Point = { x: number; y: number };
type Landmarks = Point[];

// MediaPipe 468-point indices. The oval is the closed face contour; the lips
// ring is the outer lip contour. The anchors are deliberately eye corners,
// nose bridge and temples only — no jaw, no lips. Fitting on points that move
// with expression makes the whole mask swim when the user talks.
const LS_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,
                 378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,
                 162,21,54,103,67,109];
const LS_LIPS = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,
                 37,39,40,185];
const LS_ANCHORS = [33,133,362,263,168,6,1,197,234,454,127,356];

interface SwapMsg {
  type: string;
  id?: number;
  t?: number;
  image?: string;
  mask?: string;
  message?: string;
  hair?: boolean;
  hair_reason?: string;
}

interface SwapResult { img: HTMLImageElement; lm: Landmarks | null; patch: HTMLCanvasElement | null; t: number }
interface Similarity { a: number; b: number; e: number; f: number; scale: number }

let liveStream: MediaStream | null = null;
let liveWs: WebSocket | null = null;
let liveRunning = false;
let liveRafId: number | null = null;

// The latest reply, plus the landmarks from the moment the frame it answers
// was captured. The image says what the face looks like, the landmarks say
// where it was standing when it was photographed.
let lsResult: SwapResult | null = null;
let lsInFlight: { id: number; t0: number } | null = null;   // exactly one frame on the wire
const lsPending = new Map<number, Landmarks>();
let lsSeq = 0;

let lsTracker: FaceLandmarker | null = null;
let lsTrackerLoading = false;
let lsTrackTs = -1;
let lsCur: Landmarks | null = null;
let lsLost = 0;
let lsStat = { ms: 0, frames: 0, since: 0, fps: 0 };
let lsFellBack = false;   // V2 → V1 drop already used this session
let lsStatTimer: number | null = null;

const lsCvs = document.createElement('canvas');
const lsCtx = lsCvs.getContext('2d')!;
const lsPatchCvs = document.createElement('canvas');
const lsMouthCvs = document.createElement('canvas');
const lsAlphaCvs = document.createElement('canvas');
let lsHairNote = '';   // why hair is off, when the user asked for it

// A result older than this is not worth pasting on a live face any more.
const LS_MAX_STALE_MS = 1500;

// Safari only gained CanvasRenderingContext2D.filter in version 18. Where it
// is missing the feathered mask degrades to a hard edge instead of throwing.
const CANVAS_FILTER = typeof CanvasRenderingContext2D !== 'undefined' && 'filter' in CanvasRenderingContext2D.prototype;
function setBlur(x: CanvasRenderingContext2D, px: number): void {
  if (CANVAS_FILTER) x.filter = px > 0 ? `blur(${px}px)` : 'none';
}

function lsOpt(id: string, dflt: boolean): boolean {
  const el = document.getElementById(id) as HTMLInputElement | null;
  return el ? el.checked : dflt;
}
function lsCaptureSize(): number {
  const el = document.getElementById('lsQuality') as HTMLSelectElement | null;
  return el ? +el.value : 384;
}

export function checkLiveReady(): void {
  if (avatarState.liveSource && liveStream) byId<HTMLButtonElement>('liveSwapBtn').disabled = false;
}

export async function startLiveCamera(): Promise<void> {
  const blocked = cameraUnavailableReason();
  if (blocked) { setStatus('liveStatus', 'error', blocked); return; }
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    await attachStream(byId<HTMLVideoElement>('liveVideo'), liveStream);
    const btn = byId('liveCamBtn');
    btn.textContent = 'Camera on'; btn.className = 'btn btn-active';
    checkLiveReady();
    // Fetch the tracker now, while the user is still picking a face. It is a
    // ~10 MB download; paying for it after they press Start is the difference
    // between the mask locking on immediately and the first few seconds
    // looking like the old behaviour.
    void lsInitTracker();
  } catch (e) { setStatus('liveStatus', 'error', cameraErrorText(e)); }
}

async function lsInitTracker(): Promise<FaceLandmarker | null> {
  if (lsTracker || lsTrackerLoading) return lsTracker;
  lsTrackerLoading = true;
  try {
    lsTracker = await createFaceLandmarker();
  } catch (e) {
    // No tracker means no re-projection, so the renderer falls back to the
    // old full-frame blit rather than showing nothing.
    console.warn('[liveswap] face tracker unavailable, falling back to full-frame paste', e);
    lsTracker = null;
  } finally {
    lsTrackerLoading = false;
  }
  return lsTracker;
}

// Landmarks in the same normalised square-crop space the server sees: centre
// crop to a square, then mirror if the feature mirrors.
function lsTrack(video: HTMLVideoElement): Landmarks | null {
  if (!lsTracker || video.readyState < 2 || !video.videoWidth) return null;
  const ts = performance.now();
  if (ts <= lsTrackTs) return null;
  lsTrackTs = ts;
  let res;
  try { res = lsTracker.detectForVideo(video, ts); } catch { return null; }
  const face = res && res.faceLandmarks && res.faceLandmarks[0];
  if (!face) return null;
  const vw = video.videoWidth, vh = video.videoHeight;
  const side = Math.min(vw, vh), sx = (vw - side) / 2, sy = (vh - side) / 2;
  const out: Landmarks = new Array(face.length);
  for (let i = 0; i < face.length; i++) {
    let x = (face[i].x * vw - sx) / side;
    const y = (face[i].y * vh - sy) / side;
    if (MIRROR.ls) x = 1 - x;
    out[i] = { x, y };
  }
  return out;
}

// Least-squares similarity (uniform scale + rotation + translation) taking the
// anchor points of `from` onto those of `to`. Closed form, no iteration.
function lsFit(from: Landmarks, to: Landmarks): Similarity | null {
  const n = LS_ANCHORS.length;
  let mfx = 0, mfy = 0, mtx = 0, mty = 0;
  for (const i of LS_ANCHORS) {
    if (!from[i] || !to[i]) return null;
    mfx += from[i].x; mfy += from[i].y; mtx += to[i].x; mty += to[i].y;
  }
  mfx /= n; mfy /= n; mtx /= n; mty /= n;
  let dot = 0, cross = 0, den = 0;
  for (const i of LS_ANCHORS) {
    const fx = from[i].x - mfx, fy = from[i].y - mfy;
    const tx = to[i].x - mtx,   ty = to[i].y - mty;
    dot += fx * tx + fy * ty;
    cross += fx * ty - fy * tx;
    den += fx * fx + fy * fy;
  }
  if (den < 1e-9) return null;
  const a = dot / den, b = cross / den;          // a = s·cosθ, b = s·sinθ
  return {
    a, b,
    e: mtx - (a * mfx - b * mfy),
    f: mty - (b * mfx + a * mfy),
    scale: Math.hypot(a, b),
  };
}

// Trace a landmark loop, scaled about its own centroid, into `ctx` at `size`.
function lsPath(ctx: CanvasRenderingContext2D, lm: Landmarks, idx: number[], size: number, scale: number): void {
  let cx = 0, cy = 0;
  for (const i of idx) { cx += lm[i].x; cy += lm[i].y; }
  cx /= idx.length; cy /= idx.length;
  ctx.beginPath();
  for (let k = 0; k < idx.length; k++) {
    const p = lm[idx[k]];
    const x = (cx + (p.x - cx) * scale) * size;
    const y = (cy + (p.y - cy) * scale) * size;
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

// Cut the server's full frame down to a feathered face oval. Pulled in to
// 0.94: inswapper's own paste stops around the hairline, so a mask cut on
// MediaPipe's oval exactly would carry a rim of un-swapped forehead with it.
function lsBuildPatch(img: HTMLImageElement, lm: Landmarks, maskImg: HTMLImageElement | null): HTMLCanvasElement {
  const R = img.width || 384;
  const c = lsPatchCvs, x = c.getContext('2d')!;
  if (c.width !== R) { c.width = R; c.height = R; }
  x.setTransform(1, 0, 0, 1, 0, 0);
  x.clearRect(0, 0, R, R);
  x.drawImage(img, 0, 0, R, R);
  x.globalCompositeOperation = 'destination-in';
  if (maskImg && maskImg.width) {
    x.drawImage(lsAlphaFrom(lm, maskImg, R), 0, 0);
  } else {
    setBlur(x, Math.max(4, Math.round(R * 0.03)));
    x.fillStyle = '#fff';
    lsPath(x, lm, LS_OVAL, R, 0.94);
    x.fill();
    setBlur(x, 0);
  }
  x.globalCompositeOperation = 'source-over';
  return c;
}

// When hair is being swapped the face oval is the wrong cut. The server sends
// the pixels it actually touched as a grey JPEG: its brightness is the alpha
// we want. Union it with the face oval, then move luminance into alpha.
function lsAlphaFrom(lm: Landmarks, maskImg: HTMLImageElement, R: number): HTMLCanvasElement {
  const c = lsAlphaCvs, x = c.getContext('2d', { willReadFrequently: true })!;
  if (c.width !== R) { c.width = R; c.height = R; }
  x.setTransform(1, 0, 0, 1, 0, 0);
  x.globalCompositeOperation = 'source-over';
  setBlur(x, 0);
  x.fillStyle = '#000';
  x.fillRect(0, 0, R, R);
  x.drawImage(maskImg, 0, 0, R, R);
  x.globalCompositeOperation = 'lighter';           // union, not replace
  setBlur(x, Math.max(4, Math.round(R * 0.03)));
  x.fillStyle = '#fff';
  lsPath(x, lm, LS_OVAL, R, 0.94);
  x.fill();
  setBlur(x, 0);
  x.globalCompositeOperation = 'source-over';
  const d = x.getImageData(0, 0, R, R), px = d.data;
  for (let i = 0; i < px.length; i += 4) {
    px[i + 3] = px[i];                              // grey level → alpha
    px[i] = px[i + 1] = px[i + 2] = 255;
  }
  x.putImageData(d, 0, 0);
  return c;
}

// The user's real, current lips over the swapped face, so speech is not dubbed.
function lsDrawLiveMouth(ctx: CanvasRenderingContext2D, video: HTMLVideoElement, sz: number, lm: Landmarks): void {
  const c = lsMouthCvs, x = c.getContext('2d')!;
  if (c.width !== sz) { c.width = sz; c.height = sz; }
  x.setTransform(1, 0, 0, 1, 0, 0);
  x.clearRect(0, 0, sz, sz);
  drawCam(x, video, sz, MIRROR.ls);
  x.globalCompositeOperation = 'destination-in';
  setBlur(x, Math.max(3, Math.round(sz * 0.012)));
  x.fillStyle = '#fff';
  lsPath(x, lm, LS_LIPS, sz, 1.18);
  x.fill();
  setBlur(x, 0);
  x.globalCompositeOperation = 'source-over';
  ctx.drawImage(c, 0, 0);
}

export function toggleLiveSwap(): void {
  lsFellBack = false;          // an explicit Start re-arms the V2 → V1 drop
  if (liveRunning) stopLiveSwap(); else startLiveSwap();
}

export function startLiveSwap(): void {
  const liveSource = avatarState.liveSource;
  if (!liveSource) { setStatus('liveStatus', 'error', 'Select a face first'); return; }
  lsResult = null; lsInFlight = null; lsPending.clear(); lsCur = null; lsLost = 0;
  lsStat = { ms: 0, frames: 0, since: performance.now(), fps: 0 };
  void lsInitTracker();
  const ws = new WebSocket(selectedSwapWs());
  liveWs = ws;
  ws.onopen = () => {
    setDot('ls', true);
    const baseInit: Record<string, unknown> = { type: 'init', session_id: LUCY_SESSION_ID, hair: lsOpt('lsHair', true) };
    const initMsg = (typeof liveSource === 'object' && liveSource.avatar_id)
      ? Object.assign(baseInit, { avatar_id: liveSource.avatar_id })
      : Object.assign(baseInit, { source_image: liveSource });
    ws.send(JSON.stringify(initMsg));
    const btn = byId('liveSwapBtn');
    btn.textContent = 'Stop';
    btn.className = 'btn btn-stop';
    liveRunning = true;
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data as string) as SwapMsg;
    if (msg.type === 'ping') {
      try { ws.send(JSON.stringify({ type: 'pong', t: msg.t })); } catch { /* ignore */ }
      return;
    }
    if (msg.type === 'pong') return;
    if (msg.type === 'ready') {
      // Hair can be asked for and still not happen — carry the reason into
      // the status line rather than reading as a broken feature.
      lsHairNote = (lsOpt('lsHair', true) && !msg.hair)
        ? ' · hair off: ' + (msg.hair_reason || 'unavailable')
        : (msg.hair ? ' · hair on' : '');
      setStatus('liveStatus', 'success', 'Live swap active' + lsHairNote);
      startLsStats(); startLsRender();
      // Voice auto-starts with the swap — the user gesture from the Start
      // click is what satisfies the browser's audioCtx.resume() rule.
      if (!isVoiceActive()) {
        startStreamingVoice().catch((err) => console.warn('[voice] auto-start failed', err));
      }
    } else if (msg.type === 'result') {
      // Read the capture landmarks BEFORE settling — settling releases the
      // wire and drops the pending entry for the frame it belongs to.
      const key = (msg.id != null) ? msg.id : (lsInFlight ? lsInFlight.id : null);
      const lm  = (key != null) ? lsPending.get(key) ?? null : null;
      lsSettle(msg.id);
      if (key != null) lsPending.delete(key);
      const img     = new Image();
      const maskImg = msg.mask ? new Image() : null;
      let waiting   = maskImg ? 2 : 1;
      const settled = () => {
        if (--waiting) return;
        lsResult = {
          img, lm: lm || null, t: performance.now(),
          patch: lm ? lsBuildPatch(img, lm, maskImg) : null,
        };
        lsStat.frames++;
      };
      img.onload = settled;
      if (maskImg) {
        // A mask that fails to decode is not worth dropping the frame for.
        maskImg.onload = maskImg.onerror = settled;
        maskImg.src = 'data:image/jpeg;base64,' + msg.mask;
      }
      img.src = 'data:image/jpeg;base64,' + msg.image;
    } else if (msg.type === 'no_face' || msg.type === 'dropped') {
      lsSettle(msg.id);
    } else if (msg.type === 'error') {
      // V2 closes the socket outright when its weights are not on the box.
      // V1 speaks the identical protocol, so drop to it and say why.
      if (/V2 swap not available/i.test(msg.message || '') && !lsFellBack) {
        lsFellBack = true;
        const v1 = document.querySelector<HTMLInputElement>('input[name="swapEngine"][value="v1"]');
        if (v1) v1.checked = true;
        setStatus('liveStatus', 'error',
          'V2 weights are not on the server yet — running V1. They download on the next backend restart.');
        try { ws.close(); } catch { /* ignore */ }
        setTimeout(() => { if (!liveRunning) startLiveSwap(); }, 400);
        return;
      }
      setStatus('liveStatus', 'error', msg.message ?? 'error');
    }
  };
  ws.onclose = () => { setDot('ls', false); stopLsStats(); stopLsRender(); liveRunning = false; };
  ws.onerror = () => { setStatus('liveStatus', 'error', 'Connection error'); stopLiveSwap(); };
}

export function stopLiveSwap(): void {
  if (liveWs) liveWs.close();
  stopLsStats(); stopLsRender(); liveRunning = false;
  lsResult = null; lsInFlight = null; lsPending.clear();
  const btn = byId('liveSwapBtn');
  btn.textContent = 'Start Live Swap';
  btn.className = 'btn btn-primary';
  if (isVoiceActive()) stopStreamingVoice();
}

// A reply frees the wire. An older server that does not echo the id has
// nothing to match on, so any reply is taken as the answer to whatever is in
// flight — correct, because only one frame is ever out there.
function lsSettle(id: number | undefined): void {
  if (!lsInFlight) return;
  if (id == null || id === lsInFlight.id) {
    lsStat.ms = performance.now() - lsInFlight.t0;
    lsPending.delete(lsInFlight.id);
    lsInFlight = null;
  }
}

function startLsStats(): void {
  if (lsStatTimer) clearInterval(lsStatTimer);
  lsStatTimer = window.setInterval(() => {
    const now = performance.now();
    const dt  = (now - lsStat.since) / 1000;
    if (dt > 0) { lsStat.fps = lsStat.frames / dt; }
    lsStat.frames = 0; lsStat.since = now;
    if (!liveRunning) return;
    const locked = lsTracker ? 'locked' : 'unlocked';
    setStatus('liveStatus', 'success',
      `Live swap active · ${lsStat.fps.toFixed(1)} swap fps · ${Math.round(lsStat.ms)} ms · mask ${locked}${lsHairNote}`);
  }, 1000);
}
function stopLsStats(): void { if (lsStatTimer) { clearInterval(lsStatTimer); lsStatTimer = null; } }

function startLsRender(): void {
  if (liveRafId) cancelAnimationFrame(liveRafId);
  const canvas = byId<HTMLCanvasElement>('liveCanvas');
  const video  = byId<HTMLVideoElement>('liveVideo');
  const ctx    = canvas.getContext('2d')!;
  (function render() {
    liveRafId = liveRunning ? requestAnimationFrame(render) : null;
    const vw = video.videoWidth || 480, vh = video.videoHeight || 480;
    // Deliberately not the camera's full 720: holding the canvas near the
    // capture size keeps both halves of the picture at one resolution.
    const sz = Math.min(vw, vh, Math.max(512, lsCaptureSize()));
    if (canvas.width !== sz || canvas.height !== sz) { canvas.width = sz; canvas.height = sz; }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    // The live camera is always the base layer.
    if (video.readyState >= 2) drawCam(ctx, video, sz, MIRROR.ls);

    const cur = lsTrack(video);
    if (cur) { lsCur = cur; lsLost = 0; } else if (lsTracker) { lsLost++; }

    const fresh = !!lsResult && (performance.now() - lsResult.t) < LS_MAX_STALE_MS;
    const lock  = lsOpt('lsLockMask', true) && lsTracker
                  && fresh && lsResult!.patch && lsResult!.lm && lsCur && lsLost < 8;
    if (lock) {
      const T = lsFit(lsResult!.lm!, lsCur!);
      // A wild fit means the tracker jumped to something that is not the same
      // face. Better to show the untouched camera for a frame.
      if (T && T.scale > 0.4 && T.scale < 2.5) {
        ctx.save();
        ctx.scale(sz, sz);
        ctx.transform(T.a, T.b, -T.b, T.a, T.e, T.f);
        ctx.drawImage(lsResult!.patch!, 0, 0, 1, 1);
        ctx.restore();
        if (lsOpt('lsLiveMouth', true)) lsDrawLiveMouth(ctx, video, sz, lsCur!);
      }
    } else if (lsResult && fresh) {
      // No tracker, or the mask lock was turned off: the old whole-frame paste.
      ctx.drawImage(lsResult.img, 0, 0, sz, sz);
    }
    lsPaceSend();
  })();
}
function stopLsRender(): void { if (liveRafId) { cancelAnimationFrame(liveRafId); liveRafId = null; } }

// Exactly one frame on the wire, and the next one is captured the instant the
// previous reply lands — never on a timer.
function lsPaceSend(): void {
  if (!liveWs || liveWs.readyState !== WebSocket.OPEN) return;
  const now = performance.now();
  if (lsInFlight) {
    if (now - lsInFlight.t0 < 2500) return;   // still working — wait
    lsPending.delete(lsInFlight.id);          // lost reply — reopen the wire
    lsInFlight = null;
  }
  // Nothing on camera to swap: a frame with no face comes back `no_face` and
  // would leave the stale mask on screen for another 200 ms.
  if (lsTracker && !lsCur) return;
  const video = byId<HTMLVideoElement>('liveVideo');
  const vw = video.videoWidth || 0, vh = video.videoHeight || 0;
  if (!vw || !vh) return;
  const sz = Math.min(vw, vh, lsCaptureSize());
  if (lsCvs.width !== sz) { lsCvs.width = sz; lsCvs.height = sz; }
  drawCam(lsCtx, video, sz, MIRROR.ls);
  const id = ++lsSeq;
  if (lsCur) {
    lsPending.set(id, lsCur);
    // Bounded: a server that never echoes an id would otherwise add one
    // entry per frame for the whole session.
    while (lsPending.size > 12) lsPending.delete(lsPending.keys().next().value as number);
  }
  lsInFlight = { id, t0: now };
  liveWs.send(JSON.stringify({
    type: 'frame', id,
    image: lsCvs.toDataURL('image/jpeg', 0.9).split(',')[1],
  }));
}

// Coming back from a hidden tab: the reply to the frame that was on the wire
// may never arrive, and until the slot is free nothing new is sent.
export function releaseInFlightIfRunning(): void {
  if (liveRunning) lsInFlight = null;
}
