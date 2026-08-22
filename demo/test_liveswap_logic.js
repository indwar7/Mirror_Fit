// Offline test for the Live Face Swap client in index.html.
//
//   node demo/test_liveswap_logic.js
//
// The tab's hard parts — one frame on the wire at a time, pairing each reply
// with the landmarks captured alongside the frame it answers, and re-projecting
// that reply onto a head that has since moved — are all pure logic, and all of
// it used to be checkable only by pointing a webcam at a GPU box in Mumbai.
// This lifts the block straight out of index.html, stubs the handful of DOM
// and network surfaces it touches, and drives a whole session through it.
const fs = require('fs');
const path = require('path');

const drawn = [];        // every drawImage onto the visible canvas
const transforms = [];   // every transform() applied before one

function ctxStub(tag) {
  return {
    tag, globalCompositeOperation: 'source-over', filter: 'none', fillStyle: '#000',
    _path: [],
    setTransform(){}, clearRect(){}, save(){}, restore(){},
    scale(s){ this._scale = s; },
    transform(a,b,c,d,e,f){ if (tag === 'display') transforms.push([a,b,c,d,e,f]); },
    fillRect(){ this._cleared = true; },
    getImageData(x, y, w, h){ return { data: new Uint8ClampedArray(w * h * 4), width: w, height: h }; },
    putImageData(){ this._put = true; },
    beginPath(){ this._path = []; }, closePath(){},
    moveTo(x,y){ this._path.push([x,y]); }, lineTo(x,y){ this._path.push([x,y]); },
    fill(){ this._filled = this._path.length; this._fillFilter = this.filter;
            this._fillOp = this.globalCompositeOperation; },
    drawImage(src){ if (tag === 'display') drawn.push(src && src._id || 'video'); },
  };
}
function canvasStub(id) {
  const c = { _id: id, width: 0, height: 0 };
  const ctx = ctxStub(id === 'liveCanvas' ? 'display' : id);
  c.getContext = () => ctx;      // options arg (willReadFrequently) ignored
  c._ctx = ctx;
  return c;
}

const els = {
  liveCanvas: canvasStub('liveCanvas'),
  liveVideo:  { _id: 'video', readyState: 4, videoWidth: 1280, videoHeight: 720, style: {} },
  liveSwapBtn: { textContent: '', className: '', disabled: true },
  liveCamBtn: { textContent: '', className: '' },
  lsLockMask: { checked: true },
  lsLiveMouth: { checked: true },
  lsHair: { checked: true },
  lsQuality:  { value: '384' },
  liveStatus: { textContent: '' },
};
const radios = { v1: { value: 'v1', checked: true }, v2: { value: 'v2', checked: false } };
global.document = {
  querySelector: sel => {
    const m = /value="(v1|v2)"/.exec(sel);
    if (m) return radios[m[1]];
    return radios.v1.checked ? radios.v1 : radios.v2;   // :checked
  },
  getElementById: id => els[id] || null,
  createElement: () => canvasStub('offscreen'),
  addEventListener(){},
};
let clock = 1000;
global.performance = { now: () => (clock += 16) };
global.window = {};
global.requestAnimationFrame = () => 1;
global.cancelAnimationFrame = () => {};
global.setInterval = () => 1; global.clearInterval = () => {};
global.console = console;

let lastImg = 0;
global.Image = class {
  constructor(){ this.width = 384; this.height = 384; this._id = 'result#' + (++lastImg); }
  set src(v){ this._src = v; queueMicrotask(() => this.onload && this.onload()); }
  get src(){ return this._src; }
};

const sent = [];
global.WebSocket = class {
  constructor(url){ this.url = url; this.readyState = 1; WS = this; }
  send(s){ sent.push(JSON.parse(s)); }
  close(){ this.readyState = 3; this.onclose && this.onclose(); }
};
global.WebSocket.OPEN = 1;
let WS = null;

// Page globals the block leans on
global.MIRROR = { ls: true };
global.LUCY_SESSION_ID = 'sess_test';
global._vsActive = false;
global.setStatus = (id, kind, text) => { els.liveStatus.textContent = text; };
global.setDot = () => {};
global.selectedSwapWs = () => 'ws://test/ws/live-swap';
global.drawCam = () => {};
global._lsCvs = canvasStub('_lsCvs');
global._lsCtx = global._lsCvs._ctx;
global._lsCvs.toDataURL = () => 'data:image/jpeg;base64,QUJD';

// Landmarks: a synthetic 468-point face we can rigidly move around.
function face(scale, degrees, tx, ty) {
  const th = degrees * Math.PI / 180, a = scale * Math.cos(th), b = scale * Math.sin(th);
  return Array.from({length: 468}, (_, i) => {
    const x = 0.5 + 0.16 * Math.cos(i * 2.399), y = 0.5 + 0.20 * Math.sin(i * 2.399);
    return { x: a*x - b*y + tx, y: b*x + a*y + ty };
  });
}
let TRACK = face(1, 0, 0, 0);
const tracker = {
  detectForVideo() {
    return { faceLandmarks: TRACK ? [TRACK.map(p => ({ x: p.x, y: p.y }))] : [] };
  },
};
// Landmarks come back in raw video coordinates and the block maps them into
// mirrored square-crop space, so feed it the inverse of that mapping.
function toVideoSpace(pts) {
  const vw = 1280, vh = 720, side = 720, sx = (vw-side)/2, sy = 0;
  return pts.map(p => ({ x: ((1 - p.x) * side + sx) / vw, y: (p.y * side + sy) / vh }));
}
const rawTracker = { detectForVideo(){ return TRACK ? { faceLandmarks: [toVideoSpace(TRACK)] } : {}; } };
global.window._lucyCreateFaceLandmarker = async () => rawTracker;

// Lift the block out of the page itself, so this cannot drift from what ships.
const page  = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const from  = page.indexOf('// \u2500\u2500 Live Face Swap \u2500');
const to    = page.indexOf('// \u2500\u2500 Live Portrait (expression-driven swap, port 7862) \u2500');
if (from < 0 || to < 0) {
  console.error('Could not find the Live Face Swap block in index.html — did its banner comment change?');
  process.exit(2);
}
const block = page.slice(from, to);
// Run the block in global scope, then hand back handles to the internals the
// test needs to poke at.
const api = new Function(block + `
  return {
    startLiveSwap, stopLiveSwap, _lsPaceSend, _lsSettle, _startLsRender,
    _lsInitTracker, _lsFit,
    state: () => ({ inFlight: _lsInFlight, pending: _lsPending, result: _lsResult,
                    cur: _lsCur, tracker: _lsTracker, seq: _lsSeq }),
    setSource: v => { liveSource = v; },
    setRunning: v => { liveRunning = v; },
  };
`)();

(async () => {
  let fails = 0;
  const ok  = (cond, msg) => { console.log((cond ? '  PASS  ' : '  FAIL  ') + msg); if (!cond) fails++; };

  await api._lsInitTracker();
  ok(api.state().tracker === rawTracker, 'face tracker initialises');

  api.setSource({ avatar_id: 'av_1' });
  api.startLiveSwap();
  WS.onopen();
  ok(sent.length === 1 && sent[0].type === 'init', 'init sent on open');
  ok(sent[0].hair === true, 'init asks the server for hair swap');
  WS.onmessage({ data: JSON.stringify({ type: 'ready', session_id: 'sess_test' }) });

  const render = api._startLsRender;
  // _startLsRender's loop self-schedules through requestAnimationFrame, which
  // is stubbed out, so each call here is exactly one displayed frame.
  const frame = () => { api.setRunning(true); render(); };

  // ── one frame in flight, never two ────────────────────────────────────────
  // `ready` starts the render loop and its first pass already captures a
  // frame, so the wire is busy before the test drives anything.
  const caps = sent.filter(m => m.type === 'frame');
  ok(caps.length === 1, 'ready starts the loop and captures exactly one frame');
  sent.length = 0;
  const id1 = caps[0].id;
  ok(api.state().pending.has(id1), 'capture landmarks stored against the frame id');
  frame(); frame(); frame(); frame();
  ok(sent.length === 0, 'no further captures while one is on the wire (was: one per 60 ms timer)');

  // ── reply pairs with its capture and becomes a masked patch ───────────────
  WS.onmessage({ data: JSON.stringify({ type: 'result', id: id1, image: 'QUJD' }) });
  await new Promise(r => setTimeout(r, 0));
  const st = api.state();
  ok(st.inFlight === null, 'reply frees the wire');
  ok(st.result && st.result.lm, 'result carries the landmarks captured with it');
  ok(st.result && st.result.patch, 'result is cut down to a face patch');
  ok(st.pending.size === 0, 'pending map does not leak');

  // ── the head moves; the mask must follow it without a new server frame ────
  drawn.length = 0; transforms.length = 0;
  TRACK = face(1.15, 14, 0.04, -0.03);
  frame();
  ok(drawn.some(d => d === 'offscreen'), 'patch is composited over the live camera');
  ok(transforms.length === 1, 'exactly one re-projection transform per displayed frame');
  const [a, b] = transforms[0] || [];
  const gotScale = Math.hypot(a, b), gotRot = Math.atan2(b, a) * 180 / Math.PI;
  ok(Math.abs(gotScale - 1.15) < 1e-6, `re-projection recovers the 1.15x scale (got ${gotScale.toFixed(6)})`);
  ok(Math.abs(gotRot - 14) < 1e-6, `re-projection recovers the 14 deg roll (got ${gotRot.toFixed(4)})`);

  // ── face leaves the frame ────────────────────────────────────────────────
  sent.length = 0;
  TRACK = null;
  for (let i = 0; i < 12; i++) frame();
  ok(sent.length === 0, 'no captures sent while there is no face on camera');
  drawn.length = 0; transforms.length = 0;
  frame();
  ok(transforms.length === 0, 'stale mask is not stamped on an empty frame');

  // ── stale results are dropped rather than pasted on a moved head ─────────
  TRACK = face(1, 0, 0, 0);
  clock += 3000;
  drawn.length = 0; transforms.length = 0;
  frame();
  ok(transforms.length === 0, 'a result older than 1.5 s is dropped, not re-projected');

  // ── a reply that never arrives must not wedge the stream ─────────────────
  // Fresh session so this does not depend on what the earlier sections left
  // on the wire.
  TRACK = face(1, 0, 0, 0);
  api.stopLiveSwap(); api.startLiveSwap(); WS.onopen();
  sent.length = 0;
  WS.onmessage({ data: JSON.stringify({ type: 'ready' }) });
  ok(sent.filter(m => m.type === 'frame').length === 1, 'a fresh session captures one frame');
  sent.length = 0;
  frame(); frame();
  ok(sent.length === 0, 'still one at a time while that frame is unanswered');
  clock += 3000;                       // blow through the 2500 ms watchdog
  frame();
  ok(sent.length === 1, 'watchdog reopens the wire after a lost reply');

  // ── a server that does not echo ids still gets a locked mask ─────────────
  sent.length = 0;
  api.stopLiveSwap();
  api.startLiveSwap(); WS.onopen();
  WS.onmessage({ data: JSON.stringify({ type: 'ready' }) });
  sent.length = 0;
  frame();
  WS.onmessage({ data: JSON.stringify({ type: 'result', image: 'QUJD' }) });   // no id
  await new Promise(r => setTimeout(r, 0));
  ok(api.state().result && api.state().result.lm, 'id-less reply still pairs with its capture');
  ok(api.state().inFlight === null, 'id-less reply still frees the wire');

  // ── hair: the reply carries a mask, and the patch waits for it ──────────
  // Cutting the reply to the face oval would crop the hair straight off, so
  // when the server sends the pixels it touched, that is what gets composited.
  api.stopLiveSwap(); api.startLiveSwap(); WS.onopen();
  WS.onmessage({ data: JSON.stringify({ type: 'ready', hair: true }) });
  ok(/hair on/.test(els.liveStatus.textContent), 'ready reports hair engaged');
  const hid = sent.filter(m => m.type === 'frame').pop().id;
  ok(api.state().result === null, 'fresh session starts with no result');
  WS.onmessage({ data: JSON.stringify({ type: 'result', id: hid, image: 'QUJD', mask: 'QUJD' }) });
  ok(api.state().result === null, 'nothing composited until both images decode');
  await new Promise(r => setTimeout(r, 0));
  ok(api.state().result && api.state().result.patch, 'patch built once image and mask are both in');

  // ── hair asked for but impossible must say why, not fail silently ───────
  api.stopLiveSwap(); api.startLiveSwap(); WS.onopen();
  WS.onmessage({ data: JSON.stringify({ type: 'ready', hair: false,
    hair_reason: 'face_parser.onnx is not on the server, so hair cannot be segmented' }) });
  ok(/hair off: face_parser\.onnx/.test(els.liveStatus.textContent),
     `the reason reaches the user ("${els.liveStatus.textContent.slice(0, 80)}")`);

  // ── V2 with no weights on the box must not dead-end the user ────────────
  radios.v1.checked = false; radios.v2.checked = true;
  api.stopLiveSwap(); api.startLiveSwap(); WS.onopen();
  WS.onmessage({ data: JSON.stringify({ type: 'error', message:
    'V2 swap not available — server is missing inswapper_128_fp16.onnx and/or GFPGANv1.4.onnx.' }) });
  ok(radios.v1.checked === true, 'a V2 box with no weights falls back to the V1 engine');
  ok(/running V1/i.test(els.liveStatus.textContent), `and says so ("${els.liveStatus.textContent}")`);
  await new Promise(r => setTimeout(r, 500));
  ok(WS.readyState === 1 && sent.some(m => m.type === 'init'), 'and reconnects on its own');

  // A genuine error is still surfaced rather than swallowed by the fallback.
  WS.onmessage({ data: JSON.stringify({ type: 'error', message: 'No face detected in source image' }) });
  ok(els.liveStatus.textContent === 'No face detected in source image', 'other errors still shown verbatim');

  console.log(fails ? `\n${fails} FAILURES` : '\nall checks passed');
  process.exit(fails ? 1 : 0);
})();
