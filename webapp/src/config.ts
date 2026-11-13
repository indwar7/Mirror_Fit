// Backend host. The demo page used to be served by tryon_backend itself, so
// the right backend was simply the host the page was loaded from — a new EC2
// public IP needed no code change. That still holds when this build is served
// from the box. Served from anywhere else (Vercel, localhost dev server), set
// VITE_BACKEND_HOST in .env; the literal is the last-resort fallback for a
// page opened off disk, where hostname is empty.
const FALLBACK_HOST = '13.126.77.241';

export const BACKEND_HOST: string =
  (import.meta.env.VITE_BACKEND_HOST as string | undefined) || location.hostname || FALLBACK_HOST;

// Every backend URL follows the page's own scheme. A browser blocks ws:// and
// http:// requests from an https:// page as mixed content, and the failure
// looks like a dead backend. The demo hard-coded ws:// on three of these and
// surfaced it as the misleading "is port 7862 running?" error. Consequence
// for hosting: an HTTPS deployment needs TLS in front of the box (Caddy with
// a domain), because the backends themselves speak plain HTTP.
const SECURE = location.protocol === 'https:';
const HTTP = SECURE ? 'https://' : 'http://';
const WS   = SECURE ? 'wss://'   : 'ws://';

export const FACE_API = `${HTTP}${BACKEND_HOST}:7860`;
export const TWIN_API = `${HTTP}${BACKEND_HOST}:7863`;
export const TRYON_WS = `${WS}${BACKEND_HOST}:8000/ws/tryon`;

// Live Face Swap uses inswapper-128 on port 7860 — fast, identity-stable,
// ~5 fps. Live Portrait uses the LivePortrait pipeline on port 7862, which
// drives the source's expression from the camera frame. Independent backends,
// independent UI, so either can be demoed without touching the other.
export const SWAP_WS_V1 = `${WS}${BACKEND_HOST}:7860/ws/live-swap`;
export const SWAP_WS_V2 = `${WS}${BACKEND_HOST}:7860/ws/live-swap-v2`;
export const LIVEPORTRAIT_WS = `${WS}${BACKEND_HOST}:7862/ws/liveportrait-swap`;
export const VOICE_WS_URL = `${WS}${BACKEND_HOST}:7860/ws/voice-stream`;

// Read the engine radio at connect time; default V1 if nothing selected.
export function selectedSwapWs(): string {
  const el = document.querySelector<HTMLInputElement>('input[name="swapEngine"]:checked');
  return el && el.value === 'v2' ? SWAP_WS_V2 : SWAP_WS_V1;
}

// Same session_id is sent on both /ws/live-swap and /ws/voice-stream so the
// server can correlate audio with frames for Wav2Lip lip-sync.
export const LUCY_SESSION_ID =
  'sess_' + Math.random().toString(36).slice(2, 10) + '_' + Date.now().toString(36);
