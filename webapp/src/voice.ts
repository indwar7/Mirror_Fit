// Streaming voice: mic PCM → WebSocket → pitch-shift → speakers.
// Continuous full-duplex: while you speak, the server pitch-shifts your audio
// in ~500 ms chunks and streams it back. The browser schedules contiguous
// playback in a single AudioContext so it sounds like one continuous voice.
import { FACE_API, LUCY_SESSION_ID, VOICE_WS_URL } from './config';
import { avatarState } from './avatar-state';
import { byId, maybe } from './dom';
import { setStatus } from './ui';

const STREAM_SR    = 16000;
const STREAM_CHUNK = 4096;   // ScriptProcessor buffer ≈ 256 ms @ 16 kHz

let vsCtx: AudioContext | null = null;          // capture AudioContext
let vsPlayCtx: AudioContext | null = null;      // playback AudioContext (separate so capture echo is avoided)
let vsStream: MediaStream | null = null;        // mic MediaStream
let vsProcessor: ScriptProcessorNode | null = null;
let vsWs: WebSocket | null = null;
let vsActive = false;
let vsPlayTime = 0;                             // next scheduled playback start (sec)
let vsReconnected = false;                      // one-shot guard for the auto-reconnect attempt

interface VoiceMsg { type: string; t?: number; semitones?: number; message?: string }

export function isVoiceActive(): boolean { return vsActive; }

function audioContextCtor(): typeof AudioContext {
  return window.AudioContext || window.webkitAudioContext!;
}

export function resetVoiceUI(): void {
  byId('voiceBtnLabel').textContent = 'Start live voice';
  byId('recDot').style.display = 'none';
  byId('voiceBtn').classList.remove('active');
  const st = maybe('voiceStatus');
  if (st) { st.className = 'status-msg'; st.textContent = ''; }
}

export async function toggleStreamingVoice(): Promise<void> {
  if (vsActive) stopStreamingVoice();
  else          await startStreamingVoice();
}
// Back-compat aliases — older buttons may still call these names.
export async function toggleRealtimeVoice(): Promise<void> { return toggleStreamingVoice(); }
export async function toggleLipSyncRecord(): Promise<void> { return toggleStreamingVoice(); }

export async function startStreamingVoice(): Promise<void> {
  if (!avatarState.selected) { setStatus('voiceStatus', 'error', 'Select an avatar first'); return; }
  console.log('[voice] starting…');
  try {
    vsStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: STREAM_SR, channelCount: 1,
               echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    console.log('[voice] mic ok');
    const AC = audioContextCtor();
    vsCtx     = new AC({ sampleRate: STREAM_SR });
    vsPlayCtx = new AC({ sampleRate: STREAM_SR });
    // AudioContexts start SUSPENDED in modern browsers — must resume() with a
    // user gesture or audio will silently never play. The button click is the
    // user gesture; call resume now.
    if (vsCtx.state     === 'suspended') await vsCtx.resume();
    if (vsPlayCtx.state === 'suspended') await vsPlayCtx.resume();
    console.log('[voice] audioCtx state:', vsCtx.state, vsPlayCtx.state);
    vsPlayTime = vsPlayCtx.currentTime + 0.1;

    const handleMessage = (e: MessageEvent) => {
      const playCtx = vsPlayCtx;
      if (!playCtx) return;
      if (e.data instanceof ArrayBuffer) {
        const pcm = new Float32Array(e.data);
        if (pcm.length === 0) return;
        const buf = playCtx.createBuffer(1, pcm.length, STREAM_SR);
        buf.copyToChannel(pcm, 0);
        const src = playCtx.createBufferSource();
        src.buffer = buf;
        src.connect(playCtx.destination);
        const now   = playCtx.currentTime;
        const start = Math.max(now + 0.02, vsPlayTime);
        src.start(start);
        vsPlayTime = start + buf.duration;
      } else {
        try {
          const m = JSON.parse(e.data as string) as VoiceMsg;
          // Keepalive: server pings us every 20s; reply with pong so any
          // intermediate proxy sees bidirectional traffic. Swallow pongs.
          if (m.type === 'ping') {
            try { vsWs?.send(JSON.stringify({ type: 'pong', t: m.t })); } catch { /* ignore */ }
            return;
          }
          if (m.type === 'pong') return;
          console.log('[voice] ws msg:', m);
          if (m.type === 'ready') {
            const av = avatarState.all.find((a) => a.id === avatarState.selected);
            const semis = m.semitones ?? 0;
            setStatus('voiceStatus', 'success',
                      `Live: ${av?.name || 'avatar'} (${semis > 0 ? '+' : ''}${semis} semitones)`);
          } else if (m.type === 'error') {
            setStatus('voiceStatus', 'error', m.message || 'stream error');
          }
        } catch { /* not JSON */ }
      }
    };
    const handleError = (err: Event) => {
      console.error('[voice] ws error', err);
      setStatus('voiceStatus', 'error', 'Voice stream WS error — check console');
    };
    const attach = (sock: WebSocket) => {
      sock.binaryType = 'arraybuffer';
      sock.onopen = () => {
        console.log('[voice] ws open', 'audioCtx=', vsCtx?.state, 'playCtx=', vsPlayCtx?.state);
        sock.send(JSON.stringify({
          type: 'init',
          avatar_id:  avatarState.selected,
          session_id: LUCY_SESSION_ID,
        }));
      };
      sock.onmessage = handleMessage;
      sock.onerror   = handleError;
      sock.onclose   = (ev) => {
        console.log('[voice] ws close', ev.code, ev.reason, 'wasClean=', ev.wasClean, 'audioCtx=', vsCtx?.state);
        // Auto-reconnect ONCE if the user didn't intentionally stop — covers
        // transient ALB/SG/NAT cuts. Mic + audio contexts are still live;
        // only the socket needs replacing.
        if (vsActive && !vsReconnected) {
          vsReconnected = true;
          console.log('[voice] attempting one reconnect…');
          setTimeout(() => {
            if (!vsActive) return;
            try {
              vsWs = new WebSocket(VOICE_WS_URL);
              attach(vsWs);
            } catch (e) {
              console.error('[voice] reconnect failed', e);
              stopStreamingVoice();
            }
          }, 600);
          return;
        }
        if (vsActive) stopStreamingVoice();
      };
    };
    vsWs = new WebSocket(VOICE_WS_URL);
    attach(vsWs);

    const source = vsCtx.createMediaStreamSource(vsStream);
    vsProcessor  = vsCtx.createScriptProcessor(STREAM_CHUNK, 1, 1);
    source.connect(vsProcessor);
    vsProcessor.connect(vsCtx.destination);
    let firstAudioLogged = false;
    vsProcessor.onaudioprocess = (ev) => {
      if (!vsWs || vsWs.readyState !== WebSocket.OPEN) return;
      const inp = ev.inputBuffer.getChannelData(0);
      // Simple level check so we can confirm the mic is actually receiving sound.
      if (!firstAudioLogged) {
        let s = 0; for (let i = 0; i < inp.length; i++) s += Math.abs(inp[i]);
        console.log('[voice] first audio chunk len=', inp.length, 'sumAbs=', s.toFixed(3));
        firstAudioLogged = true;
      }
      const out = new Float32Array(inp.length);
      out.set(inp);
      vsWs.send(out.buffer);
    };

    vsActive = true;
    byId('voiceBtnLabel').textContent = 'Stop live voice';
    byId('recDot').style.display = 'inline-block';
    byId('voiceBtn').classList.add('active');
    setStatus('voiceStatus', 'info', 'Connecting… speak when you see "Live:" status.');
  } catch (e) {
    console.error('[voice] start failed', e);
    setStatus('voiceStatus', 'error', 'Mic/Audio error: ' + (e as Error).message);
    stopStreamingVoice();
  }
}

export function stopStreamingVoice(): void {
  vsActive = false;
  vsReconnected = false;
  try { if (vsWs)        vsWs.close();          } catch { /* ignore */ }
  try { if (vsProcessor) vsProcessor.disconnect(); } catch { /* ignore */ }
  try { if (vsStream)    vsStream.getTracks().forEach((t) => t.stop()); } catch { /* ignore */ }
  try { if (vsCtx)       void vsCtx.close();     } catch { /* ignore */ }
  try { if (vsPlayCtx)   void vsPlayCtx.close(); } catch { /* ignore */ }
  vsWs = null; vsProcessor = null; vsStream = null; vsCtx = null; vsPlayCtx = null;
  resetVoiceUI();
}

// Browser-side WAV encoder — avoids needing ffmpeg on the server.
export function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numCh   = 1;                              // mono
  const sr      = buffer.sampleRate;
  const samples = buffer.getChannelData(0);
  const buf     = new ArrayBuffer(44 + samples.length * 2);
  const view    = new DataView(buf);
  const w = (off: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true);
  w(8, 'WAVE'); w(12, 'fmt '); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);              // PCM
  view.setUint16(22, numCh, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * numCh * 2, true); // byte rate
  view.setUint16(32, numCh * 2, true);      // block align
  view.setUint16(34, 16, true);             // bits per sample
  w(36, 'data'); view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

// One-shot transform of a recorded clip (the pre-streaming path). Kept for
// the /voice/transform endpoint; nothing in the current UI calls it.
export async function sendVoiceForTransform(blob: Blob): Promise<void> {
  try {
    // Decode webm/opus → AudioBuffer → re-encode as WAV. If decodeAudioData
    // fails (rare), send the raw blob and let the server try.
    let toSend: Blob = blob;
    let filename = 'speech.webm';
    try {
      const arrayBuf = await blob.arrayBuffer();
      const AC       = audioContextCtor();
      const audioCtx = new AC({ sampleRate: 44100 });
      const audioBuf = await audioCtx.decodeAudioData(arrayBuf.slice(0));
      toSend   = audioBufferToWav(audioBuf);
      filename = 'speech.wav';
      void audioCtx.close();
    } catch (decodeErr) {
      console.warn('WebAudio decode failed, sending raw webm:', decodeErr);
    }

    const fd = new FormData();
    fd.append('audio', toSend, filename);
    fd.append('avatar_id', avatarState.selected ?? '');
    const res = await fetch(`${FACE_API}/voice/transform`, { method: 'POST', body: fd });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = ((await res.json()) as { detail: string }).detail; } catch { /* keep statusText */ }
      throw new Error(detail);
    }
    const outWav = await res.blob();
    const url    = URL.createObjectURL(outWav);
    const audio  = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    audio.play().catch((err) => console.warn('autoplay blocked:', err));
    const av = avatarState.all.find((a) => a.id === avatarState.selected);
    setStatus('voiceStatus', 'success',
      vsActive ? `Playing as ${av?.name || 'avatar'} — keep speaking…` : `Playing as ${av?.name || 'avatar'}`);
  } catch (e) {
    console.error('Voice transform error:', e);
    setStatus('voiceStatus', 'error', 'Transform failed: ' + (e as Error).message);
  }
}
