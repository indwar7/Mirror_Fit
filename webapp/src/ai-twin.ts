// AI Twin (SadTalker) — upload image + audio → talking head mp4 from
// /generate-talking-avatar on port 7863. Offline generation, ~30-80 s on A10G
// for a 5-10 s clip. No live tracking, no WebSocket — one multipart POST.
import { TWIN_API } from './config';
import { byId } from './dom';
import { setStatus } from './ui';

let twinImgFile: File | null = null;
let twinAudioFile: File | null = null;
let twinRecorder: MediaRecorder | null = null;
let twinRecChunks: Blob[] = [];
let twinSpinnerTimer: number | null = null;
let twinT0 = 0;

function twinReady(): boolean {
  const ok = !!twinImgFile && !!twinAudioFile;
  const btn = document.getElementById('twinGenBtn') as HTMLButtonElement | null;
  if (btn) btn.disabled = !ok;
  return ok;
}

export function twinPreviewImg(inp: HTMLInputElement): void {
  const f = inp.files && inp.files[0];
  if (!f) return;
  twinImgFile = f;
  const r = new FileReader();
  r.onload = (e) => {
    const img = byId<HTMLImageElement>('twinImgPreview');
    img.src = e.target!.result as string;
    img.style.display = 'block';
    byId('twinImgPh').style.display = 'none';
  };
  r.readAsDataURL(f);
  twinReady();
}

export function twinPreviewAudio(inp: HTMLInputElement): void {
  const f = inp.files && inp.files[0];
  if (!f) return;
  twinAudioFile = f;
  const a = byId<HTMLAudioElement>('twinAudioPreview');
  a.src = URL.createObjectURL(f);
  a.style.display = 'block';
  byId('twinAudPh').style.display = 'none';
  twinReady();
}

export async function twinToggleRecord(): Promise<void> {
  const btn = byId('twinRecBtn');
  if (twinRecorder && twinRecorder.state === 'recording') {
    twinRecorder.stop();
    btn.textContent = 'Or record from mic';
    btn.classList.remove('btn-stop');
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    twinRecChunks = [];
    twinRecorder = new MediaRecorder(stream);
    twinRecorder.ondataavailable = (e) => { if (e.data.size > 0) twinRecChunks.push(e.data); };
    twinRecorder.onstop = () => {
      const blob = new Blob(twinRecChunks, { type: 'audio/webm' });
      twinAudioFile = new File([blob], 'recorded.webm', { type: 'audio/webm' });
      const a = byId<HTMLAudioElement>('twinAudioPreview');
      a.src = URL.createObjectURL(blob);
      a.style.display = 'block';
      byId('twinAudPh').style.display = 'none';
      stream.getTracks().forEach((t) => t.stop());
      twinReady();
    };
    twinRecorder.start();
    btn.textContent = 'Stop recording';
    btn.classList.add('btn-stop');
  } catch (e) {
    setStatus('twinStatus', 'error', 'Mic access denied: ' + (e as Error).message);
  }
}

export async function twinGenerate(): Promise<void> {
  if (!twinReady()) return;
  const size = document.querySelector<HTMLInputElement>('input[name="twinSize"]:checked')!.value;
  const useEnh = byId<HTMLInputElement>('twinEnhancer').checked;

  byId<HTMLButtonElement>('twinGenBtn').disabled = true;
  byId('twinResultVideo').style.display = 'none';
  byId('twinPlaceholder').style.display = 'none';
  byId('twinSpinner').style.display = 'flex';
  byId('twinDot').classList.add('on');
  byId('twinStatusText').textContent = 'Generating…';
  setStatus('twinStatus', 'info', `SadTalker running on the GPU — ~30-80 s for a 5-10 s clip…`);

  twinT0 = Date.now();
  twinSpinnerTimer = window.setInterval(() => {
    const secs = Math.round((Date.now() - twinT0) / 1000);
    byId('twinSpinnerSub').textContent = `${secs}s elapsed`;
  }, 500);

  const fd = new FormData();
  fd.append('image', twinImgFile!, twinImgFile!.name || 'source.png');
  fd.append('audio', twinAudioFile!, twinAudioFile!.name || 'driven.wav');

  try {
    // Quality knobs go on as query string.
    const qs = new URLSearchParams({ size, enhancer: useEnh ? 'gfpgan' : '' });
    const resp = await fetch(`${TWIN_API}/generate-talking-avatar?` + qs, { method: 'POST', body: fd });
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`server ${resp.status}: ${txt.slice(0, 200)}`);
    }
    const elapsed = resp.headers.get('X-Inference-Seconds') || '?';
    const blob = await resp.blob();
    const url  = URL.createObjectURL(blob);
    const video = byId<HTMLVideoElement>('twinResultVideo');
    video.src = url;
    video.style.display = 'block';
    video.play().catch(() => { /* autoplay blocked */ });
    setStatus('twinStatus', 'success', `Done in ${elapsed} s — video ready`);
    byId('twinStatusText').textContent = 'Idle';
  } catch (e) {
    setStatus('twinStatus', 'error', 'Generation failed: ' + (e as Error).message);
    byId('twinPlaceholder').style.display = 'block';
    byId('twinStatusText').textContent = 'Error';
  } finally {
    if (twinSpinnerTimer) clearInterval(twinSpinnerTimer);
    byId('twinSpinner').style.display = 'none';
    byId('twinDot').classList.remove('on');
    byId<HTMLButtonElement>('twinGenBtn').disabled = false;
  }
}
