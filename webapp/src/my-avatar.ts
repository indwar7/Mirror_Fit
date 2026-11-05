// My Avatar: upload a photo, pick male/female, press Create. Measurements are
// an optional refinement: they choose a closer one of the eight curated base
// bodies. Both endpoints are on FACE_API (:7860).
import { FACE_API } from './config';
import { byId } from './dom';

let avPhotoFile: File | null = null;
let avGender: 'male' | 'female' = 'male';
let avId: string | null = null;
let avCategory: 'upper' | 'lower' | 'dress' = 'upper';

interface BaseBodiesStatus { ready: number; total: number; bodies?: { id: string; state: string }[] }
interface BodyResponse {
  image_url: string;
  selection?: { from_defaults?: boolean; gender?: string; build?: string };
  base_body?: { id?: string; license?: string };
  skin_tone_match?: string;
}

export function avSetPhoto(input: HTMLInputElement): void {
  avPhotoFile = (input.files && input.files[0]) || null;
  const img = byId<HTMLImageElement>('avPhotoPreview');
  if (avPhotoFile) {
    img.src = URL.createObjectURL(avPhotoFile);
    img.style.display = 'block';
    byId('avPhotoHint').style.display = 'none';
  }
  byId<HTMLButtonElement>('avCreateBtn').disabled = !avPhotoFile;
}

export function avSetGender(g: 'male' | 'female', btn: HTMLElement): void {
  avGender = g;
  document.querySelectorAll('.av-gender-btn').forEach((b) => b.classList.toggle('active', b === btn));
}

export function avSetCategory(c: 'upper' | 'lower' | 'dress', btn: HTMLElement): void {
  avCategory = c;
  document.querySelectorAll('.av-cat-btn').forEach((b) => b.classList.toggle('active', b === btn));
}

function avStatus(el: string, kind: 'busy' | 'error' | 'success', msg: string): void {
  const colour = kind === 'error' ? 'var(--red)'
               : kind === 'success' ? 'var(--green)' : 'var(--accent-hot)';
  byId(el).innerHTML = '<span style="color:' + colour + '">' + msg + '</span>';
}

// The backend puts the real reason in `detail`.
async function avError(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { detail?: string; error?: string };
    return j.detail || j.error || ('HTTP ' + res.status);
  } catch { return 'HTTP ' + res.status; }
}

// Tracks the object URL currently on screen so it can be released when the
// next one replaces it.
let avObjectUrl: string | null = null;

function avShow(url: string): void {
  const img = byId<HTMLImageElement>('avBodyImg');

  // Cache-bust ONLY http(s) URLs. The body composite is written back over the
  // avatar's own path, so without a changing query the browser serves the
  // previous body. A blob: URL is already unique per call, and appending a
  // query to one makes it unresolvable.
  const isObjectUrl = url.startsWith('blob:') || url.startsWith('data:');
  img.src = isObjectUrl
    ? url
    : url + (url.includes('?') ? '&' : '?') + 'v=' + Date.now();

  if (avObjectUrl && avObjectUrl !== url) {
    URL.revokeObjectURL(avObjectUrl);
  }
  avObjectUrl = isObjectUrl ? url : null;

  img.style.display = 'block';
  byId('avBodyPlaceholder').style.display = 'none';
}

// Base bodies are curated photographs and nothing generates one, so an empty
// asset set is a hard blocker. Say so on arrival, not after Create.
export async function avCheckReady(): Promise<void> {
  const box = byId('avBlocker');
  const text = byId('avBlockerText');

  // "unreachable" and "reachable but answered wrongly" are different faults
  // with different fixes.
  let res: Response;
  try {
    res = await fetch(FACE_API + '/base-bodies');
  } catch {
    box.style.display = 'block';
    text.innerHTML = '<b>Cannot reach the avatar backend on port 7860.</b><br>' +
      'Start it: <code>python -m uvicorn main:app --host 0.0.0.0 --port 7860</code> ' +
      'in <code>face_swap_backend</code>.';
    return;
  }

  if (res.status === 404) {
    box.style.display = 'block';
    text.innerHTML = '<b>The backend on 7860 is running older code.</b><br>' +
      '<code>/base-bodies</code> returned 404, so this process started before ' +
      'that endpoint existed. Pulling updates the page but not a running ' +
      'Python process — restart it.';
    return;
  }
  if (!res.ok) {
    box.style.display = 'block';
    text.textContent = 'The avatar backend answered ' + res.status + ' for /base-bodies.';
    return;
  }

  let st: BaseBodiesStatus;
  try {
    st = (await res.json()) as BaseBodiesStatus;
  } catch {
    box.style.display = 'block';
    text.textContent = '/base-bodies did not return JSON.';
    return;
  }

  if (st.ready === st.total) { box.style.display = 'none'; return; }

  const missing = (st.bodies || []).filter((b) => b.state !== 'ok').map((b) => b.id);
  box.style.display = 'block';
  text.innerHTML =
    '<b>' + st.ready + ' of ' + st.total + ' base bodies available.</b><br>' +
    'Base bodies are curated photographs — nothing generates one, so a build ' +
    'with no photograph cannot be substituted for. Create will fail with 501 ' +
    'until they are added, and Step 3 (garments) only appears once a body ' +
    'exists.<br><br>Missing: ' + missing.join(', ') +
    '<br><br>Add them with <code>tools/curate_base_bodies.py add</code>.';
}

export async function avCreate(): Promise<void> {
  if (!avPhotoFile) return;
  const btn = byId<HTMLButtonElement>('avCreateBtn');
  btn.disabled = true;
  try {
    avStatus('avCreateStatus', 'busy', 'Checking the photo…');
    const fd = new FormData();
    fd.append('photo', avPhotoFile);
    fd.append('name', 'You');
    fd.append('gender', avGender);

    let res: Response;
    try {
      res = await fetch(FACE_API + '/avatars/create', { method: 'POST', body: fd });
    } catch {
      avStatus('avCreateStatus', 'error', 'Could not reach the avatar backend on port 7860.');
      return;
    }
    if (!res.ok) { avStatus('avCreateStatus', 'error', await avError(res)); return; }
    avId = ((await res.json()) as { id: string }).id;

    avStatus('avCreateStatus', 'busy', 'Putting you on a body…');
    const bodyFd = new FormData();
    bodyFd.append('gender', avGender);
    const chest = byId<HTMLInputElement>('avChest').value;
    const waist = byId<HTMLInputElement>('avWaist').value;
    if (chest) bodyFd.append('chest_cm', chest);
    if (waist) bodyFd.append('waist_cm', waist);

    let bodyRes: Response;
    try {
      bodyRes = await fetch(FACE_API + '/avatars/' + avId + '/body', { method: 'POST', body: bodyFd });
    } catch {
      avStatus('avCreateStatus', 'error', 'Could not reach the avatar backend on port 7860.');
      return;
    }
    if (!bodyRes.ok) {
      // 501 means the bin has no approved photograph yet.
      const detail = await avError(bodyRes);
      avStatus('avCreateStatus', 'error',
        bodyRes.status === 501 ? 'No base body for this build yet — ' + detail : detail);
      return;
    }

    const data = (await bodyRes.json()) as BodyResponse;
    avShow(FACE_API + data.image_url);
    const sel = data.selection || {};
    byId('avSelection').textContent =
      (sel.from_defaults ? 'Standard ' : '') + sel.gender + ' · ' + sel.build +
      ' build. Base photo ' + (data.base_body || {}).id +
      ' (' + ((data.base_body || {}).license || 'licence unrecorded') + '). ' +
      'Skin tone: ' + data.skin_tone_match + '.';
    byId('avTryonCard').style.display = 'block';
    avStatus('avCreateStatus', 'success', 'Done.');
  } finally {
    btn.disabled = !avPhotoFile;
  }
}

export async function avTryon(input: HTMLInputElement): Promise<void> {
  const file = input.files && input.files[0];
  if (!file || !avId) return;
  avStatus('avTryonStatus', 'busy', 'Fitting the garment…');

  const fd = new FormData();
  fd.append('garment_image', file);
  fd.append('category', avCategory);

  let res: Response;
  try {
    res = await fetch(FACE_API + '/avatars/' + avId + '/tryon', { method: 'POST', body: fd });
  } catch {
    avStatus('avTryonStatus', 'error', 'Could not reach the avatar backend on port 7860.');
    return;
  }
  if (!res.ok) { avStatus('avTryonStatus', 'error', await avError(res)); return; }

  // The endpoint returns the image itself, not JSON.
  const blob = await res.blob();
  if (!blob.size) {
    avStatus('avTryonStatus', 'error', 'The server returned an empty image.');
    return;
  }
  const img = byId<HTMLImageElement>('avBodyImg');
  // Report a load failure rather than leaving a blank pane under a success message.
  img.onerror = () => avStatus('avTryonStatus', 'error', 'The fitted image could not be displayed.');
  img.onload = () => avStatus('avTryonStatus', 'success', 'Fitted — your face is untouched.');
  avShow(URL.createObjectURL(blob));
}
