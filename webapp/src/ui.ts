import { byId, maybe } from './dom';
import { avCheckReady } from './my-avatar';

export type TabId =
  | 'liveswap' | 'liveportrait' | 'tryon' | 'ar' | 'twin' | 'fit' | 'myavatar';

// switchTab() indexes .tab positionally against this list, so the order of
// the buttons in index.html is load-bearing: removing or reordering one
// silently points every later tab at the wrong panel.
const TAB_IDS: TabId[] = ['liveswap', 'liveportrait', 'tryon', 'ar', 'twin', 'fit', 'myavatar'];

// In the single-file demo two later scripts *wrapped* window.switchTab and
// window.enterFeature to bolt on behaviour (the fit readout stops when you
// leave Try-On; the credit meter charges a launch). Modules cannot reassign
// each other's exports, so the same thing is done with explicit hooks.
type TabListener = (t: TabId) => void;
const tabListeners: TabListener[] = [];
export function onTabSwitch(fn: TabListener): void { tabListeners.push(fn); }

// A guard returning false blocks the launch.
type EnterGuard = (name: TabId) => boolean | void;
const enterGuards: EnterGuard[] = [];
export function onEnterFeature(fn: EnterGuard): void { enterGuards.push(fn); }

export function switchTab(t: TabId): void {
  for (const fn of tabListeners) fn(t);
  const tabs = document.querySelectorAll('.tab');
  TAB_IDS.forEach((id, i) => {
    tabs[i].classList.toggle('active', id === t);
    byId('panel-' + id).classList.toggle('active', id === t);
  });
  byId('myAvatarWrap').style.display = t === 'myavatar' ? 'block' : 'none';
  if (t === 'myavatar') void avCheckReady();
  byId('liveWrap').style.display    = t === 'liveswap'     ? 'grid' : 'none';
  byId('lpWrap').style.display      = t === 'liveportrait' ? 'grid' : 'none';
  byId('tryonWrap').style.display   = t === 'tryon'        ? 'grid' : 'none';
  byId('arWrap').style.display      = t === 'ar'           ? 'grid' : 'none';
  byId('twinWrap').style.display    = t === 'twin'         ? 'grid' : 'none';
  byId('fitWrap').style.display     = t === 'fit'          ? 'block' : 'none';
  byId('placeholder').style.display = 'none';
}

// Landing → feature entry. Hides the intro page, reveals the app shell and
// switches to the requested tab.
export function enterFeature(name: TabId): void {
  for (const g of enterGuards) if (g(name) === false) return;
  // AR Avatar and AI Twin are hidden; nothing was deleted, their panels are
  // untouched.
  if (name === 'ar') return;
  if (name === 'twin') return;
  document.body.classList.remove('landing-active');
  const landing = maybe('landing');
  if (landing) landing.classList.add('hide');
  switchTab(name);
}

// Logo click takes the user back to the landing page.
export function goLanding(): void {
  const landing = maybe('landing');
  if (landing) landing.classList.remove('hide');
  document.body.classList.add('landing-active');
}

// Preview toggle: each feature has a checkbox above its result screen.
// Unchecked (default) = only the result is shown. Checked = the user's camera
// pane appears next to it in a two-column grid.
export function setPreview(feature: 'live' | 'lp' | 'tryon', on: boolean): void {
  const map = {
    live:  { grid: 'liveGrid',  pane: 'liveCameraPane'  },
    lp:    { grid: 'lpGrid',    pane: 'lpCameraPane'    },
    tryon: { grid: 'tryonGrid', pane: 'tryonCameraPane' },
  } as const;
  const ids = map[feature];
  if (!ids) return;
  const grid = maybe(ids.grid);
  const pane = maybe(ids.pane);
  if (!grid || !pane) return;
  grid.style.gridTemplateColumns = on ? '1fr 1fr' : '1fr';
  pane.style.display             = on ? '' : 'none';
}

export function previewFile(inp: HTMLInputElement, imgId: string, phId: string): void {
  const f = inp.files && inp.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = (e) => {
    const img = byId<HTMLImageElement>(imgId);
    img.src = e.target!.result as string; img.style.display = 'block';
    const ph = maybe(phId);
    if (ph) ph.style.display = 'none';
    const ph2 = maybe(phId + '2');
    if (ph2) ph2.style.display = 'none';
  };
  r.readAsDataURL(f);
}

export type StatusKind = 'loading' | 'info' | 'success' | 'error' | 'busy';

export function setStatus(id: string, type: StatusKind, msg: string): void {
  const el = maybe(id);
  if (!el) return;
  el.className = 'status-msg ' + (type === 'loading' ? 'info' : type);
  el.textContent = msg;
}

export function setDot(prefix: string, on: boolean): void {
  ['', '2'].forEach((sfx) => {
    const d = maybe(prefix + 'Dot' + sfx);
    if (d) d.className = 'dot' + (on ? ' on' : '');
  });
  const lbl = maybe(prefix + 'StatusText');
  if (lbl) lbl.textContent = on ? 'Connected' : 'Disconnected';
}
