// The AR avatar presets and the sidebar grid. Kept apart from ar-avatar.ts so
// the grid can be painted at boot without pulling three.js and MediaPipe into
// the initial download; those load the first time the camera or Start is used.
import { byId, maybe } from './dom';

export interface ArPreset { id: string; name: string; emoji: string; skin: number; hair: number; hat: number; eye: number }

// Procedural Bitmoji-style 3D heads, built from primitives in code so there
// is no GLB download to fail. Sized to match a real face.
export const AR_AVATARS: ArPreset[] = [
  { id: 'av_1', name: 'Toby',  emoji: '🧒', skin: 0xf2c2a4, hair: 0x3b2417, hat: 0x7c3aed, eye: 0x2a1d12 },
  { id: 'av_2', name: 'Maya',  emoji: '👧', skin: 0xf5d0b5, hair: 0xc9a04c, hat: 0xef4444, eye: 0x1f3a5f },
  { id: 'av_3', name: 'Kai',   emoji: '🧑', skin: 0xc69774, hair: 0x141414, hat: 0x22c55e, eye: 0x2a1d12 },
  { id: 'av_4', name: 'Zara',  emoji: '👩', skin: 0xa57555, hair: 0x1c0e07, hat: 0xf59e0b, eye: 0x351f0f },
];

export const arSelection = { preset: AR_AVATARS[0] };

// ar-avatar.ts registers this once loaded, to hot-swap the head mid-session.
let onSelect: ((p: ArPreset) => void) | null = null;
export function setArSelectListener(fn: (p: ArPreset) => void): void { onSelect = fn; }

export function populateArGrid(): void {
  const grid = maybe('arAvatarGrid');
  if (!grid) return;
  grid.innerHTML = AR_AVATARS.map((a, i) => `
    <div class="av-card${i === 0 ? ' selected' : ''}" data-avid="${a.id}" onclick="_selectArAvatar('${a.id}')">
      <div style="font-size:36px;text-align:center;padding:14px 0">${a.emoji}</div>
      <div style="text-align:center;font-size:12px;font-weight:600">${a.name}</div>
    </div>
  `).join('');
}

export function selectArAvatar(id: string): void {
  const av = AR_AVATARS.find((a) => a.id === id);
  if (!av) return;
  arSelection.preset = av;
  document.querySelectorAll<HTMLElement>('#arAvatarGrid .av-card').forEach((c) =>
    c.classList.toggle('selected', c.dataset.avid === id));
  if (onSelect) onSelect(av);
  byId<HTMLButtonElement>('arStartBtn').disabled = false;
}
