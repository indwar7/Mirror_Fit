// The Step 2.6 measurement inputs (chest, shoulder, height) are hidden on the
// Try-On sidebar but stay in the DOM: the outfit rating, the live fit readout
// and the Returns Guard handoff all read them from here.
import type { BodyInput } from './outfit-score';

export function osNum(id: string): number {
  const el = document.getElementById(id) as HTMLInputElement | null;
  const v = parseFloat(el ? el.value : '');
  return Number.isFinite(v) ? v : 0;
}

export function readBody(): BodyInput {
  return {
    chest_cm: osNum('osChest'),
    shoulder_cm: osNum('osShoulder'),
    height_cm: osNum('osHeight'),
  };
}

export function readSize(): string {
  const el = document.getElementById('osSize') as HTMLSelectElement | null;
  return el ? el.value : '';
}
