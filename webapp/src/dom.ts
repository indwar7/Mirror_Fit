// The demo reaches for elements by id everywhere. `byId` is for the places
// the original dereferenced the result unconditionally (a missing element
// threw there too, just less clearly); `maybe` is for the places it checked.
export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} is not in the document`);
  return el as T;
}

export function maybe<T extends HTMLElement = HTMLElement>(id: string): T | null {
  return document.getElementById(id) as T | null;
}

export function esc(s: unknown): string {
  return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' } as Record<string, string>)[c]);
}
