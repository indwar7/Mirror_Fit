// Injected by vite.config.ts from `git rev-parse --short HEAD`.
declare const __BUILD__: string;

interface Window {
  // Safari still exposes the prefixed constructor only.
  webkitAudioContext?: typeof AudioContext;
  // Scroll-audit debounce handle (build-stamp.ts).
  __sa?: number;
  mfScrollAudit?: () => void;
}
