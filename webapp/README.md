# Mirror Fit web app

The `demo/index.html` single-file demo, split into typed modules. Same markup,
same CSS, same behaviour; the JavaScript is now TypeScript under `src/`, one
module per feature, built with Vite.

```
npm install
npm run dev        # http://localhost:5173 — camera works on localhost without flags
npm run build      # tsc --noEmit, then a static build into dist/
npm run preview    # serve dist/ locally
```

## Backend host

The page talks to the GPU box on ports 7860 / 7862 / 8000 / 7863. The host is,
in order: `VITE_BACKEND_HOST` from a `.env` file, then the hostname the page was
served from, then a hard-coded fallback IP. For local development against the
box, create `.env`:

```
VITE_BACKEND_HOST=52.66.250.134
```

## Layout

| File | What |
| --- | --- |
| `index.html` | The demo's markup, unchanged except that the inline scripts are replaced by `src/main.ts`. Inline `onclick` handlers are kept. |
| `src/main.ts` | Publishes every handler the markup calls on `window`, then boots the features in the demo's original order. |
| `src/ui.ts` | Tabs, landing entry, status helpers. `onTabSwitch` / `onEnterFeature` hooks replace the demo's monkey-patching of `switchTab` / `enterFeature`. |
| `src/config.ts` | Backend URLs and the session id. |
| `src/camera.ts` | Mirror policy and the one camera-draw routine. |
| `src/liveswap.ts`, `src/liveportrait.ts`, `src/tryon.ts`, `src/my-avatar.ts`, `src/ai-twin.ts` | The server-backed features. |
| `src/ar-avatar.ts` | Browser-only face filter (MediaPipe + three.js). |
| `src/outfit-score.ts`, `src/fit-report.ts` | The fit models, ported from `demo/outfit-score.js` and `demo/fit-report.js`. |
| `src/fit-analyser.ts`, `src/returns-guard.ts`, `src/fit-readout.ts` | The three screens that share a fit reading. |
| `src/credits.ts`, `src/build-stamp.ts` | Demo credit meter; console build stamp and scroll audit. |
| `public/*.json` | The model files, served next to the page as before. |

Shared mutable state that the demo kept in globals lives in small import-free
modules (`avatar-state.ts`, `fit-session.ts`) so no two feature modules import
each other in a cycle.
