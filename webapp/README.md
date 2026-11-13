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

## Hosting

Static site. On Vercel or Cloudflare Pages set the project root to `webapp/`;
the build command is `npm run build` and the output directory is `dist/`.

Two things change when the page is served over HTTPS:

- Every backend URL follows the page's scheme, so the box is reached as
  `https://` and `wss://`. The backends speak plain HTTP, so put TLS in front
  of them (Caddy with a domain) before pointing a hosted build at the box.
  The dev server on `localhost` is plain http and talks to the box directly.
- Phones can open the camera. Browsers only allow `getUserMedia` on `https://`
  or `localhost`, which is why the old insecure-origin Chrome flag was needed.

`public/manifest.webmanifest` and the icons make the site installable to the
home screen. Regenerate the PNGs with `npm run icons` after editing `icon.svg`.

## Checks

```
npm run typecheck   # strict TypeScript, no emit
npm run smoke       # builds nothing; serves dist/ and drives it in headless Chromium
```

The smoke test opens a fake camera and mocks the try-on WebSocket, so the
whole Live Try-On loop (init payload, frame pacing, result drawing, garment
switch mid-session, fit readout, stop) is exercised without the GPU box. It
also loads the page at phone width and checks nothing overflows, and that
three.js and MediaPipe are not downloaded until a feature needs them.
`.github/workflows/webapp.yml` runs build + smoke on every push touching
`webapp/`.
