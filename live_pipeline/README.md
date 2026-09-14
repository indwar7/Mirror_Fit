# Live try-on pipeline

`BUILD_PLAN.md` implemented, Phases 0-4. A browser streams the 3D
composite over WebRTC, a GPU worker restyles each frame with
StreamDiffusionV2 on Wan 2.1 1.3B, the result streams back.

## Layout

| Path | Phase | What it is |
|---|---|---|
| `setup_gpu_box.sh` | 0 | Provisions a Linux GPU box, ending in a real smoke test |
| `config.py` | — | Every tunable: resolution, fps, strength, session length |
| `worker/backend.py` | 1 | StreamDiffusionV2, with a passthrough fallback |
| `server/queue_manager.py` | 1 | One GPU, one session, FIFO with wait estimates |
| `server/track.py` | 1 | Runs each frame through the model |
| `server/app.py` | 1 | WebRTC signaling, status, session lifecycle |
| `client/mirror.html` | 1 | The 3D guide layer (three.js + MediaPipe) |
| `client/webrtc.js` | 1 | Captures the 3D canvas, not the raw camera |
| `client/session_ui.js` | 1 | Queue position, countdown, fallback messaging |
| `training/RECORDING.md` | 2 | Shoot checklist and movement script |
| `training/chroma.py` | 3 | Blue-screen keying, despill, garment erasure |
| `training/render_guide.py` | 3 | Renders the guide using the app's own rig |
| `training/prep.py` | 3 | Footage -> training pairs, split by person |
| `training/dataset.py` | 4 | Paired loader |
| `training/train.py` | 4 | LoRA fine-tune |
| `training/eval.py` | 4 | PSNR / SSIM / LPIPS on held-out people, plus previews |
| `product/db.py` | 5 | SQLite schema; the credit ledger is append-only |
| `product/accounts.py` | 5 | Signup, tokens, credits |
| `product/sessions.py` | 5 | Session records, captures, usage rollups |
| `product/api.py` | 5 | HTTP routes for accounts, history and admin |
| `product/admin_cli.py` | 5 | First admin, manual grants, usage report |
| `client/auth.js` | 5 | Sign-in strip and credit count |
| `client/admin.html` | 6 | Usage dashboard and the add-a-card verdict |
| `product/monitor.py` | 6 | Queue and GPU report, peak wait vs budget |
| `rig_jacket.py` | — | Produces `jacket_rigged.glb` from the FBX-derived GLB |

## Run the live pipeline

```bash
pip install -r requirements.txt
python -m server.app          # client and signaling on :8080
```

Open `http://localhost:8080/mirror.html` and press **Go live**.

Without CUDA the worker logs a warning and runs `PassthroughBackend` —
the composite streams through untouched, which is also what a waiting
user sees. The whole pipeline is testable this way.

## Train a garment

```bash
# 3. footage -> pairs (see training/RECORDING.md first)
python -m training.prep --footage data/footage --out data/pairs \
    --holdout priya,arjun --garment-hue 0,12

# 4. fine-tune, then score against the stock model
python -m training.train --pairs data/pairs --out runs/jacket-v1
python -m training.eval  --pairs data/pairs --out runs/baseline
python -m training.eval  --pairs data/pairs --out runs/jacket-v1-eval \
    --lora runs/jacket-v1/final

LIVE_LORA_PATH=runs/jacket-v1/final python -m server.app
```

## Why it is built this way

- The client sends the **3D composite**, not the raw camera. The garment
  is already placed, so the model only has to make it look real. That is
  what keeps `strength` at 0.35 and the fine-tune cheap.
- `render_guide.py` drives the **actual** `mirror.html` in headless
  Chromium rather than reimplementing the rig in Python. A second
  implementation would drift from production within a week, and the LoRA
  would learn to correct a composite that never occurs live.
- Training samples timesteps only up to `STRENGTH`. Inference never
  enters the high-noise regime, so training there spends budget on a
  case that cannot happen.
- One frame is in the model at a time; overflow is **dropped**, not
  queued. Queuing only adds latency to a live view.
- Hold-out is by person. Frames from one person are near-copies, so a
  frame-level split scores memorisation.

## Accounts and credits

A live session costs one credit, charged when the GPU is actually
granted — not when the user joins the queue, and refunded if the session
delivers no frames. Everything else on the page works signed out.

```bash
python -m product.admin_cli make-admin --email you@example.com
python -m product.admin_cli grant --email a@b.com --amount 20
python -m product.admin_cli usage --hours 24
```

The admin dashboard is at `/admin.html`. It shows sessions, GPU
utilisation, mean and peak wait, and says plainly whether the box needs
a second card — the plan's threshold is a few seconds of peak wait.

**No payment provider is wired in.** The ledger and its idempotency are
built for one (`grant(..., ref=...)` is safe to replay, which is what a
payment webhook needs), but credits are added by an admin for now.

## Status

Phases 0-6 are written, minus payments.

Verified on this machine: queue behaviour including early leave; credit
ledger under 6 concurrent spends; refund idempotency; billing charged at
GPU grant and not at queue join; API auth, validation and admin gating;
rejection of offers carrying no video track; chroma keying and garment
erasure; torso crop; evaluation metrics; dataset pairing.

**Not yet run against a GPU or a real camera.** The model call in
`worker/backend.py`, the training step, and the headless renderer need a
box with CUDA and a first real footage clip. WebRTC has not been carried
end to end through a browser.
