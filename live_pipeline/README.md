# Live try-on pipeline

Phase 1 of `BUILD_PLAN.md`: browser streams the 3D composite over WebRTC,
a GPU worker restyles each frame, the result streams back. Stock model,
no training yet — this phase proves the plumbing.

## Layout

| Path | What it is |
|---|---|
| `config.py` | Every tunable: resolution, fps, strength, session length |
| `worker/backend.py` | StreamDiffusionV2 on Wan 2.1 1.3B, with a passthrough fallback |
| `server/queue_manager.py` | One GPU, one session, FIFO queue with wait estimates |
| `server/track.py` | The video track that runs each frame through the model |
| `server/app.py` | WebRTC signaling (`/offer`), status, session lifecycle |
| `client/mirror.html` | The existing 3D guide layer (three.js + MediaPipe) |
| `client/webrtc.js` | Browser connection; captures the 3D canvas, not the raw camera |
| `client/session_ui.js` | Queue position, countdown, fallback messaging |
| `rig_jacket.py` | Produces `jacket_rigged.glb` from the FBX-derived GLB |

## Run

```bash
pip install -r requirements.txt
python -m server.app          # serves the client and signaling on :8080
```

Open `http://localhost:8080/mirror.html`.

Without CUDA the worker logs a warning and runs `PassthroughBackend` —
the 3D composite streams through untouched, which is also what a user
sees while waiting for a GPU. The whole pipeline is testable this way.

## Notes

- The client sends the **3D composite**, not the raw camera. The garment
  is already placed, so the model only has to make it look real — that
  is what keeps the Phase 4 fine-tune cheap.
- `strength` stays low (0.35) for the same reason. Higher values repaint
  the frame and lose the garment.
- One frame is in the model at a time. If the camera pushes faster than
  the GPU drains, frames are dropped rather than queued — queuing would
  only add latency to a live view.
- Phase 4's LoRA drops in via `LIVE_LORA_PATH`; nothing else changes.
