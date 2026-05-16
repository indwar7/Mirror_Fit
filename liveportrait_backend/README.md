# LUCY LivePortrait Backend

Driving-frame expression transfer for live face swap. Source image
(avatar) + driving video (user's webcam frame) → animated source frame
matching the user's expression. Designed for blinks, smiles, brow raises
to carry through — fixes the "static swap face" problem that
InstantID/PuLID could not solve.

Runs as a separate FastAPI server on port `7862` in a Python 3.11
conda env (PyTorch CUDA wheels exist for 3.11 but not 3.14).

## What it does

- Per WebSocket session, takes one avatar image (preset or uploaded).
- Crops + extracts appearance feature, source keypoints, head rotation.
- Per webcam frame, extracts the driving frame's motion keypoints +
  rotation and applies relative motion to the source — producing an
  animated source frame that mirrors your live expression and head pose.
- ~22-28 fps on g5.xlarge (A10G), 90-130 ms glass-to-glass with a
  local browser.

## One-time setup on the server

```powershell
cd C:\virtual-try-on\liveportrait_backend
powershell .\setup.ps1
```

This will:
1. Create conda env `liveportrait` with Python 3.11.
2. Install PyTorch 2.3 (CUDA 12.1), insightface, onnxruntime-gpu, etc.
3. Clone the official KwaiVGI/LivePortrait repo as a sibling folder.
4. Download model weights into `LivePortrait/pretrained_weights/`
   (~2.2 GB: appearance extractor, motion extractor, warping module,
   SPADE generator, stitching/retargeting, landmark.onnx, plus
   InsightFace buffalo_l).

Total ~10-15 min on AWS Mumbai.

## Running

```powershell
cd C:\virtual-try-on\liveportrait_backend
powershell .\start.ps1
```

Cold load ~15-30 s. After `[LP] engine ready for traffic`, the demo
client (with `SWAP_WS` pointing to port 7862) will use it.

## AWS Security Group

Open inbound TCP **7862** on the EC2 instance. Port range 7860-7900
was already opened for the existing backends — 7862 should already be
covered.

## First-frame calibration

LivePortrait uses *relative-motion* driving — the first driving frame
captured for each session becomes the "neutral" baseline. For best
results, face the camera with a neutral expression for ~200 ms when
the swap starts. Subsequent expressions are computed as deltas from
that baseline. The pipeline automatically captures this on first frame.

## Protocol

The WebSocket protocol on `/ws/liveportrait-swap` is identical to
`/ws/instantid-swap` and `/ws/live-swap`, so the demo client can
switch backends by changing one constant.

| Direction | Type      | Payload                                    |
| --------- | --------- | ------------------------------------------ |
| C → S     | `init`    | `{avatar_id, session_id?}` or `{source_image: <b64>}` |
| C → S     | `frame`   | `{image: <base64 JPEG>}`                   |
| C → S     | `ping`    | `{t?}` — keepalive                         |
| S → C     | `ready`   | `{session_id}`                             |
| S → C     | `result`  | `{image: <base64 JPEG>}`                   |
| S → C     | `no_face` | no face detected in this frame             |
| S → C     | `dropped` | server still processing previous frame     |
| S → C     | `error`   | `{message}`                                |

## Latency budget on A10G

| Stage                              | ms  |
| ---------------------------------- | --- |
| Network India ↔ Mumbai RTT         | 80  |
| Frame decode (b64 → JPEG → ndarray)| 5   |
| Cropper detection (every 10th frame, ~3 ms amortized) | 3 |
| Motion extractor + warping + SPADE | 30-45 |
| Stitching + paste-back             | 8   |
| JPEG encode + base64               | 8   |
| **Total per frame (steady state)** | **~135-150 ms** |

Compare with InstantID's ~1700 ms per frame. ~10x faster, AND it
actually transfers driving-frame expressions (InstantID does not).

## Co-existence with other backends

VRAM on A10G (24 GB):
- face_swap_backend (inswapper + Wav2Lip + BiSeNet): ~4 GB
- liveportrait_backend: ~5 GB
- Together: ~9 GB → fits comfortably.

You can run face_swap_backend without `LUCY_MINIMAL_MODE` (keep voice
+ Wav2Lip lipsync alive) AND liveportrait_backend on the same GPU.

## Known limitations

- Fixed 256x256 model — micro-detail in eyes may look slightly soft
  after paste-back to 720p webcam frame.
- 1-pixel jitter on lip corners in per-frame mode (no temporal Kalman).
- Large source yaw (>30°) produces stretching artifacts. Pick a
  near-frontal source avatar.
- Eye-blink amplitude can over-estimate on low-light webcams ("shocked"
  look). Mitigated by `flag_normalize_lip=True` and reasonable lighting.

## Troubleshooting

- **`No module named 'src'`** — `git clone` of LivePortrait didn't run
  or path is wrong. Check `liveportrait_backend/LivePortrait/src/`
  exists.
- **`antelopev2 / buffalo_l not found`** — re-run setup.ps1; HF download
  may have been rate-limited.
- **`cuDNN missing / DLL load failed`** — pip install nvidia-cudnn-cu12
  did not complete or PATH is wrong. Restart PowerShell as admin and
  reinstall.
- **First swap is gibberish** — that frame was the neutral baseline.
  Send 2-3 more frames and the output stabilizes.
