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
- Crops + extracts appearance feature, source keypoints, head rotation,
  and the paste-back mask for that source.
- Per webcam frame, face-aligns the driving frame, extracts its motion
  keypoints, and applies the expression delta (optionally head rotation
  too) to the source — then pastes the animated face back into the full
  source portrait.
- The first ~10 frames are spent averaging the driver's neutral face;
  the client is told (`calibrating`) so it can say "hold still" instead
  of looking frozen.

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
5. Verify every weight the engine loads is actually on disk, then run
   the offline logic tests.

Total ~10-15 min on AWS Mumbai. Safe to re-run — completed steps are
skipped.

Every step runs through `conda run -n liveportrait`, never
`conda activate`: activation is a silent no-op in a non-interactive
shell, and when it fails the 2.2 GB of CUDA wheels land in `base` while
the env stays empty.

## Running

```powershell
cd C:\virtual-try-on\liveportrait_backend
powershell .\start.ps1
```

Cold load ~15-30 s, including a synthetic warmup pass so the first real
frame isn't the one that pays for cuDNN autotuning. After
`[LP] engine ready for traffic`, the demo client's Live Portrait tab
will use it.

Pushing to `main` also restarts it: `.github/workflows/deploy.yml` stops
all Python processes and relaunches try-on, face-swap, **and** this
backend. If the conda env is missing on the box, the deploy logs
`SKIP LivePortrait` rather than failing the run.

### Environment variables

| Var               | Default | Meaning                                       |
| ----------------- | ------- | --------------------------------------------- |
| `LP_PORT`         | 7862    | Listen port                                    |
| `LP_JPEG_QUALITY` | 85      | Result JPEG quality                            |
| `LP_VERBOSE`      | off     | Log every frame instead of a summary per 100   |

### Health and introspection

- `GET /health` — engine state, per-session frame counts, rolling
  inference ms, whether each session finished calibrating, and its
  current tuning params. Also reports *why* the engine failed to load
  rather than just refusing connections.
- `GET /avatars` — the preset ids `init` will accept.

## AWS Security Group

Open inbound TCP **7862** on the EC2 instance. Port range 7860-7900
was already opened for the existing backends — 7862 should already be
covered.

## First-frame calibration

LivePortrait uses *relative-motion* driving, so the driver's neutral
face is the reference every expression is measured against. The first
10 frames are averaged into that baseline — averaged, because a single
captured frame that happened to catch a blink or a half-smile
miscalibrates every subsequent frame for the whole session. During
collection the server returns the untouched portrait and reports
`calibrating`.

Send `{"type": "recalibrate"}` to re-measure at any time (the demo
exposes this as **Recalibrate neutral face**) — useful after the user
changes seat or the lighting shifts.

## Protocol

WebSocket `/ws/liveportrait-swap`. The JSON protocol is identical to
`/ws/instantid-swap` and `/ws/live-swap`, so the demo client can switch
backends by changing one constant.

| Direction | Type            | Payload                                              |
| --------- | --------------- | ---------------------------------------------------- |
| C → S     | `init`          | `{avatar_id, session_id?}` or `{source_image: <b64>}`, plus optional `exp_amp` / `pose_gain` / `smooth` |
| C → S     | `frame`         | `{image: <base64 JPEG>}`                             |
| C → S     | *binary*        | raw JPEG bytes — same as `frame`, a third fewer bytes |
| C → S     | `config`        | `{exp_amp?, pose_gain?, smooth?}` — live retune       |
| C → S     | `recalibrate`   | re-measure the neutral baseline                       |
| C → S     | `ping`          | `{t?}` — keepalive                                    |
| S → C     | `ready`         | `{session_id, params}`                                |
| S → C     | `result`        | `{image: <base64 JPEG>, ms}`                          |
| S → C     | *binary*        | raw JPEG bytes — the reply to a binary frame          |
| S → C     | `calibrating`   | `{image}` — still measuring the neutral face          |
| S → C     | `no_face`       | no face detected in this frame                        |
| S → C     | `dropped`       | server still processing the previous frame            |
| S → C     | `config`        | `{params}` — acknowledges a retune                    |
| S → C     | `error`         | `{message}`                                           |

A binary frame gets a binary reply; status messages are always JSON, so
a client using the binary transport still has to handle text frames.
Exactly one frame should be in flight at a time — send the next one when
a reply arrives. (Sending on a timer *and* on every reply turns each
`dropped` into another doomed frame, which makes latency worse the more
the server is struggling.)

## Tuning knobs

Settable at `init` and live via `config`:

| Param       | Default | Effect                                                        |
| ----------- | ------- | ------------------------------------------------------------- |
| `exp_amp`   | 1.1     | Expression amplification. Above ~1.35 geometry starts to distort. |
| `pose_gain` | 0.0     | Share of the driver's head rotation applied to the portrait. 0 locks the head to the source pose; 0.6-0.8 feels like a mirror on a front-facing source. |
| `smooth`    | 0.7     | EMA weight on the newest expression delta (~110 ms tau). Lower is smoother but laggier. |

Head translation and scale stay locked to the source at any `pose_gain`:
carrying them through slides the head around the frame and tears the
paste-back seam.

## Driving-frame alignment

The motion extractor only produces meaningful coefficients when each
driving frame is cropped the way the source was — face-aligned, same
scale, same vertical offset. So, exactly as upstream does for video:
detect the face, then track landmarks frame-to-frame with
`landmark.onnx` (~2 ms) and crop through `crop_image`. Detection re-runs
every 15 frames to correct drift, immediately whenever the tracker is
lost, and whenever the tracked face jumps implausibly far in one frame
(a tracker that lost the face will happily lock onto a wall).

## Latency budget on A10G

Estimates, not a fresh measurement — the server reports its own numbers,
so read the real ones off `GET /health` or the per-100-frame log line.

| Stage                                   | ms   |
| --------------------------------------- | ---- |
| Network India ↔ Mumbai RTT              | 80   |
| Frame decode (JPEG → ndarray)           | 5    |
| Landmark tracking (every frame)         | 2-3  |
| Face detection (every 15th, amortized)  | 1-2  |
| Motion extractor + warping + SPADE      | 30-45 |
| Stitching + paste-back                  | 8    |
| JPEG encode                             | 6    |
| **Total per frame (steady state)**      | **~135-150** |

Compare with InstantID's ~1700 ms per frame. ~10x faster, AND it
actually transfers driving-frame expressions (InstantID does not).

Source portraits are downscaled to 720 px on the long edge at `init`.
Paste-back and JPEG encode both scale with that number, and anything
larger is invisible in a webcam-sized canvas.

## Tests

```powershell
conda run -n liveportrait python test_engine_logic.py
```

Runs without a GPU, weights, or network — the upstream repo is replaced
by stand-ins that record how they were called. Covers crop scheduling,
baseline averaging, pose gain, paste-back wiring, no-face and
tracker-loss recovery, session lifecycle, and the server's input
validation. `setup.ps1` runs it as its last step.

## Co-existence with other backends

VRAM on A10G (24 GB):
- face_swap_backend (inswapper + Wav2Lip + BiSeNet): ~4 GB
- liveportrait_backend: ~5 GB
- Together: ~9 GB → fits comfortably.

You can run face_swap_backend without `LUCY_MINIMAL_MODE` (keep voice
+ Wav2Lip lipsync alive) AND liveportrait_backend on the same GPU.

Abandoned sessions (closed laptop, dropped LTE) are swept after 5
minutes idle so their cached appearance features don't camp in VRAM
until the next restart.

## Known limitations

- Fixed 256x256 model — micro-detail in eyes may look slightly soft
  after paste-back to a 720p portrait.
- Large source yaw (>30°) produces stretching artifacts, and more so
  with `pose_gain` raised. Pick a near-frontal source avatar.
- Eye-blink amplitude can over-estimate on low-light webcams ("shocked"
  look). Lower `exp_amp` or improve the lighting.
- One frame at a time per GPU: concurrent sessions serialize behind one
  lock, so N viewers get roughly 1/N of the frame rate.

## Troubleshooting

- **`No module named 'src'`** — `git clone` of LivePortrait didn't run
  or path is wrong. Check `liveportrait_backend/LivePortrait/src/`
  exists.
- **`antelopev2 / buffalo_l not found`** — re-run setup.ps1; HF download
  may have been rate-limited. setup.ps1 now names the exact missing
  files instead of failing on one probe.
- **`cuDNN missing / DLL load failed`** — pip install nvidia-cudnn-cu12
  did not complete or PATH is wrong. Restart PowerShell as admin and
  reinstall.
- **Portrait animates weakly, or "breathes" on a fixed cycle** — that
  was the old crop path feeding misaligned frames to the motion
  extractor. Fixed; if it reappears, check that `landmark.onnx` loaded —
  the engine refuses to start when the cropper exposes neither
  `human_landmark_runner` nor `landmark_runner`, and `/health` reports
  that as the load error.
- **Portrait locked in a skewed expression** — the neutral baseline was
  captured mid-expression. Hit **Recalibrate neutral face**.
- **Live Portrait tab says "is port 7862 running?"** — check
  `GET /health` on 7862 and `C:\logs\liveportrait_err.log`.
