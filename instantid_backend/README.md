# LUCY InstantID Backend

Photo-real face identity transfer using InstantID + SDXL Lightning 4-step.
Runs as a separate FastAPI server on port `7861` so it can use a Python 3.11
conda env (PyTorch CUDA wheels do not exist for Python 3.14 yet).

## What it does

- Per WebSocket session, takes one avatar image.
- Computes the avatar's 512-d ArcFace identity embedding **once** at init.
- Per webcam frame, generates a new face that has the avatar's identity
  but the user's expression, head pose, eye direction, and lighting.
- ~95 % photo-real at 1024×1024, ~1.5–2.0 s per frame on g5.xlarge (A10G).

## One-time setup on the server

```powershell
cd C:\virtual-try-on\instantid_backend
powershell .\setup.ps1
```

This will:
1. Create conda env `instantid` with Python 3.11.
2. Install PyTorch (CUDA 12.1), diffusers, transformers, insightface.
3. Download SDXL base, SDXL Lightning 4-step UNet, InstantID weights, and
   the `antelopev2` InsightFace bundle into `models/`.

Total ~12 GB download, ~20 min on the AWS Mumbai region.

## Running

```powershell
cd C:\virtual-try-on\instantid_backend
powershell .\start.ps1
```

Server logs the cold-load progress. First model load takes ~30–60 s.
After `[InstantID] engine ready for traffic`, the demo client will use it.

## AWS Security Group

Open inbound TCP **7861** on the EC2 instance, scope `0.0.0.0/0`.

## Protocol

The WebSocket protocol on `/ws/instantid-swap` is identical to
`/ws/live-swap` on `face_swap_backend` so the demo client can swap
between them without any other changes:

| Direction | Type      | Payload                                                    |
| --------- | --------- | ---------------------------------------------------------- |
| C → S     | `init`    | `{avatar_id, session_id?}`                                 |
| C → S     | `frame`   | `{image: <base64 JPEG>}`                                   |
| C → S     | `ping`    | `{t?}` — keepalive                                         |
| S → C     | `ready`   | `{session_id}`                                             |
| S → C     | `result`  | `{image: <base64 JPEG>}`                                   |
| S → C     | `no_face` | no face detected in this frame                             |
| S → C     | `dropped` | server is still processing the previous frame              |
| S → C     | `error`   | `{message}`                                                |

## Latency budget

| Stage                                | ms  |
| ------------------------------------ | --- |
| Network India ↔ Mumbai RTT           | 80  |
| Frame decode (base64 → JPEG → ndarray) | 5  |
| InsightFace detection (640 px)       | 30  |
| Diffusion 4-step (1024 px)            | 1500 |
| VAE decode + crop + resize           | 100 |
| JPEG encode + base64                  | 30  |
| **Total per frame**                  | **~1.7 s** |

That's ~0.6 fps. Acceptable for "photo-real mirror" UX, NOT for fluid
mirror-like motion. If you want fluid motion, drop to 512 px which
roughly halves the diffusion time but visibly degrades skin detail.

## Co-existence with face_swap_backend

Both servers can run on the same machine, on different ports:
- `face_swap_backend` (Python 3.14): port 7860 — fast inswapper-128 swap.
- `instantid_backend` (Python 3.11): port 7861 — slow but photo-real.

The demo client tries InstantID first; if it fails to connect within
3 seconds, it falls back to the inswapper backend.

## Troubleshooting

- **`No CUDA capable device`** on first `import torch`: you ran the
  install in the wrong env. `conda activate instantid` first.
- **`antelopev2 not found`**: re-run `setup.ps1`; the GH download may
  have rate-limited. Check for `models/antelopev2/glintr100.onnx`.
- **`first frame takes 12 s`**: kernel autotune. Subsequent frames are
  the documented 1.5–2 s.
- **OOM on a different instance type**: pipeline needs ~14 GB VRAM.
  A10G (24 GB) is fine. T4 (16 GB) only fits at 768 px.
