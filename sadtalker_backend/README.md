# LUCY AI Twin — SadTalker backend

Wraps [OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker)
behind a FastAPI endpoint so the Lucy demo can request a talking-head
video for any uploaded photo + audio combination.

Runs on port **7863** alongside the other Lucy backends.

---

## EC2 setup (Windows, one-time)

> The Windows box already has the other Lucy backends. SadTalker needs its
> **own conda env** because it pins older torch/CUDA than the rest of the
> stack. The two coexist fine — only the conda env you activate at server
> start matters.

```powershell
# 1. Clone the SadTalker repo next to this backend
cd C:\virtual-try-on
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker

# 2. Fresh conda env (Python 3.8 — required by SadTalker)
conda create -n sadtalker python=3.8 -y
conda activate sadtalker

# 3. Pin torch to the exact version SadTalker expects
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 `
    --extra-index-url https://download.pytorch.org/whl/cu113

# 4. ffmpeg via conda (SadTalker shells out to ffmpeg for video assembly)
conda install -c conda-forge ffmpeg -y

# 5. SadTalker's own Python deps
pip install -r requirements.txt

# 6. Download model checkpoints (~1 GB total)
#    These mirrors are pinned by the upstream repo. If any 404, check the
#    SadTalker README for current URLs.
mkdir checkpoints
mkdir gfpgan\weights

# Core SadTalker weights
$base = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/"
Invoke-WebRequest "$base/mapping_00229-model.pth.tar"        -OutFile checkpoints\mapping_00229-model.pth.tar
Invoke-WebRequest "$base/SadTalker_V0.0.2_256.safetensors"   -OutFile checkpoints\SadTalker_V0.0.2_256.safetensors
Invoke-WebRequest "$base/SadTalker_V0.0.2_512.safetensors"   -OutFile checkpoints\SadTalker_V0.0.2_512.safetensors

# GFPGAN weights for the optional face enhancer (sharper output)
$gfp = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/"
Invoke-WebRequest "$gfp/alignment_WFLW_4HG.pth"   -OutFile gfpgan\weights\alignment_WFLW_4HG.pth
Invoke-WebRequest "$gfp/detection_Resnet50_Final.pth" -OutFile gfpgan\weights\detection_Resnet50_Final.pth
$gfp2 = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/"
Invoke-WebRequest "$gfp2/GFPGANv1.4.pth" -OutFile gfpgan\weights\GFPGANv1.4.pth

# 7. Smoke test the inference once with the bundled sample
python inference.py `
    --driven_audio examples/driven_audio/sample.wav `
    --source_image examples/source_image/sample.png `
    --result_dir   examples/_test_out `
    --size 256 --still --enhancer gfpgan
# An mp4 should appear under examples/_test_out/. If it does, you're done.

# 8. Install THIS wrapper's deps
cd ..\sadtalker_backend
pip install -r requirements.txt
```

---

## Running the server

```powershell
conda activate sadtalker
cd C:\virtual-try-on\sadtalker_backend
python -m uvicorn server:app --host 0.0.0.0 --port 7863
```

Server starts in <5 s (no models loaded eagerly — SadTalker loads its
checkpoints per inference). Confirm:

```powershell
(Invoke-WebRequest "http://localhost:7863/health" -UseBasicParsing).Content
```

Expected:
```json
{"ok": true, "sadtalker_found": true, ...}
```

---

## Tunables (env vars)

| Var                       | Default  | Effect |
|---------------------------|----------|--------|
| `SADTALKER_HOME`          | `../SadTalker` | Where the SadTalker repo lives |
| `SADTALKER_JOBS`          | `./jobs` | Per-job working dirs (mp4 outputs land here) |
| `SADTALKER_PREPROCESS`    | `full`   | `crop` / `extcrop` / `full` — `full` keeps the whole head visible |
| `SADTALKER_ENHANCER`      | `gfpgan` | `gfpgan` / `RestoreFormer` / empty to skip |
| `SADTALKER_SIZE`          | `512`    | 256 (faster) or 512 (sharper) |
| `SADTALKER_STILL`         | `1`      | Locks the head pose (no random nodding) |
| `SADTALKER_MAX_AUDIO_SEC` | `20`     | Soft cap on driving audio length — protects the GPU |

---

## API

### `GET /health`
Liveness + config dump. No GPU touched.

### `POST /generate-talking-avatar`
Multipart form:
- `image` — JPG/PNG portrait, head clearly visible
- `audio` — WAV/MP3 driving audio

Returns: `video/mp4` of the portrait lip-syncing the audio.
Header `X-Inference-Seconds` reports wall-clock time on the server.

### `DELETE /jobs/{job_id}`
Remove a job's working dir + output mp4. Useful during demos to keep
disk usage bounded.

---

## Expected speed (A10G, 512 px, GFPGAN on)

| Audio length | Inference time |
|--------------|----------------|
| 5 s          | ~25-40 s       |
| 10 s         | ~50-80 s       |
| 20 s         | ~100-150 s     |

Faster with `SADTALKER_SIZE=256` and `SADTALKER_ENHANCER=""` (drops GFPGAN).
