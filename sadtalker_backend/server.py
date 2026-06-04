"""
LUCY AI Twin — SadTalker FastAPI wrapper.

Generates a talking-head video from:
  • a still portrait image (JPG/PNG)
  • a driving audio clip   (WAV/MP3)

Wraps the OpenTalker/SadTalker `inference.py` CLI as a subprocess. The
SadTalker repo + checkpoints are expected to live in `SADTALKER_HOME`
(default: ../SadTalker relative to this file). Conda env `sadtalker` must
be activated before running this server (SadTalker uses Python 3.8 +
torch 1.12 + CUDA 11.3, separate from the rest of Lucy's stack).

Endpoints:
  GET  /health                       liveness probe
  POST /generate-talking-avatar      multipart form: image, audio
                                     -> streams the generated mp4

Run on the GPU box:
  conda activate sadtalker
  cd lucy/sadtalker_backend
  uvicorn server:app --host 0.0.0.0 --port 7863
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sadtalker")

HERE           = Path(__file__).parent.resolve()
SADTALKER_HOME = Path(os.environ.get("SADTALKER_HOME", HERE.parent / "SadTalker")).resolve()
JOBS_DIR       = Path(os.environ.get("SADTALKER_JOBS", HERE / "jobs")).resolve()
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Tunables — read at server start so they can be set via env without code edits.
PREPROCESS = os.environ.get("SADTALKER_PREPROCESS", "full")     # crop / extcrop / full
ENHANCER   = os.environ.get("SADTALKER_ENHANCER",   "gfpgan")   # gfpgan / RestoreFormer / ""
SIZE       = int(os.environ.get("SADTALKER_SIZE",   "512"))     # 256 or 512
STILL      = os.environ.get("SADTALKER_STILL", "1") == "1"      # add --still flag
MAX_AUDIO_SEC = float(os.environ.get("SADTALKER_MAX_AUDIO_SEC", "20"))

app = FastAPI(title="LUCY AI Twin")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # local demo only — tighten for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Basic liveness — checks that the SadTalker repo is reachable on disk."""
    return {
        "ok":              True,
        "sadtalker_home":  str(SADTALKER_HOME),
        "sadtalker_found": (SADTALKER_HOME / "inference.py").exists(),
        "preprocess":      PREPROCESS,
        "enhancer":        ENHANCER,
        "size":            SIZE,
        "jobs_dir":        str(JOBS_DIR),
    }


@app.post("/generate-talking-avatar")
async def generate_talking_avatar(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    """
    Run one SadTalker inference. Saves inputs to a per-job dir, invokes
    `python inference.py …`, finds the resulting mp4 in that dir and
    streams it back.
    """
    if not (SADTALKER_HOME / "inference.py").exists():
        raise HTTPException(
            500,
            detail=(
                f"SadTalker repo not found at {SADTALKER_HOME}. Clone it and "
                "set SADTALKER_HOME, or place it at the default path."
            ),
        )

    job_id  = uuid.uuid4().hex[:10]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log.info("[%s] new job", job_id)

    # Persist uploads with their original extensions so SadTalker's
    # format sniffing works (.png vs .jpg matters for the preprocess step).
    img_suffix = Path(image.filename or "source.png").suffix or ".png"
    aud_suffix = Path(audio.filename or "driven.wav").suffix or ".wav"
    img_path = job_dir / f"source{img_suffix}"
    aud_path = job_dir / f"driven{aud_suffix}"

    with img_path.open("wb") as f:
        f.write(await image.read())
    with aud_path.open("wb") as f:
        f.write(await audio.read())

    # Build the SadTalker CLI command. We pass --result_dir so the output
    # mp4 lands inside our job dir (it nests under a timestamped subdir).
    cmd = [
        "python", "inference.py",
        "--driven_audio", str(aud_path),
        "--source_image", str(img_path),
        "--result_dir",   str(job_dir),
        "--size",         str(SIZE),
        "--preprocess",   PREPROCESS,
    ]
    if STILL:
        cmd.append("--still")
    if ENHANCER:
        cmd += ["--enhancer", ENHANCER]

    log.info("[%s] running: %s", job_id, " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(SADTALKER_HOME),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    log.info("[%s] finished in %.1fs (returncode=%d)", job_id, elapsed, proc.returncode)

    if proc.returncode != 0:
        # Surface the failure to the caller — but never to the public
        # demo, since stderr can leak filesystem paths.
        log.error("[%s] stderr:\n%s", job_id, proc.stderr[-2000:])
        return JSONResponse(
            status_code=500,
            content={
                "error":    "SadTalker inference failed",
                "stderr":   proc.stderr[-1500:],
                "elapsed":  elapsed,
            },
        )

    # SadTalker nests the result under a timestamped subdir. Find the mp4.
    mp4_files = list(job_dir.rglob("*.mp4"))
    if not mp4_files:
        return JSONResponse(
            status_code=500,
            content={
                "error":   "SadTalker produced no mp4",
                "stdout":  proc.stdout[-1500:],
                "elapsed": elapsed,
            },
        )

    # Pick the most-recently modified mp4 (typically there's only one).
    mp4_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out_mp4 = mp4_files[0]
    log.info("[%s] serving %s (%.1f MB)", job_id, out_mp4, out_mp4.stat().st_size / 1e6)

    return FileResponse(
        out_mp4,
        media_type="video/mp4",
        filename=f"twin_{job_id}.mp4",
        headers={"X-Inference-Seconds": f"{elapsed:.1f}"},
    )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Hand-roll cleanup endpoint so we don't fill the disk during a demo."""
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, detail="unknown job_id")
    shutil.rmtree(job_dir, ignore_errors=True)
    return {"deleted": job_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7863)
