import asyncio
import base64
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model import TryOnModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

tryon_model: Optional[TryOnModel] = None

# 2 threads: one for GPU inference, one for JPEG encode/decode overlap
_thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vton-worker")


def _decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _decode_image_raw(b64: str) -> Image.Image:
    """Decode without forcing RGB — preserves RGBA alpha for garment PNG."""
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _encode_jpeg(img: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


async def run_in_thread(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_thread_pool, fn, *args)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tryon_model
    log.info("Loading try-on model...")
    tryon_model = TryOnModel()
    await run_in_thread(tryon_model.load)
    log.info("Model ready.")
    yield
    _thread_pool.shutdown(wait=False)
    log.info("Shutdown.")


app = FastAPI(title="LUCY Try-On Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tryon_model is not None}


@app.websocket("/ws/tryon")
async def tryon_ws(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")

    garment_img: Optional[Image.Image] = None
    processing  = False   # frame-drop gate: only one inference at a time

    async def send(obj: dict):
        await ws.send_text(json.dumps(obj))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            kind = msg.get("type")

            # ── Init: receive garment once ────────────────────────────────────
            if kind == "init":
                # Raw decode preserves RGBA so set_garment can extract the alpha mask
                garment_img = await run_in_thread(_decode_image_raw, msg["garment_image"])

                # Cache resized garment on model — avoids resize every frame
                await run_in_thread(tryon_model.set_garment, garment_img)
                # Reset temporal state for new session
                tryon_model.reset_temporal()

                await send({"type": "ready"})
                log.info("Garment set and cached on model.")

            # ── Frame: camera frame → try-on result ───────────────────────────
            elif kind == "frame":
                if garment_img is None:
                    await send({"type": "error", "message": "Send garment first (type: init)"})
                    continue

                # Frame drop: skip if still processing previous frame
                # This is the Decart real-time trick — never queue up stale frames
                if processing:
                    await send({"type": "dropped"})
                    continue

                processing = True
                try:
                    # Decode on thread pool (CPU-bound)
                    person_img = await run_in_thread(_decode_image, msg["image"])

                    # Inference (GPU-bound, released by thread pool)
                    result_img = await run_in_thread(tryon_model.tryon, person_img)

                    # JPEG encode on thread pool — never blocks the async loop
                    result_b64 = await run_in_thread(_encode_jpeg, result_img)

                    await send({"type": "result", "image": result_b64})

                finally:
                    processing = False

            # ── Reset temporal state (new person / scene cut) ─────────────────
            elif kind == "reset":
                tryon_model.reset_temporal()
                await send({"type": "ready"})

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        try:
            await send({"type": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        # single worker — model lives in process memory, don't fork
        workers=1,
    )
