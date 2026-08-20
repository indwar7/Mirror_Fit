import asyncio
import base64
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
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

_demo_dir = Path(__file__).resolve().parent.parent / "demo"
if _demo_dir.exists():
    app.mount("/demo", StaticFiles(directory=str(_demo_dir), html=True), name="demo")


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

                # Garment type from UI ("tshirt" / "shirt" / "jacket"). Lets the
                # SD prompt describe the actual garment shape SD should generate.
                garment_type = msg.get("garment_type") or "tshirt"

                # Cache resized garment + type on model — avoids resize every frame
                await run_in_thread(tryon_model.set_garment, garment_img, garment_type)

                # Optional fabric / design pattern in same init message.
                fabric_b64 = msg.get("fabric_image")
                if fabric_b64:
                    fabric_img = await run_in_thread(_decode_image_raw, fabric_b64)
                    await run_in_thread(tryon_model.set_fabric, fabric_img)
                    log.info("Fabric applied in init.")

                # Reset temporal state for new session
                tryon_model.reset_temporal()

                await send({"type": "ready"})
                log.info("Garment set (type=%s, fabric=%s).",
                         garment_type, bool(fabric_b64))

            # ── Fabric: apply uploaded fabric pattern onto cached garment ────
            elif kind == "fabric":
                fabric_img = await run_in_thread(_decode_image_raw, msg["fabric_image"])
                await run_in_thread(tryon_model.set_fabric, fabric_img)
                tryon_model.reset_temporal()
                await send({"type": "fabric_set"})
                log.info("Fabric applied to garment.")

            elif kind == "fabric_clear":
                await run_in_thread(tryon_model.set_fabric, None)
                tryon_model.reset_temporal()
                await send({"type": "fabric_cleared"})
                log.info("Fabric cleared, garment restored.")

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
                    t0 = time.perf_counter()
                    person_img = await run_in_thread(_decode_image, msg["image"])
                    t_decode = time.perf_counter() - t0

                    # Inference (GPU-bound, released by thread pool).
                    # `tryon` internally measures + logs its sub-stages.
                    t1 = time.perf_counter()
                    result_img = await run_in_thread(tryon_model.tryon, person_img)
                    t_infer = time.perf_counter() - t1

                    # JPEG encode on thread pool — never blocks the async loop
                    t2 = time.perf_counter()
                    result_b64 = await run_in_thread(_encode_jpeg, result_img)
                    t_encode = time.perf_counter() - t2

                    total = time.perf_counter() - t0
                    log.info(
                        f"[lat] decode={t_decode*1000:.0f}ms "
                        f"infer={t_infer*1000:.0f}ms "
                        f"encode={t_encode*1000:.0f}ms "
                        f"total={total*1000:.0f}ms "
                        f"fps={1.0/max(total, 1e-3):.2f}"
                    )

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
