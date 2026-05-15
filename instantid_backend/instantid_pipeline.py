"""
InstantID + SDXL-Lightning pipeline for LUCY photo-real face transfer.

This is a separate process from face_swap_backend so it can use a Python 3.11
conda env (PyTorch CUDA wheels do not exist for Python 3.14 yet).

Per-frame contract:
  load_avatar(avatar_path) -> session id; cached for the duration of the WS
  transfer(session_id, frame_bgr) -> result_bgr  (~1.5-2.0 s on A10G)

Architecture choices and their reasons are documented inline so the pipeline
can be tuned in one place without re-reading the docs:
  * SDXL base + Lightning 4-step UNet -> minimum viable photo-real at 4 steps
  * InstantX/InstantID weights (official SDXL build) -> identity ControlNet
  * antelopev2 InsightFace bundle -> 512-d ArcFace embedding
  * 1024x1024 inference -> 95 % photo-real ceiling at the cost of latency
  * Avatar embedding pre-computed once and reused per frame
  * Prompt embeds also pre-computed once
  * `set_progress_bar_config(disable=True)` so logs stay readable

Refs:
  https://huggingface.co/InstantX/InstantID
  https://huggingface.co/ByteDance/SDXL-Lightning
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

INFERENCE_RESOLUTION = 1024   # SDXL native
NUM_INFERENCE_STEPS  = 4      # Lightning 4-step
GUIDANCE_SCALE       = 1.0    # LCM/Lightning require ~1.0
CONTROLNET_SCALE     = 0.8    # identity strength via ControlNet
IP_ADAPTER_SCALE     = 0.8    # identity strength via IP-Adapter

# Files we expect to be present after running setup.ps1
MODEL_ROOT = Path(__file__).parent / "models"
ANTELOPE_DIR = MODEL_ROOT / "antelopev2"
INSTANTID_DIR = MODEL_ROOT / "InstantID"   # contains ControlNet/ + ip-adapter.bin
LIGHTNING_UNET = MODEL_ROOT / "sdxl_lightning_4step_unet.safetensors"


# Lazy import — torch + diffusers must only load inside the InstantID process,
# not in any other module that imports this file accidentally.
def _load_pipeline():
    import torch
    from diffusers import (
        ControlNetModel,
        EulerDiscreteScheduler,
        UNet2DConditionModel,
    )
    from huggingface_hub import hf_hub_download
    # Use InstantID's CUSTOM pipeline class (vendored next to this file).
    # Standard StableDiffusionXLControlNetPipeline does not have
    # `load_ip_adapter_instantid`, which is the method that loads
    # InstantID's per-token face projection + IP-Adapter cross-attention.
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

    if not ANTELOPE_DIR.exists():
        raise FileNotFoundError(
            f"antelopev2 not found at {ANTELOPE_DIR}. "
            f"Download from InstantID GH issue #61 mirror and unzip there."
        )
    if not INSTANTID_DIR.exists():
        raise FileNotFoundError(
            f"InstantID weights not found at {INSTANTID_DIR}. "
            f"Run: huggingface-cli download InstantX/InstantID --local-dir {INSTANTID_DIR}"
        )

    log.info("[InstantID] loading SDXL Lightning UNet")
    base_repo = "stabilityai/stable-diffusion-xl-base-1.0"
    if LIGHTNING_UNET.exists():
        unet = UNet2DConditionModel.from_config(base_repo, subfolder="unet").to(
            "cuda", torch.float16
        )
        from safetensors.torch import load_file
        unet.load_state_dict(load_file(str(LIGHTNING_UNET), device="cuda"))
    else:
        log.warning("Lightning UNet not at %s, downloading", LIGHTNING_UNET)
        ckpt_path = hf_hub_download(
            "ByteDance/SDXL-Lightning",
            "sdxl_lightning_4step_unet.safetensors",
            local_dir=str(MODEL_ROOT),
        )
        from safetensors.torch import load_file
        unet = UNet2DConditionModel.from_config(base_repo, subfolder="unet").to(
            "cuda", torch.float16
        )
        unet.load_state_dict(load_file(ckpt_path, device="cuda"))

    log.info("[InstantID] loading ControlNet")
    controlnet = ControlNetModel.from_pretrained(
        str(INSTANTID_DIR / "ControlNetModel"),
        torch_dtype=torch.float16,
    )

    log.info("[InstantID] assembling pipeline (InstantID custom class)")
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_repo,
        controlnet=controlnet,
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)

    log.info("[InstantID] loading IP-Adapter")
    pipe.load_ip_adapter_instantid(str(INSTANTID_DIR / "ip-adapter.bin"))

    return pipe


def _draw_kps(image_pil, kps, color_list=None):
    """Render the 5-point face landmarks as the InstantID ControlNet input.
    Re-implementation of InstantX/InstantID's draw_kps so we don't depend
    on the upstream utility module."""
    from PIL import Image
    if color_list is None:
        color_list = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255),
        ]
    stickwidth = 4
    limb_seq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)
    w, h = image_pil.size
    out = np.zeros([h, w, 3], dtype=np.uint8)
    for i in range(limb_seq.shape[0]):
        idx = limb_seq[i]
        kp1 = kps[idx[0]]
        kp2 = kps[idx[1]]
        x = kp1[0], kp2[0]
        y = kp1[1], kp2[1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = np.degrees(np.arctan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        out = cv2.fillConvexPoly(out.copy(), polygon, color_list[idx[0]])
    out = (out * 0.6).astype(np.uint8)
    for i, kp in enumerate(kps):
        cv2.circle(out, (int(kp[0]), int(kp[1])), 10, color_list[i], -1)
    return Image.fromarray(out.astype(np.uint8))


class InstantIDEngine:
    """One process-wide pipeline + many per-session avatar embeddings.

    Thread-safety: all per-frame work runs under a single lock because
    a diffusion pipeline cannot be entered concurrently from multiple
    Python threads (CUDA streams + python tensor ownership).
    """

    def __init__(self):
        import torch
        from insightface.app import FaceAnalysis

        self._torch = torch
        self.pipe = _load_pipeline()
        # InsightFace constructs the path as `{root}/models/{name}`.
        # Our antelopev2 files live at `instantid_backend/models/antelopev2`,
        # so root must be `instantid_backend/` (i.e. MODEL_ROOT.parent),
        # NOT `instantid_backend/models/`. Using MODEL_ROOT directly makes
        # InsightFace look in `instantid_backend/models/models/antelopev2/`
        # and crash with `assert 'detection' in self.models`.
        self.face_app = FaceAnalysis(
            name="antelopev2",
            root=str(MODEL_ROOT.parent),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Pre-encode the prompt so we don't rerun the SDXL text encoder
        # per frame (~80 ms saved). The wording is generic on purpose
        # because identity is supplied by IP-Adapter, not the prompt.
        self._prompt = "photo of a person, sharp, natural skin, even lighting"
        self._neg_prompt = (
            "low quality, blurry, deformed, plastic, oversaturated, painting, sketch"
        )
        with torch.no_grad():
            (
                self._pe, self._ne, self._pp, self._np
            ) = self.pipe.encode_prompt(
                prompt=self._prompt,
                negative_prompt=self._neg_prompt,
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )

        self._sessions: Dict[str, dict] = {}
        self._lock = threading.Lock()
        log.info("[InstantID] engine ready")

    # ── Avatar registration ────────────────────────────────────────────────
    def load_avatar(self, session_id: str, avatar_path: str) -> bool:
        """Compute the avatar's identity embedding ONCE and cache it under
        session_id. Returns True on success, False if no face found."""
        img_bgr = cv2.imread(str(avatar_path))
        if img_bgr is None:
            log.warning("[InstantID] avatar not readable: %s", avatar_path)
            return False
        faces = self.face_app.get(img_bgr)
        if not faces:
            log.warning("[InstantID] no face in avatar: %s", avatar_path)
            return False
        face = sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
        )[-1]
        emb = self._torch.from_numpy(face.normed_embedding).unsqueeze(0).half().cuda()
        self._sessions[session_id] = {
            "emb": emb,
            "loaded_at": time.time(),
        }
        log.info(
            "[InstantID] avatar loaded for session %s (%s)",
            session_id, Path(avatar_path).name,
        )
        return True

    def drop_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ── Per-frame inference ────────────────────────────────────────────────
    def transfer(self, session_id: str, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Apply the cached avatar identity to the user's frame.

        Returns a result image the SAME shape as the input frame so the
        demo client can overlay it 1-to-1. Internally we:
          1. Center-square-crop the input frame (no top-left padding -
             the previous version made the result look heavily zoomed
             because the model placed the face at the canvas centre but
             we cropped the top-left rectangle out).
          2. Resize the square crop to INFERENCE_RESOLUTION.
          3. Run InstantID with rescaled keypoints.
          4. Resize result back to the square crop size.
          5. Paste the square result back into the original-shape canvas
             at the same position we cropped from, so the face lines up
             with where it was in the original frame.
        """
        sess = self._sessions.get(session_id)
        if sess is None:
            log.warning("[InstantID] unknown session: %s", session_id)
            return None

        faces = self.face_app.get(frame_bgr)
        if not faces:
            return None
        face = sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
        )[-1]

        # 1. Center-square-crop the input frame (preserves face position)
        h0, w0 = frame_bgr.shape[:2]
        side = min(h0, w0)
        x_off = (w0 - side) // 2
        y_off = (h0 - side) // 2
        square = frame_bgr[y_off:y_off + side, x_off:x_off + side]

        # Recompute kps relative to the crop
        kps_in_crop = face.kps.copy()
        kps_in_crop[:, 0] -= x_off
        kps_in_crop[:, 1] -= y_off

        # 2. Resize the square crop to INFERENCE_RESOLUTION
        scale = INFERENCE_RESOLUTION / float(side)
        kps_scaled = kps_in_crop * scale

        from PIL import Image
        kps_img = _draw_kps(
            Image.new("RGB", (INFERENCE_RESOLUTION, INFERENCE_RESOLUTION), 0),
            kps_scaled,
        )

        # 3. Run InstantID
        with self._lock, self._torch.no_grad():
            self.pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
            out = self.pipe(
                prompt_embeds=self._pe,
                negative_prompt_embeds=self._ne,
                pooled_prompt_embeds=self._pp,
                negative_pooled_prompt_embeds=self._np,
                image_embeds=sess["emb"],
                image=kps_img,
                controlnet_conditioning_scale=CONTROLNET_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                width=INFERENCE_RESOLUTION,
                height=INFERENCE_RESOLUTION,
                output_type="np",
            ).images[0]

        # 4. Resize result back to the square crop size
        out_rgb = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        out_bgr = cv2.resize(out_bgr, (side, side), interpolation=cv2.INTER_LINEAR)

        # 5. Paste back into a copy of the original frame so dimensions match
        result = frame_bgr.copy()
        result[y_off:y_off + side, x_off:x_off + side] = out_bgr
        return result
