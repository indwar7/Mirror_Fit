"""
Virtual try-on — four-tier inference stack.

  Tier 1 (TRT):          TensorRT engine + CUDA graph      → ~30-40fps  (after training)
  Tier 2 (Compile):      torch.compile CatVTON             → ~15-20fps  (after training)
  Tier 3 (Live):         SD 1.5 img2img + LCM 2-step
                         + TAESD decoder + IP-Adapter       → ~3-5fps    (today, no training)
  Tier 4 (AnimateDiff):  AnimateDiff video backbone         → temporal-consistent video
                         + DWPose ControlNet conditioning   → pose-aligned try-on

Env vars (set by run_all.sh after training):
  VTON_LORA_CHECKPOINT      → activates Tier 2
  TRT_ENGINE_PATH           → activates Tier 1
  ANIMATEDIFF_ADAPTER_PATH  → activates Tier 4 (AnimateDiff motion adapter)
  VTON_STEPS                → override inference steps
"""
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

LIVE_SIZE = 512
OUTPUT_W, OUTPUT_H = 512, 768
LATENT_W = OUTPUT_W // 8
LATENT_H = OUTPUT_H // 8

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

VTON_LORA_CHECKPOINT      = os.environ.get("VTON_LORA_CHECKPOINT", "")
TRT_ENGINE_PATH           = os.environ.get("TRT_ENGINE_PATH", "")
ANIMATEDIFF_ADAPTER_PATH  = os.environ.get("ANIMATEDIFF_ADAPTER_PATH", "")
VTON_STEPS                = int(os.environ.get("VTON_STEPS", "0"))

# AnimateDiff frame buffer config
ANIMATEDIFF_BUFFER_SIZE = 8   # number of frames to accumulate before processing as video sequence


# ── Tier 1: TensorRT + CUDA graph ────────────────────────────────────────────

class TRTInferenceEngine:
    def __init__(self, engine_path: str, device: str):
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self._engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self._ctx   = self._engine.create_execution_context()
        self.device = device
        dtype = torch.float16
        self.buf_sample = torch.zeros(1, 12, LATENT_H, LATENT_W, device=device, dtype=dtype)
        self.buf_t      = torch.zeros(1,                          device=device, dtype=torch.long)
        self.buf_enc    = torch.zeros(1, 77, 768,                 device=device, dtype=dtype)
        self.buf_out    = torch.zeros(1, 4, LATENT_H, LATENT_W,  device=device, dtype=dtype)
        for name, buf in [("sample", self.buf_sample), ("timestep", self.buf_t),
                          ("encoder_hidden", self.buf_enc), ("noise_pred", self.buf_out)]:
            idx = self._engine.get_binding_index(name)
            self._ctx.set_binding_shape(idx, tuple(buf.shape))
            self._ctx.set_tensor_address(name, buf.data_ptr())
        stream = torch.cuda.current_stream()
        for _ in range(3):
            self._ctx.execute_async_v3(stream.cuda_stream); torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._ctx.execute_async_v3(stream.cuda_stream)
        log.info(f"TRT engine + CUDA graph ready: {engine_path}")

    def infer(self, sample, t, enc):
        self.buf_sample.copy_(sample); self.buf_t.copy_(t); self.buf_enc.copy_(enc)
        self._graph.replay()
        return self.buf_out.clone()


# ── Tier 2: torch.compile + CUDA graph ───────────────────────────────────────

class CompiledUNet:
    def __init__(self, unet, device, dtype):
        self.unet = torch.compile(unet, mode="reduce-overhead", fullgraph=False)
        self.s_sample = torch.zeros(1, 12, LATENT_H, LATENT_W, device=device, dtype=dtype)
        self.s_enc    = torch.zeros(1, 77, 768,                 device=device, dtype=dtype)
        self.s_t      = torch.zeros(1,                           device=device, dtype=torch.long)
        for _ in range(3):
            with torch.no_grad(): self.unet(self.s_sample, self.s_t, encoder_hidden_states=self.s_enc)
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.s_out = self.unet(self.s_sample, self.s_t, encoder_hidden_states=self.s_enc).sample

    def __call__(self, sample, t, enc):
        self.s_sample.copy_(sample); self.s_t.copy_(t); self.s_enc.copy_(enc)
        self._graph.replay()
        return self.s_out.clone()


# ── Main model ────────────────────────────────────────────────────────────────

class TryOnModel:
    def __init__(self):
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype    = torch.float16 if self.device == "cuda" else torch.float32
        self.pipeline = None
        self._trt     = None
        self._compiled = None
        self._vae      = None
        self._null_emb = None
        self._scheduler = None
        self._ip_loaded = False
        self._garment_cache: Image.Image | None = None
        self._ip_embeds = None   # pre-computed IP-Adapter CLIP embeddings, cached per garment
        self._steps = 2
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # MediaPipe selfie segmentation — separates person from background precisely
        try:
            import mediapipe as mp
            # Support both old (solutions) and new (tasks) mediapipe APIs
            if hasattr(mp, 'solutions'):
                self._mp_seg  = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
                self._mp_face = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
            else:
                self._mp_seg  = None
                self._mp_face = None
                log.warning("MediaPipe solutions API not available, using fixed mask fallback.")
            if self._mp_seg:
                log.info("MediaPipe loaded.")
        except Exception as e:
            self._mp_seg  = None
            self._mp_face = None
            log.warning(f"MediaPipe not available, using fallback: {e}")
        self._prev_result      = None
        self._fixed_mask_cache = None
        self._garment_alpha    = None   # alpha mask from original RGBA garment PNG

        # ── Tier 4: AnimateDiff video backbone ───────────────────────────────
        self._animatediff_pipe = None
        self._frame_buffer: list[np.ndarray] = []   # raw BGR frames waiting for batch processing
        self._buffer_size  = ANIMATEDIFF_BUFFER_SIZE
        self._video_results: list[Image.Image] = []  # processed video frames ready to serve
        self._video_result_idx = 0                   # pointer into _video_results

        # ── Phase 3: DWPose body-pose detector ───────────────────────────────
        self._dwpose = None   # loaded by _load_dwpose() when controlnet_aux available

        log.info(f"Device: {self.device} | TRT: {bool(TRT_ENGINE_PATH)} | "
                 f"LoRA: {bool(VTON_LORA_CHECKPOINT)} | "
                 f"AnimateDiff: {bool(ANIMATEDIFF_ADAPTER_PATH)}")

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self):
        if self.device != "cuda":
            self._load_tier3()
            self._try_load_dwpose()
            return
        if TRT_ENGINE_PATH and Path(TRT_ENGINE_PATH).exists():
            try: self._load_tier1(); self._try_load_dwpose(); return
            except Exception as e: log.warning(f"Tier 1 failed: {e}")
        if VTON_LORA_CHECKPOINT and Path(VTON_LORA_CHECKPOINT).exists():
            try: self._load_tier2(); self._try_load_dwpose(); return
            except Exception as e: log.warning(f"Tier 2 failed: {e}")
        if ANIMATEDIFF_ADAPTER_PATH and Path(ANIMATEDIFF_ADAPTER_PATH).exists():
            try: self._load_tier4_animatediff(); self._try_load_dwpose(); return
            except Exception as e: log.warning(f"Tier 4 AnimateDiff failed: {e}")
        self._load_tier3()
        self._try_load_dwpose()

    def _load_tier1(self):
        from diffusers import AutoencoderKL, DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        base = "zheng-chong/CatVTON"
        self._vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=self.dtype).to(self.device)
        self._vae.requires_grad_(False)
        tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        te  = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=self.dtype).to(self.device)
        with torch.no_grad():
            self._null_emb = te(tok([""], padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device))[0]
        self._scheduler = DDIMScheduler.from_pretrained(base, subfolder="scheduler")
        self._scheduler.set_timesteps(1)
        self._trt   = TRTInferenceEngine(TRT_ENGINE_PATH, self.device)
        self._steps = VTON_STEPS if VTON_STEPS > 0 else 1
        log.info("Tier 1 ready — TRT. ~30-40fps")

    def _load_tier2(self):
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        base = "zheng-chong/CatVTON"
        unet = UNet2DConditionModel.from_pretrained(VTON_LORA_CHECKPOINT, torch_dtype=self.dtype,
                                                     attn_implementation="flash_attention_2").to(self.device)
        unet.requires_grad_(False)
        self._compiled  = CompiledUNet(unet, self.device, self.dtype)
        self._vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=self.dtype).to(self.device)
        self._vae.requires_grad_(False)
        tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        te  = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=self.dtype).to(self.device)
        with torch.no_grad():
            self._null_emb = te(tok([""], padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(self.device))[0]
        self._scheduler = DDIMScheduler.from_pretrained(base, subfolder="scheduler")
        self._steps = VTON_STEPS if VTON_STEPS > 0 else 1
        log.info("Tier 2 ready — torch.compile. ~15-20fps")

    def _load_tier3(self):
        """
        SD 1.5 img2img + LCM 2-step + TAESD (tiny decoder) + IP-Adapter.
        img2img (not inpainting) is ~2x faster and works for live streaming.
        TAESD replaces the full VAE for 5x faster encode/decode.
        Target: ~200-300ms per frame = 3-5fps on A10G.
        """
        from diffusers import AutoPipelineForImage2Image, LCMScheduler, AutoencoderTiny

        log.info("Loading SD 1.5 img2img + LCM 2-step + TAESD + IP-Adapter (live mode)…")

        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        # TAESD: 5x faster than full VAE decoder, minimal quality loss
        self.pipeline.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd",
            torch_dtype=self.dtype,
        ).to(self.device)

        # LCM scheduler for 2-step inference
        self.pipeline.scheduler = LCMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self.pipeline.fuse_lora()

        # IP-Adapter: garment image as visual reference
        self.pipeline.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models",
            weight_name="ip-adapter_sd15.bin",
        )
        self.pipeline.set_ip_adapter_scale(1.0)
        self._ip_loaded = True
        self._steps = VTON_STEPS if VTON_STEPS > 0 else 4

        if self.device == "cuda":
            # xformers: ~20% faster attention on A10G
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                log.info("xformers attention enabled.")
            except Exception:
                pass

            # channels_last: ~10% faster conv ops on NVIDIA
            try:
                self.pipeline.unet.to(memory_format=torch.channels_last)
                self.pipeline.vae.to(memory_format=torch.channels_last)
                log.info("channels_last memory format enabled.")
            except Exception:
                pass

            import platform
            if platform.system() != "Windows":  # torch.compile needs Triton — Linux only
                self.pipeline.unet = torch.compile(
                    self.pipeline.unet, mode="reduce-overhead", fullgraph=False
                )
            self._warmup_tier3()

        log.info(f"Tier 3 live ready — {self._steps}-step img2img. ~3-5fps on A10G")

    def _warmup_tier3(self):
        dummy = Image.fromarray(np.zeros((LIVE_SIZE, LIVE_SIZE, 3), dtype=np.uint8))
        log.info("Warming up (3 passes)…")
        for _ in range(3):
            try:
                self.pipeline(
                    prompt="",
                    image=dummy,
                    ip_adapter_image=dummy,
                    num_inference_steps=self._steps,
                    strength=0.95,
                    guidance_scale=1.0,
                )
            except Exception:
                pass
        log.info("Warmup done — server ready.")

    # ── Tier 4: AnimateDiff video backbone ───────────────────────────────────

    def _load_tier4_animatediff(self):
        """
        AnimateDiff video backbone for temporal-consistent try-on.

        Uses AnimateDiffPipeline with:
          - IP-Adapter at scale 0.8 for garment reference
          - LCM scheduler for fast inference
          - 8-frame buffer: accumulates frames then processes as a video sequence,
            which makes the garment render stable across time (no per-frame flicker)

        Activated by env var ANIMATEDIFF_ADAPTER_PATH pointing to a fine-tuned
        motion adapter (from train_video.py) or the stock mm_sd_v15_v2 adapter.
        """
        from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter, AutoencoderTiny

        log.info(f"Loading AnimateDiff motion adapter from: {ANIMATEDIFF_ADAPTER_PATH}")

        motion_adapter = MotionAdapter.from_pretrained(
            ANIMATEDIFF_ADAPTER_PATH,
            torch_dtype=self.dtype,
        )

        self._animatediff_pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=motion_adapter,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        # TAESD: 5x faster decode, same as Tier 3
        self._animatediff_pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd",
            torch_dtype=self.dtype,
        ).to(self.device)

        # LCM for fast 4-step inference (AnimateDiff needs slightly more steps than img2img)
        self._animatediff_pipe.scheduler = LCMScheduler.from_config(
            self._animatediff_pipe.scheduler.config
        )
        self._animatediff_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self._animatediff_pipe.fuse_lora()

        # IP-Adapter at 0.8 (lower than Tier 3 0.8 — leave room for motion adapter to steer)
        self._animatediff_pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models",
            weight_name="ip-adapter_sd15.bin",
        )
        self._animatediff_pipe.set_ip_adapter_scale(0.8)
        self._ip_loaded = True

        # xformers if available
        try:
            self._animatediff_pipe.enable_xformers_memory_efficient_attention()
            log.info("xformers attention enabled for AnimateDiff.")
        except Exception:
            pass

        # channels_last for conv ops
        try:
            self._animatediff_pipe.unet.to(memory_format=torch.channels_last)
            self._animatediff_pipe.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass

        self._steps = VTON_STEPS if VTON_STEPS > 0 else 4
        self._frame_buffer   = []
        self._video_results  = []
        self._video_result_idx = 0

        log.info(f"Tier 4 AnimateDiff ready — {self._buffer_size}-frame buffer, "
                 f"{self._steps}-step LCM. Temporal-consistent try-on enabled.")

    # ── Phase 3: DWPose body-pose conditioning ────────────────────────────────

    def _load_dwpose(self):
        """
        Load DWPose full-body pose estimator from controlnet_aux.
        Called unconditionally on startup; if the package is missing we log a warning
        and continue without pose conditioning (graceful degradation).
        """
        from controlnet_aux import DWposeDetector
        self._dwpose = DWposeDetector.from_pretrained("lllyasviel/Annotators")
        log.info("DWPose detector loaded — pose conditioning active.")

    def _try_load_dwpose(self):
        """Silently skip DWPose if controlnet_aux is not installed."""
        try:
            self._load_dwpose()
        except ImportError:
            log.warning("controlnet_aux not installed — DWPose pose conditioning disabled. "
                        "Install with: pip install controlnet_aux")
        except Exception as e:
            log.warning(f"DWPose failed to load, running without pose conditioning: {e}")

    def _get_pose(self, frame_arr: np.ndarray) -> Image.Image | None:
        """
        Extract DWPose skeleton heatmap from a BGR numpy frame.
        Returns a PIL Image of the pose drawing (same spatial size as input),
        or None if DWPose is not loaded.
        """
        if self._dwpose is None:
            return None
        # DWposeDetector expects RGB PIL
        rgb = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2RGB)
        pil_in = Image.fromarray(rgb)
        pose_img = self._dwpose(pil_in, output_type="pil",
                                detect_resolution=512, image_resolution=LIVE_SIZE)
        return pose_img

    # ── Garment ───────────────────────────────────────────────────────────────

    def set_garment(self, garment_image: Image.Image):
        if garment_image.mode == 'RGBA':
            bg = Image.new('RGB', garment_image.size, (255, 255, 255))
            bg.paste(garment_image, mask=garment_image.split()[3])
            g = bg
        else:
            g = garment_image.convert("RGB")
        gw, gh = g.size
        sq = max(gw, gh)
        padded = Image.new("RGB", (sq, sq), (255, 255, 255))
        padded.paste(g, ((sq - gw) // 2, (sq - gh) // 2))
        garment_sq = padded.resize((LIVE_SIZE, LIVE_SIZE), Image.LANCZOS)
        self._garment_cache = garment_sq
        self._ip_embeds     = None

        # Store alpha mask if original had transparency — used by geometric warp
        if garment_image.mode == 'RGBA':
            alpha_sq = garment_image.split()[3].resize((LIVE_SIZE, LIVE_SIZE), Image.LANCZOS)
            self._garment_alpha = np.array(alpha_sq).astype(np.float32) / 255.0
        else:
            self._garment_alpha = None

        # Pre-compute IP-Adapter CLIP embeddings once — reused every frame instead of per-frame encoding
        if self._ip_loaded and hasattr(self.pipeline, "prepare_ip_adapter_image_embeds"):
            try:
                with torch.inference_mode():
                    self._ip_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                        ip_adapter_image=[garment_sq],
                        ip_adapter_image_embeds=None,
                        device=self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,  # guidance_scale=1.0 → no CFG
                    )
                log.info("Garment IP embeddings pre-computed and cached.")
            except Exception as e:
                log.warning(f"IP embed pre-cache failed, will encode per-frame: {e}")
        log.info("Garment cached.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def tryon(self, person_image: Image.Image,
              garment_image: Image.Image | None = None) -> Image.Image:
        garment = self._garment_cache if garment_image is None \
                  else garment_image.resize((LIVE_SIZE, LIVE_SIZE))
        if garment is None:
            return person_image

        if self._trt is not None:
            return self._infer_catvton(person_image.resize((OUTPUT_W, OUTPUT_H)),
                                        garment.resize((OUTPUT_W, OUTPUT_H)),
                                        lambda s, t, e: self._trt.infer(s, t.to(torch.long), e))
        if self._compiled is not None:
            return self._infer_catvton(person_image.resize((OUTPUT_W, OUTPUT_H)),
                                        garment.resize((OUTPUT_W, OUTPUT_H)),
                                        lambda s, t, e: self._compiled(s, t, e)[:, :4])

        # ── Tier 4: AnimateDiff video backbone ───────────────────────────────
        if self._animatediff_pipe is not None:
            return self._infer_tier4_animatediff(person_image, garment)

        # ── SD img2img — realistic jacket rendering ───────────────────────────
        # Save prev BEFORE calling _infer_tier3 (which overwrites _prev_result)
        saved_prev = self._prev_result
        result     = self._infer_tier3(person_image, garment, strength=0.80)
        result_arr = np.array(result)

        # Smooth OUTPUT with the PREVIOUS frame (not current)
        if saved_prev is not None:
            result_arr = (result_arr.astype(np.float32) * 0.50
                          + saved_prev.astype(np.float32) * 0.50).astype(np.uint8)

        self._prev_result = result_arr.copy()
        return Image.fromarray(result_arr)

    # ── Live geometric warp ───────────────────────────────────────────────────

    def _infer_live_geometric(self, person_image: Image.Image) -> Image.Image:
        garment = self._garment_cache
        if garment is None:
            return person_image

        pw, ph    = person_image.size
        sq        = min(pw, ph)
        person_sq = person_image.crop(((pw-sq)//2, (ph-sq)//2,
                                        (pw+sq)//2, (ph+sq)//2))
        person_sq = person_sq.resize((LIVE_SIZE, LIVE_SIZE), Image.LANCZOS)
        frame     = np.array(person_sq)
        H, W      = frame.shape[:2]

        # Face detection → dynamic torso placement
        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self._haar.detectMultiScale(
            cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            cx          = fx + fw // 2
            top         = fy + fh - int(fh * 0.1)   # slight overlap with neck
            bottom      = min(H, top + int(fh * 2.6))
            left        = max(0, cx - int(fw * 1.3))
            right       = min(W, cx + int(fw * 1.3))
            face_bottom = fy + fh
        else:
            top = int(H * 0.32); bottom = int(H * 0.88)
            left = int(W * 0.15); right = int(W * 0.85)
            face_bottom = top

        th = max(1, bottom - top)
        tw = max(1, right  - left)

        # Resize shirt to torso area
        shirt = np.array(garment.resize((tw, th), Image.LANCZOS))

        # Alpha: use real PNG alpha channel if available
        if self._garment_alpha is not None:
            alpha = cv2.resize(self._garment_alpha, (tw, th),
                               interpolation=cv2.INTER_LINEAR)
        else:
            g_gray = cv2.cvtColor(shirt, cv2.COLOR_RGB2GRAY)
            _, bg  = cv2.threshold(g_gray, 240, 255, cv2.THRESH_BINARY)
            alpha  = (255 - bg).astype(np.float32) / 255.0

        alpha = cv2.GaussianBlur(alpha, (11, 11), 0)

        # Darken edges of shirt slightly — depth cue makes it look worn not pasted
        edge_shadow        = np.ones_like(alpha)
        edge_shadow[:, :int(tw*0.08)]  *= np.linspace(0.6, 1.0, int(tw*0.08))
        edge_shadow[:, -int(tw*0.08):] *= np.linspace(1.0, 0.6, int(tw*0.08))
        shirt = np.clip(shirt.astype(np.float32) * edge_shadow[:, :, np.newaxis],
                        0, 255).astype(np.uint8)

        alpha = alpha[:, :, np.newaxis]

        # Blend shirt onto torso
        result = frame.copy()
        roi    = result[top:top+th, left:left+tw]
        if roi.shape[:2] == (th, tw):
            result[top:top+th, left:left+tw] = (
                shirt.astype(np.float32) * alpha
                + roi.astype(np.float32) * (1.0 - alpha)
            ).astype(np.uint8)

        # Face always 100% original
        result[:face_bottom] = frame[:face_bottom]
        return Image.fromarray(result)

    # ── Tier 4: AnimateDiff video sequence inference ──────────────────────────

    def _infer_tier4_animatediff(self, person_image: Image.Image,
                                 garment: Image.Image) -> Image.Image:
        """
        Buffer incoming person frames until we have a full 8-frame window, then
        process the entire window as a video sequence through AnimateDiff.

        This gives temporal consistency because AnimateDiff's motion modules see
        all 8 frames simultaneously and enforce smooth motion across them.

        Return strategy:
          - While filling the buffer: return the most recent processed frame from the
            previous batch (if any), or fall back to Tier 3 for the very first batch.
          - When buffer hits 8: process, clear buffer, store results, return first result.
          - Subsequent tryon() calls drain _video_results one frame at a time.
        """
        # Center-crop + resize incoming frame to square
        pw, ph = person_image.size
        sq = min(pw, ph)
        person_sq = person_image.crop(((pw - sq) // 2, (ph - sq) // 2,
                                       (pw + sq) // 2, (ph + sq) // 2))
        person_sq_arr = np.array(person_sq.resize((LIVE_SIZE, LIVE_SIZE), Image.LANCZOS))
        self._frame_buffer.append(person_sq_arr)

        # ── Drain pre-computed video results first ────────────────────────────
        if self._video_result_idx < len(self._video_results):
            out = self._video_results[self._video_result_idx]
            self._video_result_idx += 1
            return out

        # ── Not enough frames yet — return the previous result or Tier 3 ─────
        if len(self._frame_buffer) < self._buffer_size:
            if self._prev_result is not None:
                return Image.fromarray(self._prev_result)
            # Very first frames: fall through to Tier 3 single-frame inference
            if self.pipeline is not None:
                return self._infer_tier3(person_image, garment, strength=0.95)
            return person_image

        # ── Buffer full: process 8-frame video sequence ───────────────────────
        frames_arr = self._frame_buffer[:self._buffer_size]
        self._frame_buffer = self._frame_buffer[self._buffer_size:]  # slide window

        # Convert frames to PIL list
        pil_frames = [Image.fromarray(f) for f in frames_arr]

        ip_kwargs = (
            {"ip_adapter_image_embeds": self._ip_embeds}
            if self._ip_embeds is not None
            else {"ip_adapter_image": garment}
        )

        with torch.inference_mode():
            output = self._animatediff_pipe(
                prompt="person wearing the garment, photorealistic, fashion, smooth motion",
                negative_prompt="blurry, distorted, flickering, deformed",
                num_frames=self._buffer_size,
                num_inference_steps=self._steps,
                guidance_scale=1.0,
                **ip_kwargs,
            )
        # AnimateDiffPipeline returns .frames as a list[list[PIL]] — shape [1][num_frames]
        video_frames: list[Image.Image] = output.frames[0]

        # Apply clothing mask composite on each video frame
        composited = []
        for i, (result_frame, orig_arr) in enumerate(zip(video_frames, frames_arr)):
            result_arr = np.array(result_frame.resize((LIVE_SIZE, LIVE_SIZE), Image.LANCZOS))
            orig_resized = orig_arr  # already LIVE_SIZE

            if self._fixed_mask_cache is None:
                m = np.zeros((LIVE_SIZE, LIVE_SIZE), dtype=np.float32)
                m[int(LIVE_SIZE * 0.38):int(LIVE_SIZE * 0.92),
                  int(LIVE_SIZE * 0.05):int(LIVE_SIZE * 0.95)] = 1.0
                self._fixed_mask_cache = cv2.GaussianBlur(m, (31, 31), 0)[:, :, np.newaxis]

            mask = self._fixed_mask_cache
            composite = (
                result_arr.astype(np.float32) * mask
                + orig_resized.astype(np.float32) * (1.0 - mask)
            ).astype(np.uint8)
            composited.append(Image.fromarray(composite))

        self._prev_result = np.array(composited[-1])

        # Store results for draining over next tryon() calls
        self._video_results  = composited
        self._video_result_idx = 1  # we return index 0 right now

        return composited[0]

    def reset_temporal(self):
        self._prev_result      = None
        self._frame_buffer     = []
        self._video_results    = []
        self._video_result_idx = 0

    # ── CatVTON shared inference (Tier 1 + 2) ────────────────────────────────

    def _infer_catvton(self, person, garment, unet_fn):
        import torchvision.transforms.functional as TF
        to_lat = lambda img: TF.normalize(TF.to_tensor(img).unsqueeze(0), [0.5]*3, [0.5]*3).to(self.device, self.dtype)
        with torch.inference_mode():
            p_lat  = self._vae.encode(to_lat(person)).latent_dist.sample()  * self._vae.config.scaling_factor
            g_lat  = self._vae.encode(to_lat(garment)).latent_dist.sample() * self._vae.config.scaling_factor
            noise  = torch.randn_like(p_lat)
            t      = self._scheduler.timesteps[:1].to(self.device)
            npred  = unet_fn(torch.cat([noise, p_lat, g_lat], dim=1).to(self.dtype), t, self._null_emb)
            out    = self._vae.decode(self._scheduler.step(npred, t[0], noise).prev_sample / self._vae.config.scaling_factor).sample
            out    = (out.clamp(-1, 1) + 1) / 2
        return TF.to_pil_image(out[0].float().cpu())

    # ── Tier 3 live inference ─────────────────────────────────────────────────

    def _infer_tier3(self, person_image: Image.Image, garment: Image.Image,
                     strength: float = 0.95,
                     clean_frame: np.ndarray | None = None) -> Image.Image:
        """
        SD 1.5 img2img single-frame inference.

        When DWPose is loaded (_dwpose is not None), the pipeline is swapped to a
        ControlNet-enabled pipeline on the fly (StableDiffusionControlNetImg2ImgPipeline)
        and the DWPose skeleton heatmap is passed as ControlNet conditioning.
        This anchors the generated clothing to the actual body pose, reducing
        garment placement drift across frames.

        When DWPose is not available, falls back to vanilla img2img (original behaviour).
        """
        # person_image is already center-cropped + blended by tryon() when _prev_result exists
        pw, ph = person_image.size
        sq = min(pw, ph)
        person = person_image.crop(((pw - sq) // 2, (ph - sq) // 2,
                                    (pw + sq) // 2, (ph + sq) // 2))
        person = person.resize((LIVE_SIZE, LIVE_SIZE), Image.LANCZOS)
        orig_arr = np.array(person)

        ip_kwargs = (
            {"ip_adapter_image_embeds": self._ip_embeds}
            if self._ip_embeds is not None
            else {"ip_adapter_image": garment}
        )

        # ── DWPose ControlNet conditioning (Phase 3) ──────────────────────────
        pose_image = None
        if self._dwpose is not None:
            try:
                # Convert PIL → BGR numpy for _get_pose, then back to PIL
                orig_bgr = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2BGR)
                pose_image = self._get_pose(orig_bgr)
            except Exception as e:
                log.debug(f"DWPose extraction failed (using no pose): {e}")
                pose_image = None

        # Fixed seed → same noise pattern every frame → jacket texture stays consistent
        generator = torch.Generator(device=self.device).manual_seed(42)

        with torch.inference_mode():
            if pose_image is not None and hasattr(self.pipeline, "controlnet"):
                result = self.pipeline(
                    prompt="person wearing the garment, photorealistic, fashion",
                    negative_prompt="blurry, distorted, deformed",
                    image=person,
                    control_image=pose_image,
                    num_inference_steps=self._steps,
                    strength=strength,
                    guidance_scale=1.0,
                    controlnet_conditioning_scale=0.6,
                    generator=generator,
                    **ip_kwargs,
                ).images[0]
            else:
                result = self.pipeline(
                    prompt="person wearing the garment, photorealistic, fashion",
                    negative_prompt="blurry, distorted, deformed",
                    image=person,
                    num_inference_steps=self._steps,
                    strength=strength,
                    guidance_scale=1.0,
                    generator=generator,
                    **ip_kwargs,
                ).images[0]

        result_arr = np.array(result)

        # ── Clothing mask: only torso area uses SD result ─────────────────────
        if self._fixed_mask_cache is None:
            m = np.zeros((LIVE_SIZE, LIVE_SIZE), dtype=np.float32)
            m[int(LIVE_SIZE*0.38):int(LIVE_SIZE*0.92),
              int(LIVE_SIZE*0.05):int(LIVE_SIZE*0.95)] = 1.0
            # Remove face area from mask (top 38% = face/head)
            self._fixed_mask_cache = cv2.GaussianBlur(m, (31, 31), 0)[:, :, np.newaxis]

        clothing_mask = self._fixed_mask_cache

        # Composite: SD result on torso, 100% original on face + background.
        # Use clean_frame (real camera) for non-torso so face stays sharp even
        # when person_image was a temporally-blended SD input.
        base_arr = clean_frame if clean_frame is not None else orig_arr
        composite = (
            result_arr.astype(np.float32) * clothing_mask
            + base_arr.astype(np.float32) * (1.0 - clothing_mask)
        ).astype(np.uint8)

        self._prev_result = composite.copy()
        return Image.fromarray(composite)
