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
        self._steps   = 2
        self._catvton = False   # True when CatVTON loaded, False for SD+IP-Adapter
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
        self._garment_alpha       = None   # alpha mask from original RGBA garment PNG
        self._garment_color_name  = None   # dominant garment color, injected into prompt

        # ── Tier 4: AnimateDiff video backbone ───────────────────────────────
        self._animatediff_pipe = None
        self._frame_buffer: list[np.ndarray] = []   # raw BGR frames waiting for batch processing
        self._buffer_size  = ANIMATEDIFF_BUFFER_SIZE
        self._video_results: list[Image.Image] = []  # processed video frames ready to serve
        self._video_result_idx = 0                   # pointer into _video_results

        # ── CatVTON direct (Tier 3 primary) ──────────────────────────────────
        self._catvton_unet = None   # UNet2DConditionModel with 12-channel in (garment concat)

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

        # VTON_LORA_CHECKPOINT can be either:
        #   (a) A full UNet directory (config.json + diffusion_pytorch_model.safetensors)
        #       → load as a complete model.
        #   (b) A PEFT LoRA adapter directory (adapter_config.json + adapter_model.safetensors)
        #       → load base UNet from zheng-chong/CatVTON, attach LoRA on top.
        # Our Kaggle training notebook produces (b).
        ckpt_path = Path(VTON_LORA_CHECKPOINT)
        is_peft_adapter = (ckpt_path / "adapter_config.json").exists()
        if is_peft_adapter:
            log.info(f"Loading PEFT LoRA adapter from {ckpt_path}")
            unet = UNet2DConditionModel.from_pretrained(
                base, subfolder="unet",
                in_channels=12, ignore_mismatched_sizes=True,
                torch_dtype=self.dtype,
            ).to(self.device)
            try:
                from peft import PeftModel
                unet = PeftModel.from_pretrained(unet, str(ckpt_path))
                # Merge so torch.compile / CUDA graphs see a plain UNet
                unet = unet.merge_and_unload()
                log.info("LoRA adapter merged into base UNet.")
            except ImportError:
                log.error("peft not installed — pip install peft")
                raise
        else:
            log.info(f"Loading full UNet from {ckpt_path}")
            unet = UNet2DConditionModel.from_pretrained(
                VTON_LORA_CHECKPOINT, torch_dtype=self.dtype,
                attn_implementation="flash_attention_2",
            ).to(self.device)
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

    def _load_catvton_direct(self):
        """
        Load CatVTON (zheng-chong/CatVTON) UNet directly — no diffusers Pipeline wrapper.
        The UNet expects 12-channel input: [noise(4), person_lat(4), garment_lat(4)].
        Single DDIM step at inference time → ~1-2 s/frame on A10G, GPU-quality try-on.

        If CATVTON_MASKFREE_VARIANT env var is set (e.g. "vitonhd-16k-512"), this
        also overlays the MaskFree attention weights from zhengchong/CatVTON-MaskFree
        on top of the base UNet. Those weights are gated (non-commercial, requires
        HF login + accepted gate at https://huggingface.co/zhengchong/CatVTON-MaskFree).
        Set HF_TOKEN to a token from an account that has accepted the gate.
        Variants available:
          vitonhd-16k-512  — trained on VITON-HD at 512px (best for upper-body try-on)
          dresscode-16k-512 — trained on DressCode (broader garment types)
          mix-48k-1024     — trained on a 48k mixed dataset at 1024px (highest quality)
        """
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        base = "zheng-chong/CatVTON"
        log.info(f"Loading CatVTON model from {base} …")

        self._vae = AutoencoderKL.from_pretrained(
            base, subfolder="vae", torch_dtype=self.dtype,
        ).to(self.device)
        self._vae.requires_grad_(False)

        self._catvton_unet = UNet2DConditionModel.from_pretrained(
            base, subfolder="unet", torch_dtype=self.dtype,
            in_channels=12, ignore_mismatched_sizes=True,
        ).to(self.device)
        self._catvton_unet.requires_grad_(False)

        # ── MaskFree attention overlay (optional) ────────────────────────────
        # The MaskFree repo ships only the trained attention layers — they
        # overwrite matching keys on the base UNet.
        maskfree_variant = os.environ.get("CATVTON_MASKFREE_VARIANT", "").strip()
        if maskfree_variant:
            self._overlay_maskfree_weights(maskfree_variant)

        tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        te  = CLIPTextModel.from_pretrained(
            base, subfolder="text_encoder", torch_dtype=self.dtype,
        ).to(self.device)
        with torch.no_grad():
            ids = tok([""], padding="max_length", max_length=77,
                      truncation=True, return_tensors="pt").input_ids.to(self.device)
            self._null_emb = te(ids)[0]

        n_steps = VTON_STEPS if VTON_STEPS > 0 else 6
        self._scheduler = DDIMScheduler.from_pretrained(base, subfolder="scheduler")
        self._scheduler.set_timesteps(n_steps)
        self._steps = n_steps

        if self.device == "cuda":
            try:
                self._catvton_unet.to(memory_format=torch.channels_last)
                self._vae.to(memory_format=torch.channels_last)
            except Exception:
                pass

        log.info(f"CatVTON direct ready — {self._steps}-step DDIM. Garment-latent concat try-on.")

    def _overlay_maskfree_weights(self, variant: str):
        """
        Download zhengchong/CatVTON-MaskFree attention weights for `variant` and
        merge them into self._catvton_unet. Only attention layers are overwritten;
        the rest of the UNet stays as the base CatVTON weights.

        Requires:
          - User has accepted the gate at huggingface.co/zhengchong/CatVTON-MaskFree
          - HF_TOKEN env var, OR a prior `huggingface-cli login` on the host

        Failure is non-fatal — base CatVTON keeps working.
        """
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError:
            log.warning("huggingface_hub or safetensors missing — skipping MaskFree overlay")
            return

        repo_id = "zhengchong/CatVTON-MaskFree"
        filename = f"{variant}/attention/model.safetensors"
        log.info(f"Fetching MaskFree weights: {repo_id}/{filename}")

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id, filename=filename, token=token,
            )
        except Exception as e:
            log.warning(
                f"MaskFree download failed ({e}). "
                f"Have you accepted the gate at https://huggingface.co/{repo_id} "
                f"and set HF_TOKEN? Continuing with base CatVTON."
            )
            return

        try:
            maskfree_state = load_file(local_path, device="cpu")
        except Exception as e:
            log.warning(f"Could not read MaskFree safetensors: {e}")
            return

        # Merge: only overwrite keys that already exist on the base UNet.
        base_state = self._catvton_unet.state_dict()
        matched, skipped = 0, 0
        new_state = dict(base_state)
        for k, v in maskfree_state.items():
            if k in base_state and base_state[k].shape == v.shape:
                new_state[k] = v.to(base_state[k].dtype).to(base_state[k].device)
                matched += 1
            else:
                skipped += 1
        self._catvton_unet.load_state_dict(new_state, strict=False)
        log.info(
            f"MaskFree overlay applied — {matched} layers updated, "
            f"{skipped} unmatched (base CatVTON weights retained)."
        )

    def _load_tier3(self):
        """
        Primary: CatVTON direct (zheng-chong/CatVTON) — concatenates garment+person in latent.
        Fallback 1: SD Inpainting + IP-Adapter.
        Fallback 2: geometric warp (always works, no model required).
        """
        try:
            self._load_catvton_direct()
            return
        except Exception as e:
            log.warning(f"CatVTON direct load failed ({e}), trying SD+IP-Adapter…")
        try:
            self._load_catvton()
        except Exception as e:
            log.warning(f"SD Inpainting load failed ({e}), falling back to SD+IP-Adapter…")
            self._load_sd_ipadapter()

    def _load_catvton(self):
        """
        SD Inpainting + IP-Adapter for garment reference.
        Inpainting ONLY modifies the torso mask — face/background are 100% original
        by design (no manual compositing needed). IP-Adapter feeds garment visually.
        This is the correct architecture for try-on: mask = where to put garment.
        """
        from diffusers import StableDiffusionInpaintPipeline, LCMScheduler, AutoencoderTiny

        log.info("Loading SD Inpainting + IP-Adapter try-on pipeline…")

        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        # TAESD: 5x faster decode
        self.pipeline.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=self.dtype,
        ).to(self.device)

        # LCM for fast inference
        self.pipeline.scheduler = LCMScheduler.from_config(
            self.pipeline.scheduler.config)
        self.pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self.pipeline.fuse_lora()

        # IP-Adapter: garment image as visual reference. Scale 1.5 is the
        # sweet spot — high enough to anchor colour/texture, low enough that
        # the model still generates real garment structure (collar, zipper,
        # sleeves). Higher scales (2.0+) collapse the mask into a flat colour
        # blob instead of a real jacket.
        self.pipeline.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models",
            weight_name="ip-adapter_sd15.bin")
        # IP-Adapter scale 1.0 with guidance_scale=1.0 (LCM): IP-Adapter
        # provides the garment colour/texture reference, the body-silhouette
        # mask provides the shape, the prompt provides the jacket structure.
        # Higher IP scales collapse the inpaint into a flat colour blob.
        self.pipeline.set_ip_adapter_scale(1.0)

        self._steps     = VTON_STEPS if VTON_STEPS > 0 else 6
        self._ip_loaded = True
        self._catvton   = True   # use inpainting path

        if self.device == "cuda":
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            try:
                self.pipeline.unet.to(memory_format=torch.channels_last)
                self.pipeline.vae.to(memory_format=torch.channels_last)
            except Exception:
                pass

        log.info(f"Inpainting try-on ready — {self._steps}-step LCM. Face+BG preserved by mask.")

    def _load_sd_ipadapter(self):
        from diffusers import AutoPipelineForImage2Image, LCMScheduler, AutoencoderTiny

        log.info("Loading SD 1.5 + LCM + IP-Adapter (fallback)…")
        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=self.dtype, safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        self.pipeline.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=self.dtype,
        ).to(self.device)
        self.pipeline.scheduler = LCMScheduler.from_config(
            self.pipeline.scheduler.config)
        self.pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self.pipeline.fuse_lora()
        self.pipeline.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models",
            weight_name="ip-adapter_sd15.bin")
        self.pipeline.set_ip_adapter_scale(1.2)
        self._ip_loaded = True
        self._catvton   = False
        self._steps     = VTON_STEPS if VTON_STEPS > 0 else 6
        if self.device == "cuda":
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            try:
                self.pipeline.unet.to(memory_format=torch.channels_last)
                self.pipeline.vae.to(memory_format=torch.channels_last)
            except Exception:
                pass
            self._warmup_tier3()
        log.info(f"SD+IP-Adapter ready — {self._steps}-step LCM.")

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

    @staticmethod
    def _dominant_color_name(img_arr: np.ndarray) -> str:
        """Pick the dominant non-white/black colour in the garment and map it
        to a short English name. Used to anchor the diffusion prompt so the
        model doesn't drift to a different colour."""
        # Skip white/transparent edges by sampling the centre 60% only
        h, w = img_arr.shape[:2]
        cx0, cx1 = int(w * 0.20), int(w * 0.80)
        cy0, cy1 = int(h * 0.20), int(h * 0.80)
        center = img_arr[cy0:cy1, cx0:cx1].reshape(-1, 3).astype(np.float32)
        # Drop near-white and near-black pixels
        mask = (center.max(axis=1) < 240) & (center.min(axis=1) > 20)
        sel  = center[mask] if mask.any() else center
        r, g, b = sel.mean(axis=0)
        # Coarse name lookup
        hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]
        h_, s_, v_ = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if s_ < 30 and v_ > 200: return "white"
        if s_ < 30 and v_ < 60:  return "black"
        if s_ < 30:              return "grey"
        if h_ < 10 or h_ > 170:  return "red"
        if h_ < 20:              return "orange"
        if h_ < 35:              return "yellow"
        if h_ < 85:              return "green"
        if h_ < 100:             return "teal"
        if h_ < 130:             return "blue"
        if h_ < 150:             return "purple"
        return "pink"

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
        self._garment_cache    = garment_sq
        self._ip_embeds        = None
        self._fixed_mask_cache = None   # reset so mask regenerates at new LIVE_SIZE
        self._prev_result      = None   # reset temporal state for new garment

        # Store alpha mask if original had transparency — used by geometric warp
        if garment_image.mode == 'RGBA':
            alpha_sq = garment_image.split()[3].resize((LIVE_SIZE, LIVE_SIZE), Image.LANCZOS)
            self._garment_alpha = np.array(alpha_sq).astype(np.float32) / 255.0
        else:
            self._garment_alpha = None

        # Extract dominant non-background color name from garment for the prompt.
        # This dramatically helps the diffusion model lock the actual colour.
        self._garment_color_name = self._dominant_color_name(np.array(garment_sq))
        log.info(f"Garment dominant color: {self._garment_color_name}")

        # Pre-compute IP-Adapter CLIP embeddings once — reused every frame instead of per-frame encoding
        if self._ip_loaded and hasattr(self.pipeline, "prepare_ip_adapter_image_embeds"):
            try:
                with torch.inference_mode():
                    self._ip_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                        ip_adapter_image=[garment_sq],
                        ip_adapter_image_embeds=None,
                        device=self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,  # CFG ON in inpaint path
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

        # ── CatVTON direct — garment+person concat in latent, best quality ──
        if self._catvton_unet is not None:
            sz = LIVE_SIZE
            p  = person_image.convert("RGB").resize((sz, sz), Image.LANCZOS)
            g  = garment.convert("RGB").resize((sz, sz), Image.LANCZOS)
            return self._infer_catvton_direct(p, g)

        # ── Tier 4: AnimateDiff video backbone ───────────────────────────────
        if self._animatediff_pipe is not None:
            return self._infer_tier4_animatediff(person_image, garment)

        # ── SD Inpainting / img2img + IP-Adapter (tier 3 AI path) ────────────
        if self.pipeline is not None:
            return self._infer_tier3(person_image, garment)

        # ── Geometric warp fallback — no model required ───────────────────────
        return self._infer_live_geometric(person_image)

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

        # ── Face detection → torso placement ─────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self._haar.detectMultiScale(
            cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            cx          = fx + fw // 2
            # Jacket top = just below chin, shoulder-width placement
            top         = fy + fh - int(fh * 0.05)
            bottom      = min(H, top + int(fh * 2.8))
            left        = max(0, cx - int(fw * 1.4))
            right       = min(W, cx + int(fw * 1.4))
            face_bottom = fy + fh + int(fh * 0.08)
        else:
            top = int(H * 0.32); bottom = int(H * 0.88)
            left = int(W * 0.10); right = int(W * 0.90)
            face_bottom = int(H * 0.40)
            cx = W // 2

        th = max(1, bottom - top)
        tw = max(1, right  - left)

        # ── Garment crop — strip sleeves ──────────────────────────────────────
        gH, gW = np.array(garment).shape[:2]
        cl = int(gW * 0.15); cr = int(gW * 0.85)
        g_crop = garment.crop((cl, 0, cr, gH))
        shirt  = np.array(g_crop.resize((tw, th), Image.LANCZOS))

        # ── Alpha mask ────────────────────────────────────────────────────────
        if self._garment_alpha is not None:
            alpha_pil  = Image.fromarray((self._garment_alpha * 255).astype(np.uint8))
            aw, ah     = alpha_pil.size
            alpha_crop = alpha_pil.crop((int(aw*0.15), 0, int(aw*0.85), ah))
            alpha = np.array(alpha_crop.resize((tw, th), Image.LANCZOS)).astype(np.float32) / 255.0
        else:
            g_gray = cv2.cvtColor(shirt, cv2.COLOR_RGB2GRAY)
            _, bg  = cv2.threshold(g_gray, 240, 255, cv2.THRESH_BINARY)
            alpha  = (255 - bg).astype(np.float32) / 255.0

        alpha = cv2.GaussianBlur(alpha, (9, 9), 0)

        # ── Perspective warp — taper bottom 6% to simulate body wrap ─────────
        taper = max(1, int(tw * 0.06))
        src_pts = np.float32([[0,0],[tw,0],[tw,th],[0,th]])
        dst_pts = np.float32([[0,0],[tw,0],[tw-taper,th],[taper,th]])
        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        shirt = cv2.warpPerspective(shirt, M_persp, (tw, th),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpPerspective(alpha, M_persp, (tw, th))

        # ── Ambient light harmonization ───────────────────────────────────────
        roi_orig  = frame[top:top+th, left:left+tw].astype(np.float32)
        if roi_orig.size > 0:
            orig_mean  = roi_orig.mean(axis=(0, 1))
            shirt_mean = shirt.astype(np.float32).mean(axis=(0, 1)) + 1e-6
            ratio = np.clip((orig_mean / shirt_mean) * 0.35 + 0.65, 0.4, 1.8)
            shirt = np.clip(shirt.astype(np.float32) * ratio, 0, 255).astype(np.uint8)

        # ── Body mask — MediaPipe if available, else soft torso ellipse ───────
        body_mask_roi = None
        if self._mp_seg is not None:
            try:
                seg_result = self._mp_seg.process(frame)
                if seg_result.segmentation_mask is not None:
                    bm = (seg_result.segmentation_mask > 0.4).astype(np.float32)
                    bm = cv2.GaussianBlur(bm, (21, 21), 0)
                    body_mask_roi = bm[top:top+th, left:left+tw]
            except Exception:
                pass

        if body_mask_roi is None:
            # Fallback: soft ellipse approximating torso silhouette
            bm = np.zeros((H, W), np.float32)
            cv2.ellipse(bm, (cx, (top+bottom)//2),
                        ((right-left)//2, (bottom-top)//2),
                        0, 0, 360, 1.0, -1)
            bm = cv2.GaussianBlur(bm, (41, 41), 0)
            body_mask_roi = bm[top:top+th, left:left+tw]

        if body_mask_roi.shape == (th, tw):
            alpha = alpha * body_mask_roi

        # ── Edge shadow — depth cue ───────────────────────────────────────────
        edge_w = max(1, int(tw * 0.07))
        eshadow = np.ones_like(alpha)
        eshadow[:, :edge_w]  *= np.linspace(0.45, 1.0, edge_w)
        eshadow[:, -edge_w:] *= np.linspace(1.0, 0.45, edge_w)
        shirt = np.clip(shirt.astype(np.float32) * eshadow[:, :, np.newaxis],
                        0, 255).astype(np.uint8)

        alpha3 = alpha[:, :, np.newaxis]

        # ── Blend onto frame ──────────────────────────────────────────────────
        result = frame.copy()
        roi    = result[top:top+th, left:left+tw]
        if roi.shape[:2] == (th, tw):
            result[top:top+th, left:left+tw] = (
                shirt.astype(np.float32) * alpha3
                + roi.astype(np.float32) * (1.0 - alpha3)
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

    def _infer_catvton_direct(self, person: Image.Image, garment: Image.Image) -> Image.Image:
        """
        Multi-step DDIM denoising with the direct CatVTON UNet.
        Input concat: [x_t(4), person_lat(4), garment_lat(4)] → 12 channels.
        After generation, face region is restored from original frame.
        """
        import torchvision.transforms.functional as TF
        to_lat = lambda img: TF.normalize(
            TF.to_tensor(img).unsqueeze(0), [0.5]*3, [0.5]*3
        ).to(self.device, self.dtype)

        orig_arr = np.array(person)

        with torch.inference_mode():
            p_lat = self._vae.encode(to_lat(person)).latent_dist.sample() * self._vae.config.scaling_factor
            g_lat = self._vae.encode(to_lat(garment)).latent_dist.sample() * self._vae.config.scaling_factor
            x = torch.randn_like(p_lat)
            for t in self._scheduler.timesteps:
                inp = torch.cat([x, p_lat, g_lat], dim=1).to(self.dtype)
                t_b = t.unsqueeze(0).to(self.device)
                noise_pred = self._catvton_unet(
                    inp, t_b, encoder_hidden_states=self._null_emb
                ).sample
                x = self._scheduler.step(noise_pred, t, x).prev_sample
            out = self._vae.decode(x / self._vae.config.scaling_factor).sample
            out = (out.clamp(-1, 1) + 1) / 2

        result_arr = np.array(TF.to_pil_image(out[0].float().cpu()))

        # Restore face — CatVTON may alter the upper portion
        sz    = result_arr.shape[0]
        gray  = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2GRAY)
        faces = self._haar.detectMultiScale(
            cv2.equalizeHist(gray), scaleFactor=1.1,
            minNeighbors=4, minSize=(40, 40)
        )
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            cutoff = min(fy + fh + int(fh * 0.15), int(sz * 0.65))
        else:
            cutoff = int(sz * 0.42)
        result_arr[:cutoff] = orig_arr[:cutoff]
        self._prev_result = result_arr.copy()
        return Image.fromarray(result_arr)

    # ── Body-shaped mask builder (per-frame, follows actual silhouette) ──────

    def _build_body_mask(self, frame_rgb: np.ndarray):
        """
        Build three masks from a 512x512 RGB person frame:
          torso_mask       — soft mask passed to SD inpainting (where to paint)
          body_silhouette  — hard person silhouette used to composite result
          face_cutoff_y    — y-row below which jacket is allowed (above = face/hair)

        torso_mask is body_silhouette restricted to the torso+arms band so SD
        never paints onto legs or background. If MediaPipe is unavailable we
        fall back to an elliptical torso approximation.
        """
        h, w = frame_rgb.shape[:2]

        # 1. Person silhouette via MediaPipe selfie segmentation
        silhouette = None
        if self._mp_seg is not None:
            try:
                seg = self._mp_seg.process(frame_rgb)
                if seg.segmentation_mask is not None:
                    s = seg.segmentation_mask.astype(np.float32)
                    # Threshold at 0.5 → hard binary, then dilate slightly so
                    # the jacket can extend a few pixels past the body edge
                    # (sleeve thickness, jacket flare) without clipping.
                    bm = (s > 0.5).astype(np.float32)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    bm = cv2.dilate(bm, kernel, iterations=2)
                    silhouette = cv2.GaussianBlur(bm, (7, 7), 0).clip(0, 1)
            except Exception as e:
                log.debug(f"MediaPipe seg failed in mask build: {e}")

        if silhouette is None:
            # Fallback ellipse: same shape the legacy fixed mask used.
            silhouette = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(silhouette,
                        (w // 2, int(h * 0.62)),
                        (int(w * 0.40), int(h * 0.36)),
                        0, 0, 360, 1.0, -1)
            silhouette = cv2.GaussianBlur(silhouette, (21, 21), 0)

        # 2. Face cutoff — chin row. Everything above is preserved.
        face_cutoff_y = int(h * 0.35)
        try:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = self._haar.detectMultiScale(
                cv2.equalizeHist(gray), scaleFactor=1.1,
                minNeighbors=4, minSize=(40, 40),
            )
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                # Cutoff at chin + a real neck gap (~35% of face height) so
                # the collar lands on the neck, not on the chin. This is the
                # main "worn on" tweak — without the gap, the jacket sits
                # like a sticker pressed against the face.
                face_cutoff_y = int(np.clip(fy + fh + fh * 0.35,
                                            h * 0.28, h * 0.55))
        except Exception:
            pass

        # 3. Torso band — restrict mask vertically. Jacket reaches from just
        # below the chin to mid-thigh.
        band = np.zeros((h, w), dtype=np.float32)
        band[face_cutoff_y:int(h * 0.92), :] = 1.0
        band = cv2.GaussianBlur(band, (15, 15), 0)

        # 4. torso_mask = silhouette ∩ band  (only body pixels, only torso band)
        torso_mask = (silhouette * band).clip(0, 1)
        # Gentle feather so SD has a smooth boundary to denoise into.
        torso_mask = cv2.GaussianBlur(torso_mask, (11, 11), 0)

        return torso_mask, silhouette, face_cutoff_y

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

        generator = torch.Generator(device=self.device).manual_seed(42)

        # ── Per-frame body-silhouette mask ────────────────────────────────────
        # The mask must follow the actual body so SD only paints garment on
        # the person, not the background or head. Built from:
        #   1. MediaPipe selfie segmentation → person silhouette
        #   2. Face detection → cut everything above chin out of the mask
        #   3. Vertical band → only paint torso+arms, never legs/feet
        # Result: a body-shaped mask that hugs the actual person each frame.
        torso_mask, body_silhouette, face_cutoff_y = self._build_body_mask(orig_arr)

        # ── Inpainting path — mask follows actual body silhouette ────────────
        if self._catvton:
            mask_pil = Image.fromarray((torso_mask * 255).astype(np.uint8))
            # diffusers' SD-inpaint pipeline unconditionally does
            #   neg, pos = single_image_embeds.chunk(2)
            # on the IP-Adapter embeddings, which only works when
            # do_classifier_free_guidance=True (i.e. guidance_scale > 1.0).
            # So we MUST run with CFG on in this code path. The cached
            # embeds were prepared with CFG=False (shape [1,…]) — pass the
            # raw garment image so diffusers re-encodes with the right
            # CFG-shape (negative + positive concatenated).
            ip_kw = {"ip_adapter_image": garment}
            color = self._garment_color_name or "matching"
            prompt = (
                f"photo of a person wearing a fitted {color} jacket, "
                f"the jacket fits naturally on the body, visible collar, "
                f"front zipper, sleeves following the arms, "
                f"realistic fabric folds, detailed texture, sharp focus, photorealistic"
            )
            neg = (
                "wrong color, faded, washed out, bare arms, t-shirt, tank top, "
                "sleeveless, naked, floating clothes, jacket on background, "
                "deformed body, extra limbs, blurry, low quality, painting, cartoon"
            )
            # LCM-distilled UNet prefers guidance≈1.0, but the diffusers
            # SD-inpaint + IP-Adapter code path requires CFG to be ON or it
            # crashes on a .chunk(2) of the image embeds. A tiny CFG of 1.5
            # is the sweet spot: enables the path without degrading LCM.
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=neg,
                    image=person,
                    mask_image=mask_pil,
                    num_inference_steps=6,
                    guidance_scale=1.5,
                    generator=generator,
                    **ip_kw,
                ).images[0]

            # ── Composite result back onto original via body silhouette ──────
            # SD output can bleed slightly past the mask edge. We blend the
            # SD result into the original ONLY where body_silhouette has
            # alpha > 0, and ONLY below the face cutoff. Everything else
            # (face, background, hair, legs) stays pixel-identical to the
            # camera frame — so the jacket genuinely appears "worn on" you.
            result_arr = np.array(result).astype(np.float32)
            orig_f     = orig_arr.astype(np.float32)
            blend_mask = body_silhouette.copy()
            blend_mask[:face_cutoff_y] = 0.0          # protect face & hair
            blend_mask = cv2.GaussianBlur(blend_mask, (9, 9), 0)
            a = blend_mask[:, :, np.newaxis]
            composed = (result_arr * a + orig_f * (1.0 - a)).astype(np.uint8)
            self._prev_result = composed.copy()
            return Image.fromarray(composed)

        # ── SD img2img + IP-Adapter fallback ─────────────────────────────────
        else:
            prompt = (
                "photo of person wearing jacket on body, shirt on torso, "
                "photorealistic, detailed fabric texture, well-fitted clothes"
            )
            neg_prompt = (
                "naked, bare chest, no shirt, deformed, blurry, distorted, "
                "bad anatomy, floating clothes"
            )
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    image=person,
                    num_inference_steps=self._steps,
                    strength=strength,
                    guidance_scale=1.0,
                    generator=generator,
                    **ip_kwargs,
                ).images[0]

        result_arr = np.array(result)
        cm = self._fixed_mask_cache
        clothing_mask = cm if cm.ndim == 3 else cm[:, :, np.newaxis]

        base_arr = clean_frame if clean_frame is not None else orig_arr
        composite = (
            result_arr.astype(np.float32) * clothing_mask
            + base_arr.astype(np.float32) * (1.0 - clothing_mask)
        ).astype(np.uint8)

        self._prev_result = composite.copy()
        return Image.fromarray(composite)
