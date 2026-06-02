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

# When set, skips every AI tier and always uses the geometric warp.
# Default OFF — AI inpaint with strong colour anchor tends to win on
# realism. Set TRYON_FORCE_GEOMETRIC=1 to flip back to the deterministic
# overlay if the AI path is misbehaving.
TRYON_FORCE_GEOMETRIC     = os.environ.get("TRYON_FORCE_GEOMETRIC", "0") == "1"

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
        # MediaPipe selfie segmentation — separates person from background precisely.
        # Also load Hands detector so we can carve hand regions OUT of the garment
        # mask. Without this, a hand crossing in front of the camera gets painted
        # over with the garment (user-reported "hand crosses → painted as jacket").
        try:
            import mediapipe as mp
            # MediaPipe 0.10.30+ on Python 3.14 makes `mp.solutions` lazy and
            # `hasattr(mp, 'solutions')` returns False until the submodule
            # is explicitly imported. Force-import each solution submodule
            # so the namespace exists. If any of these imports fails, the
            # whole block falls through to the fallback.
            from mediapipe.python.solutions import selfie_segmentation as _mp_ss
            from mediapipe.python.solutions import face_detection      as _mp_fd
            from mediapipe.python.solutions import hands               as _mp_hd
            self._mp_seg  = _mp_ss.SelfieSegmentation(model_selection=1)
            self._mp_face = _mp_fd.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            # Hands: detect up to 2 hands, lower confidence so a partial /
            # blurry hand crossing still gets picked up. Performance-mode
            # model (model_complexity=0) is ~5-7 ms / frame on CPU.
            self._mp_hands = _mp_hd.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            log.info("MediaPipe loaded (seg + face + hands).")
        except Exception as e:
            self._mp_seg   = None
            self._mp_face  = None
            self._mp_hands = None
            log.warning(f"MediaPipe not available, using fallback: {e}")
        self._prev_result      = None
        self._prev_silhouette  = None      # smoothed MediaPipe silhouette (per-pixel EMA)
        self._prev_face_bbox   = None      # EMA-smoothed Haar face bbox (fx, fy, fw, fh)
        self._prev_torso_mask  = None      # EMA-smoothed final torso_mask (kills jitter)
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
        # IP-Adapter scale 1.2 paired with CFG 2.5 (see _infer_tier3). The
        # IP-Adapter back to 1.2 — the working value that produced clean
        # jacket results in the user's May 26 screenshot. 1.5 was tried
        # for stronger garment lock and produced over-saturated garbage
        # output (rainbow stripes / random colors). Reverted to known good.
        self.pipeline.set_ip_adapter_scale(1.2)

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
        # Bumped 1.2 -> 1.5 for the manager demo: stronger IP-Adapter
        # IP-Adapter 1.2 — known good value, see comment above.
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

    def set_garment(self, garment_image: Image.Image, garment_type: str = "tshirt"):
        self._garment_type = (
            garment_type if garment_type in ("tshirt", "shirt", "jacket") else "tshirt"
        )
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
        self._prev_silhouette  = None
        self._prev_face_bbox   = None
        self._prev_torso_mask  = None

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

        # Geometric override: skip every AI tier when the operator wants
        # the deterministic, lag-free overlay (TRYON_FORCE_GEOMETRIC=1).
        if TRYON_FORCE_GEOMETRIC:
            return self._infer_live_geometric(person_image)

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

    # ── Helpers: hand + skin exclusion ────────────────────────────────────────

    def _hand_exclusion_mask(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Build a soft mask where hands are present so we can KEEP the user's
        skin pixels (not paint garment over them).

        Combines two signals:
          1. MediaPipe Hands landmarks → bounding polygon for each detected hand,
             dilated so it covers the full hand silhouette + a bit of wrist.
          2. YCrCb skin-tone detection inside the same ROI (catches forearm
             skin that MediaPipe's landmarks don't cover).

        Returns: float32 mask of shape (H, W), values in [0, 1] where 1 means
        "this is a hand/arm, KEEP original". Or None if MediaPipe is unavailable
        and skin detection finds nothing meaningful.
        """
        H, W = frame_rgb.shape[:2]
        hand_mask = np.zeros((H, W), np.float32)

        # 1. MediaPipe hand landmarks → convex hull per hand
        if self._mp_hands is not None:
            try:
                res = self._mp_hands.process(frame_rgb)
                if res.multi_hand_landmarks:
                    for hand_lmk in res.multi_hand_landmarks:
                        pts = np.array(
                            [[int(lm.x * W), int(lm.y * H)] for lm in hand_lmk.landmark],
                            dtype=np.int32,
                        )
                        if len(pts) >= 3:
                            hull = cv2.convexHull(pts)
                            cv2.fillPoly(hand_mask, [hull], 1.0)
            except Exception:
                pass

        # 2. YCrCb skin-tone secondary signal. Restricted to the lower body
        # half so we don't accidentally pull face skin into the exclusion.
        try:
            ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
            # Standard skin gamut on YCrCb
            lower = np.array([0, 133, 77], dtype=np.uint8)
            upper = np.array([255, 173, 127], dtype=np.uint8)
            skin = cv2.inRange(ycrcb, lower, upper).astype(np.float32) / 255.0
            # Cut the top 40% of the frame (face area) — we never want to
            # exclude face skin from the garment region, only arms/hands.
            skin[: int(H * 0.40)] = 0.0
            # Dilate hand_mask region a bit, then OR with skin signal LIMITED
            # to inside the dilated hand neighborhood. This keeps the skin
            # signal scoped — random skin-colored objects elsewhere in the
            # frame won't trigger.
            if hand_mask.max() > 0:
                hand_dil = cv2.dilate(
                    hand_mask, np.ones((45, 45), np.uint8), iterations=2
                )
                hand_mask = np.maximum(hand_mask, skin * (hand_dil > 0))
        except Exception:
            pass

        if hand_mask.max() < 0.05:
            return None

        # Dilate so the mask covers the full hand thickness, then soften
        hand_mask = cv2.dilate(hand_mask, np.ones((15, 15), np.uint8), iterations=1)
        hand_mask = cv2.GaussianBlur(hand_mask, (31, 31), 0)
        return np.clip(hand_mask, 0.0, 1.0)


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
            # Shirt top = just below the chin (≈18% face-height neck gap)
            # so the collar lands on the neck, not on the jaw. Width = 2× the
            # face width — that's roughly shoulder span for a forward-facing
            # subject.
            # Shirt top sits a small neck gap below the chin so the collar
            # lands on the neck. Width = ≈2.7× face width (≈ realistic
            # shoulder span). Body silhouette will clip whatever extends
            # past the actual body.
            top         = fy + fh + int(fh * 0.15)
            bottom      = min(H, top + int(fh * 3.0))
            left        = max(0, cx - int(fw * 1.35))
            right       = min(W, cx + int(fw * 1.35))
            face_bottom = fy + fh + int(fh * 0.10)
        else:
            top = int(H * 0.32); bottom = int(H * 0.88)
            left = int(W * 0.10); right = int(W * 0.90)
            face_bottom = int(H * 0.40)
            cx = W // 2

        th = max(1, bottom - top)
        tw = max(1, right  - left)

        # ── Garment crop — strip a small margin so transparent edges of
        # the PNG don't bleed; keep enough of the shirt body that the collar
        # is still visible.
        gH, gW = np.array(garment).shape[:2]
        cl = int(gW * 0.08); cr = int(gW * 0.92)
        g_crop = garment.crop((cl, 0, cr, gH))
        shirt  = np.array(g_crop.resize((tw, th), Image.LANCZOS))

        # ── Alpha mask ────────────────────────────────────────────────────────
        if self._garment_alpha is not None:
            alpha_pil  = Image.fromarray((self._garment_alpha * 255).astype(np.uint8))
            aw, ah     = alpha_pil.size
            alpha_crop = alpha_pil.crop((int(aw*0.08), 0, int(aw*0.92), ah))
            alpha = np.array(alpha_crop.resize((tw, th), Image.LANCZOS)).astype(np.float32) / 255.0
        else:
            # Treat only nearly-pure-white as background; threshold at 250
            # so light-grey shirts aren't accidentally cut out. Also clean
            # up isolated speckles with a small morphological close.
            g_gray = cv2.cvtColor(shirt, cv2.COLOR_RGB2GRAY)
            _, bg  = cv2.threshold(g_gray, 250, 255, cv2.THRESH_BINARY)
            alpha  = (255 - bg).astype(np.float32) / 255.0
            alpha  = cv2.morphologyEx(
                alpha, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )

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

        # ── Hand / arm exclusion ─────────────────────────────────────────────
        # If a hand crosses in front of the user's torso, we want the original
        # hand to stay visible (not get painted with garment). The exclusion
        # mask multiplies (1 - hand) into the alpha so hand pixels go alpha=0.
        hand_full = self._hand_exclusion_mask(frame)
        if hand_full is not None:
            hand_roi = hand_full[top:top + th, left:left + tw]
            if hand_roi.shape == (th, tw):
                alpha = alpha * (1.0 - hand_roi)

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
        self._prev_silhouette  = None
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
                    # Threshold at 0.6 (was 0.5) for a tighter person edge.
                    # Single iteration of dilation gives the jacket just
                    # enough room for sleeve thickness without producing
                    # the ghost outline we were seeing past the body.
                    # Threshold 0.25 (was 0.4 → 0.6 originally) catches
                    # the soft edges where arms / hands fade into the
                    # background. User reported background visible past
                    # jacket edges and arms not painted; the mask was
                    # cutting too tight.
                    bm = (s > 0.25).astype(np.float32)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    # 5 dilation iterations gives the silhouette enough
                    # bulk for full shoulder + arm + hem coverage.
                    bm = cv2.dilate(bm, kernel, iterations=5)
                    bm = cv2.GaussianBlur(bm, (5, 5), 0).clip(0, 1)
                    # Per-pixel temporal EMA: damps the 1-2 px shimmer
                    # MediaPipe produces per frame. alpha=0.6 on the new
                    # frame is faster than the result EMA (0.45) so the
                    # mask still tracks body motion; the 40% prev term
                    # eliminates edge wobble that would otherwise show
                    # once the final alpha is tightened to (3,3).
                    if (self._prev_silhouette is not None
                            and self._prev_silhouette.shape == bm.shape):
                        bm = (0.6 * bm + 0.4 * self._prev_silhouette).clip(0, 1)
                    self._prev_silhouette = bm
                    silhouette = bm
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

        # Safety: OR a generous bbox-derived rectangle covering the
        # expected shoulders + arms + torso area. This guarantees the
        # silhouette never undershoots even when MediaPipe is conservative
        # on a frame. Anchored to a Haar face detection here (inline so
        # this block doesn't depend on the face_cutoff section below).
        try:
            safety = np.zeros((h, w), dtype=np.float32)
            gray_for_safety = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            safety_faces = self._haar.detectMultiScale(
                cv2.equalizeHist(gray_for_safety), scaleFactor=1.1,
                minNeighbors=4, minSize=(40, 40),
            )
            # If Haar fails on this frame (user too close / tilt / motion
            # blur), reuse the last successful bbox instead of falling
            # back to the wide horizontal band. Without this fallback,
            # one missed-detection frame painted a giant teal rectangle
            # because the fallback band is 64% of frame width.
            if len(safety_faces) == 0 and self._prev_face_bbox is not None:
                pfx, pfy, pfw, pfh = self._prev_face_bbox
                safety_faces = np.array([[pfx, pfy, pfw, pfh]])
            if len(safety_faces) > 0:
                fx2, fy2, fw2, fh2 = max(safety_faces, key=lambda r: r[2] * r[3])
                # EMA on bbox: 70% prev + 30% new. Haar wobbles by 2-3 px
                # per frame which propagates to the safety rect and makes
                # the painted garment jitter / drift / ghost on the body
                # (the "jacket not stable" the user reported when MediaPipe
                # was unavailable). Heavy prev weight locks the rect.
                if self._prev_face_bbox is not None:
                    pfx, pfy, pfw, pfh = self._prev_face_bbox
                    fx2 = int(0.7 * pfx + 0.3 * fx2)
                    fy2 = int(0.7 * pfy + 0.3 * fy2)
                    fw2 = int(0.7 * pfw + 0.3 * fw2)
                    fh2 = int(0.7 * pfh + 0.3 * fh2)
                self._prev_face_bbox = (fx2, fy2, fw2, fh2)
                cx2 = fx2 + fw2 // 2
                # Per-garment safety rectangle. Jacket UNTOUCHED — kept
                # exactly at the 'jacket done' values (width 4x face,
                # height chin → frame bottom) because the user confirmed
                # jacket renders perfectly with these. T-shirt and shirt
                # are narrower / shorter than a jacket; using the jacket
                # rectangle for them left an empty band SD painted as a
                # dark rectangle ("black square behind garment"). Narrow
                # widths kill that band.
                # Per-garment polygon. Jacket gets the proven 'jacket done'
                # width (2x face). T-shirt and shirt are slightly narrower
                # so the polygon doesn't extend past the body into the
                # background — that was leaving a blue/teal rectangle
                # visible behind the user (user: "yeh blue rectangle
                # hatado"). Skin-tone hand extension below still gives
                # arm/hand coverage on movement.
                gtype  = getattr(self, "_garment_type", "tshirt")
                if gtype == "jacket":
                    half_w     = fw2 * 2
                    rect_bottom = h
                elif gtype == "shirt":
                    half_w     = int(fw2 * 1.65)
                    rect_bottom = int(h * 0.88)
                else:  # tshirt
                    half_w     = int(fw2 * 1.50)
                    rect_bottom = int(h * 0.80)
                rect_top   = fy2 + fh2                         # chin row
                rect_left  = max(0, cx2 - half_w)
                rect_right = min(w, cx2 + half_w)
                # Per-garment collar geometry on the top edge. Notch
                # depths bumped UP because the 31px Gaussian below
                # softens them — shallower notches were dissolving into
                # a flat horizontal line.
                if gtype == "tshirt":
                    neck_dip  = int(fh2 * 0.55)
                    neck_half = int(fw2 * 0.65)
                elif gtype == "shirt":
                    neck_dip  = int(fh2 * 0.40)
                    neck_half = int(fw2 * 0.45)
                else:  # jacket
                    neck_dip  = int(fh2 * 0.20)
                    neck_half = int(fw2 * 0.35)
                neck_l = max(0, cx2 - neck_half)
                neck_r = min(w, cx2 + neck_half)
                poly = np.array(
                    [[rect_left, rect_top],
                     [neck_l, rect_top],
                     [cx2, rect_top + neck_dip],
                     [neck_r, rect_top],
                     [rect_right, rect_top],
                     [rect_right, rect_bottom],
                     [rect_left, rect_bottom]],
                    dtype=np.int32,
                )
                cv2.fillPoly(safety, [poly], 1.0)
            else:
                # No face → soft tapered band (narrower than full width)
                cv2.rectangle(safety, (int(w * 0.18), int(h * 0.32)),
                              (int(w * 0.82), int(h * 0.92)), 1.0, -1)
            # 31px feather — heavy enough to soften corners but small
            # enough to preserve the per-garment U/V neckline notch
            # carved into the polygon top. 71px was dissolving the notch
            # into a flat horizontal line at the chin.
            safety = cv2.GaussianBlur(safety, (31, 31), 0).clip(0, 1)
            silhouette = np.maximum(silhouette, safety * 0.85)

            # ── Skin-tone hands extension (no MediaPipe needed) ────────
            # Detect skin pixels via YCrCb (works for all skin tones)
            # BELOW the face_cutoff_y line (so face skin is excluded).
            # OR these into the silhouette so the painted garment can
            # extend onto raised hands / arms. Without this the polygon
            # only covers the chest band and hands stay un-painted when
            # the user moves them (user: "hands cover with movements").
            try:
                ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
                skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
                skin = skin.astype(np.float32) / 255.0
                # Restrict to below the chin so we don't catch the face.
                chin_y = int(fy2 + fh2 * 0.95) if len(safety_faces) > 0 else int(h * 0.30)
                skin_mask = np.zeros_like(skin)
                skin_mask[chin_y:, :] = skin[chin_y:, :]
                # Clean noise, then dilate so the hand/arm region is a
                # solid blob rather than skin-texture speckle.
                skin_mask = cv2.morphologyEx(
                    skin_mask, cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                )
                skin_mask = cv2.dilate(
                    skin_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                    iterations=2,
                )
                skin_mask = cv2.GaussianBlur(skin_mask, (21, 21), 0).clip(0, 1)
                silhouette = np.maximum(silhouette, skin_mask * 0.80)
            except Exception as e:
                log.debug(f"skin extension skipped: {e}")
        except Exception as e:
            log.debug(f"safety rect skipped: {e}")

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
                # Cutoff right at the chin row (fy + fh). Tried higher
                # (inside jaw) → paint on face. Tried lower (1.05*fh) →
                # collar dropped to mid-chest with a wallpaper gap above.
                # Exact chin = collar sits at the neck like a real
                # crew-neck (user: "collar sahi karo sabka").
                face_cutoff_y = int(np.clip(fy + fh,
                                            h * 0.20, h * 0.48))
        except Exception:
            pass

        # 3. Torso band — restrict mask vertically. Extended bottom to
        # 0.98 (was 0.92) so the jacket reaches the bottom of the frame
        # rather than cutting off at mid-thigh leaving a visible hem oval.
        band = np.zeros((h, w), dtype=np.float32)
        band[face_cutoff_y:int(h * 0.98), :] = 1.0
        band = cv2.GaussianBlur(band, (15, 15), 0)

        # 4. torso_mask = silhouette ∩ band  (only body pixels, only torso band)
        torso_mask = (silhouette * band).clip(0, 1)

        # ── GrabCut body extraction: garment ONLY on body, not behind ───
        # User: "background impaint kyu kr rhe ho jab sirf tshrt body p
        # aani h". Polygon alone extends past the body into background;
        # GrabCut takes the polygon as "probable foreground", the corners
        # as "definite background", and the face area as "definite
        # foreground" — then finds the real body silhouette. Downscale
        # to 256x256 for speed (iterCount=1 ≈ 30 ms at that size).
        try:
            small_size = 256
            small_rgb  = cv2.resize(frame_rgb, (small_size, small_size))
            small_poly = cv2.resize(safety, (small_size, small_size))
            gc_mask = np.full((small_size, small_size),
                              cv2.GC_PR_BGD, dtype=np.uint8)
            gc_mask[small_poly > 0.30] = cv2.GC_PR_FGD
            # Definite background: 25 px at each corner.
            bc = 25
            gc_mask[:bc, :bc]  = cv2.GC_BGD
            gc_mask[:bc, -bc:] = cv2.GC_BGD
            gc_mask[-bc:, :bc] = cv2.GC_BGD
            gc_mask[-bc:, -bc:] = cv2.GC_BGD
            # Definite foreground: face bbox + a chest band BELOW the
            # face. The chest band guarantees GrabCut keeps the neck/
            # upper chest as foreground — without it the neck-skin region
            # sometimes got classified as background and the painted
            # garment dropped to mid-chest with a visible gap above.
            if len(safety_faces) > 0:
                s = small_size / float(h)
                sfx = int(fx2 * s); sfy = int(fy2 * s)
                sfw = int(fw2 * s); sfh = int(fh2 * s)
                gc_mask[sfy:sfy+sfh, sfx:sfx+sfw] = cv2.GC_FGD
                # Chest stripe: half-face-width wide, from chin down 1.5x
                # face height. Definite foreground hint.
                cx_s = sfx + sfw // 2
                stripe_l = max(0, cx_s - sfw // 2)
                stripe_r = min(small_size, cx_s + sfw // 2)
                stripe_t = sfy + sfh
                stripe_b = min(small_size, stripe_t + int(sfh * 1.5))
                gc_mask[stripe_t:stripe_b, stripe_l:stripe_r] = cv2.GC_FGD
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(small_rgb, gc_mask, None, bgd_model, fgd_model,
                        iterCount=1, mode=cv2.GC_INIT_WITH_MASK)
            body = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                1.0, 0.0
            ).astype(np.float32)
            body = cv2.resize(body, (w, h), interpolation=cv2.INTER_LINEAR)
            body = cv2.dilate(
                body,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            body = cv2.GaussianBlur(body, (15, 15), 0).clip(0, 1)
            torso_mask = torso_mask * body
        except Exception as e:
            log.debug(f"grabcut body extraction skipped: {e}")

        # Gentle feather so SD has a smooth boundary to denoise into.
        torso_mask = cv2.GaussianBlur(torso_mask, (11, 11), 0)
        # Per-pixel EMA on the final mask (0.65 new + 0.35 prev). Even
        # with bbox EMA above, the silhouette / safety blur can still
        # produce 1-2 px boundary wobble between frames; the painted
        # jacket inherits that wobble and looks unstable on the body.
        # Mask EMA glues it down.
        if (self._prev_torso_mask is not None
                and self._prev_torso_mask.shape == torso_mask.shape):
            torso_mask = (0.65 * torso_mask + 0.35 * self._prev_torso_mask).clip(0, 1)
        self._prev_torso_mask = torso_mask

        # NOTE: hand exclusion was tried here (subtracting MediaPipe Hands +
        # YCrCb skin from torso_mask) but the skin detector was matching
        # the user's brown shirt and dark skin tones across the entire
        # torso, leaving SD inpaint with no area to paint. Result: the
        # output frame looked identical to input (jacket never appeared).
        # Removed from the AI tier mask builder. The geometric path still
        # uses _hand_exclusion_mask separately on the warped alpha — that
        # is safer because it only kills the garment alpha, not the SD
        # paint region.

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

            # ── Colour anchor: prefill the masked torso with the garment's
            # mean colour BEFORE inpainting. The pipeline starts denoising
            # from this seed, so SD has a much stronger pull toward the
            # right colour than from prompt + IP-Adapter alone. This is the
            # cheapest fix for the brown-drift we were seeing.
            try:
                g_arr = np.array(garment.convert("RGB"))
                # Use the dark/saturated centre of the garment as the seed
                # (skip white/transparent edges).
                gh, gw = g_arr.shape[:2]
                cx0, cx1 = int(gw * 0.30), int(gw * 0.70)
                cy0, cy1 = int(gh * 0.30), int(gh * 0.70)
                centre = g_arr[cy0:cy1, cx0:cx1].reshape(-1, 3).astype(np.float32)
                mk = (centre.max(axis=1) < 245) & (centre.min(axis=1) > 8)
                seed_rgb = centre[mk].mean(axis=0) if mk.any() else centre.mean(axis=0)
                seed_rgb = np.clip(seed_rgb, 0, 255).astype(np.uint8)
                # Blend the seed colour into the orig_arr ONLY where the
                # torso mask is high — outside the mask we keep the camera
                # pixels untouched so the seed colour doesn't leak.
                m = torso_mask[:, :, np.newaxis]
                anchor = (
                    seed_rgb[np.newaxis, np.newaxis, :].astype(np.float32) * m
                    + orig_arr.astype(np.float32) * (1.0 - m)
                ).astype(np.uint8)
                person = Image.fromarray(anchor)
            except Exception as e:
                log.debug(f"colour anchor prefill failed (using raw frame): {e}")
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
                f"photo of a person wearing a fitted {color} button-up shirt, "
                f"solid {color} fabric, neutral {color} colour, "
                f"the shirt fits naturally on the body, visible collar around the neck, "
                f"long sleeves following the arms down to the wrists, "
                f"realistic fabric folds, detailed texture, sharp focus, photorealistic"
            )
            # Anti-drift negatives. Includes purple/violet because at higher
            # IP-Adapter scales grey shirts pick up a mauve tint from the
            # mid-tone bias of the embedding.
            neg = (
                "wrong color, brown, dark brown, beige, tan, purple, violet, mauve, "
                "saturated, oversaturated, tinted, faded, washed out, "
                "bare arms, t-shirt, tank top, sleeveless, naked, "
                "floating clothes, shirt on background, shirt outline, "
                "deformed body, extra limbs, blurry, low quality, painting, cartoon"
            )
            # CFG 2.5: stronger than the bare minimum (1.5) needed to keep
            # diffusers happy. With LCM, 2.5 still converges in 6 steps
            # and is what finally beats the brown-drift problem. The
            # masked region is pre-seeded with the garment's mean colour
            # (see colour-anchor block above) which is the other half of
            # the fix.
            # 4-step LCM: ~33% faster than 6 steps, quality drop negligible
            # because the colour-anchor prefill + IP-Adapter carry most of
            # the signal. Saves ~250 ms per frame which is the difference
            # between "feels laggy" and "feels live".
            # 4-step LCM @ CFG 2.5 — known good combination from the
            # user's working May 26 demo state. Bumping to 6/4.0 with
            # IP scale 1.5 produced over-conditioning + rainbow-output
            # corruption. Reverted to known good.
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=neg,
                    image=person,
                    mask_image=mask_pil,
                    num_inference_steps=4,
                    guidance_scale=2.5,
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
            # Compose ONLY inside the torso_mask (the same region SD was
            # actually told to paint). Using body_silhouette here was
            # letting the inpaint bleed out below the chest band, leaving
            # a faint colour ghost where the shirt outline floated past
            # the body. torso_mask is already silhouette ∩ torso-band, so
            # this gives a clean cut at the bottom of the jacket too.
            # Tight final alpha — blur FIRST with a small (3,3) kernel
            # for just enough antialiasing to avoid jaggies, then zero
            # the face region. Old order (zero, then blur 7x7) was
            # smearing the chin row by 7 px and making the collar look
            # half-transparent.
            blend_mask = cv2.GaussianBlur(torso_mask, (3, 3), 0)
            blend_mask[:face_cutoff_y] = 0.0

            # Edge contact shadow — narrow ring just INSIDE the alpha
            # gets darkened to 85% in the SD result. Fakes the depth
            # cue of fabric sitting proud of the shoulder/sleeve. Set
            # SHADOW_STRENGTH = 0.0 to disable as a kill-switch.
            SHADOW_STRENGTH = 0.15
            mask_hard = (blend_mask > 0.5).astype(np.float32)
            inner = cv2.erode(
                mask_hard,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            edge_ring = cv2.GaussianBlur(
                (mask_hard - inner).clip(0, 1), (3, 3), 0,
            ).clip(0, 1)
            shadow = 1.0 - SHADOW_STRENGTH * edge_ring
            result_arr = result_arr * shadow[:, :, np.newaxis]

            a = blend_mask[:, :, np.newaxis]
            composed = (result_arr * a + orig_f * (1.0 - a)).astype(np.uint8)

            # ── Temporal stability: lock the painted shirt to previous
            # frame, so colour stops flickering every 3 s. Only blend
            # inside the masked region; outside, the live camera passes
            # through untouched. Strong prev weight (0.7) keeps the shirt
            # rock-steady even as small generation differences come and
            # go. Body movement still shows because the silhouette mask
            # itself is updating per frame from MediaPipe.
            if self._prev_result is not None and self._prev_result.shape == composed.shape:
                # 0.45 = "live filter" balance — new frame contributes
                # nearly half, so body movement tracks faster (~1 sec lag
                # at 800ms capture interval) while the 55% prev term still
                # damps colour flicker between samples.
                alpha_new = 0.45
                ema = (
                    composed.astype(np.float32) * alpha_new
                    + self._prev_result.astype(np.float32) * (1.0 - alpha_new)
                ).astype(np.uint8)
                stable_region = (a > 0.05)
                composed = np.where(stable_region, ema, composed)

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
