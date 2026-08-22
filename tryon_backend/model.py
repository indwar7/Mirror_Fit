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


# ── MediaPipe Tasks API (selfie segmentation + hand landmarks) ──────────────
# The legacy `mediapipe.python.solutions` API isn't shipped in the Windows
# PyPI wheels — only `mediapipe.tasks` is available there. These two model
# bundles are downloaded once (cached locally) and used via Tasks API on
# every platform for consistent behaviour.

_MP_MODELS_DIR = Path(__file__).resolve().parent / "mp_models"
_MP_SEG_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)
_MP_HANDS_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_MP_POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Pose landmark indices we care about (MediaPipe's 33-point topology).
POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12
POSE_L_ELBOW, POSE_R_ELBOW = 13, 14
POSE_L_WRIST, POSE_R_WRIST = 15, 16
POSE_L_HIP, POSE_R_HIP = 23, 24


def _mp_download(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    log.info(f"Downloading MediaPipe model: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def _load_mediapipe_tasks():
    """Returns (segmenter, hand_landmarker, pose_landmarker) via the Tasks
    API — works on Windows, Linux and Mac (unlike the legacy `solutions`
    API).

    The pose landmarker is what makes garment placement independent of
    posture: shoulders and hips locate the torso wherever the body happens
    to be, instead of assuming it sits directly below the chin.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        ImageSegmenter, ImageSegmenterOptions,
        HandLandmarker, HandLandmarkerOptions,
        PoseLandmarker, PoseLandmarkerOptions,
        RunningMode,
    )

    seg_path = _MP_MODELS_DIR / "selfie_segmenter.tflite"
    hands_path = _MP_MODELS_DIR / "hand_landmarker.task"
    pose_path = _MP_MODELS_DIR / "pose_landmarker_lite.task"
    _mp_download(_MP_SEG_URL, seg_path)
    _mp_download(_MP_HANDS_URL, hands_path)
    _mp_download(_MP_POSE_URL, pose_path)

    segmenter = ImageSegmenter.create_from_options(ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(seg_path)),
        running_mode=RunningMode.IMAGE,
        output_category_mask=False,
        output_confidence_masks=True,
    ))
    hand_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hands_path)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    ))
    pose_landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_path)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_segmentation_masks=False,
    ))
    return segmenter, hand_landmarker, pose_landmarker


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
        #
        # NOTE: the legacy `mediapipe.python.solutions` API is not shipped in the
        # Windows PyPI wheels (any version) — only the newer Tasks API
        # (`mediapipe.tasks`) is available there. Using Tasks API unconditionally
        # so this works the same on Windows, Linux and Mac.
        try:
            self._mp_seg, self._mp_hands, self._mp_pose = _load_mediapipe_tasks()
            log.info("MediaPipe loaded (seg + hands + pose, Tasks API).")
        except Exception as e:
            self._mp_seg   = None
            self._mp_hands = None
            self._mp_pose  = None
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
        base = "abhay07080/CatVTON-bucket"
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
        base = "abhay07080/CatVTON-bucket"

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

        base = "abhay07080/CatVTON-bucket"
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

    _fabric_overlay = None    # numpy RGB (LIVE_SIZE, LIVE_SIZE, 3) or None

    def set_fabric(self, fabric_image):
        """Store a fabric pattern to overlay on top of the final SD result.

        The garment (tshirt / shirt / jacket) is left COMPLETELY untouched
        — the SD pipeline still produces the clean garment on the body
        as before. Only the final composited result gets a multiply
        overlay of the fabric pattern inside the torso mask. Result:
        garment fit/shape stays correct, fabric design shows on top.

        Passing None clears the overlay.
        """
        if fabric_image is None:
            self._fabric_overlay = None
            log.info("Fabric overlay cleared.")
            return
        if fabric_image.mode == "RGBA":
            bg = Image.new("RGB", fabric_image.size, (255, 255, 255))
            bg.paste(fabric_image, mask=fabric_image.split()[3])
            fabric_rgb = bg
        else:
            fabric_rgb = fabric_image.convert("RGB")
        fabric_full = fabric_rgb.resize((LIVE_SIZE, LIVE_SIZE), Image.LANCZOS)
        self._fabric_overlay = np.array(fabric_full).astype(np.float32)
        log.info("Fabric overlay stored — will composite over SD result.")

    def recolor_garment(self, color: str):
        """Recolour the cached garment to a hex colour while preserving
        the original shading / texture. Re-encodes IP-Adapter embeds and
        updates the dominant colour name so the SD prompt also reflects
        the new colour. 'original' restores the original garment."""
        if not hasattr(self, "_garment_original") or self._garment_original is None:
            self._garment_original = self._garment_cache
        if self._garment_original is None:
            log.warning("recolor_garment: no garment uploaded yet")
            return

        if color == "original" or not color:
            new_garment = self._garment_original
        else:
            try:
                tr = int(color[1:3], 16)
                tg = int(color[3:5], 16)
                tb = int(color[5:7], 16)
            except Exception:
                log.warning("recolor_garment: bad hex %r, ignoring", color)
                return
            src = np.array(self._garment_original).astype(np.float32)
            # Mask out the white padding bg — only tint the garment.
            is_garment = ~((src[:, :, 0] > 240)
                           & (src[:, :, 1] > 240)
                           & (src[:, :, 2] > 240))
            # Apply tint in HSV: replace hue+sat with target, keep value.
            src_hsv = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            tgt_rgb = np.array([[[tr, tg, tb]]], dtype=np.uint8)
            tgt_hsv = cv2.cvtColor(tgt_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            th, ts, _ = tgt_hsv[0, 0]
            out_hsv = src_hsv.copy()
            out_hsv[is_garment, 0] = th
            # Keep some original saturation so dark/light tones don't
            # collapse to flat colour. Mix 70% target sat + 30% original.
            out_hsv[is_garment, 1] = 0.70 * ts + 0.30 * src_hsv[is_garment, 1]
            # value unchanged → shading / wrinkles preserved
            recolored = cv2.cvtColor(out_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            new_garment = Image.fromarray(recolored)

        self._garment_cache = new_garment
        # Update dominant-colour name so the SD prompt picks up the change.
        self._garment_color_name = self._dominant_color_name(np.array(new_garment))
        # Re-encode IP-Adapter embeds with the new-colour garment.
        if self._ip_loaded and hasattr(self.pipeline, "prepare_ip_adapter_image_embeds"):
            try:
                with torch.inference_mode():
                    self._ip_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                        ip_adapter_image=[new_garment],
                        ip_adapter_image_embeds=None,
                        device=self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                    )
            except Exception as e:
                log.warning(f"recolor: IP embed re-cache failed: {e}")
                self._ip_embeds = None
        log.info("Garment recoloured to %s (prompt colour=%s).",
                 color, self._garment_color_name)

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
        self._garment_original = garment_sq   # for recolor_garment()
        self._fabric_overlay   = None         # reset on new garment upload
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
                import mediapipe as mp
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(frame_rgb),
                )
                res = self._mp_hands.detect(mp_image)
                if res.hand_landmarks:
                    for hand_lmk in res.hand_landmarks:
                        pts = np.array(
                            [[int(lm.x * W), int(lm.y * H)] for lm in hand_lmk],
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
                import mediapipe as mp
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(frame),
                )
                seg_result = self._mp_seg.segment(mp_image)
                if seg_result.confidence_masks:
                    mask_arr = seg_result.confidence_masks[0].numpy_view()
                    bm = (mask_arr > 0.4).astype(np.float32)
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

    def _pose_torso_region(self, frame_rgb: np.ndarray):
        """Locate the torso from body landmarks instead of from the chin.

        Returns (region, neck_y) where `region` is a soft HxW mask covering
        torso + sleeves, or None when pose is unavailable or too uncertain.

        Why this exists: the fallback geometry defines the torso as "the
        horizontal band below the detected chin", which silently assumes the
        wearer is upright. Someone reclining, leaning far over, or lying
        down has a torso that is beside or behind their head in image space,
        not below it — and the garment lands on their face. Shoulders and
        hips locate the torso whatever the posture, so the mask follows the
        body rather than the frame.
        """
        if self._mp_pose is None:
            return None
        try:
            import mediapipe as mp

            h, w = frame_rgb.shape[:2]
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb),
            )
            result = self._mp_pose.detect(image)
            if not result.pose_landmarks:
                return None
            marks = result.pose_landmarks[0]

            def point(index: int):
                lm = marks[index]
                return (
                    np.array([lm.x * w, lm.y * h], dtype=np.float32),
                    float(getattr(lm, "visibility", 1.0)),
                )

            l_sh, v_lsh = point(POSE_L_SHOULDER)
            r_sh, v_rsh = point(POSE_R_SHOULDER)
            l_hip, v_lhip = point(POSE_L_HIP)
            r_hip, v_rhip = point(POSE_R_HIP)

            # Both shoulders must be credible; hips may be out of frame on a
            # head-and-shoulders crop, so they are allowed to be inferred.
            # MediaPipe drops `visibility` for landmarks at or past the
            # frame border, so a close-up head-and-shoulders shot -- the
            # framing people actually use -- was scoring under the old 0.5
            # gate and falling through to the band. The predicted positions
            # are still good there; the span check below is what actually
            # rejects a bad detection.
            if min(v_lsh, v_rsh) < 0.30:
                return None

            shoulder_span = float(np.linalg.norm(l_sh - r_sh))
            if shoulder_span < 0.06 * w:      # implausibly small — bad detection
                return None

            shoulder_mid = (l_sh + r_sh) / 2.0
            if min(v_lhip, v_rhip) < 0.35:
                # Hips not visible: project a torso length down the body axis.
                axis = shoulder_mid - (point(0)[0])       # nose -> shoulders
                norm = np.linalg.norm(axis)
                axis = axis / norm if norm > 1e-3 else np.array([0.0, 1.0], np.float32)
                hip_mid = shoulder_mid + axis * (1.55 * shoulder_span)
                offset = (l_sh - r_sh) * 0.42
                l_hip, r_hip = hip_mid + offset, hip_mid - offset
            hip_mid = (l_hip + r_hip) / 2.0

            gtype = getattr(self, "_garment_type", "tshirt")
            hem_extend = {"tshirt": 0.10, "shirt": 0.24, "jacket": 0.34}.get(gtype, 0.10)
            body_axis = hip_mid - shoulder_mid
            l_hem = l_hip + body_axis * hem_extend
            r_hem = r_hip + body_axis * hem_extend

            # Widen away from the body centre so the garment has bulk.
            centre = (shoulder_mid + hip_mid) / 2.0
            def widen(p, factor=1.20):
                return centre + (p - centre) * factor

            region = np.zeros((h, w), dtype=np.float32)
            # ── Neckline ────────────────────────────────────────────
            # A straight edge across the shoulders leaves no collar: the
            # garment covers the neck flat and the shoulder line reads as
            # a bar rather than a seam. Cut a notch between the shoulders
            # so SD has a neck opening to paint a collar around, and push
            # the shoulder points outward past the joint so the seam sits
            # on the edge of the body rather than inside it.
            shoulder_dir = (r_sh - l_sh)
            span = np.linalg.norm(shoulder_dir)
            shoulder_dir = shoulder_dir / span if span > 1e-3 else np.array([1.0, 0.0], np.float32)
            down = hip_mid - shoulder_mid
            dn = np.linalg.norm(down)
            down = down / dn if dn > 1e-3 else np.array([0.0, 1.0], np.float32)

            # Wider, shallower for a tee; narrower and higher for a jacket
            # worn closed. These mirror the per-garment neckline the
            # face-bbox path used, which is where the collar came from.
            # Fractions of shoulder span. A crew neck opening is about a
            # third of shoulder span across, so half of it is ~0.18, and it
            # sits shallow -- roughly 0.14 of span below the shoulder line.
            # The old 0.26 / 0.30 cut an opening two-thirds as wide as the
            # chest and deep enough to reach the sternum, which reads as a
            # scoop-neck vest, not a collar.
            #
            # shoulder_out_f pushes the seam outward from the shoulder
            # joint. At 1.16 the seam hung past the arm and the garment
            # looked draped over the wearer rather than fitted; 1.06 puts it
            # just outside the joint, where a real seam sits.
            neck_half_f, neck_dip_f, shoulder_out_f = {
                "tshirt": (0.18, 0.14, 1.06),
                "shirt":  (0.15, 0.12, 1.05),
                "jacket": (0.13, 0.10, 1.07),
            }.get(gtype, (0.18, 0.14, 1.06))

            neck_l = shoulder_mid - shoulder_dir * (span * neck_half_f)
            neck_r = shoulder_mid + shoulder_dir * (span * neck_half_f)
            neck_b = shoulder_mid + down * (span * neck_dip_f)

            def out(p):
                """Push a shoulder point outward along the shoulder line."""
                return shoulder_mid + (p - shoulder_mid) * shoulder_out_f

            # Drop the outer seam a little below the joint. A shoulder seam
            # runs from the neck outward and slightly down; a level edge
            # between neck and arm reads as a bar laid across the chest.
            seam_drop = down * (span * 0.05)
            torso = np.array(
                [out(l_sh) + seam_drop, neck_l, neck_b, neck_r,
                 out(r_sh) + seam_drop,
                 widen(r_hem), widen(l_hem)],
                dtype=np.int32,
            )
            cv2.fillPoly(region, [torso], 1.0)

            # Sleeves follow the arm chain. A tee stops at the upper arm; a
            # shirt or jacket runs to the wrist.
            sleeve_thickness = max(6, int(shoulder_span * 0.36))
            for shoulder_i, elbow_i, wrist_i in (
                (POSE_L_SHOULDER, POSE_L_ELBOW, POSE_L_WRIST),
                (POSE_R_SHOULDER, POSE_R_ELBOW, POSE_R_WRIST),
            ):
                shoulder, v_s = point(shoulder_i)
                elbow, v_e = point(elbow_i)
                if min(v_s, v_e) < 0.4:
                    continue
                if gtype == "tshirt":
                    end = shoulder + (elbow - shoulder) * 0.62
                    cv2.line(region, tuple(shoulder.astype(int)), tuple(end.astype(int)),
                             1.0, sleeve_thickness)
                else:
                    cv2.line(region, tuple(shoulder.astype(int)), tuple(elbow.astype(int)),
                             1.0, sleeve_thickness)
                    wrist, v_w = point(wrist_i)
                    if v_w >= 0.4:
                        cv2.line(region, tuple(elbow.astype(int)), tuple(wrist.astype(int)),
                                 1.0, int(sleeve_thickness * 0.82))

            region = cv2.GaussianBlur(region, (31, 31), 0).clip(0, 1)

            # Neck line: the shoulder line, not a chin row. Used downstream
            # for the fabric fade so the pattern starts at the collar.
            neck_y = int(np.clip(min(l_sh[1], r_sh[1]) - 0.10 * shoulder_span, 0, h - 1))
            return region, neck_y
        except Exception as e:
            log.debug(f"pose torso unavailable: {e}")
            return None

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
        mp_silhouette_ok = False   # True once a real per-frame MediaPipe mask lands
        if self._mp_seg is not None:
            try:
                import mediapipe as mp
                # mp.Image requires a memory-contiguous buffer — frames that
                # went through PIL crop/resize aren't always contiguous,
                # which throws here silently (caught below) and falls back
                # to the crude safety rectangle for the whole session.
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(frame_rgb),
                )
                seg = self._mp_seg.segment(mp_image)
                if seg.confidence_masks:
                    s = seg.confidence_masks[0].numpy_view().astype(np.float32)
                    # Threshold at 0.6 (was 0.5) for a tighter person edge.
                    # Single iteration of dilation gives the jacket just
                    # enough room for sleeve thickness without producing
                    # the ghost outline we were seeing past the body.
                    # Threshold 0.25 (was 0.4 → 0.6 originally) catches
                    # the soft edges where arms / hands fade into the
                    # background. User reported background visible past
                    # jacket edges and arms not painted; the mask was
                    # cutting too tight.
                    # Threshold 0.2 (was 0.25) — a foreshortened shoulder
                    # (person angled toward camera) gets lower segmentation
                    # confidence on that side, undershooting the true edge
                    # and leaving that shoulder/sleeve uncovered.
                    # Threshold and dilation had been loosened step by step
                    # to chase uncovered arms and foreshortened shoulders --
                    # 0.6 -> 0.4 -> 0.25 -> 0.2, and 2 -> 5 -> 4 iterations of
                    # a 7x7 kernel. Together that grew the person by roughly
                    # 12 px at model scale, which on a 2k frame is a ~50 px
                    # apron of garment hanging past the body onto whatever is
                    # behind it. The garment stopped reading as worn.
                    #
                    # Tight edge here instead. Arm and sleeve coverage does
                    # not need a fat silhouette: the pose path already draws
                    # sleeves along the shoulder-elbow-wrist chain, and the
                    # skin-tone pass below extends onto raised hands. Both
                    # add coverage where the limb actually is, rather than
                    # everywhere at once.
                    bm = (s > 0.45).astype(np.float32)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    bm = cv2.dilate(bm, kernel, iterations=1)
                    bm = cv2.GaussianBlur(bm, (5, 5), 0).clip(0, 1)
                    # Per-pixel temporal EMA: damps the 1-2 px shimmer
                    # MediaPipe produces per frame. 0.8 new / 0.2 prev (was
                    # 0.6/0.4) — the heavier prev term was smearing the mask
                    # edge into a visible ghost/blur whenever the body moved
                    # between frames.
                    if (self._prev_silhouette is not None
                            and self._prev_silhouette.shape == bm.shape):
                        bm = (0.8 * bm + 0.2 * self._prev_silhouette).clip(0, 1)
                    self._prev_silhouette = bm
                    silhouette = bm
                    mp_silhouette_ok = True
            except Exception as e:
                log.debug(f"MediaPipe seg failed in mask build: {e}")

        if silhouette is None and self._prev_silhouette is not None:
            # A transient per-frame MediaPipe miss (motion blur, lighting
            # flicker) shouldn't revert to the crude ellipse/rectangle —
            # that jarring shape-jump is exactly what read as the garment
            # "fading away" every few seconds. Hold the last known-good
            # real silhouette instead; it'll self-correct next successful
            # frame since _prev_silhouette only updates on real hits.
            silhouette = self._prev_silhouette
            mp_silhouette_ok = True

        if silhouette is None:
            # No real silhouette ever obtained this session yet — genuine
            # fallback ellipse (same shape the legacy fixed mask used).
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
                # Safety clamp: cap half_w at 42% of frame width. Without
                # this, a single bad Haar detection (fw2 abnormally large)
                # blows the rectangle up to near-full-frame width, painting
                # over background objects and other people in frame.
                half_w = min(half_w, int(w * 0.42))
                rect_top   = fy2 + fh2                         # chin row
                rect_left  = max(0, cx2 - half_w)
                rect_right = min(w, cx2 + half_w)
                # Per-garment collar — deeper / wider notches for clearer
                # neckline definition (user: "make the collar more clear
                # and defined").
                if gtype == "tshirt":
                    neck_dip  = int(fh2 * 0.70)
                    neck_half = int(fw2 * 0.75)
                elif gtype == "shirt":
                    neck_dip  = int(fh2 * 0.55)
                    neck_half = int(fw2 * 0.55)
                else:  # jacket
                    neck_dip  = int(fh2 * 0.30)
                    neck_half = int(fw2 * 0.40)
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
            # Only fall back to the crude face-bbox rectangle when real
            # MediaPipe segmentation didn't land this frame. Blending it in
            # unconditionally (old behaviour) dragged the collar/shoulder/
            # waist edges away from the actual body toward this generic
            # per-garment-type shape, even when the real silhouette was
            # available and more accurate.
            if not mp_silhouette_ok:
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
        face_box = None
        try:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = self._haar.detectMultiScale(
                cv2.equalizeHist(gray), scaleFactor=1.1,
                minNeighbors=4, minSize=(40, 40),
            )
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                # Cutoff at chin row. The 25-px soft fade below (in the
                # blend block) hides the seam, so we can keep the cutoff
                # right at the chin — collar sits at the neck naturally.
                face_cutoff_y = int(np.clip(fy + fh,
                                            h * 0.20, h * 0.48))
                face_box = (int(fx), int(fy), int(fw), int(fh))
        except Exception:
            pass

        # 3. Torso region.
        #
        # Preferred: a polygon built from the wearer's own shoulders and
        # hips, which is correct at any posture.
        #
        # Fallback: the horizontal band below the chin. That band is only
        # right for an upright wearer — reclined or leaning, the region
        # below the chin is the wearer's face and the garment lands there.
        # It is kept solely because pose detection can fail, and a mask in
        # roughly the wrong place still beats no mask at all.
        pose_region = self._pose_torso_region(frame_rgb)
        if pose_region is not None:
            band, pose_neck_y = pose_region
            face_cutoff_y = int(np.clip(pose_neck_y, h * 0.05, h * 0.90))
        elif face_box is not None:
            # Pose failed, but a face was found. Size a torso from the face.
            #
            # This used to be `band[face_cutoff_y:, :] = 1.0` — the full
            # width of the frame. On a head-and-shoulders crop, where pose
            # most often fails, that paints garment across the entire lower
            # frame including the wall and furniture behind the wearer, with
            # a straight edge under the chin and no collar. GrabCut is meant
            # to trim it back to the body and cannot reliably do so against a
            # busy background.
            #
            # A face is a dependable ruler: shoulder span runs about 3x face
            # width, so the body can be bounded without any pose landmarks.
            fx3, fy3, fw3, fh3 = face_box
            cx = fx3 + fw3 * 0.5
            sh_half = fw3 * 1.55            # shoulder span ~= 3.1 face widths
            sh_y    = face_cutoff_y + fh3 * 0.22
            hem_y   = float(h) * 0.99
            hem_half = sh_half * 1.12       # hem slightly wider than shoulders

            gtype = getattr(self, "_garment_type", "tshirt")
            # Same proportions as the pose path, re-expressed in face
            # box dimensions: shoulder span is ~3.1 face widths, so a
            # 0.18-of-span half-opening is ~0.56 face widths; the dip is
            # scaled by face height, which Haar returns near-square, so
            # ~0.43 matches the 0.14-of-span drop. Agreeing means the collar
            # does not change shape when pose detection drops out.
            neck_half_f, neck_dip_f = {
                "tshirt": (0.56, 0.43),
                "shirt":  (0.47, 0.36),
                "jacket": (0.40, 0.30),
            }.get(gtype, (0.56, 0.43))
            neck_half = fw3 * neck_half_f
            neck_dip  = fh3 * neck_dip_f

            band = np.zeros((h, w), dtype=np.float32)
            torso_poly = np.array([
                [cx - sh_half,  sh_y],                    # left shoulder
                [cx - neck_half, sh_y],                   # neckline left
                [cx,            sh_y + neck_dip],         # neckline dip
                [cx + neck_half, sh_y],                   # neckline right
                [cx + sh_half,  sh_y],                    # right shoulder
                [cx + hem_half, hem_y],                   # right hem
                [cx - hem_half, hem_y],                   # left hem
            ], dtype=np.int32)
            cv2.fillPoly(band, [torso_poly], 1.0)
            band = cv2.GaussianBlur(band, (31, 31), 0).clip(0, 1)
        else:
            # No pose and no face: nothing reliable to aim at. Keep the old
            # band so a frame still renders, but hold it to the middle half
            # of the frame rather than edge to edge.
            band = np.zeros((h, w), dtype=np.float32)
            band[face_cutoff_y:int(h * 0.98), int(w * 0.24):int(w * 0.76)] = 1.0
            band = cv2.GaussianBlur(band, (21, 21), 0)

        # Keep the throat clear.
        #
        # The neckline notch is cut into the torso polygon, so it protects
        # the neck only when that polygon is positioned well. Whenever the
        # geometry rides high -- an odd pose, a mis-sized face box -- paint
        # climbs to the jaw and the garment reads as a turtleneck
        # swallowing the chin, with no collar line anywhere. A collar is
        # only legible if bare neck shows above it.
        #
        # Subtracting a soft ellipse over the throat guarantees that gap
        # regardless of how the polygon came out, in both paths.
        neck_hole = None          # (centre, axes) of the opening, for the collar
        if face_box is not None:
            fxn, fyn, fwn, fhn = face_box
            nc = (int(fxn + fwn * 0.5), int(fyn + fhn))   # centred on the chin
            na = (int(fwn * 0.30), int(fhn * 0.52))       # neck column
            throat = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(throat, nc, na, 0, 0, 360, 1.0, -1)
            throat = cv2.GaussianBlur(throat, (21, 21), 0).clip(0, 1)
            band = (band * (1.0 - throat)).clip(0, 1)
            neck_hole = (nc, na, fwn)

        # 4. torso_mask = silhouette ∩ region  (body pixels, torso only)
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
                        iterCount=2, mode=cv2.GC_INIT_WITH_MASK)
            body = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                1.0, 0.0
            ).astype(np.float32)
            body = cv2.resize(body, (w, h), interpolation=cv2.INTER_LINEAR)
            # Tight body silhouette: 3x3 dilation (was 9x9) so the
            # painted garment hugs the body edge without extending into
            # background. 9x9 was over-dilating and leaving a soft halo
            # past the body — user: "stick to body no extra square".
            body = cv2.dilate(
                body,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            body = cv2.GaussianBlur(body, (7, 7), 0).clip(0, 1)
            # Temporal EMA on the body mask — body doesn't change shape
            # per frame, so blending the previous body mask kills the
            # 1-2 px GrabCut wobble that otherwise jiggles the garment.
            if (getattr(self, "_prev_body_mask", None) is not None
                    and self._prev_body_mask.shape == body.shape):
                body = (0.6 * body + 0.4 * self._prev_body_mask).clip(0, 1)
            self._prev_body_mask = body
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

        # Collar ring.
        #
        # Every previous attempt cut a neck hole and left the collar to the
        # diffusion model. It never arrived, and it was never going to: a
        # collar is the few pixels at the rim of the opening, that rim is
        # deliberately feathered so SD has something smooth to denoise
        # into, and six LCM steps will not resolve a ribbed band there
        # anyway. Tuning the hole's shape cannot fix a detail that is not
        # being drawn.
        #
        # So draw it. The ring between the neck opening and a slightly
        # larger ellipse, clipped to wherever garment actually ended up, is
        # a collar's footprint. _infer_tier3 shades it.
        collar_band = np.zeros((h, w), dtype=np.float32)
        if neck_hole is not None:
            nc, na, fwn = neck_hole
            thick = max(4, int(fwn * 0.11))
            outer = np.zeros((h, w), dtype=np.float32)
            inner = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(outer, nc, (na[0] + thick, na[1] + thick), 0, 0, 360, 1.0, -1)
            cv2.ellipse(inner, nc, na, 0, 0, 360, 1.0, -1)
            ring = cv2.GaussianBlur((outer - inner).clip(0, 1), (5, 5), 0)
            # Only where garment was actually painted -- otherwise the ring
            # would be drawn across bare neck below an open collar.
            collar_band = (ring * torso_mask).clip(0, 1)

        return torso_mask, silhouette, face_cutoff_y, collar_band

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
        torso_mask, body_silhouette, face_cutoff_y, collar_band = \
            self._build_body_mask(orig_arr)

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
                gh, gw = g_arr.shape[:2]
                cx0, cx1 = int(gw * 0.30), int(gw * 0.70)
                cy0, cy1 = int(gh * 0.30), int(gh * 0.70)
                centre = g_arr[cy0:cy1, cx0:cx1].reshape(-1, 3).astype(np.float32)
                mk = (centre.max(axis=1) < 245) & (centre.min(axis=1) > 8)
                seed_rgb = centre[mk].mean(axis=0) if mk.any() else centre.mean(axis=0)
                seed_rgb = np.clip(seed_rgb, 0, 255).astype(np.uint8)
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
            gtype_p = getattr(self, "_garment_type", "tshirt")
            color = self._garment_color_name or "matching"
            garment_word = {"tshirt": "t-shirt", "shirt": "button-up shirt",
                            "jacket": "jacket"}.get(gtype_p, "shirt")
            prompt = (
                f"photograph of a person wearing a {color} {garment_word}, "
                f"the fabric drapes over the chest and follows the shoulders, "
                f"soft fabric folds gathering at the waist and under the arms, "
                f"visible seams at the shoulder and a defined collar at the neck, "
                f"cloth catching the light from above with soft shadows in the creases, "
                f"woven fabric texture, natural cloth weight, "
                f"solid {color} colour, sharp focus, photorealistic, studio lighting"
            )
            neg = (
                # Colour drift
                "wrong color, brown, beige, tan, purple, violet, mauve, "
                "oversaturated, faded, washed out, "
                # The failure mode that makes it look pasted rather than worn
                "flat shading, uniform flat colour, no folds, no wrinkles, "
                "sticker, cutout, pasted on, decal, printed on skin, 2d overlay, "
                "rigid fabric, cardboard, plastic sheen, "
                # Structure
                "bare chest, naked, sleeveless, floating clothes, garment on "
                "background, garment outline, deformed body, extra limbs, "
                "blurry, low quality, painting, cartoon, illustration"
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
                    # 6 steps, not 4. Four is enough to get the colour and
                    # silhouette right, but fold structure and seam detail
                    # are still forming at that point and the cloth reads
                    # flat. Six costs roughly 250ms more per frame and is
                    # where the drape starts to look like fabric.
                    num_inference_steps=6,
                    guidance_scale=2.8,
                    generator=generator,
                    **ip_kw,
                ).images[0]

            # ── Fabric overlay (post-SD) ─────────────────────────────────
            # HSV composite: take FABRIC's hue + saturation (the colour /
            # pattern) and SD result's value (the body folds / shading).
            # Uses a SHIFTED-DOWN mask so the fabric starts at the
            # collar/clavicle, not at the chin (user: "yeh mere neck pe
            # fabric overlay ho rhi hai, it should be only till collar").
            if self._fabric_overlay is not None:
                try:
                    r_arr = np.array(result).astype(np.uint8)
                    if self._fabric_overlay.shape == r_arr.shape:
                        f_arr = self._fabric_overlay.astype(np.uint8)
                    else:
                        f_arr = cv2.resize(
                            self._fabric_overlay,
                            (r_arr.shape[1], r_arr.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        ).astype(np.uint8)
                    r_hsv = cv2.cvtColor(r_arr, cv2.COLOR_RGB2HSV).astype(np.float32)
                    f_hsv = cv2.cvtColor(f_arr, cv2.COLOR_RGB2HSV).astype(np.float32)
                    # H + S from fabric, V from SD (keeps SD's shading).
                    out_hsv = r_hsv.copy()
                    out_hsv[:, :, 0] = f_hsv[:, :, 0]
                    out_hsv[:, :, 1] = f_hsv[:, :, 1]
                    out_rgb = cv2.cvtColor(out_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
                    # Fabric mask = torso_mask with the neck strip zeroed
                    # out. The 'neck band' is the region from face_cutoff_y
                    # down to clavicle (~80 px). A linear ramp 0->1 lets
                    # the fabric fade in at the clavicle so there's no
                    # hard edge.
                    fabric_mask = torso_mask.copy()
                    H_im = fabric_mask.shape[0]
                    band_top = max(0, int(face_cutoff_y))
                    band_bot = min(H_im, band_top + 80)
                    if band_bot > band_top:
                        ramp = np.linspace(0, 1, band_bot - band_top,
                                           dtype=np.float32)
                        fabric_mask[band_top:band_bot] = \
                            fabric_mask[band_top:band_bot] * ramp[:, None]
                        fabric_mask[:band_top] = 0
                    m = fabric_mask[:, :, np.newaxis]
                    mixed = (0.80 * out_rgb + 0.20 * r_arr.astype(np.float32)) * m \
                            + r_arr.astype(np.float32) * (1.0 - m)
                    result = Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))
                except Exception as e:
                    log.debug(f"fabric overlay skipped: {e}")

            # ── Composite result back onto original via body silhouette ──────
            # SD output can bleed slightly past the mask edge. We blend the
            # SD result into the original ONLY where body_silhouette has
            # alpha > 0, and ONLY below the face cutoff. Everything else
            # (face, background, hair, legs) stays pixel-identical to the
            # camera frame — so the jacket genuinely appears "worn on" you.
            result_arr = np.array(result).astype(np.float32)
            orig_f     = orig_arr.astype(np.float32)

            # ── Body shading transfer ────────────────────────────────────
            # This is what separates "a garment painted on" from "a garment
            # being worn". SD renders cloth with its own invented lighting,
            # which does not match the room the shopper is standing in, so
            # the result reads as a flat cutout however good the colour is.
            #
            # The camera frame already contains the correct lighting: where
            # the chest catches light, where the arm casts shade, where the
            # body curves away. Dividing the frame's luminance by a heavily
            # blurred copy of itself isolates exactly that — local shading
            # and fold structure — while discarding absolute brightness,
            # which belongs to whatever the shopper was already wearing.
            #
            # Multiplying the generated cloth by that ratio grounds it in
            # the real scene: it picks up the body's contours and the room's
            # light without inheriting the old garment's colour.
            try:
                lum = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
                base = cv2.GaussianBlur(lum, (0, 0), sigmaX=21)
                ratio = lum / np.maximum(base, 1.0)
                # Clamp hard: beyond this, sensor noise and the old
                # garment's own pattern start printing through the new one.
                ratio = np.clip(ratio, 0.78, 1.28)
                # Low-pass the shading map before applying it. Body form is
                # mid-frequency; the weave of whatever the shopper is
                # already wearing is high-frequency. Without this blur the
                # two are transferred together and the old garment's
                # texture prints through the new one. Measured on a
                # synthetic torso, sigma 5 raises the form-to-texture ratio
                # from 0.56 to 2.98 — which is what makes it safe to run a
                # stronger effect and get deeper folds rather than a
                # louder copy of the old shirt.
                ratio = cv2.GaussianBlur(ratio, (0, 0), sigmaX=5)
                SHADING_STRENGTH = 0.90
                shading = 1.0 + (ratio - 1.0) * SHADING_STRENGTH
                # Only where the garment was actually painted, feathered so
                # the effect fades out with the mask rather than ending on
                # a hard line.
                sm = cv2.GaussianBlur(torso_mask, (0, 0), sigmaX=3).clip(0, 1)
                shading = 1.0 + (shading - 1.0) * sm
                result_arr = np.clip(result_arr * shading[:, :, np.newaxis], 0, 255)
            except Exception as e:
                log.debug(f"shading transfer skipped: {e}")
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

            # Saturate the interior before anything else touches the alpha.
            #
            # torso_mask is a product of three soft masks -- silhouette (0.85
            # where it came from the safety rect, 0.80 from skin), the torso
            # band, and the GrabCut body -- each blurred. Multiplying them
            # leaves the middle of the chest around 0.75, not 1.0, so a
            # quarter of the bare body is blended back in over the whole
            # garment. That is what makes the shirt read as a translucent
            # projection you can see through rather than cloth.
            #
            # Map 0.28 -> 0 and 0.62 -> 1: anything that is clearly inside
            # becomes fully opaque, and the soft ramp survives only across
            # the boundary, where feathering is actually wanted.
            blend_mask = np.clip((blend_mask - 0.28) / 0.34, 0.0, 1.0)

            # Soft fade above the cutoff instead of a hard 0.0 cut.
            # The hard cut produced a visible horizontal line on the
            # chin (user: "face p chin ko ek line cut kr rhi hai").
            # 25-pixel linear ramp blends paint smoothly into face.
            blend_mask[:face_cutoff_y] = 0.0
            fade_band = 25
            for i in range(fade_band):
                y = face_cutoff_y + i
                if 0 <= y < blend_mask.shape[0]:
                    blend_mask[y] *= (i / fade_band)

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

            # ── Collar ──────────────────────────────────────────────────
            # Shade the ring around the neck opening rather than hoping the
            # sampler renders one. A collar reads as a band of the same
            # cloth turned back on itself: same hue, less light, with a
            # defined inner edge. Darkening what is already there gives
            # exactly that and cannot clash with the garment colour, since
            # it takes the colour from the render.
            #
            # Only inside the mask, so an open neckline stays open.
            try:
                cb = collar_band * blend_mask
                if cb.max() > 0.05:
                    cb3 = cb[:, :, np.newaxis]
                    COLLAR_DARKEN = 0.74
                    composed = (
                        composed.astype(np.float32) * (1.0 - cb3)
                        + composed.astype(np.float32) * COLLAR_DARKEN * cb3
                    ).clip(0, 255).astype(np.uint8)
            except Exception as e:
                log.debug(f"collar shading skipped: {e}")

            # ── Temporal stability: lock the painted shirt to previous
            # frame, so colour stops flickering every 3 s. Only blend
            # inside the masked region; outside, the live camera passes
            # through untouched. Strong prev weight (0.7) keeps the shirt
            # rock-steady even as small generation differences come and
            # go. Body movement still shows because the silhouette mask
            # itself is updating per frame from MediaPipe.
            if self._prev_result is not None and self._prev_result.shape == composed.shape:
                # 0.7 (was 0.45) — new frame now dominates. The heavier
                # 55% prev term was smearing a visible ghost/double-edge at
                # the garment boundary whenever the body moved between
                # frames. Still keeps some damping against flicker.
                alpha_new = 0.7
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
                "ribbed crew neck collar at the neckline, "
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
