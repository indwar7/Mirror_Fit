"""
BiSeNet face-parsing ONNX wrapper.

Outputs a per-pixel label map (19 classes from the CelebAMask-HQ palette).
We care most about class 17 = 'hair' and class 13 = 'cloth'. These are what
let us do proper hair transfer for the live face swap.

Model file expected at: face_swap_backend/models/models/face_parser.onnx
Try any of these to download:
  https://huggingface.co/datasets/dragn/face-parser/resolve/main/face_parser.onnx
  https://huggingface.co/manhcuong02/face-parsing/resolve/main/79999_iter.onnx
  https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx

Input:  RGB image, resized to (512, 512), normalized with ImageNet mean/std
Output: (1, 19, 512, 512) class logits OR (1, 512, 512) argmax (handled both)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# CelebAMask-HQ class IDs (standard BiSeNet output)
LABEL_BACKGROUND = 0
LABEL_SKIN       = 1
LABEL_NOSE       = 2
LABEL_EYE_G      = 3   # eyeglasses
LABEL_L_EYE      = 4
LABEL_R_EYE      = 5
LABEL_L_BROW     = 6
LABEL_R_BROW     = 7
LABEL_L_EAR      = 8
LABEL_R_EAR      = 9
LABEL_MOUTH      = 10
LABEL_U_LIP      = 11
LABEL_L_LIP      = 12
LABEL_HAIR       = 13   # in some palettes hair=17; we accept both
LABEL_HAT        = 14
LABEL_EARRING    = 15
LABEL_NECKLACE   = 16
LABEL_NECK       = 17
LABEL_CLOTH      = 18

# Some ONNX exports use a permuted palette where hair=17 and neck=14 etc.
# We pick whichever id corresponds to the dominant top-of-head region.
HAIR_CANDIDATES = (17, 13)

PARSE_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FaceParserONNX:
    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort
        p = str(onnx_path)
        if not Path(p).exists():
            raise FileNotFoundError(p)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session   = ort.InferenceSession(
            p,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(f"[FaceParser] loaded {p}")

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (PARSE_SIZE, PARSE_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb = rgb.transpose(2, 0, 1)[np.newaxis]      # 1, 3, 512, 512
        return rgb.astype(np.float32)

    def parse(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return label map of shape (H, W) at the input image resolution."""
        h, w = img_bgr.shape[:2]
        x = self._preprocess(img_bgr)
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        # Accept (1, C, H, W) or (1, H, W)
        if out.ndim == 4:
            labels = out[0].argmax(axis=0).astype(np.uint8)
        elif out.ndim == 3:
            labels = out[0].astype(np.uint8)
        else:
            raise RuntimeError(f"unexpected parser output shape: {out.shape}")
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        return labels

    def hair_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Binary 0/255 hair mask at input resolution. Auto-detects which
        class id the export uses for hair by picking whichever candidate
        has the most pixels in the top half of the image."""
        labels = self.parse(img_bgr)
        h = img_bgr.shape[0]
        top = labels[:h // 2]
        best_cls = HAIR_CANDIDATES[0]
        best_n = -1
        for cid in HAIR_CANDIDATES:
            n = int((top == cid).sum())
            if n > best_n:
                best_n, best_cls = n, cid
        mask = ((labels == best_cls).astype(np.uint8)) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask


def transfer_hair(
    src_bgr: np.ndarray,
    src_hair_mask: np.ndarray,
    src_face_kps: np.ndarray,
    tgt_bgr: np.ndarray,
    tgt_hair_mask: np.ndarray,
    tgt_face_kps: np.ndarray,
    tgt_face_exclusion_mask: Optional[np.ndarray] = None,
    tgt_head_region_mask:    Optional[np.ndarray] = None,
) -> np.ndarray:
    """Warp avatar hair onto the target image. Clean version, no wig artifacts.

    Critical correctness rules:
      1. Source is PRE-MASKED to hair-only (background -> 0) BEFORE warping.
         Otherwise warpAffine produces an opaque rectangle that then
         shows as a hard wig outline.
      2. Paint area is the warped HAIR mask only (NOT union with user hair).
         Union with user hair previously bled the avatar background into
         the user's existing hair region.
      3. Paint is intersected with a HEAD-REGION mask (oval grown from the
         user's bbox). Guarantees we never paint outside a plausible
         head silhouette, even if BiSeNet hallucinates.
      4. Paint subtracts the FACE-EXCLUSION mask (convex hull of user's
         106 landmarks). Hair never paints over the inswapper face.
      5. LAB colour-match warps the avatar hair tone toward the user's
         scene lighting so it blends instead of looking pasted.
      6. Wide Gaussian (41 px) feather, no alpha boost. Soft natural edge.
    """
    th, tw = tgt_bgr.shape[:2]

    M, _ = cv2.estimateAffinePartial2D(
        src_face_kps.astype(np.float32),
        tgt_face_kps.astype(np.float32),
    )
    if M is None:
        return tgt_bgr

    # 1. Pre-mask source so warp produces no background.
    src_alpha     = (src_hair_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    src_hair_only = (src_bgr.astype(np.float32) * src_alpha).astype(np.uint8)

    src_warp = cv2.warpAffine(
        src_hair_only, M, (tw, th),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    mask_warp = cv2.warpAffine(
        src_hair_mask, M, (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # 2. Paint = warped avatar hair only.
    paint = mask_warp.copy()

    # 3. Clip to a head-region mask if supplied.
    if tgt_head_region_mask is not None:
        if tgt_head_region_mask.shape[:2] != (th, tw):
            tgt_head_region_mask = cv2.resize(
                tgt_head_region_mask, (tw, th), interpolation=cv2.INTER_NEAREST
            )
        paint = cv2.bitwise_and(paint, tgt_head_region_mask)

    # 4. Subtract the face exclusion hull.
    if tgt_face_exclusion_mask is not None:
        if tgt_face_exclusion_mask.shape[:2] != (th, tw):
            tgt_face_exclusion_mask = cv2.resize(
                tgt_face_exclusion_mask, (tw, th), interpolation=cv2.INTER_NEAREST
            )
        paint = cv2.bitwise_and(paint, cv2.bitwise_not(tgt_face_exclusion_mask))

    n_pix = int((paint > 0).sum())
    if n_pix < 800:
        return tgt_bgr

    # 5. LAB colour transfer: avatar hair tone -> user scene lighting.
    try:
        if int((tgt_hair_mask > 128).sum()) > 300 and int((mask_warp > 128).sum()) > 300:
            src_lab_full = cv2.cvtColor(src_warp, cv2.COLOR_BGR2LAB).astype(np.float32)
            src_pix = src_warp[mask_warp > 128].reshape(-1, 3).astype(np.uint8)
            tgt_pix = tgt_bgr[tgt_hair_mask > 128].reshape(-1, 3).astype(np.uint8)
            sm_lab  = cv2.cvtColor(src_pix.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0]
            tm_lab  = cv2.cvtColor(tgt_pix.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0]
            for c in range(3):
                ss = sm_lab[:, c].std() + 1e-6
                ts = tm_lab[:, c].std() + 1e-6
                src_lab_full[:, :, c] = (
                    (src_lab_full[:, :, c] - sm_lab[:, c].mean()) * (ts / ss)
                    + tm_lab[:, c].mean()
                )
            src_warp = cv2.cvtColor(
                np.clip(src_lab_full, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
            )
    except Exception:
        pass  # colour match is a nice-to-have

    # 6. Soft feather, no boost.
    alpha = cv2.GaussianBlur(paint, (41, 41), 0).astype(np.float32) / 255.0
    a3    = alpha[:, :, np.newaxis]
    out   = (src_warp.astype(np.float32) * a3 +
             tgt_bgr.astype(np.float32) * (1.0 - a3)).astype(np.uint8)
    return out
