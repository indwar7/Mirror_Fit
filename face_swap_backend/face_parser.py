"""
BiSeNet face-parsing ONNX wrapper.

Outputs a per-pixel label map (19 classes from the CelebAMask-HQ palette).
We care most about class 17 = 'hair' and class 13 = 'cloth' — those are what
let us do proper hair transfer for the live face swap.

Model file expected at: face_swap_backend/models/models/face_parser.onnx
Try any of these to download:
  https://huggingface.co/datasets/dragn/face-parser/resolve/main/face_parser.onnx
  https://huggingface.co/manhcuong02/face-parsing/resolve/main/79999_iter.onnx
  https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx

Input:  RGB image, resized to (512, 512), normalized with ImageNet mean/std
Output: (1, 19, 512, 512) class logits OR (1, 512, 512) argmax — handled both
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
            labels = out[0].argmax(axis=0).astype(np.uint8)   # (H, W)
        elif out.ndim == 3:
            labels = out[0].astype(np.uint8)
        else:
            raise RuntimeError(f"unexpected parser output shape: {out.shape}")
        # Upsample back to original resolution
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
        # Clean up — close small holes, soften edges
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
) -> np.ndarray:
    """Warp the avatar's hair onto the target image using a face-aligned
    affine and clip with the target's own hair silhouette.

    Pipeline:
      1. Affine src->tgt from 5 face keypoints — anchors the warp to the
         user's actual face geometry.
      2. Combined mask = (warped avatar hair) ∪ (user hair), intersected
         with a generously-dilated user-hair region (no floating blocks).
      3. SUBTRACT tgt_face_exclusion_mask (the user's face convex hull,
         slightly dilated) from the combined mask. This is what stops
         the avatar's forehead/cheek pixels from bleeding over the
         inswapper face — hair only paints ABOVE the eyebrow line and
         OUTSIDE the face oval.
      4. Feathered Gaussian alpha for a soft edge.
    """
    th, tw = tgt_bgr.shape[:2]

    M, _ = cv2.estimateAffinePartial2D(
        src_face_kps.astype(np.float32),
        tgt_face_kps.astype(np.float32),
    )
    if M is None:
        return tgt_bgr

    src_warp = cv2.warpAffine(
        src_bgr, M, (tw, th),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    mask_warp = cv2.warpAffine(
        src_hair_mask, M, (tw, th),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Use the UNION of warped avatar hair and user hair as the paint area.
    # We don't intersect with a dilated user-hair-only region because that
    # collapses to ~user-hair when avatar and user hair shapes are similar
    # (the visible-color-change problem on Lucas-like avatars).
    combined = cv2.bitwise_or(mask_warp, tgt_hair_mask)

    # Hard-exclude the user's face region — prevents forehead bleed.
    if tgt_face_exclusion_mask is not None:
        if tgt_face_exclusion_mask.shape[:2] != (th, tw):
            tgt_face_exclusion_mask = cv2.resize(
                tgt_face_exclusion_mask, (tw, th), interpolation=cv2.INTER_NEAREST
            )
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(tgt_face_exclusion_mask))

    n_pix = int(combined.sum() // 255)
    if n_pix < 500:
        print(f"[hair] skip: combined too small ({n_pix} px)")
        return tgt_bgr

    # Diagnostic so we can see at a glance whether hair is actually painting
    print(f"[hair] paint: combined={n_pix}px  warp={int(mask_warp.sum()//255)}px  "
          f"tgt={int(tgt_hair_mask.sum()//255)}px  excl="
          f"{0 if tgt_face_exclusion_mask is None else int(tgt_face_exclusion_mask.sum()//255)}px")

    alpha = cv2.GaussianBlur(combined, (15, 15), 0).astype(np.float32) / 255.0
    alpha = np.clip(alpha * 1.4, 0.0, 1.0)
    a3    = alpha[:, :, np.newaxis]
    out   = (src_warp.astype(np.float32) * a3 +
             tgt_bgr.astype(np.float32) * (1 - a3)).astype(np.uint8)
    return out
