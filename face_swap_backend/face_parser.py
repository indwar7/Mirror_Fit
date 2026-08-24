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


def _border_fade(mask: np.ndarray, fade_px: int) -> np.ndarray:
    """Ramp a mask to zero near the image border.

    Hair almost always runs off the edge of an avatar portrait, so its mask
    ends in a straight line at the crop. Warp that onto a head and the straight
    line comes with it — the hard vertical and horizontal seams that read as a
    rectangle laid over the picture rather than as hair. Fading the mask into
    the border turns that cut into a gradient the later feather hides.
    """
    if fade_px <= 0:
        return mask
    h, w = mask.shape[:2]
    fade = min(int(fade_px), h // 2, w // 2)
    if fade < 2:
        return mask
    edge = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    ry = np.ones(h, np.float32); ry[:fade], ry[-fade:] = edge, edge[::-1]
    rx = np.ones(w, np.float32); rx[:fade], rx[-fade:] = edge, edge[::-1]
    return (mask.astype(np.float32) * ry[:, None] * rx[None, :]).astype(np.uint8)


def _fill_outward(color: np.ndarray, alpha: np.ndarray,
                  valid: float = 0.06) -> np.ndarray:
    """Smear the colour outward into transparent pixels.

    Everything downstream that grows or feathers the paint mask samples pixels
    just outside the hair — and just outside the hair the warped source is
    black. That is where the dark rim along the hairline and the black slab
    under the chin came from, and it is why an ear strip with no avatar hair
    over it would otherwise be painted pure black.

    Pyramid push-pull: halve the premultiplied colour and its weight until the
    image is a couple of pixels across, then walk back up filling each level's
    gaps from the coarser one. Going through the pyramid rather than blurring
    in place matters — a fixed blur only reaches as far as its radius, so a
    region further from the hair than that stays at zero weight and divides
    out to black. Every pixel gets a colour here however far it is.

    Pixels that were already valid come back untouched, so the hair stays sharp.
    """
    a = alpha.astype(np.float32) / 255.0
    if float(a.max()) <= 0.0:
        return color
    c = color.astype(np.float32) * a[:, :, np.newaxis]

    pyr_c, pyr_w = [c], [a]
    while min(pyr_c[-1].shape[:2]) > 2:
        pyr_c.append(cv2.pyrDown(pyr_c[-1]))
        pyr_w.append(cv2.pyrDown(pyr_w[-1]))

    up_c, up_w = pyr_c[-1], pyr_w[-1]
    for i in range(len(pyr_c) - 2, -1, -1):
        hh, ww = pyr_c[i].shape[:2]
        up_c = cv2.resize(up_c, (ww, hh), interpolation=cv2.INTER_LINEAR)
        up_w = cv2.resize(up_w, (ww, hh), interpolation=cv2.INTER_LINEAR)
        gap  = 1.0 - np.minimum(pyr_w[i], 1.0)
        up_c = pyr_c[i] + up_c * gap[:, :, np.newaxis]
        up_w = pyr_w[i] + up_w * gap

    filled = up_c / np.maximum(up_w, 1e-6)[:, :, np.newaxis]
    keep   = (a > valid)[:, :, np.newaxis]
    return np.clip(np.where(keep, color.astype(np.float32), filled),
                   0, 255).astype(np.uint8)


def transfer_hair(
    src_bgr: np.ndarray,
    src_hair_mask: np.ndarray,
    src_face_kps: np.ndarray,
    tgt_bgr: np.ndarray,
    tgt_hair_mask: np.ndarray,
    tgt_face_kps: np.ndarray,
    tgt_face_exclusion_mask: Optional[np.ndarray] = None,
    tgt_head_region_mask:    Optional[np.ndarray] = None,
    feather_px: int = 21,
    erode_px:   int = 7,
    tgt_extra_paint_mask: Optional[np.ndarray] = None,
    border_fade_px: int = 24,
    grow_px: int = 0,
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

    `tgt_extra_paint_mask` is painted whether or not the warped avatar hair
    covers it — used for the ear strips, since hair that parts around the
    avatar's own ears otherwise leaves the user's ears showing through.
    `grow_px` dilates the warped hair before painting, to close small gaps.
    """
    th, tw = tgt_bgr.shape[:2]

    M, _ = cv2.estimateAffinePartial2D(
        src_face_kps.astype(np.float32),
        tgt_face_kps.astype(np.float32),
    )
    if M is None:
        return tgt_bgr

    # 1. Fade the mask into the source border, then warp colour and mask
    #    together as a premultiplied pair so warpAffine cannot drag background
    #    in along the edges.
    src_mask_f = _border_fade(src_hair_mask, border_fade_px)
    src_alpha  = (src_mask_f.astype(np.float32) / 255.0)[:, :, np.newaxis]
    src_premul = (src_bgr.astype(np.float32) * src_alpha).astype(np.uint8)

    src_warp = cv2.warpAffine(
        src_premul, M, (tw, th),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    mask_warp = cv2.warpAffine(
        src_mask_f, M, (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # 1b. Divide the alpha back out. Premultiplying is what keeps the warp
    #     clean, but the colour has to be un-premultiplied afterwards or the
    #     blend at the end multiplies by alpha a SECOND time. That double
    #     multiply is what darkened every soft hair edge towards black — the
    #     rim around the hairline and the black slab under the chin — because
    #     hair_mask() Gaussian-blurs its output, so the whole hair silhouette
    #     is soft-edged, not binary.
    aw = (mask_warp.astype(np.float32) / 255.0)[:, :, np.newaxis]
    src_warp = np.clip(src_warp.astype(np.float32) / np.maximum(aw, 0.06),
                       0, 255).astype(np.uint8)
    src_warp = _fill_outward(src_warp, mask_warp)

    # 2. Paint = warped avatar hair, grown a little, plus anything the caller
    #    insists on covering (the ear strips).
    paint = mask_warp.copy()
    if tgt_extra_paint_mask is not None:
        extra = tgt_extra_paint_mask
        if extra.shape[:2] != (th, tw):
            extra = cv2.resize(extra, (tw, th), interpolation=cv2.INTER_NEAREST)
        paint = cv2.max(paint, extra)
    if grow_px > 0:
        k = max(3, int(grow_px) | 1)
        paint = cv2.dilate(paint,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

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

    # Erode the paint mask by a few pixels so the Gaussian feather (next
    # step) doesn't extend hair pixels INTO the background — that's what
    # created the visible "blur halo / rectangular halo" around the head
    # in the earlier broken version. Erode pulls the hair edge slightly
    # inward, the feather then expands it back to roughly the original
    # silhouette but with a soft alpha rather than a hard cut.
    if erode_px > 0:
        k = max(3, erode_px | 1)
        paint = cv2.erode(paint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

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

    # 6. Soft feather. Kernel is configurable; 21 px (default) gives a
    # ~10 px soft edge — tight enough to look like real hair, wide enough
    # to hide the warp seam. Old default was 41 px which created a visible
    # halo extending into the background.
    k = max(3, feather_px | 1)
    alpha = cv2.GaussianBlur(paint, (k, k), 0).astype(np.float32) / 255.0
    a3    = alpha[:, :, np.newaxis]
    out   = (src_warp.astype(np.float32) * a3 +
             tgt_bgr.astype(np.float32) * (1.0 - a3)).astype(np.uint8)
    return out
