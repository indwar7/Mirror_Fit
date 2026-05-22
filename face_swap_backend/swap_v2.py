"""
LUCY face swap V2 — fresh Deep-Live-Cam-style pipeline.

Why this exists:
  V1 (main.py:_swap_live) accumulated MIXED_CLONE seamless blending,
  CodeFormer on the full frame, optional hair swap, optional mouth mask,
  smoothing — stacked transforms that visibly degraded output quality on
  real cross-ethnicity swaps (visible blur, halo, jaw smear).

  V2 mirrors the open-source SOTA real-time face-swap pipeline shipped by
  hacksider/Deep-Live-Cam (60k+ stars on GitHub, 2025). It uses the SAME
  base inswapper model but in FP16 (faster), restores with GFPGAN-ONNX
  (not the broken pip-install GFPGAN), and composites the user's real
  mouth back over the swap so lip motion is verbatim with zero model lag.

Pipeline per frame:
  1. Detect target face (InsightFace buffalo_l)
  2. inswapper_128_fp16 — produce 128 px swapped face patch
  3. Warp back to original frame coords, soft-alpha blend on the face hull
  4. GFPGAN v1.4 ONNX — restore the swapped face area to look photoreal
  5. Mouth mask — overlay user's actual mouth/lip pixels for real lip sync
  6. Output

Models required (place under face_swap_backend/models/models/):
  - inswapper_128_fp16.onnx     (278 MB)
  - GFPGANv1.4.onnx             (340 MB)
  - buffalo_l (auto-downloaded by InsightFace on first run)
"""
from __future__ import annotations

import pathlib
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  ArcFace 5-point template
# ─────────────────────────────────────────────────────────────────────────────
_ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _arcface_template(size: int) -> np.ndarray:
    return _ARCFACE_TEMPLATE_112 * (size / 112.0)


# 106-landmark mouth indices for InsightFace 2d106det
_LMK_MOUTH = list(range(52, 72))


class FaceSwapV2:
    """Stateless face-swap engine. One instance per process — thread-safe
    inference is delegated to the ONNX session's internal locking."""

    def __init__(
        self,
        inswapper_fp16_path: str,
        gfpgan_path: Optional[str] = None,
    ):
        import onnxruntime as ort
        import onnx
        from onnx.numpy_helper import to_array

        # ── inswapper_128_fp16 ────────────────────────────────────────────────
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.inswapper = ort.InferenceSession(
            inswapper_fp16_path,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.inswapper_in  = [i.name for i in self.inswapper.get_inputs()]
        self.inswapper_out = [o.name for o in self.inswapper.get_outputs()]
        # The fp16 model has the same emap initializer as the fp32 variant
        # (it's the last initializer in the graph). We project the source
        # ArcFace embedding through it before passing to the model.
        model_proto = onnx.load(inswapper_fp16_path)
        self.emap = to_array(model_proto.graph.initializer[-1]).astype(np.float32)
        # Probe the model's input dtype — fp16 vs fp32. We cast inputs to match.
        self._target_dtype = (
            np.float16 if self.inswapper.get_inputs()[0].type == "tensor(float16)"
            else np.float32
        )

        # ── GFPGAN v1.4 (optional) ────────────────────────────────────────────
        self.gfpgan = None
        if gfpgan_path and pathlib.Path(gfpgan_path).exists():
            from gfpgan_onnx import GFPGANOnnx
            self.gfpgan = GFPGANOnnx(gfpgan_path)

    # ── Inswapper ───────────────────────────────────────────────────────────
    def _run_inswapper(self, source_embedding: np.ndarray,
                       target_img: np.ndarray, target_kps: np.ndarray):
        """Run the swap. Returns (warped_back_swap, M) where warped_back_swap
        has the swap pixels at the target face position and M is the affine
        used (for downstream re-alignment if any)."""
        # 1. Source latent: embedding @ emap, L2-normalise
        emb    = source_embedding.reshape(1, -1).astype(np.float32)
        latent = emb @ self.emap
        latent = latent / (np.linalg.norm(latent) + 1e-9)

        # 2. Warp target face patch to 128 px ArcFace template
        M = cv2.estimateAffinePartial2D(target_kps, _arcface_template(128))[0]
        patch = cv2.warpAffine(target_img, M, (128, 128), flags=cv2.INTER_LINEAR)

        # 3. Preprocess: BGR uint8 → RGB float [0, 1], NCHW
        x = (patch.astype(np.float32) / 255.0)[:, :, ::-1]   # BGR→RGB
        x = x.transpose(2, 0, 1)[np.newaxis]
        x = x.astype(self._target_dtype)
        src = latent.astype(self._target_dtype)

        # 4. Inference
        out = self.inswapper.run(
            self.inswapper_out,
            {self.inswapper_in[0]: x, self.inswapper_in[1]: src},
        )[0][0]

        # 5. Postprocess: float RGB [0, 1] → BGR uint8 at 128 px
        out = out.transpose(1, 2, 0)[:, :, ::-1]  # RGB→BGR
        out = (np.clip(out.astype(np.float32), 0, 1) * 255).astype(np.uint8)

        # 6. Warp back to original frame coords
        M_inv = cv2.invertAffineTransform(M)
        warped = cv2.warpAffine(
            out, M_inv, (target_img.shape[1], target_img.shape[0]),
            flags=cv2.INTER_CUBIC,
        )
        return warped, M

    # ── Mouth mask compositing ──────────────────────────────────────────────
    @staticmethod
    def _apply_mouth_mask(swap_bgr: np.ndarray, orig_bgr: np.ndarray,
                          landmark_2d_106: np.ndarray,
                          dilate_px: int = 8, feather_px: int = 11):
        """Overlay user's ACTUAL mouth/lip pixels (from orig_bgr) onto the
        swapped face at the mouth landmarks. Restores real-time lip sync
        with zero model lag — Wav2Lip becomes optional.
        """
        if landmark_2d_106 is None or len(landmark_2d_106) < 72:
            return swap_bgr
        pts = landmark_2d_106[_LMK_MOUTH].astype(np.int32)
        if len(pts) < 3:
            return swap_bgr
        h, w = swap_bgr.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [cv2.convexHull(pts)], 255)
        k = max(3, dilate_px | 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        sigma = max(1, feather_px | 1)
        mask = cv2.GaussianBlur(mask, (sigma * 2 + 1, sigma * 2 + 1), 0)
        if int(mask.sum()) < 1000:
            return swap_bgr
        alpha = mask.astype(np.float32) / 255.0
        a3    = alpha[:, :, np.newaxis]
        return (orig_bgr.astype(np.float32) * a3 +
                swap_bgr.astype(np.float32) * (1.0 - a3)).astype(np.uint8)

    # ── Face hull mask (for swap-region alpha blend) ────────────────────────
    @staticmethod
    def _face_hull_mask(landmark_2d_106: np.ndarray, shape_hw: tuple,
                        feather_px: int = 21):
        """Soft-edged mask covering the face area defined by the 106 landmarks.
        Used to alpha-blend the swap pixels onto the original frame without
        a hard cut-paste edge."""
        h, w = shape_hw[:2]
        mask = np.zeros((h, w), np.uint8)
        if landmark_2d_106 is None or len(landmark_2d_106) < 10:
            return mask
        pts = landmark_2d_106.astype(np.int32)
        cv2.fillPoly(mask, [cv2.convexHull(pts)], 255)
        k = max(3, feather_px | 1)
        return cv2.GaussianBlur(mask, (k * 2 + 1, k * 2 + 1), 0)

    # ── Main entrypoint ─────────────────────────────────────────────────────
    def swap_frame(
        self,
        source_embedding: np.ndarray,
        target_img: np.ndarray,
        target_face,
        enable_gfpgan: bool = True,
        enable_mouth_mask: bool = True,
    ) -> Optional[np.ndarray]:
        """Run the full V2 pipeline on a single frame.

        Args:
          source_embedding: 512-d ArcFace embedding of the avatar (cached)
          target_img:       BGR uint8 of the live webcam frame
          target_face:      InsightFace Face object detected from target_img
          enable_gfpgan:    set False to skip restoration (faster but softer)
          enable_mouth_mask:set False to use the swap's mouth verbatim

        Returns:
          BGR uint8 of the swapped frame, or None if no face usable.
        """
        if target_face is None or getattr(target_face, "kps", None) is None:
            return None

        # 1. Inswapper FP16 → warped swap at face location
        warped_swap, _ = self._run_inswapper(
            source_embedding, target_img, target_face.kps,
        )

        # 2. Soft alpha blend the swap onto the original frame
        lmk = getattr(target_face, "landmark_2d_106", None)
        hull_mask = self._face_hull_mask(lmk, target_img.shape)
        alpha = hull_mask.astype(np.float32) / 255.0
        a3    = alpha[:, :, np.newaxis]
        composited = (warped_swap.astype(np.float32) * a3 +
                      target_img.astype(np.float32) * (1.0 - a3)).astype(np.uint8)

        # 3. GFPGAN restoration on the composited frame (aligned via kps)
        if enable_gfpgan and self.gfpgan is not None:
            try:
                composited = self.gfpgan.restore_face(composited, target_face.kps)
            except Exception:
                pass  # restoration is best-effort

        # 4. Mouth mask — user's real mouth over the restored face
        if enable_mouth_mask and lmk is not None:
            composited = self._apply_mouth_mask(composited, target_img, lmk)

        return composited
