"""
CodeFormer ONNX wrapper for LUCY face-swap backend.

Photo-realistic face restoration. Sharpens identity + skin detail on the
soft 128px inswapper output so the swap looks like the user's actual face,
not a blurred patch.

Model file expected at: face_swap_backend/models/models/codeformer.onnx
Download from:
  https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx
  (~200 MB, pure onnxruntime, no basicsr dependency)

Pipeline per call:
  1. Crop face from full image using ArcFace template (5 keypoints)
  2. Resize crop to 512x512, normalize to [-1, 1] RGB
  3. ONNX forward (returns 512x512 restored RGB)
  4. Paste back via inverse affine + soft alpha boundary
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

CF_SIZE = 512

# Standard ArcFace template at 512px (scaled from 112px reference).
_ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _arcface_template(size: int) -> np.ndarray:
    return _ARCFACE_TEMPLATE_112 * (size / 112.0)


class CodeFormerONNX:
    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort
        self._path = str(onnx_path)
        if not Path(self._path).exists():
            raise FileNotFoundError(self._path)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self._path,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        ins  = self.session.get_inputs()
        outs = self.session.get_outputs()
        self.face_input_name   = ins[0].name
        # Some exports take a fidelity weight as a second input
        self.weight_input_name = ins[1].name if len(ins) > 1 else None
        self.output_name       = outs[0].name
        log.info(f"[CodeFormer] loaded {self._path}  inputs={[i.name for i in ins]}")

    # ── Inference ──────────────────────────────────────────────────────────────
    def restore_face(
        self,
        full_bgr: np.ndarray,
        kps5: np.ndarray,
        weight: float = 0.7,
        face_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Restore the face region of `full_bgr` aligned by 5-point kps.

        Args:
            full_bgr: full BGR image containing the face (any size)
            kps5:     (5, 2) ArcFace keypoints in `full_bgr` coordinates
            weight:   fidelity (0 = pure restoration, 1 = preserve input);
                      0.7 keeps identity while sharpening detail
            face_mask: optional binary mask the same shape as full_bgr
                      to limit the paste-back area; if None, uses a soft
                      ellipse derived from the kps spread
        Returns:
            full BGR image with face region restored. Same shape as input.
        """
        h, w = full_bgr.shape[:2]
        kps5 = np.asarray(kps5, dtype=np.float32)

        # 1. Align face to template
        dst = _arcface_template(CF_SIZE)
        M, _ = cv2.estimateAffinePartial2D(kps5, dst, method=cv2.LMEDS)
        if M is None:
            return full_bgr
        face_crop = cv2.warpAffine(
            full_bgr, M, (CF_SIZE, CF_SIZE),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )

        # 2. Normalize to [-1, 1] RGB, NCHW
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        rgb = rgb.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 512, 512)

        # 3. Run ONNX
        feed = {self.face_input_name: rgb.astype(np.float32)}
        if self.weight_input_name is not None:
            feed[self.weight_input_name] = np.array([float(weight)], dtype=np.float64)
        try:
            out = self.session.run([self.output_name], feed)[0]
        except Exception:
            # Some exports want fp32 weight
            if self.weight_input_name is not None:
                feed[self.weight_input_name] = np.array([float(weight)], dtype=np.float32)
                out = self.session.run([self.output_name], feed)[0]
            else:
                raise

        restored_rgb = ((out[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)

        # 4. Paste back via inverse affine
        M_inv = cv2.invertAffineTransform(M)
        pasted = cv2.warpAffine(
            restored_bgr, M_inv, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

        # 5. Build a soft alpha mask so the paste blends seamlessly.
        if face_mask is None:
            # Soft ellipse from kps spread
            mn, mx = kps5.min(axis=0), kps5.max(axis=0)
            cx, cy = (mn + mx) / 2.0
            rx     = (mx[0] - mn[0]) * 1.05
            ry     = (mx[1] - mn[1]) * 1.30
            mask   = np.zeros((h, w), np.uint8)
            cv2.ellipse(
                mask, (int(cx), int(cy)),
                (max(8, int(rx)), max(8, int(ry))),
                0, 0, 360, 255, -1,
            )
        else:
            mask = (face_mask > 0).astype(np.uint8) * 255

        alpha = cv2.GaussianBlur(mask, (51, 51), 0).astype(np.float32) / 255.0
        a3    = alpha[:, :, np.newaxis]
        out_bgr = (pasted.astype(np.float32) * a3 +
                   full_bgr.astype(np.float32) * (1.0 - a3)).astype(np.uint8)
        return out_bgr
