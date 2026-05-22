"""
GFPGAN v1.4 ONNX wrapper — face restoration for the V2 face-swap pipeline.

Sharpens the soft 128 px inswapper output by aligning the face to the
ArcFace template at 512 px, running restoration, and pasting the result
back into the full frame with a soft alpha mask.

Spec verified against xuanandsix/GFPGAN-onnxruntime-demo (master/demo_onnx.py):
  Input  : (1, 3, 512, 512) RGB float32 normalised to [-1, 1] via (x/255-0.5)/0.5
  Output : same shape; restored face, same normalisation
  Channel order in NUMPY: NCHW (batch, channel, h, w)

Model download (340 MB):
  https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


GF_SIZE = 512

# Standard ArcFace 5-point template scaled from 112 px ref to 512 px.
_ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _arcface_template(size: int) -> np.ndarray:
    return _ARCFACE_TEMPLATE_112 * (size / 112.0)


class GFPGANOnnx:
    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort
        p = str(onnx_path)
        if not Path(p).exists():
            raise FileNotFoundError(p)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            p,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def restore_face(self, full_bgr: np.ndarray, kps5: np.ndarray) -> np.ndarray:
        """Align the face in `full_bgr` to a 512 px ArcFace patch, restore via
        GFPGAN, and paste back into the full frame with a soft alpha edge.

        Identity-preserving by design — GFPGAN learns realistic face priors
        and applies them only inside the aligned crop. Outside the face hull
        the original pixels are kept verbatim.
        """
        h, w = full_bgr.shape[:2]
        kps5 = np.asarray(kps5, dtype=np.float32)

        # 1. Align face to template at 512
        dst = _arcface_template(GF_SIZE)
        M, _ = cv2.estimateAffinePartial2D(kps5, dst, method=cv2.LMEDS)
        if M is None:
            return full_bgr
        crop = cv2.warpAffine(full_bgr, M, (GF_SIZE, GF_SIZE),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

        # 2. Preprocess: BGR uint8 → RGB float32 in [-1, 1], NCHW
        x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[np.newaxis]
        # Some GFPGAN ONNX exports expect float32, the public hacksider one
        # uses float32. If a fp16 export is supplied later, cast accordingly.
        x = x.astype(np.float32)

        # 3. Inference
        out = self.session.run([self.output_name], {self.input_name: x})[0][0]

        # 4. Postprocess: [-1, 1] RGB → BGR uint8
        rgb = (out.transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1) * 255.0
        restored_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # 5. Paste back via inverse affine
        M_inv = cv2.invertAffineTransform(M)
        pasted = cv2.warpAffine(restored_bgr, M_inv, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)

        # 6. Soft alpha mask from kps spread — feathers the paste-back edge
        #    so the restored face blends with surrounding (un-restored) pixels
        mn, mx = kps5.min(axis=0), kps5.max(axis=0)
        cx, cy = (mn + mx) / 2.0
        rx     = max(8.0, (mx[0] - mn[0]) * 1.10)
        ry     = max(8.0, (mx[1] - mn[1]) * 1.35)
        mask   = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)),
                    0, 0, 360, 255, -1)
        alpha = cv2.GaussianBlur(mask, (41, 41), 0).astype(np.float32) / 255.0
        a3    = alpha[:, :, np.newaxis]
        return (pasted.astype(np.float32) * a3 +
                full_bgr.astype(np.float32) * (1.0 - a3)).astype(np.uint8)
