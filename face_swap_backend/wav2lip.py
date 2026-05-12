"""
Wav2Lip ONNX wrapper for LUCY face-swap backend.

Generates a video of a face speaking, lip-synced to a given audio waveform.

Model file expected at: face_swap_backend/models/models/wav2lip_gan.onnx
Download from any of:
  https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2Lip/wav2lip_gan.onnx
  https://github.com/instant-high/wav2lip-onnx-256/releases

Usage:
    w2l = Wav2LipONNX("/path/to/wav2lip_gan.onnx")
    frames = w2l.generate(face_bgr, audio_pcm, sr=16000)  # returns list of 96x96 BGR frames at 25 fps
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

WAV2LIP_FPS    = 25
MEL_HOPS_PER_F = 5     # 200 ms / 8 ms hop = ~5 mel hops per video frame at 25 fps + 16 kHz audio
MEL_WINDOW     = 16    # each frame attends to 16 mel time-steps centered on the frame
TARGET_PX      = 96    # wav2lip_gan input/output size


def _build_mel(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute the mel-spectrogram Wav2Lip expects.
    Returns (80, T) where T = ceil(len/200)."""
    import librosa
    if sr != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr    = 16000
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=800, hop_length=200, win_length=800,
        n_mels=80, fmin=55, fmax=7600,
    )
    mel = np.log(np.maximum(1e-5, mel))
    mel = (mel + 5.0) / 5.0   # rough normalisation Wav2Lip uses
    return mel.astype(np.float32)


class Wav2LipONNX:
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
        # Figure out input names dynamically — different exports use different names
        ins = self.session.get_inputs()
        self.face_input_name = None
        self.mel_input_name  = None
        for inp in ins:
            sh = inp.shape
            # Mel is (?, 1, 80, 16); face is (?, 6, 96, 96)
            if len(sh) == 4 and sh[1] == 1 and sh[2] in (80, "80"):
                self.mel_input_name = inp.name
            elif len(sh) == 4 and sh[1] == 6 and sh[2] in (TARGET_PX, str(TARGET_PX)):
                self.face_input_name = inp.name
        if self.face_input_name is None:
            self.face_input_name = ins[0].name
        if self.mel_input_name is None:
            self.mel_input_name = ins[1].name if len(ins) > 1 else ins[0].name
        log.info(f"[Wav2Lip] loaded {self._path} | inputs: face={self.face_input_name} mel={self.mel_input_name}")

    # ── Inference ──────────────────────────────────────────────────────────────
    def _infer(self, face_bgr96: np.ndarray, mel_chunk: np.ndarray) -> np.ndarray:
        """face_bgr96: (96,96,3) uint8 BGR.  mel_chunk: (80, 16) float32."""
        face_rgb = cv2.cvtColor(face_bgr96, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        masked   = face_rgb.copy()
        masked[TARGET_PX // 2:, :, :] = 0.0
        # 6-channel concat: lower-half-masked face + reference face
        face_in  = np.concatenate([masked, face_rgb], axis=2)            # (96,96,6)
        face_in  = face_in.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,6,96,96)
        mel_in   = mel_chunk[np.newaxis, np.newaxis, :, :].astype(np.float32) # (1,1,80,16)
        out      = self.session.run(None, {
            self.face_input_name: face_in,
            self.mel_input_name:  mel_in,
        })[0]
        out_rgb  = (out[0].transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    def generate(
        self,
        face_bgr: np.ndarray,
        audio: np.ndarray,
        audio_sr: int = 16000,
        face_bbox: Optional[tuple] = None,
    ) -> List[np.ndarray]:
        """Generate lip-synced frames for a single source face.

        Args:
            face_bgr: full BGR image of the avatar.
            audio:   1-d numpy float32 PCM at audio_sr.
            face_bbox: (x1,y1,x2,y2) bbox of the face in face_bgr. If None,
                       uses the whole image as the crop.

        Returns:
            List of full-size BGR frames with the face region lip-synced,
            sampled at 25 fps. Length = ceil(audio_sec * 25).
        """
        H, W = face_bgr.shape[:2]
        if face_bbox is None:
            face_bbox = (0, 0, W, H)
        x1, y1, x2, y2 = (int(v) for v in face_bbox)
        # Expand bbox slightly so we have room for context
        fw, fh = x2 - x1, y2 - y1
        pad = int(0.20 * max(fw, fh))
        x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
        x2c, y2c = min(W,  x2 + pad), min(H, y2 + pad)
        crop     = face_bgr[y1c:y2c, x1c:x2c]
        crop_h, crop_w = crop.shape[:2]
        face96   = cv2.resize(crop, (TARGET_PX, TARGET_PX), interpolation=cv2.INTER_LINEAR)

        mel = _build_mel(audio.astype(np.float32), sr=audio_sr)        # (80, T)
        n_frames = max(1, int(np.ceil(len(audio) / audio_sr * WAV2LIP_FPS)))

        frames: List[np.ndarray] = []
        for fi in range(n_frames):
            center = fi * MEL_HOPS_PER_F
            start  = center - MEL_WINDOW // 2
            end    = start + MEL_WINDOW
            # Pad if window goes off the edges
            if start < 0:
                pad_left = -start
                chunk = np.pad(mel[:, :end], ((0, 0), (pad_left, 0)), mode="edge")
            elif end > mel.shape[1]:
                pad_right = end - mel.shape[1]
                chunk = np.pad(mel[:, start:], ((0, 0), (0, pad_right)), mode="edge")
            else:
                chunk = mel[:, start:end]
            chunk = chunk[:, :MEL_WINDOW]

            mouth = self._infer(face96, chunk)            # (96,96,3) BGR
            # Composite: paste mouth back at the crop region of the full frame
            full = face_bgr.copy()
            mouth_big = cv2.resize(mouth, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            # Soft blend along a vertical gradient so the mouth replacement
            # doesn't show a hard seam at the lip line
            alpha = np.zeros((crop_h, crop_w), dtype=np.float32)
            alpha[crop_h // 2:, :] = 1.0
            alpha = cv2.GaussianBlur(alpha, (51, 51), 0)[:, :, np.newaxis]
            blended = (mouth_big.astype(np.float32) * alpha +
                       crop.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            full[y1c:y2c, x1c:x2c] = blended
            frames.append(full)
        return frames
