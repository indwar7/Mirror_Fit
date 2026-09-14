"""Paired image dataset for the garment LoRA."""
from __future__ import annotations

import pathlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """(composite, real) pairs written by prep.py.

    Augmentation is deliberately thin. Colour jitter would teach the
    model that the garment's colour is negotiable, and that colour is
    the whole product.
    """

    def __init__(self, root: pathlib.Path, size: int = 512, flip: bool = True) -> None:
        self.inputs = sorted((root / "input").glob("*.png"))
        self.root = root
        self.size = size
        self.flip = flip
        if not self.inputs:
            raise FileNotFoundError(f"no pairs under {root / 'input'}")
        missing = [p.name for p in self.inputs if not (root / "target" / p.name).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} inputs have no target, first: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.inputs)

    def _load(self, path: pathlib.Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BICUBIC)
        return np.asarray(img, dtype=np.float32) / 127.5 - 1.0

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        src = self.inputs[i]
        a = self._load(src)
        b = self._load(self.root / "target" / src.name)
        # Mirror both halves together, or the pair stops being a pair.
        if self.flip and np.random.rand() < 0.5:
            a, b = a[:, ::-1].copy(), b[:, ::-1].copy()
        return {
            "input": torch.from_numpy(a).permute(2, 0, 1),
            "target": torch.from_numpy(b).permute(2, 0, 1),
            "name": src.name,
        }
