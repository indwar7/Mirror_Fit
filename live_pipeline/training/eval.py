"""Phase 4 evaluation, on people the model never saw.

Three numbers, because no single one catches what goes wrong here:

  PSNR    pixel accuracy. Rewards blur, so never read alone.
  SSIM    structure. Catches a garment in the wrong place.
  LPIPS   perceptual. Closest to "does this look real to a person",
          and the one to trust when the three disagree.

Also writes preview grids — input | prediction | target — because the
failure that matters most (the garment drifting during a turn) shows up
in a strip of images and not in a mean.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import numpy as np
import torch

import config
from .dataset import PairDataset

log = logging.getLogger("eval")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on greyscale. Enough to rank checkpoints."""
    a = a.mean(axis=2)
    b = b.mean(axis=2)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    c1, c2 = 0.01**2, 0.03**2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2))
                 / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2)))


def to_img(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) in [-1,1] -> (H,W,3) in [0,1]."""
    return ((t.detach().float().cpu().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)


def run(args) -> int:
    import cv2  # noqa: PLC0415

    device = config.DEVICE
    data = PairDataset(args.pairs / "eval", size=args.size, flip=False)
    log.info("%d held-out pairs", len(data))

    from diffusers import AutoPipelineForImage2Image  # noqa: PLC0415

    pipe = AutoPipelineForImage2Image.from_pretrained(
        config.MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    if args.lora:
        pipe.unet.load_attn_procs(args.lora)
        log.info("loaded LoRA %s", args.lora)
    else:
        log.info("no LoRA — this is the baseline to beat")

    lpips_fn = None
    try:
        import lpips  # noqa: PLC0415

        lpips_fn = lpips.LPIPS(net="alex").to(device)
    except Exception as exc:
        log.warning("LPIPS unavailable (%s); reporting PSNR and SSIM only", exc)

    args.out.mkdir(parents=True, exist_ok=True)
    rows, grids = [], []
    for i in range(len(data)):
        item = data[i]
        src = item["input"].unsqueeze(0).to(device, torch.float16)
        from PIL import Image  # noqa: PLC0415

        src_pil = Image.fromarray((to_img(item["input"]) * 255).astype(np.uint8))
        pred = pipe(
            prompt=config.PROMPT,
            negative_prompt=config.NEGATIVE_PROMPT,
            image=src_pil,
            strength=config.STRENGTH,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
        ).images[0]

        p = np.asarray(pred, dtype=np.float32) / 255.0
        t = to_img(item["target"])
        row = {"name": item["name"], "psnr": psnr(p, t), "ssim": ssim(p, t)}
        if lpips_fn is not None:
            with torch.no_grad():
                a = torch.from_numpy(p).permute(2, 0, 1)[None].to(device) * 2 - 1
                b = item["target"].unsqueeze(0).to(device)
                row["lpips"] = float(lpips_fn(a.float(), b.float()).item())
        rows.append(row)

        if len(grids) < args.previews:
            strip = np.hstack([to_img(item["input"]), p, t])
            grids.append((strip * 255).astype(np.uint8)[:, :, ::-1])

        if (i + 1) % 20 == 0:
            log.info("%d/%d", i + 1, len(data))

    summary = {
        k: round(float(np.mean([r[k] for r in rows])), 4)
        for k in ("psnr", "ssim", "lpips") if k in rows[0]
    }
    summary["pairs"] = len(rows)
    summary["lora"] = str(args.lora) if args.lora else None

    (args.out / "metrics.json").write_text(
        json.dumps({"summary": summary, "per_image": rows}, indent=2)
    )
    if grids:
        cv2.imwrite(str(args.out / "previews.png"), np.vstack(grids))

    log.info("summary: %s", summary)
    log.info("previews: input | prediction | target -> %s", args.out / "previews.png")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=pathlib.Path, required=True)
    ap.add_argument("--lora", type=pathlib.Path, default=None,
                    help="omit to score the stock model as a baseline")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--steps", type=int, default=config.STEPS)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--previews", type=int, default=8)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
