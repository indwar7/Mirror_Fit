"""Phase 4: LoRA fine-tune of Wan 2.1 1.3B on composite -> real pairs.

The model already knows what fabric and people look like. What it does
not know is this garment, on this camera, under this light. A LoRA on
the attention layers is enough for that, and leaves the base model
intact so a future base swap (LiveVVT) keeps the data.

Usage:
    python -m training.train --pairs data/pairs --out runs/jacket-v1
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import torch
from torch.utils.data import DataLoader

import config
from .dataset import PairDataset

log = logging.getLogger("train")


def build_pipeline(device: str):
    """Load the base model with a LoRA attached to its attention layers."""
    from diffusers import AutoPipelineForImage2Image  # noqa: PLC0415
    from peft import LoraConfig  # noqa: PLC0415

    pipe = AutoPipelineForImage2Image.from_pretrained(
        config.MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # Freeze everything; only the adapter learns.
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    pipe.unet.add_adapter(
        LoraConfig(
            r=32,
            lora_alpha=32,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    return pipe


def lora_params(unet):
    return [p for p in unet.parameters() if p.requires_grad]


def train(args) -> int:
    device = config.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.error("CUDA requested but not available; set LIVE_DEVICE=cpu to force")
        return 1

    train_set = PairDataset(args.pairs / "train", size=args.size)
    loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=True,
    )
    log.info("%d training pairs, %d batches/epoch", len(train_set), len(loader))

    pipe = build_pipeline(device)
    unet, vae = pipe.unet, pipe.vae
    params = lora_params(unet)
    log.info("training %d LoRA tensors (%.1fM params)",
             len(params), sum(p.numel() for p in params) / 1e6)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    scaler = torch.amp.GradScaler(device.split(":")[0])

    prompt = pipe.encode_prompt(
        config.PROMPT, device=device, num_images_per_prompt=args.batch,
        do_classifier_free_guidance=False,
    )[0]

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    step = 0
    for epoch in range(args.epochs):
        t0 = time.monotonic()
        running = 0.0
        for batch in loader:
            src = batch["input"].to(device, torch.float16)
            tgt = batch["target"].to(device, torch.float16)

            with torch.no_grad():
                # Both halves go through the same VAE, so the loss is in
                # latent space where the UNet actually operates.
                z_src = vae.encode(src).latent_dist.sample() * vae.config.scaling_factor
                z_tgt = vae.encode(tgt).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(z_tgt)
            # Low timesteps only. Inference runs at strength 0.35, so
            # training at the full noise range would spend most of its
            # budget on a regime production never enters.
            hi = int(pipe.scheduler.config.num_train_timesteps * config.STRENGTH)
            t = torch.randint(0, max(hi, 1), (z_tgt.shape[0],), device=device).long()
            noisy = pipe.scheduler.add_noise(z_tgt, noise, t)

            # The composite is the conditioning: concatenating it is what
            # makes this image-to-image rather than free generation.
            model_in = torch.cat([noisy, z_src], dim=1)

            with torch.autocast(device.split(":")[0], dtype=torch.float16):
                pred = unet(model_in, t, encoder_hidden_states=prompt).sample
                loss = torch.nn.functional.mse_loss(pred.float(), noise.float())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += loss.item()
            step += 1
            if step % args.log_every == 0:
                log.info("epoch %d step %d loss %.4f lr %.2e",
                         epoch, step, running / args.log_every, sched.get_last_lr()[0])
                history.append({"step": step, "loss": running / args.log_every})
                running = 0.0

        log.info("epoch %d done in %.1fs", epoch, time.monotonic() - t0)
        ckpt = args.out / f"epoch-{epoch:03d}"
        unet.save_attn_procs(ckpt)
        log.info("saved %s", ckpt)

    unet.save_attn_procs(args.out / "final")
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    log.info("LoRA written to %s — set LIVE_LORA_PATH to use it", args.out / "final")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
