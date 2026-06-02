# Train CatVTON LoRA on VITON-HD (free)

Fine-tune your try-on model on 13k real person+garment+result triples. Free GPU, free dataset.

## What you get

A ~20 MB LoRA adapter file. Drop it into your RunPod backend, restart, and the try-on quality for clothing types in VITON-HD (jackets, shirts, dresses) jumps.

## Steps

### 1. Open Kaggle

- Go to https://kaggle.com → Sign in
- Create New Notebook
- Click **Settings** (right side):
  - **Accelerator**: GPU P100 (or T4 ×2 if P100 unavailable)
  - **Persistence**: Files and Variables
  - **Internet**: On

### 2. Add VITON-HD dataset

In the notebook, click **Add Data** → search **"VITON-HD"** or **"zalando viton"**. Pick the dataset with `train/image/` and `train/cloth/` folders. The most common one on Kaggle is `marquis03/high-resolution-viton-zalando-dataset` — search that exact name if generic search doesn't surface it.

### 3. Upload the notebook

- File → Import Notebook → upload `train_catvton_lora_vitonhd.ipynb` from this folder
- OR: copy-paste cells from the .ipynb into a new Kaggle notebook

### 4. Run all cells

- Run → Run All
- First run: pip install (~3 min)
- Model download (~5 min)
- Training: 4-8 hours on P100 for 2000-sample subset (`cfg.train_subset = 2000`)
  - Full 11k samples: ~12 hours, may exceed Kaggle's 12hr/session limit
  - Start with 2000, see how the output looks, then scale up
- A sanity comparison plot prints at the end (input | garment | ground truth | LoRA prediction)

### 5. Download the LoRA

In Kaggle's right panel under **Output**: `lora_catvton_final.zip` (~20-40 MB). Right-click → Download.

### 6. Deploy to backend

On your RunPod/server box:

```bash
unzip lora_catvton_final.zip -d /workspace/lora_catvton
export VTON_LORA_CHECKPOINT=/workspace/lora_catvton/final
cd tryon_backend && python server.py
```

That's it. The backend's `_load_tier2()` auto-detects this is a PEFT LoRA adapter (sees `adapter_config.json`), loads the CatVTON base UNet, attaches your LoRA on top, and uses it for inference.

You should see in the logs:
```
Loading PEFT LoRA adapter from /workspace/lora_catvton/final
LoRA adapter merged into base UNet.
Tier 2 ready — torch.compile. ~15-20fps
```

## Tuning knobs

In the notebook's `Config` cell:

| Knob | Default | What it changes |
|---|---|---|
| `train_subset` | 2000 | How many pairs to train on. 0 = all 11k. More = better but slower. |
| `image_size` | 384 | Training resolution. 512 = better quality but needs more VRAM. |
| `lora_rank` | 16 | LoRA expressiveness. 32 = more learning capacity, 2× file size. |
| `epochs` | 1 | Passes over data. 2-3 for noticeably better results. |
| `lr` | 1e-4 | Learning rate. Lower if loss is unstable, higher if loss isn't dropping. |

## If something breaks

**"VITON-HD not found"** — you didn't add the dataset, or the mirror has a non-standard folder layout. Look at `/kaggle/input/` in a code cell:
```python
!find /kaggle/input -maxdepth 4 -type d
```
Identify which folder contains `image/` and `cloth/`, manually set `VITON_ROOT = Path('/kaggle/input/...')`

**OOM on P100** — drop `batch_size=1`, `image_size=320`. Last resort: enable gradient checkpointing:
```python
unet.gradient_checkpointing_enable()
```

**Kaggle session disconnects mid-training** — that's why we save checkpoints every 500 steps. Resume from the latest in `cfg.output_dir`. Persistence must be ON.

**Loss not dropping** — common causes:
- LR too low: try `lr=3e-4`
- Mask is wrong: print `batch['mask'].mean()` — should be 0.10-0.30. If 0 or 1, mask detection failed.
- Bad agnostic: visualize a `batch['agnostic']` sample. If it's identical to `person`, the agnostic fallback was wrong.

## What this WON'T fix

- **Realtime AR feel** — diffusion is still 1-5 sec/frame. This improves photo try-on quality, not realtime.
- **Garment types not in VITON-HD** — pants, shoes, accessories. VITON-HD is upper-body only.
- **Body types under-represented in VITON-HD** — the dataset is mostly slim adult women. Bias carries over.

For those, you'd need to fine-tune on more diverse data later.
