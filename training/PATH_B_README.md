# Path B — Train your own Decart-grade try-on (free)

End-to-end recipe: train a CatVTON-style LoRA on VITON-HD, distill it to a
1-step LCM-LoRA, compile to TensorRT. The inference server's Tier 1 path
(`tryon_backend/model.py:_load_tier1`) loads the resulting engine and
runs the try-on at ~30 fps with CUDA Graph replay.

| Stage | What | Where | Time | Output |
| --- | --- | --- | --- | --- |
| 1 | Dataset prep | Kaggle | 1-2 days | `viton-hd-resized` Kaggle dataset |
| 2 | Train LoRA | Kaggle T4×2 | 4-7 days | `lucy_catvton_lora.safetensors` (~100 MB) |
| 3 | Distill LCM | Kaggle T4×2 | 1-2 days | `lucy_catvton_lcm_lora.safetensors` |
| 4 | TRT export | g5.xlarge (server) | half day | `engine.plan` |
| 5 | Wire it up | g5.xlarge | 10 min | Tier 1 active, ~30 fps |

Total wall time: ~2 weeks. Total GPU cost: **$0**. Kaggle gives free T4×2
for 30 hrs/week; the inference GPU is your existing g5.xlarge.

---

## Stage 1 — Dataset prep (Kaggle, ~1 hour active work)

1. Kaggle account + API token: <https://www.kaggle.com/settings/account> → Create New Token. Saves `kaggle.json`.
2. Open <https://www.kaggle.com/code> → New notebook.
3. Settings (right rail):
   - **Accelerator**: GPU T4 ×2 (or P100)
   - **Persistence**: Files and Variables
   - **Internet**: On
4. Add Data → search **VITON-HD**. Pick `marquis03/high-resolution-viton-zalando-dataset` (~13 GB, public).
5. Notebook code:
   ```python
   !mkdir -p /kaggle/working/viton-hd
   !cp -r /kaggle/input/high-resolution-viton-zalando-dataset/* /kaggle/working/viton-hd/
   # Optional: downscale to 512² to fit T4 VRAM
   from PIL import Image
   from pathlib import Path
   for p in Path('/kaggle/working/viton-hd').rglob('*.jpg'):
       Image.open(p).resize((512, 512), Image.LANCZOS).save(p)
   ```
6. Save the notebook output as a Kaggle Dataset called `viton-hd-resized`. (Notebook → File → Save as Dataset.)
7. Stage 2 + 3 notebooks reference `/kaggle/input/viton-hd-resized`.

---

## Stage 2 — Train CatVTON LoRA

Uses [`train_catvton_lora_vitonhd.ipynb`](train_catvton_lora_vitonhd.ipynb) (already in this folder).

1. New Kaggle notebook → same GPU settings.
2. Add Data → `viton-hd-resized` (from Stage 1).
3. **File → Import Notebook** → upload `train_catvton_lora_vitonhd.ipynb`.
4. Run all cells. **Wall time ~6 hrs / epoch; total ~18 hrs for 3 epochs.** Kaggle session caps at 12 hrs, so the notebook checkpoints every epoch and resumes on rerun.
5. Output: `/kaggle/working/lucy_catvton_lora.safetensors`.
6. **Download it.** Save it somewhere safe; Stage 3 needs it.

Tips:
- Watch loss on WandB (set `WANDB_API_KEY` as a Kaggle secret).
- If T4×2 OOMs, drop batch size from 4 → 2, or resolution from 512 → 384.
- LoRA rank 32 is the sweet spot. Higher = better fit, slower convergence.

---

## Stage 3 — LCM-LoRA distillation

Uses [`distill_lcm_lora.ipynb`](distill_lcm_lora.ipynb).

1. New Kaggle notebook with the same GPU + dataset setup.
2. Upload Stage 2's `lucy_catvton_lora.safetensors` as a Kaggle Dataset (private). Reference it at `/kaggle/input/lucy-catvton-lora/lucy_catvton_lora.safetensors`.
3. Import `distill_lcm_lora.ipynb`.
4. Set `TEACHER_LORA_PATH` in Cell 1 to the input path above.
5. Run. **Wall time ~6-10 hrs for 3 epochs.** Fits one Kaggle session.
6. Output: `/kaggle/working/lucy_catvton_lcm_lora.safetensors`.
7. Download.

Sanity check before downloading: Cell 7 in the notebook runs the LCM
student at 1 inference step on a random VITON-HD sample. The output
should look close to the teacher's 8-step output. If it's noisy garbage,
train another 1-2 epochs.

---

## Stage 4 — TensorRT export (on your g5.xlarge, ~30 min)

Run [`export_trt.py`](export_trt.py) on the inference server. **Not on Kaggle** —
TensorRT engines are tied to the host GPU's driver + CUDA version.

```powershell
# On the g5.xlarge, in your tryon_backend venv:
pip install tensorrt nvidia-tensorrt onnx onnxsim

# Upload Stage 2 + 3 artifacts to the box first (scp, S3, whatever)
$env:TRYON_TEACHER_LORA = "C:\path\to\lucy_catvton_lora.safetensors"
$env:TRYON_LCM_LORA     = "C:\path\to\lucy_catvton_lcm_lora.safetensors"
$env:TRT_ENGINE_PATH    = "C:\virtual-try-on\tryon_backend\engines\engine.plan"
python training\export_trt.py
```

What it does:
1. Loads SD-inpaint, attaches both LoRAs, fuses them
2. Exports ONNX (~3.5 GB fp16) at `engine.onnx`
3. Runs `trtexec` to compile fp16 TensorRT engine (~1.5 GB)
4. Saves at `$TRT_ENGINE_PATH`

If `trtexec` isn't on PATH: install the NVIDIA NGC Docker image
`nvcr.io/nvidia/tensorrt:24.05-py3` and run the script inside it. The
engine you build inside the container will work on the host as long as
the GPU driver version matches.

---

## Stage 5 — Wire it up (10 min)

```powershell
$env:TRT_ENGINE_PATH = "C:\virtual-try-on\tryon_backend\engines\engine.plan"
$env:TRYON_FORCE_GEOMETRIC = "0"
python server.py
```

Watch the startup logs for:
```
TRT engine + CUDA graph ready: C:\virtual-try-on\tryon_backend\engines\engine.plan
Tier 1 ready — TRT. ~30-40fps
```

Hit the demo. You should see `[lat] infer=30-40ms total=60-80ms fps=15-25`
in the server logs. Compared with the current SD-inpaint path at
~900 ms/frame, that's the ~10× speedup that gets the demo to Decart-class
latency.

---

## When things break (they will)

**Kaggle session times out mid-training.** The notebooks save checkpoints
each epoch. Re-run the notebook; the training loop detects existing
checkpoints and resumes.

**LoRA training loss flatlines.** Almost always LR too high. Try 5e-5
instead of 1e-4. Diffusers' default for SD-inpaint LoRA is 1e-4 but
inpainting-specific training likes lower.

**Distillation produces noise.** Either the teacher LoRA isn't strong
enough yet (more epochs), or the LCM scheduler config doesn't match what
you trained against. Cross-check `teacher_scheduler.config.beta_schedule`.

**TRT export fails on `MultiHeadAttention`.** Use `onnxsim` to simplify
the graph first:
```bash
onnxsim engine.onnx engine.sim.onnx --overwrite-input-shape sample:1,9,64,64
```
Then point `trtexec --onnx=engine.sim.onnx`.

**TRT engine loads but inference produces garbage.** The CUDA Graph
capture in `tryon_backend/model.py:TRTInferenceEngine` assumes static
shapes. Confirm `--shapes` in `export_trt.py` matches the server's
`LATENT_H/LATENT_W = 64`.

---

## Honest expectations

After Stages 1-3 (no TRT): ~3-4 fps, **quality matches a trained virtual
try-on** rather than a generic inpainter. This alone is a big win — the
demo no longer hallucinates shirt structure.

After Stage 4 (TRT): ~15-25 fps, real-time feel. This is what "Decart
parity" means in practice.

Don't expect Stage 4 to work on the first compile attempt. Budget half a
day for trtexec arguments / opset mismatches. The good news is once it
works, it's a single `engine.plan` file you copy around.
