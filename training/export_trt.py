"""
Stage 4 of Path B — Export the trained + distilled UNet to TensorRT.

Run this ON THE INFERENCE GPU (your g5.xlarge), not Kaggle. TensorRT
versions are tied to the GPU driver; the engine produced on a T4/P100
won't run on an A10G.

Pipeline:
  1. Load SD-inpaint UNet
  2. Apply trained CatVTON LoRA (from train_catvton_lora_vitonhd.ipynb)
  3. Apply LCM-LoRA (from distill_lcm_lora.ipynb)
  4. Fuse both into the base UNet (no LoRA layers left)
  5. Export to ONNX with dynamic batch axis
  6. Compile ONNX → TensorRT engine with fp16
  7. Drop the engine at $TRT_ENGINE_PATH
  8. Tier 1 path in model.py activates automatically on next server start

Usage:
  TRYON_TEACHER_LORA=/path/to/lucy_catvton_lora.safetensors \\
  TRYON_LCM_LORA=/path/to/lucy_catvton_lcm_lora.safetensors \\
  TRT_ENGINE_PATH=/path/to/output/engine.plan \\
    python export_trt.py
"""
from __future__ import annotations
import os
import sys
import time
import subprocess
from pathlib import Path

import torch
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel


BASE_MODEL        = os.environ.get('TRYON_BASE_MODEL', 'runwayml/stable-diffusion-inpainting')
TEACHER_LORA      = os.environ.get('TRYON_TEACHER_LORA')
LCM_LORA          = os.environ.get('TRYON_LCM_LORA')
ENGINE_PATH       = os.environ.get('TRT_ENGINE_PATH', './engine.plan')
ONNX_PATH         = os.path.splitext(ENGINE_PATH)[0] + '.onnx'
H                 = int(os.environ.get('TRYON_H', '64'))   # latent height (= image 512 / 8)
W                 = int(os.environ.get('TRYON_W', '64'))
MAX_BATCH         = int(os.environ.get('TRYON_MAX_BATCH', '1'))


def must(path: str | None, label: str) -> str:
    if not path or not Path(path).exists():
        print(f'ERROR: {label} missing or not found: {path!r}', file=sys.stderr)
        print(f'       set its env var to a real path.', file=sys.stderr)
        sys.exit(2)
    return path


def load_fused_unet() -> UNet2DConditionModel:
    """Load SD-inpaint UNet + teacher LoRA + LCM-LoRA, fuse everything."""
    print('Loading base inpaint pipeline…')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
    )

    teacher = must(TEACHER_LORA, 'teacher LoRA')
    lcm     = must(LCM_LORA,     'LCM LoRA')
    print(f'Attaching teacher LoRA: {teacher}')
    pipe.load_lora_weights(teacher, adapter_name='teacher')
    print(f'Attaching LCM LoRA:     {lcm}')
    pipe.load_lora_weights(lcm,     adapter_name='lcm')
    pipe.set_adapters(['teacher', 'lcm'], adapter_weights=[1.0, 1.0])

    print('Fusing LoRAs into base UNet…')
    pipe.fuse_lora()
    # peft leaves shadow layers around; unload them so torch.onnx.export sees a plain UNet
    pipe.unload_lora_weights()
    unet = pipe.unet.to('cuda').eval()
    return unet


def export_onnx(unet: UNet2DConditionModel, out_path: str):
    """Trace UNet with dummy inputs and write an ONNX file."""
    print(f'Exporting ONNX → {out_path}')
    dtype, device = torch.float16, 'cuda'
    sample = torch.randn(MAX_BATCH, 9, H, W, dtype=dtype, device=device)
    timestep = torch.tensor([1], dtype=torch.long, device=device)
    enc = torch.randn(MAX_BATCH, 77, 768, dtype=dtype, device=device)

    class Wrap(torch.nn.Module):
        def __init__(self, u): super().__init__(); self.u = u
        def forward(self, sample, timestep, encoder_hidden_states):
            return self.u(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample

    wrap = Wrap(unet).to(device, dtype).eval()
    with torch.inference_mode():
        torch.onnx.export(
            wrap, (sample, timestep, enc), out_path,
            input_names=['sample', 'timestep', 'encoder_hidden'],
            output_names=['noise_pred'],
            dynamic_axes={
                'sample':         {0: 'batch'},
                'encoder_hidden': {0: 'batch'},
                'noise_pred':     {0: 'batch'},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f'ONNX written: {size_mb:.1f} MB')


def trtexec(onnx_path: str, engine_path: str):
    """Invoke trtexec to compile ONNX → TensorRT engine with fp16."""
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        '--fp16',
        # Static shapes — server runs single-frame inference. If you want
        # dynamic batches later, swap these for minShapes/optShapes/maxShapes.
        f'--shapes=sample:{MAX_BATCH}x9x{H}x{W},timestep:1,encoder_hidden:{MAX_BATCH}x77x768',
        '--builderOptimizationLevel=5',
        '--useSpinWait',
    ]
    print('Running:', ' '.join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        print('trtexec failed. Common fixes:', file=sys.stderr)
        print('  - Install TensorRT:        pip install tensorrt nvidia-tensorrt', file=sys.stderr)
        print('  - Or get it via NGC:       https://developer.nvidia.com/tensorrt', file=sys.stderr)
        print('  - Driver/CUDA must match the TRT version. On g5.xlarge the easy', file=sys.stderr)
        print('    path is the nvcr.io/nvidia/tensorrt:24.05-py3 Docker image.', file=sys.stderr)
        sys.exit(proc.returncode)
    print(f'TensorRT engine built in {time.time()-t0:.1f}s → {engine_path}')


def main():
    unet = load_fused_unet()
    Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)
    export_onnx(unet, ONNX_PATH)
    # Free the UNet before TRT build (the builder also needs VRAM)
    del unet
    torch.cuda.empty_cache()
    trtexec(ONNX_PATH, ENGINE_PATH)
    print()
    print('Next step:')
    print(f'  export TRT_ENGINE_PATH={ENGINE_PATH}')
    print(f'  restart tryon_backend/server.py — Tier 1 path loads the engine automatically.')


if __name__ == '__main__':
    main()
