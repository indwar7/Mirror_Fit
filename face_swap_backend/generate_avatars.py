"""
One-time avatar generator for LUCY face swap.

Generates 10 diverse AI portraits using Stable Diffusion 1.5 (already cached
from the tryon backend) and writes them as JPG into ./avatars_cache/.

Run once on the G5 from the face_swap_backend directory:
    python generate_avatars.py

Skips avatars that already exist on disk so it's safe to re-run.
"""
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

OUT = Path(__file__).parent / "avatars_cache"
OUT.mkdir(exist_ok=True)

# 10 diverse prompts — 5 female, 5 male, varied skin tones / ethnicities / ages.
# Each entry: (avatar_id, gender_label, descriptor)
AVATARS = [
    ("gen_f_01", "female", "european woman, fair skin, blonde hair, blue eyes, late 20s"),
    ("gen_f_02", "female", "south asian woman, warm brown skin, long dark hair, mid 30s"),
    ("gen_f_03", "female", "african woman, deep brown skin, short curly hair, late 20s"),
    ("gen_f_04", "female", "east asian woman, light skin, sleek black hair, early 30s"),
    ("gen_f_05", "female", "hispanic woman, olive skin, wavy brown hair, late 30s"),
    ("gen_m_01", "male",   "european man, fair skin, brown hair, trimmed beard, early 30s"),
    ("gen_m_02", "male",   "middle eastern man, warm tan skin, short dark hair, mid 30s"),
    ("gen_m_03", "male",   "african man, deep brown skin, short hair, clean shaven, late 20s"),
    ("gen_m_04", "male",   "east asian man, light skin, neat black hair, early 30s"),
    ("gen_m_05", "male",   "hispanic man, olive skin, dark hair, light stubble, mid 30s"),
    # Kids — boys (ages 8-12)
    ("gen_b_01", "male",   "young boy aged 10, mixed ethnicity, light skin, curly brown hair, freckles, gentle smile"),
    ("gen_b_02", "male",   "young boy aged 9, south asian, warm brown skin, neat short dark hair, bright eyes"),
    ("gen_b_03", "male",   "young boy aged 11, east asian, light skin, straight black hair, soft smile"),
    ("gen_b_04", "male",   "young boy aged 10, african, deep brown skin, short curly hair, cheerful expression"),
    # Kids — girls (ages 8-12)
    ("gen_g_01", "female", "young girl aged 10, european, fair skin, long blonde hair, blue eyes, friendly smile"),
    ("gen_g_02", "female", "young girl aged 9, south asian, warm brown skin, long dark hair, bright eyes"),
    ("gen_g_03", "female", "young girl aged 11, east asian, light skin, sleek black hair, gentle smile"),
    ("gen_g_04", "female", "young girl aged 10, african, deep brown skin, braided hair, joyful expression"),
    # ── Indian avatars across age groups ──────────────────────────────────────
    # Indian — Kids (8-12)
    ("gen_ib_01", "male",   "indian boy aged 10, warm brown skin, neat short black hair, bright eyes, gentle smile"),
    ("gen_ig_01", "female", "indian girl aged 9, warm brown skin, long braided black hair, bright eyes, soft smile"),
    # Indian — Teens (15-17)
    ("gen_ib_02", "male",   "indian teenage boy aged 16, warm brown skin, neat hair, light stubble, confident expression"),
    ("gen_ig_02", "female", "indian teenage girl aged 16, warm brown skin, long straight black hair, friendly smile"),
    # Indian — Young adults (25-30)
    ("gen_im_01", "male",   "indian man aged 27, warm brown skin, neat black hair, trimmed beard, confident smile"),
    ("gen_if_01", "female", "indian woman aged 26, warm brown skin, long dark wavy hair, soft smile"),
    # Indian — Middle-aged (45-50)
    ("gen_im_02", "male",   "indian man aged 47, warm brown skin, salt and pepper hair, full beard, kind expression"),
    ("gen_if_02", "female", "indian woman aged 46, warm brown skin, dark hair tied back with some grey, warm smile"),
    # Indian — Elderly (65-75)
    ("gen_im_03", "male",   "indian elderly man aged 70, weathered brown skin, fully white hair and beard, deep eyes, dignified expression"),
    ("gen_if_03", "female", "indian elderly woman aged 68, weathered brown skin, silver hair, warm gentle expression"),
]

POSITIVE_TEMPLATE = (
    "professional studio portrait photograph of a {desc}, "
    "head and shoulders framing, looking directly at camera, "
    "soft natural lighting, neutral expression, sharp focus, "
    "high detail skin texture, 85mm lens, photorealistic, 8k, "
    "clean plain background"
)
NEGATIVE = (
    "cartoon, anime, illustration, painting, render, 3d, low quality, "
    "blurry, distorted, deformed, asymmetric, watermark, text, "
    "extra limbs, multiple faces, disfigured"
)


def main() -> None:
    # Pick first model that's already cached / available
    model_id = os.environ.get("AVATAR_MODEL", "runwayml/stable-diffusion-v1-5")
    dtype    = torch.float16 if torch.cuda.is_available() else torch.float32
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[avatars] loading {model_id} on {device} ({dtype}) …")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    generator = torch.Generator(device=device).manual_seed(42)
    made, skipped = 0, 0

    for avatar_id, _gender, desc in AVATARS:
        out_path = OUT / f"{avatar_id}.jpg"
        if out_path.exists():
            print(f"[avatars] skip {avatar_id} (already exists)")
            skipped += 1
            continue

        prompt = POSITIVE_TEMPLATE.format(desc=desc)
        print(f"[avatars] generating {avatar_id}  ←  {desc}")
        img = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=640,
            generator=generator,
        ).images[0]
        img.save(out_path, "JPEG", quality=92)
        made += 1

    print(f"[avatars] done. generated={made} skipped={skipped} → {OUT}")


if __name__ == "__main__":
    main()
