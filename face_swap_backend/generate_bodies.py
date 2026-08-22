"""
One-time body-template generator for LUCY full-body avatars.

Renders the 18 figures body_shapes.py can select (2 genders x 3 sizes x 3
tapers) into ./bodies_cache/. At enrolment the user's measurements pick one of
these, their enrolled face is swapped onto it, and the result becomes that
avatar's body image for virtual try-on.

Run in an env that HAS PyTorch — the same one generate_avatars.py runs in.
Run in an env that has PyTorch and diffusers — on the G5 that is conda `base`
(C:\miniconda3\python.exe), the same interpreter deploy.yml starts this backend
with, and the one generate_avatars.py already uses.

Offline by choice, not by necessity: diffusion cannot be prompted with a waist
measurement, so generating per user would land in one of these same buckets
anyway, just slower and non-reproducibly. See body_shapes.py.

    python generate_bodies.py                 # render what's missing
    python generate_bodies.py --validate      # re-check faces are detectable
    python generate_bodies.py --force         # re-render everything

Skips templates that already exist, so it is safe to re-run.

Why the face still matters in a body render
-------------------------------------------
The template's own face is thrown away — the swap replaces it. But inswapper
has to *detect* a face in the target before it can replace it, and SD renders
faces badly at full-body scale (they come out small and mangled). So each
render is checked with the same detector the server uses, and one whose face is
undetectable or tiny is regenerated with a new seed rather than shipped. A
template that fails here would fail at swap time for every user who matched it.
"""
import argparse
import os
import zlib
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from body_shapes import SIZES, TAPERS, all_body_ids

OUT = Path(__file__).parent / "bodies_cache"
OUT.mkdir(exist_ok=True)

# Below this the face has too few pixels for inswapper to produce anything
# convincing, even when the detector does find it.
MIN_FACE_PX = 56
MAX_ATTEMPTS = 4

# ── Build descriptors ────────────────────────────────────────────────────────
# Deliberately describes the SILHOUETTE, not a body-fat judgement — the words
# that actually steer SD toward the right outline.
_SIZE_WORDS = {
    "male": {
        "slim": "slim lean build, narrow frame",
        "average": "average medium build",
        "broad": "large heavy-set build, wide frame",
    },
    "female": {
        "slim": "slim slender build, narrow frame",
        "average": "average medium build",
        "broad": "full curvy heavy-set build, wide frame",
    },
}

_TAPER_WORDS = {
    "male": {
        "tapered": "broad shoulders narrowing to a trim waist, V-shaped torso",
        "regular": "ordinary shoulder and waist proportions",
        "straight": "straight torso, shoulders and waist nearly the same width",
    },
    "female": {
        "tapered": "defined waist, hourglass proportions",
        "regular": "ordinary shoulder and waist proportions",
        "straight": "straight torso, little waist definition",
    },
}

# Plain, close-fitting clothes on a plain background: the try-on model has to
# segment the torso and replace the garment, and busy clothing or background
# is what makes that segmentation fail.
POSITIVE_TEMPLATE = (
    "full body photograph of a {gender_word}, indian, {size}, {taper}, "
    "standing straight facing the camera, arms relaxed at the sides, "
    "wearing a plain close-fitting grey t-shirt and plain dark trousers, "
    "entire figure visible from head to feet, "
    "plain light grey seamless studio background, "
    "soft even studio lighting, neutral expression, "
    "photorealistic, sharp focus, high detail, 8k"
)

NEGATIVE = (
    "cropped, close-up, headshot, portrait, half body, cut off feet, "
    "cut off head, out of frame, multiple people, crowd, "
    "deformed hands, extra fingers, extra limbs, missing limbs, "
    "distorted face, disfigured, asymmetric, "
    "busy background, patterned clothing, logo, text, watermark, "
    "cartoon, anime, illustration, 3d render, painting, low quality, blurry"
)

_GENDER_WORD = {"male": "man", "female": "woman"}


def body_id(gender: str, size: str, taper: str) -> str:
    return f"body_{gender[0]}_{size}_{taper}"


def _assert_library_matches_selector() -> None:
    """The names rendered here must be exactly the names body_shapes selects.

    These are two independent constructions of the same id, so a change to
    either format would otherwise show up as every user silently falling back
    to "no body template" — a bug that looks like a data problem rather than a
    naming one. Fail loudly at startup instead.
    """
    rendered = {
        body_id(g, size, taper)
        for g in ("male", "female")
        for size in SIZES
        for taper in TAPERS
    }
    selectable = set(all_body_ids())
    if rendered != selectable:
        raise SystemExit(
            "body id mismatch between generate_bodies.py and body_shapes.py:\n"
            f"  only here:        {sorted(rendered - selectable)}\n"
            f"  only selectable:  {sorted(selectable - rendered)}"
        )


def prompt_for(gender: str, size: str, taper: str) -> str:
    return POSITIVE_TEMPLATE.format(
        gender_word=_GENDER_WORD[gender],
        size=_SIZE_WORDS[gender][size],
        taper=_TAPER_WORDS[gender][taper],
    )


# ── Face check ───────────────────────────────────────────────────────────────
def _load_detector():
    """The same detector the server swaps with, or None if unavailable.

    Optional on purpose: this script has to run in the PyTorch env, which is
    not necessarily the env insightface is installed in. Without it the render
    still happens — it just ships unvalidated, and says so.
    """
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))
        return app
    except Exception as e:
        print(f"[bodies] face validation OFF ({type(e).__name__}: {e}). "
              f"Re-run with --validate in an env that has insightface.")
        return None


def face_px(detector, pil_image) -> int:
    """Width in pixels of the largest detected face; 0 when none is found."""
    import numpy as np
    import cv2

    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = detector.get(bgr)
    if not faces:
        return 0
    widths = [int(f.bbox[2] - f.bbox[0]) for f in faces]
    return max(widths)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-render existing templates")
    ap.add_argument("--validate", action="store_true",
                    help="only re-check existing templates, render nothing")
    args = ap.parse_args()

    _assert_library_matches_selector()
    detector = _load_detector()

    if args.validate:
        if detector is None:
            raise SystemExit("--validate needs insightface installed in this env.")
        from PIL import Image
        bad = []
        for gender in ("male", "female"):
            for size in SIZES:
                for taper in TAPERS:
                    path = OUT / f"{body_id(gender, size, taper)}.jpg"
                    if not path.exists():
                        bad.append((path.name, "missing"))
                        continue
                    px = face_px(detector, Image.open(path).convert("RGB"))
                    if px < MIN_FACE_PX:
                        bad.append((path.name, f"face {px}px < {MIN_FACE_PX}"))
        if bad:
            print(f"[bodies] {len(bad)} template(s) need attention:")
            for name, why in bad:
                print(f"          {name}: {why}")
            raise SystemExit(1)
        print("[bodies] all 18 templates present with a usable face.")
        return

    model_id = os.environ.get("AVATAR_MODEL", "runwayml/stable-diffusion-v1-5")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[bodies] loading {model_id} on {device} ({dtype}) …")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # 2:3 is the aspect SD 1.5 handles without the duplicated-torso artefacts
    # that taller ratios produce. 640x960 rather than 512x768 because the face
    # has to survive at full-body scale — see MIN_FACE_PX.
    width = int(os.environ.get("BODY_WIDTH", 640))
    height = int(os.environ.get("BODY_HEIGHT", 960))

    made = skipped = failed = 0

    for gender in ("male", "female"):
        for size in SIZES:
            for taper in TAPERS:
                bid = body_id(gender, size, taper)
                out_path = OUT / f"{bid}.jpg"
                if out_path.exists() and not args.force:
                    print(f"[bodies] skip {bid} (already exists)")
                    skipped += 1
                    continue

                prompt = prompt_for(gender, size, taper)
                print(f"[bodies] generating {bid}")

                best_img, best_px = None, -1
                for attempt in range(MAX_ATTEMPTS):
                    # Seed varies per attempt so a retry is a genuinely
                    # different render, and is derived from the id so a rerun
                    # reproduces the same library. crc32, not hash(): Python
                    # randomises string hashing per process, so hash() would
                    # give a different library on every run.
                    seed = zlib.crc32(bid.encode()) % 100000 + attempt * 1009
                    img = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        width=width,
                        height=height,
                        generator=torch.Generator(device=device).manual_seed(seed),
                    ).images[0]

                    if detector is None:
                        best_img = img
                        break

                    px = face_px(detector, img)
                    if px > best_px:
                        best_img, best_px = img, px
                    if px >= MIN_FACE_PX:
                        break
                    print(f"[bodies]   attempt {attempt + 1}: face {px}px — retrying")

                if best_img is None:
                    failed += 1
                    continue

                if detector is not None and best_px < MIN_FACE_PX:
                    # Kept, not discarded — a mediocre template beats a missing
                    # one — but named so it cannot be mistaken for a good render.
                    print(f"[bodies]   WARNING {bid}: best face was {best_px}px "
                          f"(< {MIN_FACE_PX}). Swaps onto it will be poor.")
                    failed += 1

                best_img.save(out_path, "JPEG", quality=92)
                made += 1

    print(f"[bodies] done. generated={made} skipped={skipped} "
          f"needs-attention={failed} → {OUT}")


if __name__ == "__main__":
    main()
