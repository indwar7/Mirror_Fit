r"""
One-time body-template generator for LUCY full-body avatars.

Renders the 18 figures body_shapes.py can select (2 genders x 3 sizes x 3
tapers) into ./bodies_cache/. At enrolment the user's measurements — or just
their gender — pick one of these, their enrolled face is swapped onto it, and
the result becomes that avatar's body image for virtual try-on.

Run in an env that has PyTorch and diffusers — on the G5 that is conda `base`
(C:\miniconda3\python.exe), the same interpreter deploy.yml starts this backend
with, and the one generate_avatars.py already uses.

    python generate_bodies.py                 # render what's missing
    python generate_bodies.py --validate      # re-check existing templates
    python generate_bodies.py --force         # re-render everything

Skips templates that already exist, so it is safe to re-run.

Why SDXL and not SD 1.5
-----------------------
The first version of this script used SD 1.5 at 640x960 and shipped figures
with TWO TORSOS stacked on top of each other. SD 1.5 is trained at 512x512;
push it to a tall full-body frame and it composes the image twice. That is not
a prompt problem and no negative prompt reliably fixes it.

SDXL is trained on multiple aspect ratios including 832x1216, which is what a
standing figure needs, and does not duplicate. The weights are already on the
box — InstantID pulls the same `stable-diffusion-xl-base-1.0` repo, and the
HuggingFace cache is shared across conda envs.

Why validation checks the whole body, not just the face
-------------------------------------------------------
The earlier check only asked "is a face detectable?". A figure with two torsos
still has exactly one detectable face, so the broken templates passed and went
straight into production. Anatomy is now checked with MediaPipe pose: the
landmarks a dressable body must have, in the order a real body has them.

The face check still matters for a different reason — inswapper has to *detect*
a face in the target before it can replace it, and a face too small in frame
gives a poor swap even when detection succeeds.
"""
import argparse
import os
import zlib
from pathlib import Path

import numpy as np
import torch

from body_shapes import SIZES, TAPERS, all_body_ids

OUT = Path(__file__).parent / "bodies_cache"
OUT.mkdir(exist_ok=True)

# Below this the face has too few pixels for inswapper to produce anything
# convincing, even when the detector does find it.
MIN_FACE_PX = 56
MAX_ATTEMPTS = 6

# SDXL native portrait bucket. Do not "improve" this to something taller
# without re-reading the module docstring — aspect ratio is exactly what
# produced the two-torso figures.
DEFAULT_W, DEFAULT_H = 832, 1216

# ── Build descriptors ────────────────────────────────────────────────────────
# Deliberately describes the SILHOUETTE, not a body-fat judgement — the words
# that actually steer the model toward the right outline.
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
# segment the torso and replace the garment, and busy clothing or background is
# what makes that segmentation fail.
POSITIVE_TEMPLATE = (
    "full body studio photograph of one {gender_word}, indian, {size}, {taper}, "
    "standing straight, facing the camera, arms relaxed at the sides, "
    "wearing a plain close-fitting grey t-shirt and plain dark trousers, "
    "whole body in frame from head to feet, feet visible, "
    "plain light grey seamless studio background, "
    "soft even studio lighting, neutral expression, "
    "photorealistic, sharp focus, high detail"
)

NEGATIVE = (
    # The duplication failure first, because it is the one that actually
    # happened. The real fix is the model and the aspect ratio; this is belt
    # and braces.
    "two people, second person, duplicate torso, extra torso, two bodies, "
    "cloned body, mirrored body, extra head, two heads, "
    "cropped, close-up, headshot, portrait, half body, cut off feet, "
    "cut off head, out of frame, crowd, "
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


# ── Quality checks ───────────────────────────────────────────────────────────
class Checker:
    """Rejects a render that would fail downstream, for the reason it would fail.

    Two independent checks, because they catch different disasters:

      face  — inswapper must be able to detect a face in the target, and a face
              too small in frame swaps badly even when detection succeeds.
      body  — a dressable figure needs shoulders, hips, knees and ankles, one
              of each pair, in the vertical order a standing body has them.
              This is what a two-torso render fails and a face check does not.
    """

    def __init__(self):
        self.face_app = None
        self.pose = None

        try:
            from insightface.app import FaceAnalysis
            self.face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.face_app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))
        except Exception as e:
            print(f"[bodies] face check OFF ({type(e).__name__}: {e})")

        try:
            import mediapipe as mp
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=True, model_complexity=2,
                min_detection_confidence=0.5,
            )
        except Exception as e:
            print(f"[bodies] body check OFF ({type(e).__name__}: {e}) — "
                  f"`pip install mediapipe` to catch malformed figures")

    @property
    def enabled(self) -> bool:
        return self.face_app is not None or self.pose is not None

    def check(self, pil_image) -> tuple[bool, str]:
        """(ok, reason). reason is empty when ok."""
        import cv2
        rgb = np.array(pil_image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self.face_app is not None:
            faces = self.face_app.get(bgr)
            if not faces:
                return False, "no face detected"
            # More than one face is the loudest possible signal that the model
            # composed the scene twice.
            if len(faces) > 1:
                return False, f"{len(faces)} faces — duplicated figure"
            px = int(faces[0].bbox[2] - faces[0].bbox[0])
            if px < MIN_FACE_PX:
                return False, f"face {px}px < {MIN_FACE_PX}"

        if self.pose is not None:
            res = self.pose.process(rgb)
            if not res.pose_landmarks:
                return False, "no body pose detected"
            lm = res.pose_landmarks.landmark
            # MediaPipe indices: 11/12 shoulders, 23/24 hips, 25/26 knees,
            # 27/28 ankles. y grows downward.
            need = {
                "shoulders": (11, 12), "hips": (23, 24),
                "knees": (25, 26), "ankles": (27, 28),
            }
            mid = {}
            for part, (a, b) in need.items():
                if lm[a].visibility < 0.5 or lm[b].visibility < 0.5:
                    return False, f"{part} not visible — body is cropped or malformed"
                mid[part] = (lm[a].y + lm[b].y) / 2

            order = ["shoulders", "hips", "knees", "ankles"]
            for upper, lower in zip(order, order[1:]):
                if mid[upper] >= mid[lower]:
                    return False, f"{upper} not above {lower} — anatomy is wrong"

            # A standing person's shoulders sit in the upper half and the
            # ankles near the bottom. A stacked double-figure fails this.
            if mid["shoulders"] > 0.45:
                return False, "shoulders too low in frame — figure not full length"
            if mid["ankles"] < 0.80:
                return False, "ankles too high in frame — feet cut off or doubled"

        return True, ""


def _load_pipe():
    """SDXL, or SD 1.5 if SDXL is genuinely unavailable.

    The fallback exists so a box without the SDXL weights still produces
    something, but it says so loudly — SD 1.5 is what produced the two-torso
    figures and its output here should not be trusted without looking at it.
    """
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = os.environ.get("BODY_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
    try:
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        print(f"[bodies] loading {model_id} on {device} ({dtype}) …")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16",
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe, True
    except Exception as e:
        print(f"[bodies] SDXL unavailable ({type(e).__name__}: {e})")
        print("[bodies] WARNING falling back to SD 1.5 — this is the model that "
              "produced two-torso figures. Check every render by eye.")
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-render existing templates")
    ap.add_argument("--validate", action="store_true",
                    help="only re-check existing templates, render nothing")
    args = ap.parse_args()

    _assert_library_matches_selector()

    if args.validate:
        from PIL import Image
        checker = Checker()
        if not checker.enabled:
            raise SystemExit("--validate needs insightface and/or mediapipe installed.")
        bad = []
        for bid in all_body_ids():
            path = OUT / f"{bid}.jpg"
            if not path.exists():
                bad.append((path.name, "missing"))
                continue
            ok, why = checker.check(Image.open(path))
            if not ok:
                bad.append((path.name, why))
        if bad:
            print(f"[bodies] {len(bad)} template(s) need attention:")
            for name, why in bad:
                print(f"          {name}: {why}")
            print("[bodies] re-render those with: python generate_bodies.py --force")
            raise SystemExit(1)
        print(f"[bodies] all {len(all_body_ids())} templates present and usable.")
        return

    checker = Checker()
    pipe, is_sdxl = _load_pipe()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    width = int(os.environ.get("BODY_WIDTH", DEFAULT_W if is_sdxl else 512))
    height = int(os.environ.get("BODY_HEIGHT", DEFAULT_H if is_sdxl else 768))
    steps = int(os.environ.get("BODY_STEPS", 30))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[bodies] rendering at {width}x{height}, {steps} steps")
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

                accepted = None
                last_reason = "not attempted"
                for attempt in range(MAX_ATTEMPTS):
                    # crc32, not hash(): Python randomises string hashing per
                    # process, so hash() would give a different library on
                    # every run.
                    seed = zlib.crc32(bid.encode()) % 100000 + attempt * 1009
                    img = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE,
                        num_inference_steps=steps,
                        guidance_scale=7.0,
                        width=width,
                        height=height,
                        generator=torch.Generator(device=device).manual_seed(seed),
                    ).images[0]

                    if not checker.enabled:
                        accepted = img
                        break
                    ok, why = checker.check(img)
                    if ok:
                        accepted = img
                        break
                    last_reason = why
                    print(f"[bodies]   attempt {attempt + 1} rejected: {why}")

                if accepted is None:
                    # Not written at all. A rejected figure is worse than a
                    # missing one: the endpoint reports a missing template
                    # clearly, while a bad one silently produces the
                    # two-torso avatars this check exists to stop.
                    print(f"[bodies]   FAILED {bid} after {MAX_ATTEMPTS} attempts "
                          f"({last_reason}) — not written")
                    failed += 1
                    continue

                accepted.save(out_path, "JPEG", quality=92)
                made += 1

    print(f"[bodies] done. generated={made} skipped={skipped} failed={failed} → {OUT}")
    if failed:
        print(f"[bodies] {failed} template(s) were not written. Re-run to retry them.")


if __name__ == "__main__":
    main()
