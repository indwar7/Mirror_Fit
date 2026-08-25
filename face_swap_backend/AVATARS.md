# LUCY avatars — enrol a person, dress them

## The one rule

**Base bodies are curated photographs. Nothing generates one at runtime.**

They used to be produced at build time by a text-to-image model. Generation is
stochastic: a fraction of renders came out with two torsos stacked on top of
each other. A validator was added — and it only asked *"is a face
detectable?"*, which a two-torso image answers yes. Eighteen broken figures
shipped. The validator was then made stricter but left **optional**: when its
model was missing it printed a warning and fell back to face detection, and the
same eighteen broken figures shipped again, from a run that ended with "all
templates usable".

Moving SD 1.5 → SDXL lowered the defect rate. It did not make the output
something you can ship without a person looking at every frame. So a person
looks at every frame, once, offline, and the result is an asset in the repo.

`tools/generate_bodies_OFFLINE.py` still exists, but only to produce
*candidates* for review. Nothing in `face_swap_backend` imports it.

## The seam everything hangs off

Every consumer — static swap, live swap V1/V2, Wav2Lip lipsync, hair transfer,
try-on — resolves an avatar the same way: **`avatars_cache/{id}.jpg`**.

Giving an avatar a body does not add a second file. The composite is written
back over that same path, which is why the rest of the system picks it up with
no changes.

| id prefix | what it is |
|---|---|
| `gen_*`, `ai_*` | preset portraits, hard-coded list in `main.py` |
| `usr_*` | an enrolled person, recorded in `avatars_cache/user_avatars.json` |
| `body_*` | a curated base body in `assets/base_bodies/` |

## Pipeline

```
selfie ──> POST /avatars/create      face validated before anything is written
   │
   └────> POST /avatars/{id}/body    measurements -> bin -> curated photograph
                                     face swap + LAB skin-tone match to the
                                     body's own neck, anatomy-validated, then
                                     written over avatars_cache/{id}.jpg
   │
   └────> POST /avatars/{id}/tryon   parse + densepose -> agnostic mask ->
                                     CatVTON -> hard paste-back
```

### Measurements → bin

`body_shapes.py` is a **pure function**. No I/O, no models, no image creation.
Four builds per gender — `slim`, `average`, `athletic`, `plus` — because that
is what the measurements can actually separate.

- girth from chest/bust, cut at common Indian ready-to-wear bands
  (men 94 / 108 cm, women 86 / 100 cm), with a waist override so a large waist
  on an ordinary chest still reads as `plus`
- `athletic` is ordinary girth with a pronounced chest-to-waist drop — the
  tailor's own measure of taper
- **plus is decided before athletic**: a garment that does not go round the
  chest is not saved by having the right drop
- chest without waist is ignored, not half-used — one alone gives no ratio, and
  using it for girth would look like the measurement had counted
- no measurements at all → the gender's `average` bin

If a bin has no approved photograph the endpoint returns **501 naming the id**.
It does not fall back to a neighbouring build. A silent substitution shows
someone a body their measurements did not ask for with nothing saying so.

### Identity

Face swap only — InsightFace inswapper, then GFPGAN/CodeFormer restore, then
`skin_match.py` moves the face's LAB statistics toward the base body's own neck
skin. Direction matters: the **face** is corrected toward the **body**, never
the reverse, because the body is the large continuous region a viewer reads as
that person's skin tone.

The correction is refused if it would be implausibly large, and only pixels
inside the face mask are written — a full LAB round trip perturbs every pixel
in the image by a few levels, so the body and background are copied through
bit-identical.

### Try-on

The try-on **must never alter identity**, and that is enforced arithmetically,
not hoped for:

- the agnostic mask is built by union of replaceable ATR labels, then
  subtraction of protected ones. The subtraction is not redundant — parsers
  routinely bleed `upper_clothes` a few pixels onto the jaw, and those pixels
  are exactly the ones a viewer reads as the face changing.
- `composite()` is a **hard binary copy**, not a blend. A feathered edge leaves
  the face *nearly* unchanged, and "nearly" is not something a test can hold
  anyone to.
- shape mismatch raises rather than resampling the original
- `tryon()` re-asserts `identity_preserved()` on its own output before
  returning, so a future change to `composite()` fails loudly

Parse and DensePose are cached per avatar id — they depend only on the avatar,
so ten garments run them once. Rebuilding the body invalidates the cache.

## The validation gate

`avatar_validation.py`. One validator, used by both the curation script and
every avatar write.

**It fails closed.** If the pose model is unavailable, every call raises
`ValidatorUnavailable`. There is no skip flag in the module, and a test asserts
none is ever added. Bypassing it has to happen at a call site, visibly, in
review.

Checks, in order, each reported separately:

| check | catches |
|---|---|
| `single_person` | 0 or 2+ people — the two-torso signature |
| `complete_skeleton` | missing or low-confidence keypoints — cropped figures |
| `vertical_order` | nose < shoulders < hips < knees < ankles |
| `proportions` | limb ratios outside human range |
| `single_blob` | a second figure the pose model missed |

Proportions use Drillis & Contini's body-segment tables (shoulder 0.818 of
height, hip 0.530, knee 0.285, ankle 0.039, biacromial width 0.259), expressed
against the nose-to-ankle span because that is what is measurable in an image.
Tolerances are wide — the job is to reject a shape that is not human, not to
grade posture.

`ValidationResult` is per-check by design. "18 templates passed" is the shape
of claim that let two rounds of broken assets through.

## Adding a base body

```bash
python tools/curate_base_bodies.py add candidate.jpg \
    --id body_male_average --height-cm 178 \
    --license "Unsplash Licence" --source "https://…" --approved-by "abhay"

python tools/curate_base_bodies.py status    # per-id readiness
python tools/curate_base_bodies.py verify    # re-validate everything admitted
```

`add` refuses ids `body_shapes` cannot select, runs the anatomy validator,
prints the per-check result, and then **requires the operator to confirm they
have looked at the image**. That step is not ceremony: the validator catches
two torsos, not "this photograph is unusable for other reasons".

`manifest.json` records gender, build, the model's real height, licence,
source, sha256, approver and timestamp. The sha256 is checked on read — an
image swapped after review has not been reviewed.

## Running the tests

```bash
cd face_swap_backend
python -m unittest test_avatar_validation test_body_shapes test_tryon test_skin_match -v
```

55 tests, stdlib only, no models needed — that is the point. The most important
one asserts that a **missing pose model makes avatar creation fail** rather
than pass.

## Current state

- **Base bodies: 0 of 8 admitted.** The asset set is empty. Every
  `POST /avatars/{id}/body` returns 501 until photographs are curated in.
  `GET /base-bodies` reports each bin separately.
- Try-on backends (SCHP, DensePose, CatVTON) are adapters against upstream
  projects that must be installed on the box; `POST /avatars/{id}/tryon`
  returns 503 naming what is missing until they are.
