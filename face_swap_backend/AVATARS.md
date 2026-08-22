# LUCY avatars — enrolment and full-body

How a person becomes an avatar they can try clothes on.

## The seam everything hangs off

Every consumer — static swap, live swap V1/V2, Wav2Lip lipsync, hair transfer —
resolves an avatar the same way: **`avatars_cache/{id}.jpg`**. Enrolment writes
into that path, which is why a user's own face works everywhere without any of
those code paths knowing enrolment exists.

| id prefix | what it is | where it comes from |
|---|---|---|
| `gen_*`, `ai_*` | preset portraits | `generate_avatars.py`, hard-coded list in `main.py` |
| `usr_*` | an enrolled person | `POST /avatars/create`, recorded in `avatars_cache/user_avatars.json` |
| `body_*` | body templates | `generate_bodies.py` → `bodies_cache/` |

## L0 — enrolment

`POST /avatars/create` (multipart: `photo`, `name`, optional `gender`)

Validates a face is detectable **before** writing anything — an avatar with no
findable face fails later inside the swap with a far less obvious error. Gender
is auto-detected from the face when InsightFace's full pipeline is loaded, and
is used later to pick a body template.

`DELETE /avatars/{id}` removes the record and both images. Presets return 403.

Voice: `_transform_voice` picks its pitch shift by id prefix and falls through
to 0 semitones for anything unrecognised, so a `usr_` avatar keeps the
speaker's natural voice. That is the correct default when the avatar *is* the
speaker — the absence of a `usr_` rule is deliberate, not an oversight.

## L1 — giving that avatar a body

`POST /avatars/{id}/body` (form: `chest_cm`, `waist_cm`, optional
`shoulder_cm`, `height_cm`, `weight_kg`)

The enrolled selfie is head-and-shoulders. Try-on runs MediaPipe pose to build
a torso mask from shoulders **and hips** (`tryon_backend/model.py`), so a face
alone cannot be dressed — there is nothing for the garment to sit on. This
endpoint picks a pre-rendered body matching the measurements and swaps the
person's face onto it.

Result: `avatars_cache/{id}_body.jpg`, served at `/avatars/{id}/body-image`.

### Why bodies are pre-rendered, not generated per user

1. **Diffusion does not obey numbers.** "waist 82 cm" is not promptable. You
   can only describe a build in words, and words land you in a bucket anyway —
   so per-user generation would produce one of a handful of silhouettes
   regardless, just slower and non-reproducibly.
2. **A rendered library is reviewable.** You can look at all 18 figures and
   approve them. You cannot approve a diffusion sample nobody has seen yet,
   and a bad one reaches the shopper directly.
3. **Selection stays testable.** Measurements → template is plain code with
   unit tests, not a GPU round-trip whose output varies per call.
4. **No render latency.** Enrolment is a swap, not a 10-second diffusion wait.

> **Correction.** An earlier version of this file claimed this backend *could
> not* run Stable Diffusion — Python 3.14, no PyTorch CUDA wheels. That is not
> true of the deployed box: `.github/workflows/deploy.yml` starts
> face_swap_backend with `C:\miniconda3\python.exe` (conda `base`), which has
> torch 2.6.0+cu124 with CUDA available. Runtime generation is therefore
> *possible* here; it is just not worth it, for the reasons above.

Note that `torch`/`diffusers` are still absent from this backend's
`requirements.txt`, so `generate_bodies.py` depends on the ambient conda env
rather than on anything this service declares.

The measurements are stored **verbatim** on the record alongside the bucket, so
fit grading (`fit_score/`) works from the real numbers, not the approximation.

### The 18 templates

`body_shapes.py` selects on two axes, because two are what the measurements
support:

- **size** — chest/bust circumference, cut at common Indian ready-to-wear
  S/M/L/XL bands (men 94/106 cm, women 86/97 cm)
- **taper** — chest-to-waist ratio, the tailor's drop measurement
  (≥1.22 tapered, ≥1.08 regular, below that straight)

Shoulder is a *width* while chest and waist are *circumferences*, so it is not
comparable to them and is used only to nudge taper by one band. Its thresholds
are **per-gender** and must stay that way: a single shared threshold read every
ordinary female frame as narrow-shouldered and cancelled out genuine tapers.

2 genders × 3 sizes × 3 tapers = 18. `generate_bodies.py` renders exactly the
set `body_shapes.all_body_ids()` returns, and asserts the two agree on startup —
a naming drift would otherwise show up as every user silently getting "no body
template", which looks like a data problem rather than a naming one.

### Running the generator

Must run in an env that **has PyTorch** — the same one `generate_avatars.py`
uses, not this backend's:

```bash
python generate_bodies.py              # render what's missing
python generate_bodies.py --validate   # re-check faces are detectable
python generate_bodies.py --force      # re-render everything
```

The template's own face is discarded by the swap, but inswapper must still
*detect* one to replace it — and SD renders faces badly at full-body scale. So
each render is checked with the same detector the server uses and retried with
a new seed if the face is missing or under 56 px. A template that fails this
would fail at swap time for every user who matched it.

Check what is actually on disk with `GET /body-templates`.

## Known gaps

- **No UI.** The endpoints and the Flutter service methods exist; nothing in
  the app calls them yet.
- **`bodies_cache/` is empty** until `generate_bodies.py` runs on a GPU box.
  Until then `POST /avatars/{id}/body` returns 503 naming the missing template.
- **Height is collected but unused** in template selection — it mostly affects
  scale rather than silhouette, and adding a height axis would triple the
  library. Stored on the record for fit grading regardless.
- **L2/L3 not built** — feeding the stored measurements into `fit_score`, and
  running the body image plus an uploaded garment through CatVTON.
