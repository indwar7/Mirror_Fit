# Base bodies — curated, never generated

Eight photographs: `{male,female}` x `{slim, average, athletic, plus}`.
Every one is admitted by `tools/curate_base_bodies.py`, which runs the same
anatomy validator the runtime uses and then requires a person to confirm they
have looked at the image.

`manifest.json` records, per body: gender, build, the model's real height,
licence, source, sha256, who approved it and when. The sha256 is checked on
read — an image swapped after review has not been reviewed.

## Requirements for a candidate

- one person, full length, head to feet in frame, feet visible
- front-facing, standing, arms relaxed and away from the torso
- plain uncluttered background
- plain close-fitting clothing, so the try-on parser can find the torso
- even lighting, no heavy shadow across the body
- resolution high enough that the face is at least ~120 px wide, because the
  enrolled face is swapped onto it

## Licence

Only images whose licence permits this use. Record the exact licence string and
a resolvable source URL — "found online" is not a licence.

## Status

Currently **0 of 8** admitted. `python tools/curate_base_bodies.py status`
