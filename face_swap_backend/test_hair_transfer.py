"""Offline test for transfer_hair in face_parser.py.

    python3 face_swap_backend/test_hair_transfer.py

Pins the three things that were visibly wrong in the swapped output: a dark
rim along every soft hair edge, a hard rectangular seam where the avatar's
hair ran off the edge of its own photo, and the user's ears showing through a
parting in the avatar's hair. All three are pure image maths — no BiSeNet, no
GPU — so they are checkable here rather than by staring at a webcam.
"""
import pathlib, sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from face_parser import transfer_hair, _border_fade, _fill_outward

fails = []


def ok(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


N = 256
HAIR = 90          # a dark hair tone — the case the old bug punished hardest
SCENE = 200        # a bright scene, so any darkening is unmistakable

# Source: hair across the top, running off the top and left edges exactly the
# way hair does in a real portrait crop.
src = np.full((N, N, 3), 240, np.uint8)
hair_area = np.zeros((N, N), np.uint8)
cv2.ellipse(hair_area, (110, 70), (120, 80), 0, 0, 360, 255, -1)
src[hair_area > 0] = HAIR
# hair_mask() closes then Gaussian-blurs, so the real mask is soft-edged. That
# softness is precisely what the premultiply bug turned into a black rim.
src_mask = cv2.GaussianBlur(hair_area, (5, 5), 0)

tgt = np.full((N, N, 3), SCENE, np.uint8)

kps = np.array([[100, 120], [150, 120], [125, 150], [105, 175], [145, 175]], np.float32)

out = transfer_hair(src, src_mask, kps, tgt, np.zeros((N, N), np.uint8), kps)

print("\n1. no dark rim along the hair edge")
# Measure well inside the hair and away from the source border — the paint
# mask is eroded then feathered, so pixels near the silhouette are meant to be
# part scene, and pixels near the border are meant to be fading out (test 2).
deep = cv2.erode(hair_area, np.ones((41, 41), np.uint8)) > 250
deep[:34, :] = False
deep[:, :34] = False
ok(abs(int(out[deep].mean()) - HAIR) < 8,
   f"hair reproduces the source tone (got {int(out[deep].mean())}, source {HAIR})")
# The transition band: soft mask, so these are genuine blends of hair and scene.
band = (src_mask > 20) & (src_mask < 235)
vals = out[band]
ok(vals.size > 0, f"there is a soft transition band to check ({vals.size} px)")
ok(int(vals.min()) >= HAIR - 20,
   f"nothing in the blend is darker than the hair itself (min {int(vals.min())}, hair {HAIR})")
ok(int(out.min()) >= HAIR - 20,
   f"and nothing anywhere went towards black (frame min {int(out.min())})")

print("\n2. the source border does not warp in as a straight seam")
faded = _border_fade(src_mask, 24)
ok(int(faded[:, 0].max()) == 0 and int(faded[0, :].max()) == 0,
   "the mask is zero along the source border")
col = faded[60, :40].astype(int)
ok(np.all(np.diff(col) >= -1) and col[0] == 0 and col[-1] > 200,
   "and ramps up from it monotonically rather than stepping")
# The cliff was never inside the mask, it was AT the border: full hair on the
# last column, nothing beyond it, so warping carried a straight edge across.
ok(int(src_mask[60, 0]) > 240, "unfaded, the mask hits the border at full strength")
ok(int(np.abs(np.diff(col)).max()) < 24,
   f"faded, the steepest step is {int(np.abs(np.diff(col)).max())} instead of a 255 cliff")
# And it shows in the output: hair that ran off the source edge arrives as a
# gradient into the scene rather than as a wall of hair ending in a straight line.
edge_col = out[40:120, 2].astype(int).mean()
deep_col = out[40:120, 60].astype(int).mean()
ok(edge_col > deep_col + 30,
   f"at the border the hair has faded into the scene ({int(deep_col)} → {int(edge_col)})")

print("\n3. ears get covered even where the avatar had no hair there")
ears = np.zeros((N, N), np.uint8)
ears[150:200, 30:60] = 255          # a strip with no warped hair over it
ok(int(src_mask[150:200, 30:60].max()) == 0, "the strip really is bare in the source mask")
out2 = transfer_hair(src, src_mask, kps, tgt, np.zeros((N, N), np.uint8), kps,
                     tgt_extra_paint_mask=ears)
strip = out2[160:190, 35:55]
ok(int(np.abs(strip.astype(int) - SCENE).max()) > 25,
   "the strip is painted over, not left as the user's own ear")
ok(int(strip.min()) > 30,
   f"and painted with hair colour, not black (min {int(strip.min())})")

print("\n4. the fill only touches empty pixels")
colr = np.full((N, N, 3), 77, np.uint8)
alpha = np.zeros((N, N), np.uint8)
alpha[100:150, 100:150] = 255
filled = _fill_outward(colr, alpha)
ok(np.array_equal(filled[100:150, 100:150], colr[100:150, 100:150]),
   "valid pixels come back bit-identical, so the hair itself stays sharp")
ok(int(filled[:90, :90].max()) > 0, "empty pixels got a colour rather than staying black")

print("\n5. an empty source mask is a no-op, not a black frame")
out3 = transfer_hair(src, np.zeros((N, N), np.uint8), kps, tgt,
                     np.zeros((N, N), np.uint8), kps)
ok(np.array_equal(out3, tgt), "nothing to transfer leaves the target untouched")

print("\n6. the ear strips land beside the face, and nowhere else")
# _ear_region_mask lives in main.py; lift it out rather than copying it.
_src = pathlib.Path(__file__).with_name("main.py").read_text(encoding="utf-8")
_ns = {"cv2": cv2, "np": np, "Optional": object}
exec(_src[_src.index("def _ear_region_mask("):_src.index("def _changed_mask(")], _ns)
ear_mask = _ns["_ear_region_mask"]


class _Face:
    def __init__(self, lmk): self.landmark_2d_106 = lmk


# A face roughly 120 px wide centred at (128, 130).
ang = np.linspace(0, 2 * np.pi, 106, endpoint=False)
lmk = np.stack([128 + 60 * np.cos(ang), 130 + 78 * np.sin(ang)], 1).astype(np.float32)
ears = ear_mask(_Face(lmk), (N, N))
hull = np.zeros((N, N), np.uint8)
cv2.fillPoly(hull, [cv2.convexHull(lmk.astype(np.int32))], 255)

ok(ears is not None and int(ears.max()) == 255, "a mask was produced")
ok(int(cv2.bitwise_and(ears, hull).max()) == 0,
   "nothing inside the face hull — the swapped face is never painted over")
ok(int(ears[:, :128].max()) == 255 and int(ears[:, 128:].max()) == 255,
   "both ears, not just one")
ys = np.where(ears.any(axis=1))[0]
ok(ys.min() > 130 - 78, f"nothing up on the forehead (top row {ys.min()}, face top {130 - 78})")
ok(ys.max() < 130 + 78, f"nothing down on the neck (bottom row {ys.max()}, face bottom {130 + 78})")
left = int((ears[:, :128] > 0).sum())
right = int((ears[:, 128:] > 0).sum())
ok(abs(left - right) < max(left, right) * 0.15,
   f"roughly symmetric ({left} px left, {right} px right)")
ok(ear_mask(_Face(None), (N, N)) is None, "no landmarks means no mask, not a crash")

print(f"\n{len(fails)} FAILURES" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
