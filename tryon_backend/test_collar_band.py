"""Offline test for the collar band in model.py.

    python3 tryon_backend/test_collar_band.py

The collar is drawn rather than sampled — the garment's top edge is feathered
so SD has something smooth to denoise into, and a few LCM steps will not
resolve a ribbed band there. Drawn shading is easy to get wrong in ways that
only show on a face: a flat multiply over a closed ellipse reads as a black
choker, which is exactly what it did. This pins the shape offline, with no
GPU and no webcam.
"""
import io, pathlib, cv2, numpy as np
src = io.open(pathlib.Path(__file__).with_name("model.py"), encoding="utf-8").read()
a = src.index("        collar_band = np.zeros((h, w), dtype=np.float32)")
b = src.index("        return torso_mask, silhouette, face_cutoff_y, collar_band")
block = "\n".join(l[8:] for l in src[a:b].rstrip().split("\n"))

h = w = 512
face_box = (200, 60, 120, 150)              # fx, fy, fw, fh
torso_mask = np.zeros((h, w), np.float32)
torso_mask[190:, 120:400] = 1.0             # a body below the chin
ns = {"cv2": cv2, "np": np, "h": h, "w": w, "face_box": face_box, "torso_mask": torso_mask}
exec(block, ns)
cb = ns["collar_band"]

fails = []
def ok(c, m):
    print(("  PASS  " if c else "  FAIL  ") + m)
    if not c: fails.append(m)

ncy = face_box[1] + face_box[3]             # chin row = 210
print(f"band max {cb.max():.3f}  ncy={ncy}")
ok(cb.max() > 0.2, "a collar band exists at all")
ok(float(cb[:ncy - 25].max()) < 0.05,
   f"nothing above the chin — no choker around the neck (max {float(cb[:ncy-25].max()):.3f})")
ys, xs = np.where(cb > 0.15)
ok(ys.min() >= ncy - 25, f"the shading starts at the neckline, not on the jaw (top row {ys.min()})")

# Gradient: strongest right at the neck opening, fading outward.
rx = int(face_box[2] * 0.30)
cx = face_box[0] + face_box[2] // 2
row = cb[ncy + 40]
nz = np.where(row > 0.02)[0]
if len(nz):
    prof = row[nz.min():nz.max() + 1]
    ok(len(set(np.round(prof, 2))) > 3, f"the band is graded, not flat ({len(set(np.round(prof,2)))} levels)")
else:
    ok(False, "no band on the sample row")

ok(float((cb * (1 - torso_mask)).max()) < 0.02,
   "never shades a pixel with no garment on it")

# What the old flat ring did, for contrast.
print(f"\n  peak darkening now: {(1-0.88)*cb.max()*100:.0f}%   (was a flat 40% over the whole ring)")
print(f"\n{len(fails)} FAILURES" if fails else "\nall checks passed")
raise SystemExit(1 if fails else 0)
