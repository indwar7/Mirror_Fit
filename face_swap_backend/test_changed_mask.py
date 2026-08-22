"""Offline test for _changed_mask in main.py.

    python3 face_swap_backend/test_changed_mask.py

_changed_mask is what tells the browser which pixels the swap touched, and it
is the only reason swapped hair survives the client's face-oval crop. It needs
nothing but OpenCV, so it is testable here rather than by squinting at a
webcam feed — including the one case it deliberately does not handle, which is
what the client's oval union exists to cover.
"""
import io, pathlib, sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).parent


def _load():
    """Pull _changed_mask out of main.py rather than copying it, so this test
    cannot drift from the code that ships."""
    src = io.open(HERE / "main.py", encoding="utf-8").read()
    a = src.index("def _changed_mask(")
    b = src.index("def _swap_live(")
    ns = {"cv2": cv2, "np": np}
    exec(src[a:b], ns)
    return ns["_changed_mask"]


changed_mask = _load()
fails = []


def ok(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


rng = np.random.default_rng(7)
H = W = 384
frame = rng.integers(60, 180, (H, W, 3), dtype=np.uint8)

print("\n1. an untouched frame reports nothing touched")
m = changed_mask(frame, frame.copy())
ok(m.shape == (H, W), f"mask is single-channel at frame size {m.shape}")
ok(int(m.max()) == 0, "identical frames produce an entirely empty mask")

print("\n2. sensor noise below the threshold is not a change")
noisy = np.clip(frame.astype(np.int16) + rng.integers(-6, 7, frame.shape), 0, 255).astype(np.uint8)
m = changed_mask(frame, noisy)
ok(int((m > 40).sum()) == 0, "±6 of noise stays under the threshold")

print("\n3. a face and the hair above it are both covered")
after = frame.copy()
cv2.ellipse(after, (192, 210), (70, 95), 0, 0, 360, (250, 250, 250), -1)   # face
cv2.ellipse(after, (192, 110), (95, 60), 0, 0, 360, (10, 10, 10), -1)      # hair
m = changed_mask(frame, after)
ok(m[210, 192] > 200, "the face region is opaque in the mask")
ok(m[95, 192] > 200, "the hair above the face oval is opaque too — the whole point")
ok(int(m[:, :40].max()) == 0, "background well away from the head stays clear")

print("\n4. the edge really is feathered, not binary")
vals = set(np.unique(m).tolist())
ok(len(vals - {0, 255}) > 20, f"{len(vals)} distinct alpha levels — a soft edge, not a stencil")

print("\n5. a small interior hole is closed")
after2 = after.copy()
cv2.circle(after2, (192, 210), 4, tuple(int(v) for v in frame[210, 192]), -1)
m2 = changed_mask(frame, after2)
ok(m2[210, 192] > 200, "a 4 px unchanged speck inside the face does not punch through")

print("\n6. a LARGE interior hole is not closed — this is why the client unions")
print("   the mask with its own face oval rather than trusting it alone.")
after3 = after.copy()
after3[180:240, 165:220] = frame[180:240, 165:220]        # 55x60 unchanged block
m3 = changed_mask(frame, after3)
ok(m3[210, 192] < 60, "the hole survives, exactly as the client's oval union assumes")

print(f"\n{len(fails)} FAILURES" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
