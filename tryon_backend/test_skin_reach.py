"""Offline test for the skin-tone body extension in model.py.

    python3 tryon_backend/test_skin_reach.py

YCrCb skin detection cannot tell an arm from a warm wall, and the mask it
produces is dilated 11 px twice then blurred 21 px — a generous blob. ORed
into the silhouette unbounded, it pushed the garment out past the shoulders
onto whatever in the room was skin-coloured. This pins the rule that fixes
that: skin may extend the body, never invent one somewhere else.
"""
import pathlib, sys
import cv2
import numpy as np

H = W = 512
SKIN_REACH_PX = 41
fails = []


def ok(c, m):
    print(("  PASS  " if c else "  FAIL  ") + m)
    if not c:
        fails.append(m)


def bound(silhouette, skin_mask, mp_ok, reach=SKIN_REACH_PX):
    """The rule as implemented in model.py."""
    if mp_ok:
        r = max(3, int(reach) | 1)
        near = cv2.dilate((silhouette > 0.30).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))).astype(np.float32)
        skin_mask = skin_mask * near
    return np.maximum(silhouette, skin_mask * 0.80)


# A torso down the middle; a warm wall on the right that reads as skin; and a
# forearm just off the torso that the segmenter trimmed.
sil = np.zeros((H, W), np.float32)
cv2.rectangle(sil, (180, 140), (330, 460), 1.0, -1)

skin = np.zeros((H, W), np.float32)
cv2.rectangle(skin, (430, 100), (512, 460), 1.0, -1)     # the wall
cv2.rectangle(skin, (334, 200), (368, 300), 1.0, -1)     # forearm, 4 px off the torso
skin = cv2.GaussianBlur(skin, (21, 21), 0).clip(0, 1)

print("\n1. with a real silhouette, the wall is refused and the arm is kept")
out = bound(sil, skin, mp_ok=True)
ok(float(out[250, 470]) < 0.05, f"the warm wall is not claimed as body ({float(out[250,470]):.2f})")
ok(float(out[250, 350]) > 0.5, f"the trimmed forearm is still recovered ({float(out[250,350]):.2f})")
ok(float(out[300, 250]) > 0.99, "the torso itself is untouched")

print("\n2. the bound is a neighbourhood, not a hard clip to the silhouette")
# Just outside the torso the blurred skin blob is mid-ramp, so the test is
# that it is admitted at all — a hard clip to the silhouette would be 0.
edge = float(out[250, 332])
ok(0.05 < edge < 0.80, f"skin just outside the torso is admitted, part-strength ({edge:.2f})")

print("\n3. reach is what decides it")
ok(float(bound(sil, skin, True, reach=5)[250, 350]) < 0.5, "a tiny reach drops the forearm")
ok(float(bound(sil, skin, True, reach=401)[250, 470]) > 0.5, "a huge reach lets the wall back in")

print("\n4. with no real silhouette the blob is still better than nothing")
out2 = bound(sil, skin, mp_ok=False)
ok(float(out2[250, 470]) > 0.5, "unbounded when MediaPipe never landed a mask")

print("\n5. the rule can only add, never remove")
ok(bool((bound(sil, skin, True) >= sil - 1e-6).all()), "never takes coverage away from the silhouette")

print(f"\n{len(fails)} FAILURES" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
