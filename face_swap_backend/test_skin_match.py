"""Tests for skin-tone matching.  python -m unittest test_skin_match -v"""
from __future__ import annotations
import unittest
import cv2, numpy as np
from skin_match import match_face_to_body, optional_match, MIN_SKIN_PIXELS


def scene(face_bgr, body_bgr, size=200):
    """An image with a face patch and a body-skin patch of known colours."""
    img = np.zeros((size, size, 3), np.uint8)
    face_mask = np.zeros((size, size), np.uint8)
    body_mask = np.zeros((size, size), np.uint8)
    img[20:70, 70:130] = face_bgr;  face_mask[20:70, 70:130] = 255
    img[90:180, 60:140] = body_bgr; body_mask[90:180, 60:140] = 255
    # a little texture so std is non-zero
    rng = np.random.default_rng(0)
    noise = rng.integers(-6, 7, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, face_mask, body_mask


def mean_lab_L(img, mask):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(lab[mask > 0][:, 0].mean())


class TestMatching(unittest.TestCase):
    def test_pale_face_on_darker_body_moves_toward_the_body(self):
        img, fm, bm = scene((200, 205, 210), (140, 150, 165))
        before = mean_lab_L(img, fm)
        out, st = match_face_to_body(img, fm, bm, strength=1.0)
        after = mean_lab_L(out, fm)
        self.assertTrue(st.applied, st.reason)
        self.assertLess(after, before, "face should darken toward the body")
        # and should land near the body's lightness
        self.assertLess(abs(after - mean_lab_L(img, bm)), 12)

    def test_body_is_never_altered(self):
        img, fm, bm = scene((200, 205, 210), (140, 150, 165))
        out, st = match_face_to_body(img, fm, bm)
        self.assertTrue(st.applied)
        self.assertTrue(np.array_equal(img[bm > 0], out[bm > 0]),
                        "the body must be left exactly as it was")

    def test_nothing_outside_the_face_mask_changes(self):
        """Bit-identical, not merely close.

        The first implementation converted the whole image back from LAB,
        which is lossy, so every pixel drifted a few levels including the body
        and background. Same rule as the try-on paste-back.
        """
        img, fm, bm = scene((200, 205, 210), (140, 150, 165))
        out, _ = match_face_to_body(img, fm, bm)
        outside = fm == 0
        self.assertTrue(np.array_equal(img[outside], out[outside]))

    def test_strength_zero_leaves_the_face_essentially_alone(self):
        # "Essentially": at strength 0 the maths is a no-op but the pixels
        # still make a BGR->LAB->BGR trip inside the mask, which costs a few
        # levels. Outside the mask nothing moves at all — asserted below.
        img, fm, bm = scene((200, 205, 210), (140, 150, 165))
        out, st = match_face_to_body(img, fm, bm, strength=0.0)
        self.assertTrue(st.applied)
        self.assertLessEqual(float(np.abs(img.astype(int) - out.astype(int)).max()), 5)

    def test_already_matching_tones_barely_move(self):
        img, fm, bm = scene((160, 168, 180), (160, 168, 180))
        before = mean_lab_L(img, fm)
        out, _ = match_face_to_body(img, fm, bm)
        self.assertLess(abs(mean_lab_L(out, fm) - before), 4)


class TestDeclines(unittest.TestCase):
    def test_tiny_masks_decline_rather_than_raise(self):
        img = np.zeros((50, 50, 3), np.uint8)
        fm = np.zeros((50, 50), np.uint8); fm[0:3, 0:3] = 255
        bm = np.zeros((50, 50), np.uint8); bm[10:13, 10:13] = 255
        out, st = match_face_to_body(img, fm, bm)
        self.assertFalse(st.applied)
        self.assertIn("not enough skin pixels", st.reason)
        self.assertTrue(np.array_equal(img, out))

    def test_absurd_shift_is_refused(self):
        # Black "face" against a white "body": not a lighting difference, a
        # bad mask. Applying it would recolour the face wildly.
        img, fm, bm = scene((5, 5, 5), (250, 250, 250))
        out, st = match_face_to_body(img, fm, bm)
        self.assertFalse(st.applied)
        self.assertIn("too large", st.reason)
        self.assertTrue(np.array_equal(img, out))

    def test_optional_match_without_masks(self):
        img = np.zeros((50, 50, 3), np.uint8)
        out, st = optional_match(img, None, None)
        self.assertFalse(st.applied)
        self.assertIn("no parse", st.reason)
        self.assertTrue(np.array_equal(img, out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
