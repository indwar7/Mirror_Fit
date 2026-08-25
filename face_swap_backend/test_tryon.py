"""
Tests for garment try-on.

The identity guarantee is asserted as bit-identical equality, not as a
similarity score — "the face looks about the same" is not something a test can
hold anyone to.

    python -m unittest test_tryon -v
"""
from __future__ import annotations

import unittest

import numpy as np

from tryon import (
    ATR,
    TryOnPipeline,
    TryOnUnavailable,
    cloth_agnostic_mask,
    composite,
    identity_preserved,
)


def parse_map(h=200, w=100) -> np.ndarray:
    """A small ATR label map: face on top, shirt, arms, trousers, shoes."""
    p = np.full((h, w), ATR["background"], np.uint8)
    p[10:30, 40:60] = ATR["hair"]
    p[30:55, 40:60] = ATR["face"]
    p[55:120, 30:70] = ATR["upper_clothes"]
    p[55:110, 20:30] = ATR["left_arm"]
    p[55:110, 70:80] = ATR["right_arm"]
    p[120:180, 30:70] = ATR["pants"]
    p[180:195, 30:48] = ATR["left_shoe"]
    p[180:195, 52:70] = ATR["right_shoe"]
    return p


class FakeParser:
    def __init__(self, p): self._p = p
    def parse(self, image_bgr): return self._p


class FakeDense:
    def densepose(self, image_bgr): return np.zeros(image_bgr.shape[:2], np.uint8)


class FakeVTON:
    """Returns a completely different image — the worst case for identity.

    If the pipeline ever returned the model output directly instead of
    compositing, every identity assertion below would fail loudly.
    """
    def __init__(self): self.calls = 0
    def generate(self, person_bgr, agnostic_mask, densepose, garment_bgr, category):
        self.calls += 1
        return np.full_like(person_bgr, 77)


class TestMask(unittest.TestCase):
    def test_upper_covers_shirt_and_arms(self):
        m = cloth_agnostic_mask(parse_map(), "upper")
        self.assertTrue(m[60, 50] > 0, "shirt should be replaceable")
        self.assertTrue(m[60, 25] > 0, "sleeve/arm should be replaceable")
        self.assertEqual(m[150, 50], 0, "trousers must not be touched by 'upper'")

    def test_face_and_hair_never_replaceable(self):
        for category in ("upper", "lower", "dress"):
            with self.subTest(category=category):
                m = cloth_agnostic_mask(parse_map(), category)
                self.assertEqual(m[40, 50], 0, "face must never be in the mask")
                self.assertEqual(m[20, 50], 0, "hair must never be in the mask")

    def test_shoes_never_replaceable(self):
        m = cloth_agnostic_mask(parse_map(), "lower")
        self.assertEqual(m[188, 40], 0, "shoes are not the garment")

    def test_protected_labels_win_over_replaceable(self):
        # A parser that bleeds shirt onto the face must not put face pixels in
        # the mask — this is the ordering the implementation relies on.
        p = parse_map()
        p[30:55, 40:60] = ATR["face"]
        p[45, 50] = ATR["upper_clothes"]      # bleed
        m = cloth_agnostic_mask(p, "upper")
        self.assertTrue(m[45, 50] > 0)        # the bled pixel is replaceable
        self.assertEqual(m[35, 50], 0)        # actual face is not

    def test_bad_category_rejected(self):
        with self.assertRaises(ValueError):
            cloth_agnostic_mask(parse_map(), "hat")


class TestCompositeIsBitExact(unittest.TestCase):
    def test_outside_mask_is_bit_identical(self):
        rng = np.random.default_rng(0)
        original = rng.integers(0, 255, (200, 100, 3), dtype=np.uint8)
        generated = np.full_like(original, 77)
        mask = cloth_agnostic_mask(parse_map(), "upper")

        result = composite(original, generated, mask)

        outside = mask <= 127
        self.assertTrue(np.array_equal(original[outside], result[outside]))
        self.assertTrue(identity_preserved(original, result, mask))

    def test_inside_mask_takes_the_generated_pixels(self):
        original = np.zeros((200, 100, 3), np.uint8)
        generated = np.full_like(original, 77)
        mask = cloth_agnostic_mask(parse_map(), "upper")
        result = composite(original, generated, mask)
        self.assertTrue(np.all(result[mask > 127] == 77))

    def test_no_blending_at_the_edge(self):
        # Every output pixel must be exactly one source or the other. A blend
        # would leave face pixels *nearly* unchanged, which is unverifiable.
        original = np.zeros((200, 100, 3), np.uint8)
        generated = np.full_like(original, 200)
        mask = cloth_agnostic_mask(parse_map(), "upper")
        result = composite(original, generated, mask)
        self.assertEqual(set(np.unique(result).tolist()), {0, 200})

    def test_shape_mismatch_raises_rather_than_resampling(self):
        original = np.zeros((200, 100, 3), np.uint8)
        with self.assertRaises(ValueError):
            composite(original, np.zeros((100, 50, 3), np.uint8),
                      np.zeros((200, 100), np.uint8))
        with self.assertRaises(ValueError):
            composite(original, np.zeros_like(original), np.zeros((10, 10), np.uint8))

    def test_identity_preserved_detects_a_violation(self):
        original = np.zeros((200, 100, 3), np.uint8)
        mask = cloth_agnostic_mask(parse_map(), "upper")
        tampered = composite(original, np.full_like(original, 77), mask)
        tampered[40, 50] = 9          # a face pixel, outside the mask
        self.assertFalse(identity_preserved(original, tampered, mask))


class TestPipeline(unittest.TestCase):
    def _pipeline(self):
        return TryOnPipeline(FakeParser(parse_map()), FakeDense(), FakeVTON())

    def test_face_survives_a_model_that_repaints_everything(self):
        rng = np.random.default_rng(1)
        avatar = rng.integers(0, 255, (200, 100, 3), dtype=np.uint8)
        garment = np.zeros((64, 64, 3), np.uint8)

        result = self._pipeline().tryon("usr_x", avatar, garment, "upper")

        mask = cloth_agnostic_mask(parse_map(), "upper")
        self.assertTrue(identity_preserved(avatar, result, mask))
        # Face pixels specifically.
        self.assertTrue(np.array_equal(avatar[30:55, 40:60], result[30:55, 40:60]))

    def test_three_garments_leave_the_face_untouched(self):
        # The end-to-end acceptance shape, with the models faked out.
        rng = np.random.default_rng(2)
        avatar = rng.integers(0, 255, (200, 100, 3), dtype=np.uint8)
        pipeline = self._pipeline()
        for i in range(3):
            garment = np.full((64, 64, 3), i * 40, np.uint8)
            result = pipeline.tryon("usr_x", avatar, garment, "upper")
            with self.subTest(garment=i):
                self.assertTrue(np.array_equal(
                    avatar[30:55, 40:60], result[30:55, 40:60]))

    def test_maps_are_cached_per_avatar(self):
        class CountingParser(FakeParser):
            def __init__(self, p): super().__init__(p); self.calls = 0
            def parse(self, image_bgr):
                self.calls += 1
                return super().parse(image_bgr)

        parser = CountingParser(parse_map())
        pipeline = TryOnPipeline(parser, FakeDense(), FakeVTON())
        avatar = np.zeros((200, 100, 3), np.uint8)
        for _ in range(3):
            pipeline.tryon("usr_x", avatar, np.zeros((8, 8, 3), np.uint8), "upper")
        self.assertEqual(parser.calls, 1, "parse should run once per avatar")

        pipeline.invalidate("usr_x")
        pipeline.tryon("usr_x", avatar, np.zeros((8, 8, 3), np.uint8), "upper")
        self.assertEqual(parser.calls, 2, "invalidate should force a re-parse")

    def test_missing_backend_raises_and_names_it(self):
        avatar = np.zeros((200, 100, 3), np.uint8)
        for parser, dense, vton, expected in (
            (None, FakeDense(), FakeVTON(), "human parser"),
            (FakeParser(parse_map()), None, FakeVTON(), "DensePose"),
            (FakeParser(parse_map()), FakeDense(), None, "VTON model"),
        ):
            with self.subTest(expected=expected):
                pipeline = TryOnPipeline(parser, dense, vton)
                self.assertFalse(pipeline.available)
                with self.assertRaises(TryOnUnavailable) as ctx:
                    pipeline.tryon("usr_x", avatar, avatar, "upper")
                self.assertIn(expected, str(ctx.exception))

    def test_missing_backend_does_not_return_the_input(self):
        # The tempting "graceful" failure is to hand back the avatar, which
        # looks exactly like a try-on that ran and changed nothing.
        pipeline = TryOnPipeline(None, None, None)
        with self.assertRaises(TryOnUnavailable):
            pipeline.tryon("usr_x", np.zeros((10, 10, 3), np.uint8),
                           np.zeros((10, 10, 3), np.uint8), "upper")

    def test_empty_region_raises(self):
        p = np.full((200, 100), ATR["background"], np.uint8)
        pipeline = TryOnPipeline(FakeParser(p), FakeDense(), FakeVTON())
        with self.assertRaises(ValueError):
            pipeline.tryon("usr_x", np.zeros((200, 100, 3), np.uint8),
                           np.zeros((8, 8, 3), np.uint8), "upper")


if __name__ == "__main__":
    unittest.main(verbosity=2)
