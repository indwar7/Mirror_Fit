"""
Tests for the base-body lookup.

Pure function, no models, so this runs anywhere:

    python -m unittest test_body_shapes -v
"""
from __future__ import annotations

import random
import unittest

import body_shapes as bs


def M(**kw) -> bs.Measurements:
    return bs.parse_measurements(kw)


class TestBinCoverage(unittest.TestCase):
    def test_eight_bins(self):
        ids = bs.all_base_body_ids()
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)

    def test_every_possible_input_maps_into_the_asset_set(self):
        """No input may name a body the asset set does not define.

        The endpoint returns 501 for a missing id rather than substituting one,
        so an id outside this set would be a permanent dead end for whoever
        happened to have those measurements.
        """
        ids = set(bs.all_base_body_ids())
        random.seed(0)
        for _ in range(5000):
            g = random.choice(["male", "female", "", "nonsense", None])
            chest = random.uniform(50, 200)
            waist = random.uniform(40, min(200, chest * 1.6))
            m = bs.Measurements(
                height_cm=random.uniform(100, 230),
                chest_cm=chest, waist_cm=waist,
                hips_cm=random.choice([None, random.uniform(50, 200)]),
            )
            self.assertIn(bs.base_body_id(g, m), ids)

    def test_no_measurements_gives_the_average_bin(self):
        for g in ("male", "female", None, "weird"):
            with self.subTest(gender=g):
                self.assertTrue(bs.base_body_id(g).endswith("_average"))
                self.assertTrue(bs.describe(g)["from_defaults"])


class TestBinning(unittest.TestCase):
    def test_real_bodies_land_where_expected(self):
        cases = [
            # gender, chest, waist, expected build
            ("male",    88,  76, "slim"),
            ("male",    99,  88, "average"),
            ("male",   102,  80, "athletic"),   # drop 1.28
            ("male",   115, 104, "plus"),
            ("male",    98, 104, "plus"),       # ordinary chest, large waist
            ("female",  82,  70, "slim"),
            ("female",  92,  80, "average"),
            ("female",  94,  70, "athletic"),   # drop 1.34
            ("female", 106,  96, "plus"),
        ]
        for g, chest, waist, expected in cases:
            with self.subTest(gender=g, chest=chest, waist=waist):
                got = bs.bin_for(g, M(chest_cm=chest, waist_cm=waist))
                self.assertEqual(got, expected,
                                 f"{g} {chest}/{waist} -> {got}, expected {expected}")

    def test_plus_is_decided_before_athletic(self):
        """A large tapered frame is plus, not athletic.

        Girth has to be satisfied before taper — a garment that does not go
        round the chest is not saved by having the right drop.
        """
        m = M(chest_cm=130, waist_cm=100)          # drop 1.30, well above plus
        self.assertEqual(bs.bin_for("male", m), "plus")

    def test_waist_alone_can_trigger_plus(self):
        self.assertEqual(bs.bin_for("male", M(chest_cm=100, waist_cm=105)), "plus")

    def test_partial_measurements_are_ignored_not_half_used(self):
        # Chest without waist gives no drop; using it for girth alone would
        # look like the measurement counted.
        self.assertEqual(bs.bin_for("male", M(chest_cm=130)), "average")
        self.assertEqual(bs.bin_for("male", M(waist_cm=130)), "average")
        self.assertTrue(bs.describe("male", M(chest_cm=130))["from_defaults"])

    def test_gender_cutoffs_differ(self):
        """The same numbers must bin differently by gender.

        92 cm sits around menswear S (≈91) but comfortably inside womenswear's
        middle band, and 102 cm is an ordinary men's L while being past the
        women's plus cutoff. A shared cutoff would misfile one of the two in
        both cases.
        """
        small = M(chest_cm=92, waist_cm=84)
        self.assertEqual(bs.bin_for("male", small), "slim")
        self.assertEqual(bs.bin_for("female", small), "average")

        big = M(chest_cm=102, waist_cm=96)
        self.assertEqual(bs.bin_for("male", big), "average")
        self.assertEqual(bs.bin_for("female", big), "plus")

    def test_determinism(self):
        m = M(chest_cm=102, waist_cm=80, height_cm=175)
        first = bs.base_body_id("male", m)
        for _ in range(50):
            self.assertEqual(bs.base_body_id("male", m), first)


class TestValidation(unittest.TestCase):
    def test_inches_typed_as_cm_rejected(self):
        with self.assertRaises(bs.MeasurementError) as ctx:
            M(chest_cm=40, waist_cm=32)
        self.assertIn("centimetres", str(ctx.exception))

    def test_swapped_chest_and_waist_rejected(self):
        with self.assertRaises(bs.MeasurementError):
            M(chest_cm=60, waist_cm=150)

    def test_non_numeric_rejected(self):
        with self.assertRaises(bs.MeasurementError):
            M(chest_cm="abc", waist_cm=80)

    def test_absent_is_not_an_error(self):
        m = M()
        self.assertIsNone(m.chest_cm)
        self.assertFalse(m.has_torso)

    def test_empty_string_treated_as_absent(self):
        # Unfilled HTML form fields arrive as "".
        m = M(chest_cm="", waist_cm="")
        self.assertFalse(m.has_torso)


class TestPurity(unittest.TestCase):
    def test_module_imports_no_model_or_io(self):
        """This module must stay a pure function.

        It used to sit next to a generator that produced the bodies; the whole
        point of the rewrite is that choosing a body and creating one are no
        longer the same concern.
        """
        import inspect
        source = inspect.getsource(bs)
        for forbidden in ("torch", "diffusers", "StableDiffusion", "requests",
                          "httpx", "open(", "Path("):
            self.assertNotIn(forbidden, source, f"{forbidden} in body_shapes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
