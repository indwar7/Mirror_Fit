#!/usr/bin/env python3
"""
Screen candidate images against the anatomy validator.

Triage only. It writes NOTHING into the asset set — it tells you which
candidates are worth opening, and prints the exact `add` command for each
survivor. Admission still goes through tools/curate_base_bodies.py, which
requires a person to confirm they have looked at the image.

    python tools/screen_candidates.py face_swap_backend/bodies_cache
    python tools/screen_candidates.py ~/photos --gender male

Expect a low pass rate on the old generated bodies. Those are the figures that
shipped with two torsos; the validator now catches exactly that, so most of
them failing is the tool working, not the tool being broken.

Passing here means "anatomically plausible", not "usable". A figure can have a
correct skeleton and still be badly lit, cropped oddly, or wearing something
the try-on parser cannot segment. That is what the human pass is for.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "face_swap_backend"))

import body_shapes                                    # noqa: E402
from avatar_validation import ValidatorUnavailable    # noqa: E402

# The old generated ids encoded (size, taper); the new bins encode a single
# build. A tapered average frame is what "athletic" now means, and "broad"
# collapses into "plus". Only a suggestion — the operator picks the real id.
_OLD_TO_BIN = {
    ("slim", "tapered"): "slim", ("slim", "regular"): "slim",
    ("slim", "straight"): "slim",
    ("average", "tapered"): "athletic",
    ("average", "regular"): "average", ("average", "straight"): "average",
    ("broad", "tapered"): "plus", ("broad", "regular"): "plus",
    ("broad", "straight"): "plus",
}


def suggest_id(stem: str) -> str | None:
    """Map an old `body_m_average_tapered` name onto a new bin id, if it looks
    like one. Returns None when the name says nothing useful."""
    parts = stem.split("_")
    if len(parts) != 4 or parts[0] != "body":
        return None
    gender = {"m": "male", "f": "female"}.get(parts[1])
    build = _OLD_TO_BIN.get((parts[2], parts[3]))
    if not gender or not build:
        return None
    return f"body_{gender}_{build}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--gender", choices=("male", "female"),
                    help="assume this gender when the filename does not say")
    args = ap.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"error: {directory} is not a directory")
        return 1

    candidates = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not candidates:
        print(f"No images in {directory}")
        return 1

    import cv2
    from pose_backends import load_validator

    validator = load_validator()
    print(f"\nScreening {len(candidates)} candidate(s) from {directory}\n")

    passed: list[tuple[Path, str | None]] = []
    failed: list[tuple[Path, str]] = []

    for path in candidates:
        image = cv2.imread(str(path))
        if image is None:
            failed.append((path, "unreadable"))
            print(f"  [FAIL] {path.name:34} unreadable")
            continue
        try:
            result = validator.validate(image)
        except ValidatorUnavailable as e:
            # Fail closed, exactly as the runtime does. Screening without the
            # anatomy check would recreate the original incident.
            print(f"\nREFUSED — {e}")
            return 2

        if result.ok:
            suggestion = suggest_id(path.stem)
            passed.append((path, suggestion))
            print(f"  [PASS] {path.name:34} -> suggest {suggestion or '(pick an id)'}")
        else:
            failed.append((path, result.reason()))
            print(f"  [FAIL] {path.name:34} {result.reason()}")

    print(f"\n{len(passed)} passed, {len(failed)} failed.")

    if not passed:
        print("\nNothing survived. On the old generated bodies that is the "
              "expected outcome — they are the two-torso figures this check "
              "exists to reject. Source photographs instead.")
        return 1

    # Which bins the survivors could actually fill, and which stay empty.
    covered = {s for _, s in passed if s}
    still_missing = [b for b in body_shapes.all_base_body_ids() if b not in covered]

    print("\nNext step — open each survivor, then admit it:\n")
    for path, suggestion in passed:
        target = suggestion or "body_<gender>_<build>"
        print(f"  python tools/curate_base_bodies.py add {path} \\\n"
              f"      --id {target} --height-cm <real height> \\\n"
              f"      --license \"<licence>\" --source \"{path}\" "
              f"--approved-by \"<you>\"\n")

    if still_missing:
        print(f"Bins no survivor covers: {', '.join(still_missing)}")
        print("Those need photographs from somewhere else.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
