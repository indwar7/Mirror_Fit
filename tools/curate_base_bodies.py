#!/usr/bin/env python3
"""
Admit one photograph into the base-body asset set.

This is the ONLY way an image gets into assets/base_bodies/. It is offline, it
is run by a person, and it records who approved what.

    python tools/curate_base_bodies.py add candidate.jpg \
        --id body_male_average --height-cm 178 \
        --license "Unsplash Licence" --source "https://unsplash.com/photos/xyz" \
        --approved-by "abhay"

    python tools/curate_base_bodies.py status
    python tools/curate_base_bodies.py verify

What `add` does, in order:
  1. Refuses unless the id is one body_shapes can actually select.
  2. Runs the SAME anatomy validator the runtime uses. Fails closed — if the
     pose model is missing, nothing is admitted.
  3. Prints the per-check result and REQUIRES the operator to confirm they have
     looked at the image. The validator catches two torsos; it does not catch
     "this photograph is unusable for other reasons", and the whole point of
     the asset set is that a person saw it.
  4. Copies the file in, records sha256, licence, source and approver.

The confirmation step is not ceremony. Two rounds of broken base bodies shipped
because a machine check said "usable" and nobody looked.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "face_swap_backend"))

import base_bodies                      # noqa: E402
import body_shapes                      # noqa: E402
from avatar_validation import ValidatorUnavailable   # noqa: E402


def _validator():
    from pose_backends import load_validator
    return load_validator()


def cmd_add(args: argparse.Namespace) -> int:
    src = Path(args.image)
    if not src.exists():
        print(f"error: {src} does not exist")
        return 1

    valid_ids = body_shapes.all_base_body_ids()
    if args.id not in valid_ids:
        print(f"error: '{args.id}' is not a selectable bin.\n"
              f"       body_shapes can only ask for: {', '.join(valid_ids)}")
        return 1

    import cv2
    image = cv2.imread(str(src))
    if image is None:
        print(f"error: {src} is not a readable image")
        return 1

    print(f"\nValidating {src.name} …")
    try:
        result = _validator().validate(image)
    except ValidatorUnavailable as e:
        # Fail closed. This is the exact path that, as a warning, let eighteen
        # broken figures through.
        print(f"\nREFUSED — {e}")
        return 2

    print(result.report())
    if not result.ok:
        print(f"\nREJECTED — {result.reason()}")
        return 1

    if not args.yes:
        print("\nThe validator is satisfied. It checks anatomy; it does not "
              "judge whether the photo is usable.")
        print(f"Open {src} and look at it now.")
        answer = input("Type 'approved' to admit it: ").strip().lower()
        if answer != "approved":
            print("Not admitted.")
            return 1

    base_bodies.ASSET_DIR.mkdir(parents=True, exist_ok=True)
    dest_name = f"{args.id}{src.suffix.lower()}"
    dest = base_bodies.ASSET_DIR / dest_name
    shutil.copy2(src, dest)

    manifest_path = base_bodies.MANIFEST_PATH
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists()
                else {"version": base_bodies.MANIFEST_VERSION, "bodies": []})

    gender, build = args.id.removeprefix("body_").split("_", 1)
    entry = {
        "id": args.id,
        "gender": gender,
        "build": build,
        "height_cm": args.height_cm,
        "license": args.license,
        "source": args.source,
        "sha256": base_bodies.sha256_of(dest),
        "file": dest_name,
        "approved_by": args.approved_by,
        "approved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest["bodies"] = [b for b in manifest["bodies"] if b.get("id") != args.id]
    manifest["bodies"].append(entry)
    manifest["bodies"].sort(key=lambda b: b["id"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nAdmitted {args.id} -> {dest_name}")
    print(f"  sha256      {entry['sha256']}")
    print(f"  licence     {entry['license']}")
    print(f"  approved by {entry['approved_by']} at {entry['approved_at']}")
    print("\nCommit assets/base_bodies/ so the asset travels with the code.")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    st = base_bodies.status()
    print(f"\nBase bodies: {st['ready']}/{st['total']} ready\n")
    for row in st["bodies"]:
        mark = {"ok": "OK     ", "missing": "MISSING", "corrupt": "CORRUPT"}[row["state"]]
        extra = row.get("detail") or f"licence={row.get('license')} by={row.get('approved_by')}"
        print(f"  [{mark}] {row['id']:24} {extra}")
    print()
    return 0 if st["ready"] == st["total"] else 1


def cmd_verify(_args: argparse.Namespace) -> int:
    """Re-run the anatomy validator over everything already admitted."""
    import cv2
    try:
        validator = _validator()
        bodies = base_bodies.all_bodies()
    except ValidatorUnavailable as e:
        print(f"REFUSED — {e}")
        return 2

    if not bodies:
        print("No base bodies admitted yet.")
        return 1

    failures = 0
    for body_id, body in sorted(bodies.items()):
        image = cv2.imread(str(body.path()))
        if image is None:
            print(f"  [FAIL] {body_id}: file unreadable")
            failures += 1
            continue
        try:
            result = validator.validate(image)
        except ValidatorUnavailable as e:
            print(f"REFUSED — {e}")
            return 2
        if result.ok:
            print(f"  [PASS] {body_id}")
        else:
            print(f"  [FAIL] {body_id}: {result.reason()}")
            failures += 1
    print(f"\n{len(bodies) - failures}/{len(bodies)} pass")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="admit one photograph")
    a.add_argument("image")
    a.add_argument("--id", required=True, help="e.g. body_male_average")
    a.add_argument("--height-cm", type=float, required=True,
                   help="the model's real standing height, for the record")
    a.add_argument("--license", required=True, help="e.g. 'Unsplash Licence'")
    a.add_argument("--source", required=True, help="where it came from")
    a.add_argument("--approved-by", required=True, help="who looked at it")
    a.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt (CI only — a human must "
                        "still have looked)")
    a.set_defaults(func=cmd_add)

    sub.add_parser("status", help="per-id readiness").set_defaults(func=cmd_status)
    sub.add_parser("verify", help="re-validate everything admitted").set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
