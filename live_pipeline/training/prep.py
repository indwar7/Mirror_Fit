"""Phase 3: turn recorded footage into training pairs.

For every frame, at 10 fps:
  input  = the frame with the real jacket erased and the 3D jacket rendered on
  target = the same frame, untouched

The model's job is to make the composite look like the target. Hold-out
is by *person*, never by frame: frames from one person are near-copies of
each other, so a frame-level split reports a score the model has not
earned.

Usage:
    python -m training.prep --footage data/footage --out data/pairs \\
        --holdout priya,arjun --garment-hue 0,12
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import cv2
import numpy as np

import config
from .chroma import blue_mask, erase_garment, garment_mask, replace_background
from .render_guide import GuideRenderer

log = logging.getLogger("prep")

FPS = 10  # the plan's sampling rate


def person_mask_from_blue(bgr: np.ndarray) -> np.ndarray:
    """1 on the subject. The inverse of the keyed screen."""
    return 1 - blue_mask(bgr)


def crop_to_torso(bgr: np.ndarray, person: np.ndarray, size: int) -> tuple[np.ndarray, tuple] | None:
    """Square crop centred on the subject's upper body.

    Training on full frames wastes most of the pixels on background; the
    garment is what we care about.
    """
    ys, xs = np.nonzero(person)
    if len(xs) < 500:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    # Upper 60% of the silhouette: torso and arms, not legs.
    y1 = y0 + int((y1 - y0) * 0.6)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = max(x1 - x0, y1 - y0) // 2
    half = int(half * 1.25)  # breathing room so arms are not clipped
    h, w = bgr.shape[:2]
    cx = np.clip(cx, half, w - half) if half * 2 < w else w // 2
    cy = np.clip(cy, half, h - half) if half * 2 < h else h // 2
    half = min(half, w // 2, h // 2)
    box = (cx - half, cy - half, cx + half, cy + half)
    crop = bgr[box[1]:box[3], box[0]:box[2]]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), box


def process_clip(
    path: pathlib.Path,
    person: str,
    renderer: GuideRenderer,
    out_dir: pathlib.Path,
    hue: tuple[int, int],
    size: int,
) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log.error("cannot open %s", path)
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / FPS))

    pairs = 0
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue

        subject = person_mask_from_blue(frame)
        if subject.sum() < 5000:
            continue  # nobody in shot

        # Target: real frame, neutral background, real jacket kept.
        target = replace_background(frame, 1 - subject)

        # Input: same frame with the real jacket erased, then the 3D
        # jacket rendered on by the app's own rig.
        jacket = garment_mask(frame, subject, hue[0], hue[1])
        if jacket.sum() < 800:
            continue  # jacket not visible — nothing to learn from here
        erased = erase_garment(target, jacket)
        guide = renderer.render(erased)
        if guide is None:
            continue  # pose not found

        tc = crop_to_torso(target, subject, size)
        gc = crop_to_torso(guide, subject, size)
        if tc is None or gc is None:
            continue

        stem = f"{person}_{path.stem}_{idx:06d}"
        cv2.imwrite(str(out_dir / "input" / f"{stem}.png"), gc[0])
        cv2.imwrite(str(out_dir / "target" / f"{stem}.png"), tc[0])
        pairs += 1

    cap.release()
    log.info("%s: %d pairs", path.name, pairs)
    return pairs


def sample_grid(out_dir: pathlib.Path, n: int = 6) -> None:
    """A strip of input/target pairs to eyeball before training.

    The plan asks for this. Bad keying or a misplaced render is obvious
    here and invisible in a loss curve.
    """
    inputs = sorted((out_dir / "input").glob("*.png"))[:: max(1, len(list((out_dir / "input").glob("*.png"))) // n)][:n]
    if not inputs:
        return
    rows = []
    for p in inputs:
        a = cv2.imread(str(p))
        b = cv2.imread(str(out_dir / "target" / p.name))
        if a is None or b is None:
            continue
        rows.append(np.hstack([a, b]))
    if rows:
        cv2.imwrite(str(out_dir / "sample_grid.png"), np.vstack(rows))
        log.info("wrote %s", out_dir / "sample_grid.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--footage", type=pathlib.Path, required=True,
                    help="directory of <person>/<clip>.mp4")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--holdout", default="",
                    help="comma-separated people kept for eval")
    ap.add_argument("--garment-hue", default="0,12",
                    help="OpenCV hue window of the real jacket, lo,hi")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    hue = tuple(int(x) for x in args.garment_hue.split(","))
    holdout = {p.strip() for p in args.holdout.split(",") if p.strip()}

    if not args.footage.is_dir():
        log.error("footage directory not found: %s", args.footage)
        return 1
    people = sorted(d for d in args.footage.iterdir() if d.is_dir())
    if not people:
        log.error("no person directories under %s", args.footage)
        return 1
    unknown = holdout - {d.name for d in people}
    if unknown:
        log.error("held-out people not found in footage: %s", ", ".join(sorted(unknown)))
        return 1

    manifest = {"train": [], "eval": [], "fps": FPS, "size": args.size}
    with GuideRenderer(config.WIDTH, config.HEIGHT) as renderer:
        for person_dir in people:
            split = "eval" if person_dir.name in holdout else "train"
            out_dir = args.out / split
            (out_dir / "input").mkdir(parents=True, exist_ok=True)
            (out_dir / "target").mkdir(parents=True, exist_ok=True)
            total = 0
            for clip in sorted(person_dir.glob("*.mp4")):
                total += process_clip(clip, person_dir.name, renderer, out_dir, hue, args.size)
            manifest[split].append({"person": person_dir.name, "pairs": total})
            log.info("%s -> %s (%d pairs)", person_dir.name, split, total)

    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for split in ("train", "eval"):
        if (args.out / split / "input").exists():
            sample_grid(args.out / split)

    n_train = sum(p["pairs"] for p in manifest["train"])
    n_eval = sum(p["pairs"] for p in manifest["eval"])
    log.info("done: %d train pairs, %d eval pairs", n_train, n_eval)
    if n_eval == 0:
        log.warning("no held-out pairs — pass --holdout with two people, "
                    "or evaluation will score memorisation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
