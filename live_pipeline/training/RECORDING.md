# Phase 2: recording checklist

The footage is the asset. It outlives every model we train on it, so
shoot it as if it has to serve a base model that does not exist yet.

## Before anyone arrives

- [ ] **Same webcam, same resolution as production.** A different lens
      teaches the model a different geometry than it will ever see live.
- [ ] Blue screen lit evenly — no hotspots, no shadow gradient. Uneven
      lighting is what makes keying leave a halo.
- [ ] The real jacket in a **solid non-blue** colour. Note its OpenCV hue
      window; `prep.py --garment-hue lo,hi` needs it.
- [ ] Subject stands ~2m from the screen so bounce light is minimal.
- [ ] Lock white balance and exposure. Auto-exposure shifts mid-clip and
      the model learns the drift as part of the garment.
- [ ] Consent form printed. Keep it with the footage, not in a drawer.

## Per person (8-10 people)

Record **with the jacket**, one clip per movement, ~10s each:

1. Face the camera, arms at sides
2. Turn 45° left, hold
3. Turn 45° right, hold
4. Arms up
5. Arms forward
6. Cross arms
7. Hands on hips
8. Walk in from off-frame, stop, walk out

Then **one short clip without the jacket** (~10s, facing camera). This is
what lets `prep.py` verify its garment mask is finding cloth and not skin.

## After the first person

Stop. Run:

```bash
python -m training.prep --footage data/footage --out /tmp/check \
    --garment-hue <lo>,<hi>
```

Open `/tmp/check/train/sample_grid.png`. Left is what the model sees,
right is what it must produce. Check:

- The blue screen is gone, with no blue rim on hair or shoulders
- The **real** jacket is erased in the left image, not showing through
- The 3D jacket sits on the shoulders, not floating or sunk into the neck
- Both halves are the same person in the same pose

Fix the lighting or the hue window before shooting the other nine. A bad
grid here means a wasted shoot day.

## Splits

Two people are held out and never trained on:

```bash
python -m training.prep ... --holdout <name1>,<name2>
```

Held out **by person**, not by frame — frames from one person are
near-copies, and a frame-level split scores memorisation.

## Keep

- Raw clips, per person, original resolution
- Consent forms
- The hue window and camera settings used

The trained LoRA is disposable. This is not.
