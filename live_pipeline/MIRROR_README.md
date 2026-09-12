# Mirror, Hybrid 1 (live 3D jacket)

## Run locally
1. Put these files in one folder: `index.html`, `jacket_rigged.glb`.
2. In that folder, start a local server (any of these):
   - Python: `python -m http.server 8000`
   - Node: `npx serve .`
   - VS Code: install the "Live Server" extension, right-click index.html, "Open with Live Server".
3. Open `http://localhost:8000` in Chrome. Allow the camera.
4. Stand back so shoulders and hips are visible.

Internet is needed on first load (three.js, MediaPipe and its two models come from CDNs and are cached after that).

## Controls
- Swatches: three built-in patterns. "Add fabric image" loads any image from disk as the jacket fabric.
- Pattern size: how many times the fabric tiles across the jacket.
- Fit: scales the jacket relative to your shoulder width.
- Hair and face in front: draws your hair and face over the collar.
- Show tracking: shows the 33 body points.
- Capture photo: downloads the current frame. This is the hook for the CatVTON step.

## Files
- `jacket_rigged.glb`: the purchased jacket mesh with a 9-bone skeleton and skin weights added.
- `rig_jacket.py`: the script that produced it from the FBX-derived GLB. Re-run it for the woman's jacket or another garment.

## Known limits of this version
- Arms track; torso twist and lean are approximate. Fast moves lag by a frame or two.
- Hands crossing the chest go behind the jacket (only hair and face are drawn in front).
- Rig is auto-generated; when you export the Unreal skeletal mesh, swap the GLB and the bone names in `MAP`.
