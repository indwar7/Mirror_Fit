# Progress Log

Running log of work sessions. Newest first. Ops details (IPs, commands, gotchas)
are kept here so they don't have to be rediscovered next time — actual
credentials (passwords, tokens) are never written here, only where they live.

---

## 2026-09-03

### Security

- **Found and removed 3 hardcoded live API keys**, committed to the public
  GitHub repo (`indwar7/Mirror_Fit`), duplicated across `lib/` and the stale
  `lucy/lib/` copy (6 occurrences total):
  - `lib/services/decart_tryon_service.dart` — Decart AI key (not in active
    use; kept the field, moved to `String.fromEnvironment('DECART_API_KEY')`)
  - `lib/services/replicate_tryon_service.dart`,
    `lib/services/face_swap_service.dart` — Replicate API token (**actively
    used** — treat as compromised, rotate on Replicate's dashboard, still
    pending as of this writing), moved to
    `String.fromEnvironment('REPLICATE_API_TOKEN')`
  - Run with: `flutter run --dart-define=REPLICATE_API_TOKEN=... --dart-define=DECART_API_KEY=...`
  - Note: `--dart-define` keeps secrets out of source control but they're
    still baked into the compiled binary — the durable fix is proxying these
    calls through a backend (like `FASHN_API_KEY` already does correctly in
    `face_swap_backend/main.py`), not yet done.
  - Keys are still present in git **history** (not just the old working
    tree) — a private repo lowers exposure but doesn't remove them; a real
    history scrub (`git filter-repo` + force-push) is a separate, bigger step
    not yet taken.
- **Made `indwar7/Mirror_Fit` private** on GitHub (`gh repo edit --visibility
  private --accept-visibility-change-consequences`).
- No other secrets found committed anywhere (current tree or history) —
  `FASHN_API_KEY` and `HF_TOKEN` are both correctly read from env vars only.

### GPU box (live try-on backend)

- Box: Windows, NVIDIA A10G (24GB, shared across `face-swap` / `try-on` /
  `ai-twin` — they're on the same GPU), public IP `35.154.220.66`.
  RDP creds live wherever you keep secrets, not here.
- Layout: repo at `C:\virtual-try-on`, conda at `C:\miniconda3`, logs at
  `C:\logs`. `face-swap` and `try-on` **share one conda env**
  (`C:\miniconda3\python.exe`); `live-portrait` and `ai-twin` each have their
  own isolated env.
- **Start a backend** (only starts what's down, safe to re-run):
  ```powershell
  cd C:\virtual-try-on
  powershell -ExecutionPolicy Bypass .\start_all.ps1 -Only tryon
  # or, no -Only, to start everything that's down
  ```
- **HF_TOKEN gotcha**: `$env:HF_TOKEN = "..."` only lasts the current
  PowerShell window. Set it permanently instead:
  ```powershell
  [Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_...", "Machine")
  # close and reopen PowerShell for it to take effect
  [Environment]::GetEnvironmentVariable("HF_TOKEN", "Machine")   # verify
  ```
- **Check backend logs**:
  ```powershell
  Get-Content C:\logs\try-on.err.log -Tail 30
  Get-Content C:\logs\try-on.log -Tail 30
  ```
- **Testing Live Try-On from a Mac browser** (backend stays on the GPU box,
  nothing runs locally on the Mac except Chrome):
  1. Don't start the full `face_swap_backend` just to serve the demo page —
     it loads heavy models (InsightFace etc.) on the same shared GPU as
     `try-on`, for no reason. Serve the static `demo/` folder directly
     instead:
     ```powershell
     New-NetFirewallRule -DisplayName "Allow 7860" -Direction Inbound -LocalPort 7860 -Protocol TCP -Action Allow
     cd C:\virtual-try-on\demo
     & C:\miniconda3\python.exe -m http.server 7860
     ```
  2. Chrome blocks camera access (`getUserMedia`) on any plain-`http://`
     origin that isn't `localhost` — a remote IP over HTTP is treated as
     insecure. Work around it (dev-only, this Mac's Chrome only):
     `chrome://flags/#unsafely-treat-insecure-origin-as-secure` → enable →
     add `http://35.154.220.66:7860,http://35.154.220.66:8000` → Relaunch.
  3. Open `http://35.154.220.66:7860/` (not `/demo/index.html` — the static
     server's root *is* the demo folder), click **Live Try-On**.
  4. If the box is behind a cloud security group (this IP looks like AWS
     `ap-south-1`), the Windows Firewall rule above isn't enough on its own —
     the security group also needs inbound rules for 7860 and 8000.

### Live try-on quality — findings

1. **Periodic "pop" every ~3s** (not continuous jitter): the live screen
   (`decart_realtime_screen.dart`) captures and sends one frame every 3
   seconds; each cycle is a fresh diffusion generation blended with the
   previous result. `tryon_backend/model.py:2003-2019` already has real
   history here — `alpha_new` (new-frame weight in that blend) was `0.45`
   (smoother, but caused visible ghosting/double-edges on movement), moved to
   `0.7` (current — less ghosting, more pop). Three independent levers, not
   yet acted on:
   - Increase the client-side crossfade duration (currently 0.6s CSS
     transition) — spreads the same change over more time, doesn't touch the
     already-tuned diffusion blend. Lowest risk.
   - Shorten the 3s capture interval — smaller change per step, costs more
     GPU load per session.
   - Re-tune `alpha_new` — highest risk, re-opens an already-fought tradeoff.
2. **DWPose pose-conditioning is silently disabled.** Log evidence from this
   session: `DWPose failed to load, running without pose conditioning:
   module 'mediapipe' has no attribute 'solutions'`. This is the same
   legacy-API breakage already fixed once in
   `face_swap_backend/pose_backends.py` (commit `1dfca03`, migrated to
   `mediapipe.tasks`) — **not yet applied to `tryon_backend`**. Complication:
   `face-swap` and `try-on` share one conda env, so a `mediapipe` version
   change to fix this would also affect the already-fixed `face_swap_backend`;
   and the loader here is third-party (`controlnet_aux`'s `DWposeDetector`),
   not our own code, so the same "rewrite on Tasks API" fix path used before
   isn't directly available — would need a newer `controlnet_aux` release or
   a vendored/patched loader. Per `model.py:1767-1797`, without DWPose the
   pipeline falls back to vanilla img2img with **no pose conditioning at
   all**, which is a plausible contributor to garment placement not adapting
   well to body movement/rotation. Not yet fixed — parked pending a decision
   on the shared-env risk.
3. **Fabric overlay sits too high on the neck** (reproduced live, see
   session). This exact complaint was already addressed once before — there's
   an 80px fade-in ramp at `model.py:1912-1926` meant to push the fabric
   pattern down to the collarbone instead of the chin. The gap is a **fixed
   80px regardless of framing**, so when the camera is close (face fills a
   large fraction of frame, as in today's test shot), 80px is proportionally
   too small a gap. Proposed fix (not yet applied — paused before
   confirmation): make the gap proportional to detected face height instead
   of a fixed pixel count. Only touches the fabric-overlay ramp, nothing else
   in the pipeline.

### Open items

- [ ] Rotate the Replicate API token (was live on a public repo).
- [ ] Decide on git history scrub for the leaked keys (separate, bigger step).
- [ ] Decide on DWPose fix approach (shared-env risk vs. leaving it disabled).
- [ ] Apply + test the fabric-overlay neck-gap proportional fix.
- [ ] Decide on the 3s-pop fix (crossfade duration vs. capture interval vs.
      re-tuning the diffusion blend).
- [ ] Cloud security group check on the GPU box (35.154.220.66) — confirm
      which ports are actually meant to be open to the internet long-term;
      currently 7860/8000 are open for testing.
