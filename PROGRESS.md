# Progress Log

Running log of work sessions. Newest first. Ops details (IPs, commands, gotchas)
are kept here so they don't have to be rediscovered next time — actual
credentials (passwords, tokens) are never written here, only where they live.

---

## 2026-11-14

### Box back up, first real-server Live Try-On pass, Returns Guard removed

- New box IP 13.233.158.170. Deploys had died because NETWORK SERVICE could
  not open the Administrator-owned `C:\logs\*.log`; `start_all.ps1` now
  falls back to `C:\logs\<account>\` (commit 5548370). Face Swap, Live
  Portrait and AI Twin came up from a deploy for the first time. Try-On did
  not: the service account has no Hugging Face cache, re-downloaded 3.5 GB
  of weights and filled the disk (0.4 GB free), which also crashed the
  runner. `tools/box-ops.ps1` (run by the user as Administrator over RDP)
  deleted that partial cache, restarted the runner and started Try-On under
  Administrator. All four backends up; web app tested against the real
  server end to end.
- Live Try-On verdict from the user: works, but the cloth twitches and
  fades while standing still. `tryon_backend/model.py` (commit 9f4c3b4,
  main) adds a per-frame motion estimate and makes every EMA on the tier-3
  path motion-aware (still → heavy prev weight, moving → light), and
  normalises the fabric overlay to the painted shirt's own mean brightness.
  Not yet restarted on the box: needs `git pull` + `start_all.ps1 -Only
  tryon -Force` there.
- Returns Guard removed from the web app completely (tab, landing card,
  panel, report, the three handoff buttons, `returns-guard.ts`,
  `returns_model.json`, its CSS). Smoke test updated and green.
- Still open: a push to main redeploys everything under NETWORK SERVICE,
  which will try to re-download the Try-On weights and fill the disk again.
  Until the runner service runs as Administrator (services.msc → Log On),
  push with `[skip ci]` and restart Try-On by hand. Disk is at 3.9 GB free;
  the Administrator HF cache holds ~50 GB incl. unused SDXL base/inpaint,
  SmolLM, InstantID. Rotate the Administrator RDP password.

## 2026-11-13

### Web app hardened for web-only delivery (branch `webapp-typescript`)

- Decision stands: product ships as the web app; Flutter app on hold.
- `webapp/` fixes: every backend URL now follows the page scheme (an HTTPS
  deployment would otherwise have mixed-content-blocked the sockets and
  fetches silently); unused Tone.js CDN script removed; three.js + MediaPipe
  load on first use instead of at boot (initial JS 665 kB → 76 kB); canvas
  blur degrades gracefully on Safari < 18; camera start gives real messages
  (permission denied / no camera / insecure page) and calls play() for iOS;
  PWA manifest + icons + mobile meta; build target includes iOS 14 Safari;
  vite 7 + playwright 1.56 (npm audit clean).
- Live Try-On: fit readout now follows the session (socket open/close) rather
  than a button click; 66 ms floor on frame sends (reply-driven loop would
  otherwise spin against a fast backend).
- Smoke test (`npm run smoke`, 50 checks) drives the WHOLE Live Try-On loop
  with a fake camera and a mocked `/ws/tryon`: camera, garment gating, init
  payload, frame pacing, result drawn, garment switch re-init, readout, stop.
  Also checks phone-width layout and that heavy chunks are not fetched at boot.
  `.github/workflows/webapp.yml` runs build + smoke on every push to webapp/.
- Still needs: a real-server pass by hand (box must be up), then merge.

---

## 2026-10-27

### GPU box - why deploys never actually restarted anything

- Box came back on a new public IP `52.66.250.134` (still no Elastic IP).
  Only 3389 answers from outside until the backends are started; 8000/7860
  closed means nothing is listening, not a firewall problem.
- The Actions runner service
  (`actions.runner.indwar7-Mirror_Fit.EC2AMAZ-28LTJ0Q`, startup Auto) runs as
  **NT AUTHORITY\NETWORK SERVICE**, not Administrator. Three consequences,
  all confirmed with the new `diag.yml` workflow:
  1. `C:\logs\*.log` are owned by Administrator with `Users:RX`. Start-Process
     still hands the child a PID; it dies on its first write; the stale log
     from the previous boot sits there looking like output. The "4 of 4
     listening in 27s" deploys on 2026-09-09 were only ever seeing backends
     started by hand over RDP (NETWORK SERVICE cannot kill Administrator's
     processes either, so `-Force` was a silent no-op).
  2. That account has no Hugging Face cache (profile is
     `C:\Windows\ServiceProfiles\NetworkService`), so even with writable logs
     try-on would re-download every model on first start.
  3. The runner kills every process still carrying `RUNNER_TRACKING_ID` when
     the job ends ("Cleaning up orphan processes").
- `start_all.ps1` now: clears the tracking id before spawning; test-opens both
  log files first and prints `FAIL` with the fix; `-PassThru` plus a `DIED`
  line with exit code and fresh log tail if the process exits within 3s.
  `deploy.yml` has a verify step that lists real processes, listening ports
  and `C:\logs` timestamps. `diag.yml` (workflow_dispatch, fixed read-only
  commands) shows the account, processes, ports, GPU, log tails and a python
  probe.
- **Durable fix, needs RDP + elevation:** make the runner service log on as
  Administrator. `services.msc` -> "GitHub Actions Runner (indwar7-Mirror_Fit.
  EC2AMAZ-28LTJ0Q)" -> Properties -> Log On -> This account -> Administrator +
  password -> OK -> restart the service. The dialog grants "Log on as a
  service" itself; `sc.exe config obj=` does not. After that a push to main
  really restarts the backends and shares Administrator's HF cache.
- From the Mac: `gh workflow run deploy.yml --ref main`,
  `gh workflow run diag.yml --ref main`, then `gh run watch`.

### Open items

- [ ] Switch the runner service to the Administrator account (above).
- [ ] Rotate the Administrator RDP password (it was pasted into a chat).
- [ ] Everything still open from 2026-09-03 below.

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
  `ai-twin` — they're on the same GPU), public IP `13.126.77.241` (changes on every instance stop/start — Elastic IP not attached).
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
     add `http://13.126.77.241:7860,http://13.126.77.241:8000` → Relaunch.
  3. Open `http://13.126.77.241:7860/` (not `/demo/index.html` — the static
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
- [ ] Cloud security group check on the GPU box (13.126.77.241) — confirm
      which ports are actually meant to be open to the internet long-term;
      currently 7860/8000 are open for testing.
