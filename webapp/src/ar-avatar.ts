// AR AVATAR engine — MediaPipe FaceLandmarker (468 pts + transformation
// matrix) + Three.js overlay. Browser-side only, no backend roundtrip.
//
// Pipeline per frame:
//   1. webcam → <video>
//   2. FaceLandmarker.detectForVideo(video)
//   3. Read facialTransformationMatrixes[0] (4x4 pose)
//   4. Apply to the Three.js avatar mesh
//   5. Render WebGL → canvas (mirrored to match the selfie cam)
import * as THREE from 'three';
import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import { byId, maybe } from './dom';
import { createFaceLandmarker } from './mediapipe';

import { arSelection, setArSelectListener, type ArPreset } from './ar-presets';

// ── State ────────────────────────────────────────────────────────────────
let arVideo: HTMLVideoElement | null = null;
let arCanvas: HTMLCanvasElement | null = null;
let arRenderer: THREE.WebGLRenderer | null = null;
let arScene: THREE.Scene | null = null;
let arCamera: THREE.PerspectiveCamera | null = null;
let arLandmarker: FaceLandmarker | null = null;
let arAvatar: THREE.Group | null = null;
let arRunning = false;
let arRafId: number | null = null;
let arVideoStream: MediaStream | null = null;
let arLastVideoTime = -1;

// ── Camera ───────────────────────────────────────────────────────────────
export async function startArCamera(): Promise<void> {
  try {
    arVideoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: 'user' },
      audio: false,
    });
    const video = byId<HTMLVideoElement>('arVideo');
    arVideo = video;
    video.srcObject = arVideoStream;
    await new Promise<void>((r) => { video.onloadedmetadata = () => r(); });
    await video.play();
    byId('arCamBtn').textContent = 'Camera on';
    byId<HTMLButtonElement>('arCamBtn').disabled = true;
    byId('arCamStatus').innerHTML = '<span style="color:var(--green)">Camera ready — pick an avatar</span>';
    byId<HTMLButtonElement>('arStartBtn').disabled = false;
  } catch (e) {
    byId('arCamStatus').innerHTML = '<span style="color:var(--red)">Camera error: ' + (e as Error).message + '</span>';
  }
}

// ── MediaPipe FaceLandmarker init (once) ─────────────────────────────────
async function initArLandmarker(): Promise<FaceLandmarker> {
  if (arLandmarker) return arLandmarker;
  arLandmarker = await createFaceLandmarker({ blendshapes: true, matrices: true });
  return arLandmarker;
}

// ── Three.js scene init (once) ───────────────────────────────────────────
function initArScene(): void {
  if (arRenderer) return;
  arCanvas   = byId<HTMLCanvasElement>('arCanvas');
  arRenderer = new THREE.WebGLRenderer({ canvas: arCanvas, alpha: true, antialias: true });
  arRenderer.setPixelRatio(window.devicePixelRatio);

  arScene  = new THREE.Scene();
  // FOV ~63° matches a typical 720p webcam. Camera sits at world origin
  // looking down -Z, matching MediaPipe's transformation-matrix convention.
  arCamera = new THREE.PerspectiveCamera(63, 1, 0.01, 100);
  arCamera.position.set(0, 0, 0);
  arCamera.lookAt(0, 0, -1);

  arScene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const light = new THREE.DirectionalLight(0xffffff, 0.6);
  light.position.set(0.4, 0.6, 0.5);
  arScene.add(light);

  resizeArCanvas();
  window.addEventListener('resize', resizeArCanvas);
}

function resizeArCanvas(): void {
  if (!arCanvas || !arVideo || !arRenderer || !arCamera) return;
  // Match the canvas to the video's intrinsic size so the overlay aligns 1:1.
  const w = arVideo.videoWidth  || arCanvas.clientWidth  || 1280;
  const h = arVideo.videoHeight || arCanvas.clientHeight || 720;
  arRenderer.setSize(w, h, false);
  arCamera.aspect = w / h;
  arCamera.updateProjectionMatrix();
}

// ── Procedural head builder ──────────────────────────────────────────────
// Sized in metres so the avatar matches a real face (~16 cm wide).
function buildArHead(preset: ArPreset): THREE.Group {
  const g = new THREE.Group();

  const skinMat  = new THREE.MeshStandardMaterial({ color: preset.skin, roughness: 0.6, metalness: 0.0 });
  const hairMat  = new THREE.MeshStandardMaterial({ color: preset.hair, roughness: 0.8, metalness: 0.0 });
  const eyeMat   = new THREE.MeshStandardMaterial({ color: preset.eye,  roughness: 0.3, metalness: 0.0 });
  const whiteMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.2 });
  const lipMat   = new THREE.MeshStandardMaterial({ color: 0xb84a4a, roughness: 0.4 });
  const hatMat   = new THREE.MeshStandardMaterial({ color: preset.hat,  roughness: 0.7, metalness: 0.0 });

  // Head — slightly egg-shaped for cartoon feel
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.085, 48, 48), skinMat);
  head.scale.set(1.0, 1.08, 0.95);
  g.add(head);

  // Hair cap — half-sphere on top of head
  const hairCap = new THREE.Mesh(
    new THREE.SphereGeometry(0.088, 48, 48, 0, Math.PI * 2, 0, Math.PI * 0.55), hairMat);
  hairCap.position.y = 0.005;
  g.add(hairCap);

  // Eyes — whites + pupils
  const eyeWhiteGeo = new THREE.SphereGeometry(0.020, 24, 24);
  const lEyeWhite = new THREE.Mesh(eyeWhiteGeo, whiteMat);
  lEyeWhite.position.set(-0.025, 0.015, 0.072);
  const rEyeWhite = new THREE.Mesh(eyeWhiteGeo, whiteMat);
  rEyeWhite.position.set( 0.025, 0.015, 0.072);
  g.add(lEyeWhite, rEyeWhite);

  const pupilGeo = new THREE.SphereGeometry(0.011, 16, 16);
  const lPupil = new THREE.Mesh(pupilGeo, eyeMat);
  lPupil.position.set(-0.025, 0.015, 0.087);
  const rPupil = new THREE.Mesh(pupilGeo, eyeMat);
  rPupil.position.set( 0.025, 0.015, 0.087);
  g.add(lPupil, rPupil);

  // Nose — tiny pyramid
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.012, 0.022, 4), skinMat);
  nose.rotation.x = Math.PI;
  nose.position.set(0, -0.005, 0.088);
  g.add(nose);

  // Mouth — small torus arc
  const mouth = new THREE.Mesh(new THREE.TorusGeometry(0.022, 0.0045, 16, 32, Math.PI), lipMat);
  mouth.rotation.set(Math.PI, 0, 0);
  mouth.position.set(0, -0.038, 0.082);
  g.add(mouth);

  // Hat — top sphere section + brim
  const hatTop = new THREE.Mesh(
    new THREE.SphereGeometry(0.08, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.45), hatMat);
  hatTop.position.y = 0.060;
  hatTop.scale.set(1.05, 1.4, 1.05);
  g.add(hatTop);

  const brim = new THREE.Mesh(new THREE.TorusGeometry(0.090, 0.012, 16, 48), hatMat);
  brim.rotation.x = Math.PI / 2;
  brim.position.y = 0.060;
  g.add(brim);

  return g;
}

function loadArAvatar(preset: ArPreset): void {
  if (!arScene) return;
  if (arAvatar) {
    arScene.remove(arAvatar);
    arAvatar.traverse((o) => { const m = o as THREE.Mesh; if (m.geometry) m.geometry.dispose(); });
  }
  arAvatar = buildArHead(preset);
  // matrixAutoUpdate off so the transform is driven from MediaPipe directly.
  arAvatar.matrixAutoUpdate = false;
  // Seed it 45 cm in front of the camera. Without this the matrix stays
  // IDENTITY until the first detection, which puts the head at the world
  // origin — exactly where the camera is — and renders nothing.
  arAvatar.matrix.makeTranslation(0, 0, -0.45);
  arAvatar.matrixWorldNeedsUpdate = true;
  arScene.add(arAvatar);
}

// MediaPipe hands back 16 floats; which order it writes has differed between
// releases. Decide it once, from the data: a real face has to land IN FRONT of
// the camera at a human distance. Held for the session.
let arMatrixOrder: 'column' | 'row' | null = null;

function arPlausible(m: THREE.Matrix4): boolean {
  const p = new THREE.Vector3().setFromMatrixPosition(m);
  const d = p.length();
  return p.z < 0 && d > 0.15 && d < 1.5;
}

function arApplyMatrix(obj: THREE.Object3D, data: ArrayLike<number>): THREE.Vector3 {
  const asColumn = new THREE.Matrix4().fromArray(data);
  if (arMatrixOrder === null) {
    const asRow = new THREE.Matrix4().fromArray(data).transpose();
    arMatrixOrder = arPlausible(asColumn) ? 'column'
                  : arPlausible(asRow)    ? 'row'
                  : 'column';               // neither looked sane — keep the documented one
  }
  const m = arMatrixOrder === 'row' ? asColumn.clone().transpose() : asColumn;
  obj.matrix.copy(m);
  obj.matrixWorldNeedsUpdate = true;
  return new THREE.Vector3().setFromMatrixPosition(m);
}

// ── Per-frame: detect face → drive avatar transform → render ─────────────
let arSeenFace = false, arLastReport = 0, arDetectErr = '';

function arFrame(): void {
  if (!arRunning || !arVideo || !arRenderer || !arScene || !arCamera) return;
  let pos: THREE.Vector3 | null = null;
  if (arVideo.readyState >= 2 && arVideo.currentTime !== arLastVideoTime) {
    arLastVideoTime = arVideo.currentTime;
    try {
      const result = arLandmarker!.detectForVideo(arVideo, performance.now());
      const mtx = result.facialTransformationMatrixes?.[0]?.data;
      if (mtx && arAvatar) {
        pos = arApplyMatrix(arAvatar, mtx);
        arSeenFace = true;
        arDetectErr = '';
      }
    } catch (e) {
      // A throw here used to kill the rAF loop outright. Keep rendering and
      // report it instead.
      arDetectErr = (e as Error).message || String(e);
    }
  }
  arRenderer.render(arScene, arCamera);

  // Say which half is working: "AR running" on its own cannot distinguish
  // "detection is dead" from "rendering is dead".
  const now = performance.now();
  if (now - arLastReport > 500) {
    arLastReport = now;
    const el = maybe('arStatus');
    if (el) {
      if (arDetectErr) {
        el.innerHTML = '<span style="color:var(--red)">Face detection error: ' + arDetectErr + '</span>';
      } else if (pos) {
        el.innerHTML = '<span style="color:var(--green)">Tracking — ' +
          Math.round(pos.length() * 100) + ' cm away (' + arMatrixOrder + '-major)</span>';
      } else if (arSeenFace) {
        el.innerHTML = '<span style="color:var(--accent-hot)">Face lost — move back into frame</span>';
      } else {
        el.innerHTML = '<span style="color:var(--accent-hot)">Rendering, but no ' +
          'face detected yet. If the head is visible, detection is the problem.</span>';
      }
    }
  }

  arRafId = requestAnimationFrame(arFrame);
}

// Hot-swap the head if the user picks another preset mid-session.
setArSelectListener((av) => { if (arRunning) loadArAvatar(av); });

// ── Start / Stop ─────────────────────────────────────────────────────────
export async function toggleArAvatar(): Promise<void> {
  if (arRunning) {
    arRunning = false;
    if (arRafId) cancelAnimationFrame(arRafId);
    arRafId = null;
    // Re-decide the matrix order and the "seen a face yet" state next run.
    arMatrixOrder = null;
    arSeenFace = false;
    arDetectErr = '';
    const btn = byId('arStartBtn');
    btn.textContent = 'Start AR';
    btn.className = 'btn btn-primary';
    byId('arStatusText').textContent = 'Stopped';
    byId('arDot').classList.remove('on');
    return;
  }
  if (!arVideoStream) {
    byId('arStatus').innerHTML = '<span style="color:var(--red)">Start the camera first</span>';
    return;
  }
  byId('arStatus').innerHTML = '<span style="color:var(--accent-hot)">Loading MediaPipe (one-time ~10 MB)…</span>';
  try {
    initArScene();
    await initArLandmarker();
    loadArAvatar(arSelection.preset);
    resizeArCanvas();
    arRunning = true;
    arLastVideoTime = -1;
    const btn = byId('arStartBtn');
    btn.textContent = 'Stop';
    btn.className = 'btn btn-stop';
    byId('arStatusText').textContent = 'AR running';
    byId('arDot').classList.add('on');
    byId('arStatus').innerHTML = '<span style="color:var(--green)">AR avatar tracking your face</span>';
    arFrame();
  } catch (e) {
    byId('arStatus').innerHTML = '<span style="color:var(--red)">Init failed: ' + (e as Error).message + '</span>';
  }
}
