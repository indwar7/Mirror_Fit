import type { FaceLandmarker } from '@mediapipe/tasks-vision';

// The MediaPipe runtime is ~130 kB of JS plus a ~10 MB WASM + model download.
// Only two hidden tabs use it, so the package is imported the first time a
// tracker is asked for rather than on every page load.
type Vision = typeof import('@mediapipe/tasks-vision');
type Fileset = Awaited<ReturnType<Vision['FilesetResolver']['forVisionTasks']>>;

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

let visionMod: Promise<Vision> | null = null;
let fileset: Promise<Fileset> | null = null;

function vision(): Promise<Vision> {
  if (!visionMod) visionMod = import('@mediapipe/tasks-vision');
  return visionMod;
}

export async function visionFileset(): Promise<Fileset> {
  const v = await vision();
  if (!fileset) fileset = v.FilesetResolver.forVisionTasks(WASM_BASE);
  return fileset;
}

// Every consumer gets its OWN FaceLandmarker: a VIDEO-mode instance demands
// strictly increasing timestamps, so two render loops feeding one instance
// would starve each other. Only the WASM + model download is shared.
export async function createFaceLandmarker(opts: { blendshapes?: boolean; matrices?: boolean } = {}): Promise<FaceLandmarker> {
  const v = await vision();
  const fs = await visionFileset();
  return v.FaceLandmarker.createFromOptions(fs, {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: !!opts.blendshapes,
    outputFacialTransformationMatrixes: !!opts.matrices,
  });
}
