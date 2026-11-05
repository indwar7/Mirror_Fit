import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// WASM runtime + the .task model from the official CDNs. ~10 MB total, fetched
// once per page and cached by the browser.
const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

type Fileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;
let fileset: Promise<Fileset> | null = null;

export function visionFileset(): Promise<Fileset> {
  if (!fileset) fileset = FilesetResolver.forVisionTasks(WASM_BASE);
  return fileset;
}

// Every consumer gets its OWN FaceLandmarker: a VIDEO-mode instance demands
// strictly increasing timestamps, so two render loops feeding one instance
// would starve each other. Only the WASM + model download is shared.
export async function createFaceLandmarker(opts: { blendshapes?: boolean; matrices?: boolean } = {}) {
  const vision = await visionFileset();
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: !!opts.blendshapes,
    outputFacialTransformationMatrixes: !!opts.matrices,
  });
}
