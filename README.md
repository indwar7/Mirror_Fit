# LUCY - Real-Time Virtual Try-On

A Flutter-based real-time virtual try-on mobile application. Users see 3D clothes on their body in real time through the camera — like a virtual trial room.

**Zero paid APIs. Fully on-device. Open-source stack.**

---

## System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│   Camera     │───>│  Pose        │───>│  Body Dimension  │
│   Stream     │    │  Detection   │    │  Estimation      │
│  (30 FPS)    │    │  (MediaPipe) │    │  (Landmarks)     │
└─────────────┘    └──────────────┘    └──────────────────┘
       │                  │                      │
       v                  v                      v
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│   Body       │    │  Garment     │    │  3D Model        │
│   Segment    │    │  Fitting     │<──│  Loader (.glb)   │
│   (TFLite)   │    │  Algorithm   │    │                  │
└─────────────┘    └──────────────┘    └──────────────────┘
       │                  │                      │
       v                  v                      v
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Occlusion   │───>│  Composite   │<──│  3D Renderer     │
│  Handler     │    │  Renderer    │    │  (Unity/OpenGL)  │
└─────────────┘    └──────────────┘    └──────────────────┘
                         │
                         v
                   ┌──────────────┐
                   │   Screen     │
                   │   Output     │
                   └──────────────┘
```

### Data Flow (per frame)

```
CameraFrame → PoseResult → BodyDimensions → GarmentTransform
    ↓              ↓                              ↓
SegmentationMask  Landmarks                  ScaledModel
    ↓              ↓                              ↓
OcclusionMask → CompositeRenderer → Final Output Frame
```

### Per-Frame Timing Target

| Stage                | Time         | Tool                  |
| -------------------- | ------------ | --------------------- |
| Camera capture       | ~0ms         | `camera` package      |
| Pose detection       | ~15ms        | MediaPipe via MLKit   |
| Body segmentation    | ~10ms        | TFLite + GPU delegate |
| Dimension estimation | ~1ms         | Pure Dart math        |
| Garment fitting      | ~1ms         | Pure Dart math        |
| Compositing          | ~2ms         | Flutter CustomPainter |
| **Total**            | **~25-30ms** | **= 30-40 FPS**       |

All processing is on-device. No cloud calls required.

---

## Project Structure

```
lib/
├── main.dart                          # App entry point
├── models/                            # Data models (ClothItem, StyleScore, etc.)
├── screens/                           # Original screens (Home, TryOn, Result, etc.)
├── services/                          # Original services (TryOnProvider, FreeTryOnService)
├── utils/                             # Theme, haptics
├── widgets/                           # Original widgets (ClothCard, PhotoUpload, etc.)
│
└── ar_tryon/                          # === NEW: Real-Time AR Pipeline ===
    ├── core/
    │   ├── pipeline_types.dart        # All shared types: Landmark, PoseResult,
    │   │                              #   BodyDimensions, GarmentTransform, etc.
    │   ├── tryon_pipeline.dart        # Main orchestrator — connects all stages
    │   └── performance_optimizer.dart # Adaptive quality, FPS monitoring
    ├── camera/
    │   └── camera_service.dart        # Camera init, frame streaming, YUV/BGRA conversion
    ├── pose/
    │   └── pose_detector_service.dart # MLKit pose detection (33 landmarks),
    │                                  #   temporal smoothing, frame dropping
    ├── segmentation/
    │   └── body_segmenter.dart        # TFLite selfie segmentation (256x256),
    │                                  #   GPU delegate, body part extraction
    ├── body/
    │   └── body_dimension_estimator.dart # Shoulder width, torso height, arm length,
    │                                    #   hip width, body angle from landmarks
    ├── fitting/
    │   └── garment_fitter.dart        # Garment-to-body mapping:
    │                                  #   Width = shoulder_dist x 1.2
    │                                  #   Height = shoulder→hip distance
    │                                  #   Rotation from body angle
    │                                  #   Perspective correction
    ├── occlusion/
    │   └── occlusion_handler.dart     # Arm-over-clothing detection via z-depth,
    │                                  #   occlusion mask generation, depth ordering
    ├── rendering/
    │   ├── garment_renderer.dart      # CustomPainter for 2D garment overlay + skeleton
    │   └── model_loader.dart          # .glb file loading + Blender export guide
    ├── unity_bridge/
    │   └── unity_bridge.dart          # Flutter↔Unity JSON message passing +
    │                                  #   complete C# GarmentController script
    ├── screens/
    │   ├── ar_tryon_screen.dart       # Main AR screen: camera + overlays + controls
    │   └── model_preview_screen.dart  # 3D model viewer (model_viewer_plus)
    └── widgets/
        ├── pose_skeleton_painter.dart # Skeleton visualization (joints + bones)
        ├── debug_overlay.dart         # FPS, processing time, body metrics
        └── garment_selector_bar.dart  # Bottom garment picker bar
```

---

## Core Pipeline Modules

### 1. Camera + Pose Detection

**Camera Service** (`camera/camera_service.dart`):

- Front camera at 480p (medium resolution for ML performance balance)
- Real-time frame streaming with FPS tracking
- YUV420 (Android) and BGRA8888 (iOS) format handling
- Camera switching (front/back)

**Pose Detector** (`pose/pose_detector_service.dart`):

- Google MLKit Pose Detection (MediaPipe under the hood)
- 33 body landmarks per frame
- Stream mode for real-time temporal tracking
- Frame dropping when pipeline is busy (prevents latency buildup)
- Exponential moving average smoothing (factor: 0.7)
- Key landmarks extracted: shoulders, hips, elbows, wrists, knees

### 2. Body Segmentation

**Body Segmenter** (`segmentation/body_segmenter.dart`):

- TensorFlow Lite with GPU delegate (3-5x faster than CPU)
- MediaPipe Selfie Segmentation model (256x256 input, ~300KB)
- Outputs probability mask: 0 = background, 255 = body
- Body part extraction using pose landmarks (torso, left/right arms)

**Model setup:**

```
# Download and place in assets/ml_models/
selfie_segmentation.tflite  (~300KB, fast)
# OR
deeplabv3_257_mv_gpu.tflite (~2.7MB, more accurate)
```

### 3. Body Dimension Estimation

**Dimension Estimator** (`body/body_dimension_estimator.dart`):

| Measurement    | Calculation                                              |
| -------------- | -------------------------------------------------------- |
| Shoulder width | `distance(leftShoulder, rightShoulder)`                  |
| Torso height   | `distance(shoulderCenter, hipCenter)`                    |
| Arm length     | `distance(shoulder→elbow) + distance(elbow→wrist)`       |
| Hip width      | `distance(leftHip, rightHip)`                            |
| Body angle     | `atan2(leftShoulder.z - rightShoulder.z, shoulderWidth)` |

- Temporal smoothing: weighted average over 5 frames
- All measurements in pixel space (sufficient for overlay rendering)

### 4. Clothing Fitting Algorithm

**Garment Fitter** (`fitting/garment_fitter.dart`):

```
FITTING FORMULA:

For tops:
  Width  = shoulder_distance × shoulder_padding (default 1.2)
  Height = torso_height × (1 + hem_extension)
  Anchor = shoulder midpoint (offset up 5% for neckline)

For bottoms:
  Width  = hip_width × 1.1
  Height = torso_height × 1.2
  Anchor = hip midpoint

For full body:
  Width  = shoulder_distance × shoulder_padding
  Height = torso_height × 2.2 (shoulder to knee)
  Anchor = shoulder midpoint

Rotation = body_angle clamped to ±25°
Perspective = scaleX × cos(body_angle) when |angle| > 5°
Smoothing = lerp(previous, current, 0.7) per frame
```

Supports three garment types: `GarmentType.top`, `GarmentType.bottom`, `GarmentType.fullBody`

### 5. Occlusion Handling

**Occlusion Handler** (`occlusion/occlusion_handler.dart`):

```
ALGORITHM:

1. Get arm z-depth from pose landmarks
2. Get torso z-depth (average of shoulders + hips)
3. If arm.z < torso.z - 0.05 → arm is in front of clothing
4. Build arm region from shoulder→elbow→wrist path (radius: 35px)
5. Cross-reference with segmentation mask (only body pixels)
6. Generate occlusion mask: 255 where arms should render over garment

RENDER ORDER (painter's algorithm):
  Camera feed (background, depth 2.0)
    → Garment overlay (middle, depth 1.0)
      → Occluding arms from camera (foreground, depth 0.5)
```

### 6. 3D Rendering

**Option A: model_viewer_plus** (simple, no native setup):

- Loads .glb/.gltf files directly
- Supports pan/zoom/rotate
- AR Quick Look (iOS) and Scene Viewer (Android)
- Good for static preview and basic overlay

**Option B: flutter_unity_widget** (advanced, requires Unity):

- Full 3D rendering with shaders
- Real-time transform updates via JSON messages
- Proper depth compositing and occlusion
- Unity C# script template included in `unity_bridge.dart`

**Message format (Flutter → Unity):**

```json
{
  "type": "update_transform",
  "model": "tshirt.glb",
  "transform": {
    "position": { "x": 0.5, "y": 0.3, "z": 0 },
    "rotation": { "x": 0, "y": 15, "z": 0 },
    "scale": { "x": 1.2, "y": 1.1, "z": 1.0 }
  }
}
```

### 7. Performance Optimization

**Adaptive Quality** (`core/performance_optimizer.dart`):

| FPS Range | Quality Level | Features                                 |
| --------- | ------------- | ---------------------------------------- |
| ≥ 30 FPS  | High          | Segmentation + Occlusion + Accurate pose |
| 20-30 FPS | Medium        | Segmentation + Base pose model           |
| 15-20 FPS | Low           | Pose only, 360p input                    |
| < 15 FPS  | Minimal       | Pose only, 240p input                    |

**Key strategies:**

- GPU delegate for TFLite (3-5x speedup)
- Frame dropping instead of queuing (prevents latency)
- 480p camera resolution (not 1080p)
- Frame rate limiter (cap at target FPS)
- EMA performance tracking over 30-frame window

---

## Blender Export Guide

### Model Preparation

1. **Target < 50,000 polygons** per garment
2. Use Decimate modifier to reduce poly count
3. Remove interior faces (not visible on mobile)
4. Merge vertices by distance (0.001m threshold)
5. Apply all modifiers before export

### Export Settings

```
File → Export → glTF 2.0 (.glb/.gltf)

Format:        glTF Binary (.glb)
Include:       ☑ Selected Objects only
Transform:     ☑ +Y Up
Mesh:          ☑ Apply Modifiers
               ☑ UVs
               ☑ Normals
               ☐ Tangents (skip)
Material:      ☑ Materials (PBR)
               Image: JPEG (or PNG if transparency)
Animation:     ☐ (skip unless cloth sim)
Compression:   ☑ Draco (60-80% size reduction)
```

### Polygon Budget

| Garment Type | Target Polys | File Size     |
| ------------ | ------------ | ------------- |
| T-Shirt      | 5K - 15K     | 200KB - 500KB |
| Button Shirt | 10K - 25K    | 300KB - 1MB   |
| Jacket       | 15K - 35K    | 500KB - 2MB   |
| Dress        | 15K - 40K    | 500KB - 2MB   |
| Full Suit    | 25K - 50K    | 1MB - 3MB     |

### Texture Optimization

- Max texture size: 1024x1024 (2048 only if critical)
- JPEG for opaque, PNG only for transparency
- Bake complex materials to simple PBR maps
- Maps needed: Base Color, Normal (optional), Roughness (optional)

---

## Dependencies

### Flutter Packages

| Package                       | Purpose                       |
| ----------------------------- | ----------------------------- |
| `camera`                      | Real-time camera stream       |
| `google_mlkit_pose_detection` | MediaPipe pose (33 landmarks) |
| `tflite_flutter`              | Body segmentation model       |
| `model_viewer_plus`           | 3D .glb model viewer          |
| `provider`                    | State management              |
| `flutter_unity_widget`        | Unity 3D renderer (optional)  |
| `ar_flutter_plugin`           | ARCore/ARKit (optional)       |

### Open-Source Tools

| Tool                             | Purpose                                 |
| -------------------------------- | --------------------------------------- |
| MediaPipe Pose                   | On-device pose detection (33 landmarks) |
| TensorFlow Lite                  | On-device body segmentation             |
| SMPL body model                  | 3D body mesh (reference architecture)   |
| Unity (via flutter_unity_widget) | 3D garment rendering                    |
| ARCore / ARKit                   | AR plane detection (optional)           |
| HR-VITON / CP-VTON+              | Cloth warping reference architecture    |

---

## Implementation Roadmap

### Week 1: Camera + Pose Detection + Skeleton Overlay

- Set up camera streaming at 30 FPS
- Integrate MLKit pose detection
- Draw skeleton overlay on camera feed
- Verify landmark extraction (shoulders, hips, arms)
- **Files:** `camera_service.dart`, `pose_detector_service.dart`, `pose_skeleton_painter.dart`

### Week 2: Body Segmentation + Dimension Estimation

- Integrate TFLite selfie segmentation model
- Generate body masks per frame
- Compute shoulder width, torso height, arm lengths
- Temporal smoothing for stable measurements
- **Files:** `body_segmenter.dart`, `body_dimension_estimator.dart`

### Week 3: Load 3D Model + Static Overlay

- Prepare .glb files from Blender (optimize for mobile)
- Load models via model_viewer_plus
- Static overlay positioning (no tracking yet)
- **Files:** `model_loader.dart`, `model_preview_screen.dart`

### Week 4: Real-Time Cloth Tracking + Scaling

- Map garment anchor points to body landmarks
- Scale garment to match detected body dimensions
- Rotation based on body orientation
- Per-frame transform updates
- **Files:** `garment_fitter.dart`, `garment_renderer.dart`

### Week 5: Unity Renderer Integration

- Set up Unity project with AR Foundation
- Export Unity as Android Library / iOS Framework
- Implement Flutter↔Unity message bridge
- Real-time 3D rendering with pose data
- **Files:** `unity_bridge.dart`, Unity C# scripts

### Week 6: Occlusion + Performance Tuning

- Arm-over-clothing detection via z-depth
- Segmentation mask-based occlusion rendering
- Adaptive quality based on FPS
- GPU delegate optimization
- **Files:** `occlusion_handler.dart`, `performance_optimizer.dart`

### Week 7: Polish + Device Testing

- Test on mid-range Android devices (target: 30 FPS)
- Test on iOS devices
- Edge cases: multiple people, partial visibility, fast movement
- Screenshot/recording functionality
- Final UI polish

---

## How Companies Do It

### Snap AR (Snapchat)

- DensePose body mesh + neural texture mapping
- Custom GPU shaders for cloth draping
- Runs lightweight models on-device via Snapdragon Neural Processing SDK
- Proprietary Lens Studio framework

### Zara / Nike

- Cloud-based approach: user uploads photo
- Server runs HR-VITON/DensePose + diffusion model
- Returns composited image (not real-time)
- Focuses on photorealism over speed

### Lucy (Decart AI)

- "World simulator" suggests diffusion-based approach
- Likely uses SMPL body model fitting from pose
- Generates 3D body mesh → wraps cloth texture via UV mapping
- Neural rendering for photorealistic output
- Runs on cloud GPU for quality

### The Full Stack (Research Grade)

```
SMPL Body Model (parametric 3D body from pose params)
    ↓
DensePose (dense UV coordinate mapping of body surface)
    ↓
Cloth Simulation (physics-based or learned deformation)
    ↓
Neural Radiance Fields / Diffusion Rendering
    ↓
Photorealistic Output
```

This is the gold standard but requires cloud GPU. Our approach trades photorealism for real-time on-device performance using geometric overlay + segmentation-based occlusion.

---

## Quick Start

```bash
# 1. Install dependencies
flutter pub get

# 2. Download segmentation model
# Place selfie_segmenter.tflite in assets/ml_models/

# 3. Add your Blender .glb models
# Place in assets/models/

# 4. Run on device (not emulator — needs camera)
flutter run

# 5. Tap "Real-Time AR Try-On" on home screen
```

---

## Platform Configuration

### Android

- Camera permission in `AndroidManifest.xml`
- Min SDK: 21 (for MLKit)
- Recommended: `minSdkVersion 24` for best TFLite GPU support

### iOS

- `NSCameraUsageDescription` in `Info.plist`
- Min iOS: 12.0 (for MLKit)
- Metal GPU delegate enabled for TFLite

---

## Tech Stack

- **Framework:** Flutter (Android + iOS)
- **Language:** Dart
- **State Management:** Provider
- **Pose Detection:** Google MLKit (MediaPipe)
- **Segmentation:** TensorFlow Lite
- **3D Rendering:** model_viewer_plus / Unity
- **Budget:** $0 (all open-source)
