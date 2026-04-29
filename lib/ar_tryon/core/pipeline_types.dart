// Core types used across the entire AR try-on pipeline.
//
// PIPELINE ARCHITECTURE (text diagram):
//
// ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
// │   Camera     │───▶│  Pose        │───▶│  Body Dimension  │
// │   Stream     │    │  Detection   │    │  Estimation      │
// │  (30 FPS)    │    │  (MediaPipe) │    │  (Landmarks)     │
// └─────────────┘    └──────────────┘    └──────────────────┘
//        │                  │                      │
//        ▼                  ▼                      ▼
// ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
// │   Body       │    │  Garment     │    │  3D Model        │
// │   Segmentati │    │  Fitting     │◀──│  Loader (.glb)   │
// │   on (TFLite)│    │  Algorithm   │    │                  │
// └─────────────┘    └──────────────┘    └──────────────────┘
//        │                  │                      │
//        ▼                  ▼                      ▼
// ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
// │   Occlusion  │───▶│  Composite   │◀──│  3D Renderer     │
// │   Handler    │    │  Renderer    │    │  (Unity/OpenGL)  │
// └─────────────┘    └──────────────┘    └──────────────────┘
//                          │
//                          ▼
//                   ┌──────────────┐
//                   │   Display    │
//                   │   Output     │
//                   └──────────────┘
//
// DATA FLOW:
//   CameraFrame → PoseResult → BodyDimensions → GarmentTransform
//       ↓              ↓                              ↓
//   SegmentationMask  Landmarks                  ScaledModel
//       ↓              ↓                              ↓
//   OcclusionMask → CompositeRenderer → Final Output Frame
//
// ALL PROCESSING ON-DEVICE:
//   - Pose detection: MediaPipe (google_mlkit_pose_detection)
//   - Segmentation:   TFLite selfie segmentation model
//   - 3D Rendering:   flutter_unity_widget or model_viewer_plus
//   - Fitting math:   Pure Dart computation
//   - No cloud calls required

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

/// Represents a single 3D landmark point from pose detection.
class Landmark {
  final double x;
  final double y;
  final double z;
  final double confidence;
  final LandmarkType type;

  const Landmark({
    required this.x,
    required this.y,
    required this.z,
    required this.confidence,
    required this.type,
  });

  Offset get offset => Offset(x, y);

  bool get isVisible => confidence > 0.5;

  double distanceTo(Landmark other) {
    final dx = x - other.x;
    final dy = y - other.y;
    final dz = z - other.z;
    return math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  double distanceTo2D(Landmark other) {
    final dx = x - other.x;
    final dy = y - other.y;
    return math.sqrt(dx * dx + dy * dy);
  }
}

/// MediaPipe Pose landmark types (33 landmarks).
enum LandmarkType {
  nose,                   // 0
  leftEyeInner,          // 1
  leftEye,               // 2
  leftEyeOuter,          // 3
  rightEyeInner,         // 4
  rightEye,              // 5
  rightEyeOuter,         // 6
  leftEar,               // 7
  rightEar,              // 8
  mouthLeft,             // 9
  mouthRight,            // 10
  leftShoulder,          // 11
  rightShoulder,         // 12
  leftElbow,             // 13
  rightElbow,            // 14
  leftWrist,             // 15
  rightWrist,            // 16
  leftPinky,             // 17
  rightPinky,            // 18
  leftIndex,             // 19
  rightIndex,            // 20
  leftThumb,             // 21
  rightThumb,            // 22
  leftHip,               // 23
  rightHip,              // 24
  leftKnee,              // 25
  rightKnee,             // 26
  leftAnkle,             // 27
  rightAnkle,            // 28
  leftHeel,              // 29
  rightHeel,             // 30
  leftFootIndex,         // 31
  rightFootIndex,        // 32
}

/// Result from a single frame of pose detection.
class PoseResult {
  final List<Landmark> landmarks;
  final int frameWidth;
  final int frameHeight;
  final int timestampMs;
  // O(1) landmark lookup — built once per frame, called N times per frame.
  // Replaces O(n) firstWhere + exception on every miss.
  final Map<LandmarkType, Landmark> _map;

  PoseResult({
    required this.landmarks,
    required this.frameWidth,
    required this.frameHeight,
    required this.timestampMs,
  }) : _map = { for (final l in landmarks) l.type: l };

  Landmark? get(LandmarkType type) => _map[type];

  Landmark? get leftShoulder => get(LandmarkType.leftShoulder);
  Landmark? get rightShoulder => get(LandmarkType.rightShoulder);
  Landmark? get leftHip => get(LandmarkType.leftHip);
  Landmark? get rightHip => get(LandmarkType.rightHip);
  Landmark? get leftElbow => get(LandmarkType.leftElbow);
  Landmark? get rightElbow => get(LandmarkType.rightElbow);
  Landmark? get leftWrist => get(LandmarkType.leftWrist);
  Landmark? get rightWrist => get(LandmarkType.rightWrist);
  Landmark? get leftKnee => get(LandmarkType.leftKnee);
  Landmark? get rightKnee => get(LandmarkType.rightKnee);
  Landmark? get nose => get(LandmarkType.nose);

  /// Minimum check: both shoulders visible with sufficient confidence.
  bool get isValid =>
      (leftShoulder?.isVisible ?? false) &&
      (rightShoulder?.isVisible ?? false);

  /// Full body check: all 4 major joints visible.
  bool get hasFullBody =>
      leftShoulder != null &&
      rightShoulder != null &&
      leftHip != null &&
      rightHip != null;

  /// Midpoint between shoulders.
  Offset? get shoulderCenter {
    final ls = leftShoulder;
    final rs = rightShoulder;
    if (ls == null || rs == null) return null;
    return Offset((ls.x + rs.x) / 2, (ls.y + rs.y) / 2);
  }

  /// Midpoint between hips.
  Offset? get hipCenter {
    final lh = leftHip;
    final rh = rightHip;
    if (lh == null || rh == null) return null;
    return Offset((lh.x + rh.x) / 2, (lh.y + rh.y) / 2);
  }
}

/// Measured body dimensions derived from pose landmarks.
class BodyDimensions {
  final double shoulderWidthPx;
  final double torsoHeightPx;
  final double leftArmLengthPx;
  final double rightArmLengthPx;
  final double hipWidthPx;
  final double bodyAngleDegrees;
  final Offset torsoCenter;

  const BodyDimensions({
    required this.shoulderWidthPx,
    required this.torsoHeightPx,
    required this.leftArmLengthPx,
    required this.rightArmLengthPx,
    required this.hipWidthPx,
    required this.bodyAngleDegrees,
    required this.torsoCenter,
  });

  double get avgArmLengthPx => (leftArmLengthPx + rightArmLengthPx) / 2;
}

/// Transform data for positioning a garment on the body.
class GarmentTransform {
  final double x;
  final double y;
  final double scaleX;
  final double scaleY;
  final double rotationDegrees;
  final double opacity;
  /// Where the shoulder seam sits in the garment image (fraction from top).
  /// Used by mesh deformer to know where torso starts.
  final double shoulderLineInImage;

  const GarmentTransform({
    required this.x,
    required this.y,
    required this.scaleX,
    required this.scaleY,
    required this.rotationDegrees,
    this.opacity = 1.0,
    this.shoulderLineInImage = 0.15,
  });

  static const identity = GarmentTransform(
    x: 0, y: 0, scaleX: 1, scaleY: 1, rotationDegrees: 0,
  );
}

/// Binary segmentation mask (kept for compatibility with segmenter/occlusion modules).
class SegmentationMask {
  final Uint8List data;
  final int width;
  final int height;
  final int channels;
  const SegmentationMask({required this.data, required this.width, required this.height, this.channels = 1});
  int valueAt(int x, int y) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0;
    return data[y * width * channels + x * channels];
  }
  bool isForeground(int x, int y, {int threshold = 128}) => valueAt(x, y) > threshold;
}

enum BodyPart { head, torso, leftArm, rightArm, leftLeg, rightLeg }

/// A compositing layer for the occlusion pipeline.
class RenderLayer {
  final BodyPart? bodyPart;
  final bool isGarment;
  final double depth;
  final Rect bounds;

  const RenderLayer({
    this.bodyPart,
    required this.isGarment,
    required this.depth,
    required this.bounds,
  });
}

/// Pipeline frame result — only what the renderer needs.
class PipelineFrame {
  final PoseResult pose;
  final BodyDimensions dimensions;
  final GarmentTransform? garmentTransform;
  final int processingTimeMs;
  /// Optional pixel-accurate body segmentation mask (256×256).
  /// Null when model not loaded or on frames where segmentation was skipped.
  final SegmentationMask? segmentationMask;
  /// How many frames ago the segmentation mask was computed.
  /// 0 = produced this frame, 1 = one frame stale, 999 = never produced.
  /// Renderer skips Layer A (mask-based erase) when this value is > 2 to
  /// avoid artifacts from a frozen mask over a moving arm.
  final int maskAgeFrames;

  const PipelineFrame({
    required this.pose,
    required this.dimensions,
    this.garmentTransform,
    required this.processingTimeMs,
    this.segmentationMask,
    this.maskAgeFrames = 999,
  });
}
