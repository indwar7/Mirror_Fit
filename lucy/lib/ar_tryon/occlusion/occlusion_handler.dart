import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';
import '../core/pipeline_types.dart';

/// Handles occlusion — when body parts (arms) appear in front of clothing.
///
/// OCCLUSION ALGORITHM:
///
/// The core challenge: When a user's arm crosses in front of their torso,
/// the arm should appear ON TOP of the virtual garment, not behind it.
///
/// APPROACH (Segmentation-Mask Based):
///
/// 1. Get full-body segmentation mask from TFLite model
/// 2. Get arm regions from pose landmarks (shoulder → elbow → wrist path)
/// 3. Compare arm z-depth vs torso z-depth from pose landmarks
/// 4. If arm.z < torso.z (arm closer to camera):
///    → arm pixels should be rendered OVER the garment
/// 5. Create occlusion mask: pixels where arm overlaps garment area
/// 6. During compositing:
///    a) Draw camera feed (background)
///    b) Draw garment overlay
///    c) Draw arm pixels FROM CAMERA FEED on top of garment
///       (using occlusion mask as alpha)
///
/// This gives the illusion that the arm is physically in front of
/// the clothing, which is critical for realism.
///
/// DEPTH ORDERING:
///   Camera feed (background)
///     → Garment (middle layer)
///       → Occluding body parts (foreground, from camera)
class OcclusionHandler {
  /// Compute the render layer order for compositing.
  ///
  /// Returns layers sorted back-to-front (painter's algorithm).
  List<RenderLayer> computeRenderOrder({
    required PoseResult pose,
    required SegmentationMask? segMask,
    required GarmentTransform garmentTransform,
    required Rect garmentBounds,
  }) {
    final layers = <RenderLayer>[];

    // Layer 1: Garment (always rendered)
    layers.add(RenderLayer(
      isGarment: true,
      depth: 1.0, // Middle depth
      bounds: garmentBounds,
    ));

    // Check arms for occlusion
    if (pose.isValid && segMask != null) {
      final leftArmOccluding = _isArmOccluding(
        shoulder: pose.leftShoulder!,
        elbow: pose.leftElbow,
        wrist: pose.leftWrist,
        torsoZ: _getTorsoZ(pose),
      );

      final rightArmOccluding = _isArmOccluding(
        shoulder: pose.rightShoulder!,
        elbow: pose.rightElbow,
        wrist: pose.rightWrist,
        torsoZ: _getTorsoZ(pose),
      );

      if (leftArmOccluding) {
        final armBounds = _getArmBounds(
          pose.leftShoulder!, pose.leftElbow, pose.leftWrist,
        );
        layers.add(RenderLayer(
          bodyPart: BodyPart.leftArm,
          isGarment: false,
          depth: 0.5, // Closer to camera than garment
          bounds: armBounds,
        ));
      }

      if (rightArmOccluding) {
        final armBounds = _getArmBounds(
          pose.rightShoulder!, pose.rightElbow, pose.rightWrist,
        );
        layers.add(RenderLayer(
          bodyPart: BodyPart.rightArm,
          isGarment: false,
          depth: 0.5,
          bounds: armBounds,
        ));
      }
    }

    // Sort back-to-front (higher depth = further away = drawn first)
    layers.sort((a, b) => (b.depth).compareTo(a.depth));
    return layers;
  }

  /// Generate an occlusion mask for compositing.
  ///
  /// Returns a mask where:
  ///   255 = this pixel should show camera feed (arm) over garment
  ///   0   = this pixel should show garment normally
  Uint8List generateOcclusionMask({
    required SegmentationMask segMask,
    required PoseResult pose,
    required int outputWidth,
    required int outputHeight,
  }) {
    final mask = Uint8List(outputWidth * outputHeight);
    if (!pose.isValid) return mask;

    final scaleX = segMask.width / pose.frameWidth;
    final scaleY = segMask.height / pose.frameHeight;
    final outScaleX = outputWidth / segMask.width;
    final outScaleY = outputHeight / segMask.height;

    final torsoZ = _getTorsoZ(pose);

    // Process left arm
    _processArmOcclusion(
      mask: mask,
      segMask: segMask,
      shoulder: pose.leftShoulder!,
      elbow: pose.leftElbow,
      wrist: pose.leftWrist,
      torsoZ: torsoZ,
      scaleX: scaleX,
      scaleY: scaleY,
      outScaleX: outScaleX,
      outScaleY: outScaleY,
      outputWidth: outputWidth,
    );

    // Process right arm
    _processArmOcclusion(
      mask: mask,
      segMask: segMask,
      shoulder: pose.rightShoulder!,
      elbow: pose.rightElbow,
      wrist: pose.rightWrist,
      torsoZ: torsoZ,
      scaleX: scaleX,
      scaleY: scaleY,
      outScaleX: outScaleX,
      outScaleY: outScaleY,
      outputWidth: outputWidth,
    );

    return mask;
  }

  /// Check if an arm is in front of the torso (occluding garment).
  bool _isArmOccluding({
    required Landmark shoulder,
    Landmark? elbow,
    Landmark? wrist,
    required double torsoZ,
  }) {
    // Check if any part of the arm is closer to camera than torso
    // In MediaPipe, smaller z = closer to camera

    if (elbow != null && elbow.isVisible) {
      if (elbow.z < torsoZ - 0.05) return true;
    }

    if (wrist != null && wrist.isVisible) {
      if (wrist.z < torsoZ - 0.05) return true;
    }

    // Also check if arm crosses the torso x-range
    // (arm is in front only if it overlaps the torso area)
    return false;
  }

  /// Get average z-depth of the torso.
  double _getTorsoZ(PoseResult pose) {
    double sum = 0;
    int count = 0;
    for (final lm in [
      pose.leftShoulder, pose.rightShoulder,
      pose.leftHip, pose.rightHip,
    ]) {
      if (lm != null && lm.isVisible) {
        sum += lm.z;
        count++;
      }
    }
    return count > 0 ? sum / count : 0;
  }

  /// Get bounding rect for an arm path.
  Rect _getArmBounds(Landmark shoulder, Landmark? elbow, Landmark? wrist) {
    double minX = shoulder.x, maxX = shoulder.x;
    double minY = shoulder.y, maxY = shoulder.y;

    if (elbow != null && elbow.isVisible) {
      minX = math.min(minX, elbow.x);
      maxX = math.max(maxX, elbow.x);
      minY = math.min(minY, elbow.y);
      maxY = math.max(maxY, elbow.y);
    }

    if (wrist != null && wrist.isVisible) {
      minX = math.min(minX, wrist.x);
      maxX = math.max(maxX, wrist.x);
      minY = math.min(minY, wrist.y);
      maxY = math.max(maxY, wrist.y);
    }

    // Add padding for arm thickness
    const padding = 40.0;
    return Rect.fromLTRB(
      minX - padding, minY - padding,
      maxX + padding, maxY + padding,
    );
  }

  /// Mark arm pixels in the occlusion mask.
  void _processArmOcclusion({
    required Uint8List mask,
    required SegmentationMask segMask,
    required Landmark shoulder,
    Landmark? elbow,
    Landmark? wrist,
    required double torsoZ,
    required double scaleX,
    required double scaleY,
    required double outScaleX,
    required double outScaleY,
    required int outputWidth,
  }) {
    if (!_isArmOccluding(
      shoulder: shoulder,
      elbow: elbow,
      wrist: wrist,
      torsoZ: torsoZ,
    )) { return; }

    // Build arm path as a thick line from shoulder → elbow → wrist
    final points = <Offset>[Offset(shoulder.x, shoulder.y)];
    if (elbow != null && elbow.isVisible) {
      points.add(Offset(elbow.x, elbow.y));
    }
    if (wrist != null && wrist.isVisible) {
      points.add(Offset(wrist.x, wrist.y));
    }

    // For each point on the arm path, mark nearby segmented pixels
    const armRadius = 35; // Pixel radius for arm thickness

    for (final pt in points) {
      final segX = (pt.dx * scaleX).toInt();
      final segY = (pt.dy * scaleY).toInt();

      for (int dy = -armRadius; dy <= armRadius; dy++) {
        for (int dx = -armRadius; dx <= armRadius; dx++) {
          if (dx * dx + dy * dy > armRadius * armRadius) continue;

          final sx = segX + dx;
          final sy = segY + dy;

          // Check if this pixel is body (from segmentation)
          if (segMask.isForeground(sx, sy, threshold: 100)) {
            // Map to output coordinates
            final ox = (sx * outScaleX).toInt();
            final oy = (sy * outScaleY).toInt();

            if (ox >= 0 && ox < outputWidth &&
                oy >= 0 && oy < mask.length ~/ outputWidth) {
              mask[oy * outputWidth + ox] = 255;
            }
          }
        }
      }
    }
  }
}
