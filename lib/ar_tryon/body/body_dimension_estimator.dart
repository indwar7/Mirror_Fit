import 'dart:math' as math;
import 'dart:ui';
import '../core/pipeline_types.dart';

/// Estimates real-world body dimensions from 2D pose landmarks.
///
/// Uses geometric relationships between landmarks to compute:
///   - Shoulder width (left shoulder ↔ right shoulder)
///   - Torso height (shoulder midpoint → hip midpoint)
///   - Arm lengths (shoulder → elbow → wrist)
///   - Hip width (left hip ↔ right hip)
///   - Body rotation angle (for 3D perspective correction)
///
/// These measurements are in pixel space and are used to scale/position
/// 3D clothing models relative to the detected body.
///
/// ACCURACY NOTE:
///   Pixel-based estimation is sufficient for overlay rendering.
///   For real-world measurements (cm), you'd need camera calibration
///   or a known reference object in frame.
class BodyDimensionEstimator {
  // Small buffer for stable measurements without lag.
  final List<BodyDimensions> _history = [];
  static const int _historySize = 3;

  /// Estimate body dimensions from pose landmarks.
  ///
  /// Works with just shoulders visible — estimates hips if missing.
  /// Returns null only if both shoulders are missing.
  BodyDimensions? estimate(PoseResult pose) {
    final ls = pose.leftShoulder;
    final rs = pose.rightShoulder;
    if (ls == null || rs == null) return null;

    // 1. Shoulder width (distance between shoulder landmarks)
    final shoulderWidth = _distance2D(ls, rs);

    // 2. Shoulder center
    final shoulderCenter = Offset(
      (ls.x + rs.x) / 2,
      (ls.y + rs.y) / 2,
    );

    // 3. Hip center — use real landmarks if available, otherwise estimate
    final lh = pose.leftHip;
    final rh = pose.rightHip;
    final Offset hipCenter;
    final double torsoHeight;
    final double hipWidth;

    if (lh != null && rh != null) {
      hipCenter = Offset((lh.x + rh.x) / 2, (lh.y + rh.y) / 2);
      torsoHeight = (hipCenter - shoulderCenter).distance;
      hipWidth = _distance2D(lh, rh);
    } else {
      // ESTIMATE: torso height ≈ 1.4x shoulder width (anatomical ratio)
      // Hip width ≈ 0.85x shoulder width
      torsoHeight = shoulderWidth * 1.4;
      hipCenter = Offset(shoulderCenter.dx, shoulderCenter.dy + torsoHeight);
      hipWidth = shoulderWidth * 0.85;
    }

    // 4. Arm lengths
    final leftArmLength = _computeArmLength(
      pose.leftShoulder, pose.leftElbow, pose.leftWrist,
    );
    final rightArmLength = _computeArmLength(
      pose.rightShoulder, pose.rightElbow, pose.rightWrist,
    );

    // 5. Body rotation angle
    final bodyAngle = _estimateBodyAngle(ls, rs);

    // 6. Torso center
    final torsoCenter = Offset(
      (shoulderCenter.dx + hipCenter.dx) / 2,
      (shoulderCenter.dy + hipCenter.dy) / 2,
    );

    final result = BodyDimensions(
      shoulderWidthPx: shoulderWidth,
      torsoHeightPx: torsoHeight,
      leftArmLengthPx: leftArmLength,
      rightArmLengthPx: rightArmLength,
      hipWidthPx: hipWidth,
      bodyAngleDegrees: bodyAngle,
      torsoCenter: torsoCenter,
    );

    // Temporal smoothing
    return _smooth(result);
  }

  /// Calculate scale factors for mapping a garment to the body.
  ///
  /// [garmentOrigWidth]  - Original garment image/model width in pixels
  /// [garmentOrigHeight] - Original garment image/model height in pixels
  ///
  /// Returns (scaleX, scaleY) to apply to the garment.
  ({double scaleX, double scaleY}) computeGarmentScale({
    required BodyDimensions body,
    required double garmentOrigWidth,
    required double garmentOrigHeight,
    double shoulderPaddingFactor = 1.2, // Add 20% for fabric overhang
  }) {
    // Scale width: garment should span shoulder width × padding
    final targetWidth = body.shoulderWidthPx * shoulderPaddingFactor;
    final scaleX = targetWidth / garmentOrigWidth;

    // Scale height: garment should span shoulder-to-hip
    final targetHeight = body.torsoHeightPx * 1.1; // 10% extra for bottom hem
    final scaleY = targetHeight / garmentOrigHeight;

    return (scaleX: scaleX, scaleY: scaleY);
  }

  /// Compute anchor point for garment placement.
  ///
  /// The anchor is the midpoint between shoulders, offset slightly
  /// upward to account for neckline.
  Offset computeGarmentAnchor(PoseResult pose) {
    final sc = pose.shoulderCenter;
    if (sc == null) return Offset.zero;

    // Offset upward by ~10% of torso height for neckline
    final hc = pose.hipCenter;
    if (hc == null) return sc;

    final torsoH = (hc - sc).distance;
    return Offset(sc.dx, sc.dy - torsoH * 0.05);
  }

  double _distance2D(Landmark a, Landmark b) {
    final dx = a.x - b.x;
    final dy = a.y - b.y;
    return math.sqrt(dx * dx + dy * dy);
  }

  double _computeArmLength(
    Landmark? shoulder,
    Landmark? elbow,
    Landmark? wrist,
  ) {
    double length = 0;
    if (shoulder != null && elbow != null) {
      length += _distance2D(shoulder, elbow);
    }
    if (elbow != null && wrist != null) {
      length += _distance2D(elbow, wrist);
    }
    return length;
  }

  /// Estimate body rotation from shoulder z-coordinates.
  ///
  /// When the body rotates, one shoulder comes closer to the camera
  /// (smaller z), and the other moves away (larger z).
  ///
  /// Returns angle in degrees:
  ///   0 = facing camera directly
  ///   positive = rotated right
  ///   negative = rotated left
  double _estimateBodyAngle(Landmark leftShoulder, Landmark rightShoulder) {
    // Use z-depth difference between shoulders
    final zDiff = leftShoulder.z - rightShoulder.z;

    // Also use x-position asymmetry as secondary signal
    final shoulderWidth = (leftShoulder.x - rightShoulder.x).abs();

    // If shoulder width appears compressed, body is more rotated
    // A rough conversion: z-diff in MediaPipe units → degrees
    // MediaPipe z is in roughly the same scale as x, relative to hip width
    final angle = math.atan2(zDiff, shoulderWidth) * (180 / math.pi);

    return angle.clamp(-45, 45);
  }

  /// Apply exponential moving average smoothing across frames.
  BodyDimensions _smooth(BodyDimensions current) {
    _history.add(current);
    if (_history.length > _historySize) {
      _history.removeAt(0);
    }

    if (_history.length < 2) return current;

    // Weighted average — recent frames have more weight
    double totalWeight = 0;
    double shoulderW = 0, torsoH = 0, leftArm = 0, rightArm = 0;
    double hipW = 0, angle = 0, cx = 0, cy = 0;

    for (int i = 0; i < _history.length; i++) {
      final weight = (i + 1).toDouble(); // Linear weight increase
      totalWeight += weight;

      final h = _history[i];
      shoulderW += h.shoulderWidthPx * weight;
      torsoH += h.torsoHeightPx * weight;
      leftArm += h.leftArmLengthPx * weight;
      rightArm += h.rightArmLengthPx * weight;
      hipW += h.hipWidthPx * weight;
      angle += h.bodyAngleDegrees * weight;
      cx += h.torsoCenter.dx * weight;
      cy += h.torsoCenter.dy * weight;
    }

    return BodyDimensions(
      shoulderWidthPx: shoulderW / totalWeight,
      torsoHeightPx: torsoH / totalWeight,
      leftArmLengthPx: leftArm / totalWeight,
      rightArmLengthPx: rightArm / totalWeight,
      hipWidthPx: hipW / totalWeight,
      bodyAngleDegrees: angle / totalWeight,
      torsoCenter: Offset(cx / totalWeight, cy / totalWeight),
    );
  }

  /// Reset smoothing history (e.g., when user changes).
  void reset() {
    _history.clear();
  }
}
