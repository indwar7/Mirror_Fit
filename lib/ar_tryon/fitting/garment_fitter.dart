import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/foundation.dart';
import '../core/pipeline_types.dart';
import '../body/body_dimension_estimator.dart';

enum GarmentType { top, bottom, fullBody }

class GarmentConfig {
  final String modelPath;
  final GarmentType type;
  final double originalWidth;
  final double originalHeight;

  /// How much wider than shoulders the garment should be (1.0 = exact match).
  /// Accounts for the fact that the garment image width includes the full sleeves,
  /// while the actual shoulder-to-shoulder body is only a fraction of that.
  /// jacket ≈ 1.7 (body is ~55% of image), flat tops ≈ 1.4 (body ~70% of image)
  final double shoulderPadding;

  /// Where the shoulder seam sits in the garment image, as a fraction from top.
  /// jacket: ~0.25 (collar above shoulders), flat tops: ~0.12 (minimal collar)
  final double shoulderLineInImage;

  final double hemExtension;
  final Offset anchorOffset;

  const GarmentConfig({
    required this.modelPath,
    required this.type,
    required this.originalWidth,
    required this.originalHeight,
    this.shoulderPadding = 1.4,
    this.shoulderLineInImage = 0.12,
    this.hemExtension = 0.0,
    this.anchorOffset = Offset.zero,
  });
}

/// Positions garment on the torso using shoulder landmarks as the primary anchor.
///
/// SNAPCHAT-GRADE APPROACH:
///   1. Scale garment width to detected shoulder width × padding factor.
///      NO minimum clamp — scale naturally with body size at any distance.
///   2. When hips visible: use torso height for perspective-correct jacket height.
///      When hips not visible: derive height from aspect ratio.
///   3. Anchor so that the shoulder seam in the garment aligns with detected shoulders.
///   4. Rotate with shoulder tilt for natural lean/tilt tracking.
///   5. Minimal extra smoothing — landmarks already One-Euro filtered.
class GarmentFitter {
  final BodyDimensionEstimator _dimensionEstimator;
  GarmentTransform? _lastTransform;

  /// Smooth alpha for garment transform tracking.
  /// 0.20 = 20% toward new value per frame → ~15-frame settle at 30fps.
  /// Landmarks are already One-Euro filtered upstream. This light secondary
  /// smoothing suppresses scale/position noise without introducing visible lag.
  static const double _smoothAlpha = 0.20;

  GarmentFitter({BodyDimensionEstimator? dimensionEstimator})
      : _dimensionEstimator = dimensionEstimator ?? BodyDimensionEstimator();

  int _debugCount = 0;

  GarmentTransform? fitGarment({
    required PoseResult pose,
    required GarmentConfig config,
    required double viewportWidth,
    required double viewportHeight,
  }) {
    if (!pose.isValid) return _lastTransform;

    final ls = pose.leftShoulder!;
    final rs = pose.rightShoulder!;

    // ── 1. Shoulder geometry ──────────────────────────────────────
    final shoulderWidth = ls.distanceTo2D(rs);
    final shoulderCenterX = (ls.x + rs.x) / 2;
    final shoulderCenterY = (ls.y + rs.y) / 2;

    // Safety: if MLKit returns degenerate shoulders (< 10px apart), skip
    if (shoulderWidth < 10) return _lastTransform;

    // ── 2. Garment width = shoulder width × padding ───────────────
    // NO minimum clamp. The jacket scales naturally:
    //   person close → big jacket, person far → small jacket.
    // A minWidth clamp makes the jacket appear huge when person is far away.
    final garmentWidth = shoulderWidth * config.shoulderPadding;
    final scaleX = garmentWidth / config.originalWidth;

    // ── 3. Garment height ─────────────────────────────────────────
    // Strategy: when hips are detectable, use the real torso height
    // for perspective-correct scaling. This is how Snapchat does it.
    // When hips aren't visible, fall back to aspect-ratio from width.
    final lh = pose.leftHip;
    final rh = pose.rightHip;

    final double garmentHeight;
    if (lh != null && rh != null && lh.isVisible && rh.isVisible) {
      final hipMidY = (lh.y + rh.y) / 2;
      final torsoHeightPx = (hipMidY - shoulderCenterY).abs();

      // The garment image body (shoulder seam → bottom hem) covers approximately
      // (1 - shoulderLineInImage - 0.05) of the image height.
      // The 0.05 accounts for bottom hem/padding in the image.
      // We map the real torso + 10% hem allowance to that portion of the image.
      final bodyFraction = 1.0 - config.shoulderLineInImage - 0.05;
      final heightFromTorso = (torsoHeightPx * 1.10) / bodyFraction;

      // Also compute from aspect ratio as a lower bound.
      // Use the larger: jacket always covers at least to hips.
      final heightFromAspect = garmentWidth * (config.originalHeight / config.originalWidth);
      garmentHeight = math.max(heightFromTorso, heightFromAspect);
    } else {
      // Hips not visible — aspect-ratio from shoulder width
      garmentHeight = garmentWidth * (config.originalHeight / config.originalWidth);
    }

    final scaleY = garmentHeight / config.originalHeight;

    // ── 4. Anchor: position garment so shoulder seam aligns with body ──
    // The garment is drawn centered at (anchorX, anchorY).
    // The shoulder seam is at (shoulderLineInImage) fraction from top of image.
    // In centered coordinates: seam is at anchorY + (shoulderLineInImage - 0.5) * garmentHeight
    // Setting that equal to shoulderCenterY gives:
    //   anchorY = shoulderCenterY - (shoulderLineInImage - 0.5) * garmentHeight
    //           = shoulderCenterY + (0.5 - shoulderLineInImage) * garmentHeight
    final anchorX = shoulderCenterX;
    final anchorY = shoulderCenterY + garmentHeight * (0.5 - config.shoulderLineInImage);

    // ── 5. Rotation from shoulder tilt ────────────────────────────
    // atan2(dy, dx): positive = right shoulder lower than left (tilt right)
    final shoulderAngle = math.atan2(rs.y - ls.y, rs.x - ls.x) * 180 / math.pi;
    final rotation = shoulderAngle.clamp(-15.0, 15.0);

    var transform = GarmentTransform(
      x: anchorX,
      y: anchorY,
      scaleX: scaleX,
      scaleY: scaleY,
      rotationDegrees: rotation,
      shoulderLineInImage: config.shoulderLineInImage,
    );

    // ── 6. Smooth the transform (gentle — landmarks already filtered) ──
    if (_lastTransform != null) {
      transform = _smooth(_lastTransform!, transform);
    }
    _lastTransform = transform;

    if (_debugCount++ < 5) {
      debugPrint('[FITTER] shoulderW=${shoulderWidth.toStringAsFixed(0)}px '
          'garmentW=${garmentWidth.toStringAsFixed(0)}px '
          'garmentH=${garmentHeight.toStringAsFixed(0)}px '
          'scaleX=${scaleX.toStringAsFixed(3)} scaleY=${scaleY.toStringAsFixed(3)} '
          'anchor=(${anchorX.toStringAsFixed(0)},${anchorY.toStringAsFixed(0)}) '
          'hips=${lh?.isVisible == true && rh?.isVisible == true}');
    }

    return transform;
  }

  GarmentTransform _smooth(GarmentTransform prev, GarmentTransform curr) {
    return GarmentTransform(
      x: prev.x + (curr.x - prev.x) * _smoothAlpha,
      y: prev.y + (curr.y - prev.y) * _smoothAlpha,
      scaleX: prev.scaleX + (curr.scaleX - prev.scaleX) * _smoothAlpha,
      scaleY: prev.scaleY + (curr.scaleY - prev.scaleY) * _smoothAlpha,
      rotationDegrees: prev.rotationDegrees + (curr.rotationDegrees - prev.rotationDegrees) * _smoothAlpha,
    );
  }

  void reset() {
    _lastTransform = null;
    _dimensionEstimator.reset();
    _debugCount = 0;
  }
}
