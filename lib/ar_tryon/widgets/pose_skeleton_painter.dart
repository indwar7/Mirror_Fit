import 'package:flutter/material.dart';
import '../core/pipeline_types.dart';

/// Paints the detected pose skeleton on top of the camera feed.
///
/// Visualizes:
///   - Joint points (circles at each landmark)
///   - Bone connections (lines between joints)
///   - Shoulder/hip measurement lines (highlighted)
///   - Joint labels (optional, for debugging)
class PoseSkeletonPainter extends CustomPainter {
  final PoseResult pose;
  final double viewportWidth;
  final double viewportHeight;
  final Color color;
  final bool showLabels;

  PoseSkeletonPainter({
    required this.pose,
    required this.viewportWidth,
    required this.viewportHeight,
    this.color = const Color(0xFF00FF00),
    this.showLabels = false,
  });

  // Bone connections (pairs of landmark types)
  static const _bones = [
    // Face
    (LandmarkType.nose, LandmarkType.leftEye),
    (LandmarkType.nose, LandmarkType.rightEye),
    (LandmarkType.leftEye, LandmarkType.leftEar),
    (LandmarkType.rightEye, LandmarkType.rightEar),
    // Torso
    (LandmarkType.leftShoulder, LandmarkType.rightShoulder),
    (LandmarkType.leftShoulder, LandmarkType.leftHip),
    (LandmarkType.rightShoulder, LandmarkType.rightHip),
    (LandmarkType.leftHip, LandmarkType.rightHip),
    // Left arm
    (LandmarkType.leftShoulder, LandmarkType.leftElbow),
    (LandmarkType.leftElbow, LandmarkType.leftWrist),
    // Right arm
    (LandmarkType.rightShoulder, LandmarkType.rightElbow),
    (LandmarkType.rightElbow, LandmarkType.rightWrist),
    // Left leg
    (LandmarkType.leftHip, LandmarkType.leftKnee),
    (LandmarkType.leftKnee, LandmarkType.leftAnkle),
    // Right leg
    (LandmarkType.rightHip, LandmarkType.rightKnee),
    (LandmarkType.rightKnee, LandmarkType.rightAnkle),
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final sx = size.width / pose.frameWidth;
    final sy = size.height / pose.frameHeight;

    // Draw bones
    final bonePaint = Paint()
      ..color = color.withValues(alpha: 0.8)
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    for (final (from, to) in _bones) {
      final a = pose.get(from);
      final b = pose.get(to);
      if (a != null && b != null && a.isVisible && b.isVisible) {
        canvas.drawLine(
          Offset(a.x * sx, a.y * sy),
          Offset(b.x * sx, b.y * sy),
          bonePaint,
        );
      }
    }

    // Draw measurement lines (shoulder + hip) in distinct color
    final measurePaint = Paint()
      ..color = const Color(0xFFFF4444)
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    final ls = pose.leftShoulder;
    final rs = pose.rightShoulder;
    if (ls != null && rs != null && ls.isVisible && rs.isVisible) {
      canvas.drawLine(
        Offset(ls.x * sx, ls.y * sy),
        Offset(rs.x * sx, rs.y * sy),
        measurePaint,
      );
    }

    final lh = pose.leftHip;
    final rh = pose.rightHip;
    if (lh != null && rh != null && lh.isVisible && rh.isVisible) {
      canvas.drawLine(
        Offset(lh.x * sx, lh.y * sy),
        Offset(rh.x * sx, rh.y * sy),
        measurePaint..color = const Color(0xFF4488FF),
      );
    }

    // Draw joints
    final jointPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final jointOutlinePaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    for (final lm in pose.landmarks) {
      if (!lm.isVisible) continue;

      final pos = Offset(lm.x * sx, lm.y * sy);
      final radius = _isKeyLandmark(lm.type) ? 6.0 : 4.0;

      // Color key landmarks differently
      if (_isKeyLandmark(lm.type)) {
        jointPaint.color = const Color(0xFFFFFF00);
      } else {
        jointPaint.color = color;
      }

      canvas.drawCircle(pos, radius, jointPaint);
      canvas.drawCircle(pos, radius, jointOutlinePaint);

      // Draw labels
      if (showLabels && _isKeyLandmark(lm.type)) {
        _drawLabel(canvas, pos, _landmarkName(lm.type));
      }
    }
  }

  bool _isKeyLandmark(LandmarkType type) {
    return type == LandmarkType.leftShoulder ||
        type == LandmarkType.rightShoulder ||
        type == LandmarkType.leftHip ||
        type == LandmarkType.rightHip ||
        type == LandmarkType.leftElbow ||
        type == LandmarkType.rightElbow ||
        type == LandmarkType.leftWrist ||
        type == LandmarkType.rightWrist;
  }

  String _landmarkName(LandmarkType type) {
    return type.name
        .replaceAllMapped(RegExp(r'[A-Z]'), (m) => ' ${m.group(0)}')
        .trim();
  }

  void _drawLabel(Canvas canvas, Offset pos, String text) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 9,
          fontWeight: FontWeight.bold,
          backgroundColor: Color(0x88000000),
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(canvas, pos + const Offset(8, -6));
  }

  @override
  bool shouldRepaint(PoseSkeletonPainter oldDelegate) {
    return pose.timestampMs != oldDelegate.pose.timestampMs;
  }
}
