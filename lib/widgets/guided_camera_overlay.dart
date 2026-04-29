import 'package:flutter/material.dart';
import '../utils/app_theme.dart';

class GuidedCameraOverlay extends StatefulWidget {
  const GuidedCameraOverlay({super.key});

  @override
  State<GuidedCameraOverlay> createState() => _GuidedCameraOverlayState();
}

class _GuidedCameraOverlayState extends State<GuidedCameraOverlay>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseCtrl;
  late Animation<double> _pulseAnim;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: 0.4, end: 1.0).animate(
      CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: AnimatedBuilder(
        animation: _pulseAnim,
        builder: (context, child) => CustomPaint(
          size: Size.infinite,
          painter: _OverlayPainter(opacity: _pulseAnim.value),
        ),
      ),
    );
  }
}

class _OverlayPainter extends CustomPainter {
  final double opacity;
  _OverlayPainter({required this.opacity});

  @override
  void paint(Canvas canvas, Size size) {
    final centerX = size.width / 2;
    final centerY = size.height / 2;
    final frameW = size.width * 0.7;
    final frameH = size.height * 0.55;
    final left = centerX - frameW / 2;
    final top = centerY - frameH / 2;
    final right = centerX + frameW / 2;
    final bottom = centerY + frameH / 2;
    final cornerLen = 28.0;

    // Dimmed overlay outside frame
    final dimPaint = Paint()..color = Colors.black.withValues(alpha: 0.5);
    final framePath = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
      ..addRRect(RRect.fromRectAndRadius(
        Rect.fromLTRB(left, top, right, bottom),
        const Radius.circular(12),
      ))
      ..fillType = PathFillType.evenOdd;
    canvas.drawPath(framePath, dimPaint);

    // Corner guides
    final cornerPaint = Paint()
      ..color = AppColors.gold.withValues(alpha: opacity)
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    // Top-left
    canvas.drawLine(
        Offset(left, top + cornerLen), Offset(left, top + 6), cornerPaint);
    canvas.drawLine(
        Offset(left + cornerLen, top), Offset(left + 6, top), cornerPaint);

    // Top-right
    canvas.drawLine(Offset(right, top + cornerLen),
        Offset(right, top + 6), cornerPaint);
    canvas.drawLine(Offset(right - cornerLen, top),
        Offset(right - 6, top), cornerPaint);

    // Bottom-left
    canvas.drawLine(Offset(left, bottom - cornerLen),
        Offset(left, bottom - 6), cornerPaint);
    canvas.drawLine(Offset(left + cornerLen, bottom),
        Offset(left + 6, bottom), cornerPaint);

    // Bottom-right
    canvas.drawLine(Offset(right, bottom - cornerLen),
        Offset(right, bottom - 6), cornerPaint);
    canvas.drawLine(Offset(right - cornerLen, bottom),
        Offset(right - 6, bottom), cornerPaint);
  }

  @override
  bool shouldRepaint(_OverlayPainter old) => old.opacity != opacity;
}
