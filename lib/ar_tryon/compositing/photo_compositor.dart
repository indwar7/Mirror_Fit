import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:flutter/painting.dart';
import '../core/pipeline_types.dart';

/// Composites a garment onto a person photo using the current AR transform.
///
/// This is instant (no network), always works, and produces a photo that
/// looks exactly like the AR preview — the person "wearing" the garment.
///
/// The transform comes from the live AR pipeline (preview coordinates).
/// We scale it to match the captured photo resolution.
class PhotoCompositor {
  static Future<Uint8List?> composite({
    required Uint8List photoBytes,
    required ui.Image garmentImage,
    required GarmentTransform transform,
    required double previewWidth,
    required double previewHeight,
  }) async {
    try {
      // Decode person photo
      final codec = await ui.instantiateImageCodec(photoBytes);
      final frame = await codec.getNextFrame();
      final personImg = frame.image;

      // On iOS with lockCaptureOrientation(portraitUp), photo should be portrait.
      // Handle both orientations just in case.
      final bool photoIsPortrait = personImg.height >= personImg.width;
      final double photoW = photoIsPortrait
          ? personImg.width.toDouble()
          : personImg.height.toDouble();
      final double photoH = photoIsPortrait
          ? personImg.height.toDouble()
          : personImg.width.toDouble();

      // Scale factor from AR preview coordinates → photo coordinates
      final double sx = photoW / previewWidth;
      final double sy = photoH / previewHeight;

      final recorder = ui.PictureRecorder();
      final canvas = ui.Canvas(
        recorder,
        Rect.fromLTWH(0, 0, photoW, photoH),
      );

      // ── 1. Draw person photo ─────────────────────────────────────
      if (photoIsPortrait) {
        canvas.drawImage(personImg, Offset.zero, Paint());
      } else {
        // Rotate landscape photo to portrait
        canvas.save();
        canvas.translate(0, photoH);
        canvas.rotate(-math.pi / 2);
        canvas.drawImage(personImg, Offset.zero, Paint());
        canvas.restore();
      }

      // ── 2. Draw garment at scaled position ───────────────────────
      final dstW = garmentImage.width * transform.scaleX * sx;
      final dstH = garmentImage.height * transform.scaleY * sy;
      final gx = transform.x * sx;
      final gy = transform.y * sy;

      canvas.save();
      canvas.translate(gx, gy);
      canvas.rotate(transform.rotationDegrees * math.pi / 180);
      canvas.drawImageRect(
        garmentImage,
        Rect.fromLTWH(
          0, 0,
          garmentImage.width.toDouble(),
          garmentImage.height.toDouble(),
        ),
        Rect.fromCenter(
          center: Offset.zero,
          width: dstW,
          height: dstH,
        ),
        Paint()
          ..filterQuality = ui.FilterQuality.high
          ..isAntiAlias = true,
      );
      canvas.restore();

      // ── 3. Convert to JPEG bytes ──────────────────────────────────
      final picture = recorder.endRecording();
      final result = await picture.toImage(photoW.toInt(), photoH.toInt());
      final byteData = await result.toByteData(format: ui.ImageByteFormat.rawRgba);
      if (byteData == null) return null;

      final pngData = await result.toByteData(format: ui.ImageByteFormat.png);
      if (pngData == null) return null;

      debugPrint('[COMPOSITOR] Done: ${photoW.toInt()}x${photoH.toInt()} '
          'garment=${dstW.toStringAsFixed(0)}x${dstH.toStringAsFixed(0)} '
          'at (${gx.toStringAsFixed(0)},${gy.toStringAsFixed(0)})');

      return pngData.buffer.asUint8List();
    } catch (e) {
      debugPrint('[COMPOSITOR] Error: $e');
      return null;
    }
  }
}
