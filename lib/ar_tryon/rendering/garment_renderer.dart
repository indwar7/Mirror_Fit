import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import '../core/pipeline_types.dart';

/// Renders garment overlay on top of camera feed.
///
/// Phase 1+2 implementation:
///   • saveLayer wrapping — correct BlendMode.dstOut behavior
///   • Pixel-accurate arm occlusion (capsule + soft blur + hand blobs)
///   • Segmentation-guided arm erasing when mask available
///   • Mesh-based sleeve deformation via drawVertices (Phase 2)
class GarmentOverlayPainter extends CustomPainter {
  final GarmentTransform? transform;
  final ui.Image? garmentImage;
  final PoseResult? pose;
  final Size viewportSize;
  final SegmentationMask? segmentationMask;
  /// Frames since last segmentation result. Layer A (mask erase) is skipped
  /// when > 2 to avoid artifacts from a frozen mask on a moving arm.
  final int maskAgeFrames;
  /// RGBA ui.Image built from the segmentation mask (alpha = body probability).
  /// When provided, jacket is clipped to body silhouette via BlendMode.dstIn —
  /// this is the key "jacket worn on body" effect.
  final ui.Image? segmentationImage;

  GarmentOverlayPainter({
    this.transform,
    this.garmentImage,
    this.pose,
    required this.viewportSize,
    this.segmentationMask,
    this.maskAgeFrames = 999,
    this.segmentationImage,
  });

  // Identity matrix for ImageShader (column-major Float64List)
  static final Float64List _identity = Float64List.fromList([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);

  // Triangle index topology for a 6×9 mesh.
  // Computed once for the app lifetime — grid topology never changes.
  static Uint16List? _cachedGridIndices;

  static Uint16List _buildGridIndices(int cols, int rows) {
    if (_cachedGridIndices != null) return _cachedGridIndices!;
    final triCount = (cols - 1) * (rows - 1) * 2;
    final indices = Uint16List(triCount * 3);
    int ip = 0;
    for (int row = 0; row < rows - 1; row++) {
      for (int col = 0; col < cols - 1; col++) {
        final i = row * cols + col;
        indices[ip++] = i;
        indices[ip++] = i + 1;
        indices[ip++] = i + cols;
        indices[ip++] = i + 1;
        indices[ip++] = i + cols + 1;
        indices[ip++] = i + cols;
      }
    }
    _cachedGridIndices = indices;
    return indices;
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (garmentImage == null) return;

    if (pose != null && pose!.isValid && transform != null && transform!.scaleX > 0.01) {
      _drawTrackedGarment(canvas, size);
    } else if (pose != null && pose!.isValid) {
      _drawFallbackGarment(canvas, size);
    } else {
      _drawGhostGarment(canvas, size);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // TRACKED: full pipeline — mesh deform + arm occlusion
  // ─────────────────────────────────────────────────────────────────────────

  void _drawTrackedGarment(Canvas canvas, Size size) {
    final img = garmentImage!;
    final t = transform!;

    // Confidence-based opacity
    final ls = pose?.leftShoulder;
    final rs = pose?.rightShoulder;
    final confidence = ls != null && rs != null
        ? math.min(ls.confidence, rs.confidence)
        : 1.0;
    final opacity = (0.7 + (confidence.clamp(0.5, 1.0) - 0.5) * 0.6).clamp(0.0, 1.0);

    // ── saveLayer: garment + arm punch-through must be in the same layer ──
    // Without saveLayer, BlendMode.dstOut operates on the entire framebuffer.
    // Inside saveLayer, dstOut only erases pixels from THIS layer, then the
    // layer composites onto the camera preview beneath — arms show through.
    canvas.saveLayer(
      Rect.fromLTWH(0, 0, size.width, size.height),
      Paint(),
    );

    // Draw garment (mesh-deformed for Phase 2, or flat rigid fallback)
    final bool hasBothElbows = (pose?.leftElbow?.isVisible ?? false) ||
        (pose?.rightElbow?.isVisible ?? false);

    if (hasBothElbows) {
      _drawMeshGarment(canvas, size, img, t, opacity);
    } else {
      _drawRigidGarment(canvas, img, t, opacity);
    }

    // ── Body silhouette clip (THE "worn" effect) ──────────────────
    // dstIn: keeps jacket pixels only where body mask has alpha > 0.
    // Jacket pixels outside the body silhouette become transparent →
    // camera feed shows through → jacket appears to be ON the body.
    if (segmentationImage != null && maskAgeFrames <= 4) {
      _clipToBodySilhouette(canvas, size);
    }

    // Punch through — arms appear IN FRONT of jacket
    _occludeArms(canvas, size);

    canvas.restore(); // composites layer → camera preview shows through erased areas
  }

  // ─────────────────────────────────────────────────────────────────────────
  // BODY SILHOUETTE CLIP — clips jacket to body shape (the "worn" effect)
  // ─────────────────────────────────────────────────────────────────────────
  //
  // How it works:
  //   segmentationImage is 256×256 RGBA where alpha = body probability.
  //   We draw it stretched to viewport with BlendMode.dstIn.
  //   dstIn formula: dst.alpha = dst.alpha × src.alpha
  //   → jacket pixels outside the body silhouette become transparent
  //   → camera feed shows through those transparent pixels
  //   → jacket appears physically on/inside the body, not floating on top

  void _clipToBodySilhouette(Canvas canvas, Size size) {
    canvas.drawImageRect(
      segmentationImage!,
      Rect.fromLTWH(
        0, 0,
        segmentationImage!.width.toDouble(),
        segmentationImage!.height.toDouble(),
      ),
      Rect.fromLTWH(0, 0, size.width, size.height),
      Paint()
        ..blendMode = BlendMode.dstIn
        ..filterQuality = FilterQuality.medium,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // RIGID DRAW (no deformation — fallback when elbows not visible)
  // ─────────────────────────────────────────────────────────────────────────

  void _drawRigidGarment(Canvas canvas, ui.Image img, GarmentTransform t, double opacity) {
    final dstW = img.width * t.scaleX;
    final dstH = img.height * t.scaleY;

    final paint = Paint()
      ..filterQuality = FilterQuality.high
      ..isAntiAlias = true
      ..color = Color.fromRGBO(255, 255, 255, opacity);

    canvas.save();
    canvas.translate(t.x, t.y);
    canvas.rotate(t.rotationDegrees * math.pi / 180);
    canvas.drawImageRect(
      img,
      Rect.fromLTWH(0, 0, img.width.toDouble(), img.height.toDouble()),
      Rect.fromCenter(center: Offset.zero, width: dstW, height: dstH),
      paint,
    );
    canvas.restore();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // PHASE 2: MESH-BASED SLEEVE DEFORMATION via drawVertices (GPU)
  // ─────────────────────────────────────────────────────────────────────────
  //
  // Divides garment into a 6×9 grid.
  // Left/right sleeve columns are displaced by elbow position.
  // This makes sleeves follow the actual arm angle — no more flat PNG.

  void _drawMeshGarment(Canvas canvas, Size size, ui.Image img,
      GarmentTransform t, double opacity) {
    const int cols = 6;
    const int rows = 9;
    final int vertCount = cols * rows;

    final gW = img.width * t.scaleX;
    final gH = img.height * t.scaleY;
    final shoulderLine = t.shoulderLineInImage;

    // Pre-compute inverse rotation for landmark → local-space transform
    final cosA = math.cos(-t.rotationDegrees * math.pi / 180);
    final sinA = math.sin(-t.rotationDegrees * math.pi / 180);

    Offset toLandmarkLocal(Landmark lm) {
      final dx = lm.x - t.x;
      final dy = lm.y - t.y;
      return Offset(dx * cosA - dy * sinA, dx * sinA + dy * cosA);
    }

    // Landmark local positions
    final lsLocal = pose?.leftShoulder != null ? toLandmarkLocal(pose!.leftShoulder!) : null;
    final rsLocal = pose?.rightShoulder != null ? toLandmarkLocal(pose!.rightShoulder!) : null;
    final leLocal = (pose?.leftElbow?.isVisible ?? false) ? toLandmarkLocal(pose!.leftElbow!) : null;
    final reLocal = (pose?.rightElbow?.isVisible ?? false) ? toLandmarkLocal(pose!.rightElbow!) : null;
    final lwLocal = (pose?.leftWrist?.isVisible ?? false) ? toLandmarkLocal(pose!.leftWrist!) : null;
    final rwLocal = (pose?.rightWrist?.isVisible ?? false) ? toLandmarkLocal(pose!.rightWrist!) : null;

    final positions = Float32List(vertCount * 2);
    final texCoords = Float32List(vertCount * 2);

    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        final u = col / (cols - 1); // 0..1 across garment width
        final v = row / (rows - 1); // 0..1 down garment height

        // Base position in local canvas space (centered at anchor)
        double px = (u - 0.5) * gW;
        double py = (v - 0.5) * gH;

        // ── Sleeve deformation ──────────────────────────────────────
        // Only deform below shoulder seam
        if (v > shoulderLine) {
          // How far down the body this point is (0 at shoulder, 1 at bottom)
          final depthT = (v - shoulderLine) / (1.0 - shoulderLine);

          // LEFT SLEEVE: u in [0, 0.35]
          if (u < 0.35 && lsLocal != null && leLocal != null) {
            // outer-edge weight: 1.0 at u=0, 0.0 at u=0.35
            final sleeveWeight = (0.35 - u) / 0.35;
            // How far into the sleeve (0=shoulder, 1=elbow)
            final sleeveFrac = depthT.clamp(0.0, 1.0);

            // Actual elbow displacement relative to shoulder
            final elbowDx = leLocal.dx - lsLocal.dx;
            final elbowDy = leLocal.dy - lsLocal.dy;

            // Wrist influence blends in gradually from depthT=0.30 (not a hard gate).
            // Previously gated at 0.50 which made the cuff not follow wrist at all
            // for the upper-half of the sleeve mesh.
            double wristBonus = 0.0;
            double wristBonusDy = 0.0;
            if (lwLocal != null) {
              final wristFrac = ((depthT - 0.30) / 0.70).clamp(0.0, 1.0);
              wristBonus = (lwLocal.dx - leLocal.dx) * wristFrac;
              wristBonusDy = (lwLocal.dy - leLocal.dy) * wristFrac;
            }

            // Y gets higher dampening than X: arm raises are the dominant motion.
            // Previous X=0.45 Y=0.30 was backwards — vertical motion under-responded.
            px += (elbowDx * sleeveFrac + wristBonus) * sleeveWeight * 0.35;
            py += (elbowDy * sleeveFrac + wristBonusDy) * sleeveWeight * 0.42;
          }

          // RIGHT SLEEVE: u in [0.65, 1.0]
          if (u > 0.65 && rsLocal != null && reLocal != null) {
            final sleeveWeight = (u - 0.65) / 0.35;
            final sleeveFrac = depthT.clamp(0.0, 1.0);

            final elbowDx = reLocal.dx - rsLocal.dx;
            final elbowDy = reLocal.dy - rsLocal.dy;

            double wristBonus = 0.0;
            double wristBonusDy = 0.0;
            if (rwLocal != null) {
              final wristFrac = ((depthT - 0.30) / 0.70).clamp(0.0, 1.0);
              wristBonus = (rwLocal.dx - reLocal.dx) * wristFrac;
              wristBonusDy = (rwLocal.dy - reLocal.dy) * wristFrac;
            }

            px += (elbowDx * sleeveFrac + wristBonus) * sleeveWeight * 0.35;
            py += (elbowDy * sleeveFrac + wristBonusDy) * sleeveWeight * 0.42;
          }
        }

        final idx = (row * cols + col) * 2;
        positions[idx]     = px;
        positions[idx + 1] = py;
        texCoords[idx]     = u * img.width;
        texCoords[idx + 1] = v * img.height;
      }
    }

    // Retrieve cached triangle indices (computed once, zero per-frame cost)
    final indices = _buildGridIndices(cols, rows);

    final vertices = ui.Vertices.raw(
      ui.VertexMode.triangles,
      positions,
      textureCoordinates: texCoords,
      indices: indices,
    );

    final shader = ui.ImageShader(img, ui.TileMode.clamp, ui.TileMode.clamp, _identity,
        filterQuality: ui.FilterQuality.medium);

    final paint = Paint()
      ..shader = shader
      ..filterQuality = FilterQuality.medium
      ..isAntiAlias = true;

    // Apply opacity via saveLayer on the mesh itself
    canvas.save();
    canvas.translate(t.x, t.y);
    canvas.rotate(t.rotationDegrees * math.pi / 180);

    if (opacity < 0.999) {
      canvas.saveLayer(
        Rect.fromCenter(
          center: Offset.zero,
          width: img.width * t.scaleX * 1.5,
          height: img.height * t.scaleY * 1.5,
        ),
        Paint()..color = Color.fromRGBO(255, 255, 255, opacity),
      );
    }

    canvas.drawVertices(vertices, ui.BlendMode.srcOver, paint);

    if (opacity < 0.999) canvas.restore();
    canvas.restore();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // PHASE 1: ARM OCCLUSION
  // Three-layer approach:
  //   1. Segmentation mask (pixel-accurate) when available
  //   2. Capsule-shaped arm path with soft blur edges
  //   3. Elliptical hand blobs at wrists
  // ─────────────────────────────────────────────────────────────────────────

  void _occludeArms(Canvas canvas, Size size) {
    if (pose == null) return;

    final ls = pose!.leftShoulder;
    final rs = pose!.rightShoulder;
    final le = pose!.leftElbow;
    final re = pose!.rightElbow;
    final lw = pose!.leftWrist;
    final rw = pose!.rightWrist;

    final shoulderWidth = (ls != null && rs != null)
        ? ls.distanceTo2D(rs)
        : size.width * 0.3;
    final armThickness = shoulderWidth * 0.26;

    // ── Layer A: Segmentation-guided pixel erase ──
    // Only apply when mask is fresh (≤2 frames old). A stale frozen mask on a
    // moving arm causes a "ghost rectangle" artifact. Layer B handles live tracking.
    if (segmentationMask != null && maskAgeFrames <= 2) {
      _eraseArmsWithMask(canvas, size, armThickness);
    }

    // ── Layer B: Capsule arm erase with soft feathered edges ──
    _eraseArmCapsule(canvas, ls, le, lw, armThickness, isSoft: true);
    _eraseArmCapsule(canvas, rs, re, rw, armThickness, isSoft: true);

    // ── Layer C: Hand/wrist blobs (ensures hands fully occlude jacket) ──
    _eraseHandBlob(canvas, lw, armThickness);
    _eraseHandBlob(canvas, rw, armThickness);
  }

  /// Convert segmentation mask to arm-region erase using landmark ROI.
  void _eraseArmsWithMask(Canvas canvas, Size size, double armThickness) {
    final mask = segmentationMask!;
    final pose_ = pose!;

    // Scale from mask space (256×256) to viewport
    final sx = size.width / mask.width;
    final sy = size.height / mask.height;

    // Build a path from body-masked pixels that fall inside arm ROI
    // Arm ROI = bounding rect of shoulder→elbow→wrist with padding
    final leftArmROI = _armROI(
      pose_.leftShoulder, pose_.leftElbow, pose_.leftWrist,
      padding: armThickness * 1.2,
    );
    final rightArmROI = _armROI(
      pose_.rightShoulder, pose_.rightElbow, pose_.rightWrist,
      padding: armThickness * 1.2,
    );

    if (leftArmROI == null && rightArmROI == null) return;

    final erasePaint = Paint()
      ..blendMode = BlendMode.dstOut
      ..style = PaintingStyle.fill;

    // Sub-sample mask every 3rd pixel; merge horizontal runs into single rects.
    // Per-pixel addRect creates O(pixels) path segments — very expensive for the
    // GPU tessellator. Span merging reduces this to O(runs) which is ~10× fewer ops.
    const step = 3;
    final path = Path();

    for (int my = 0; my < mask.height; my += step) {
      final py = my * sy;
      // Pre-filter entire rows by Y bounds — skips ~80% of rows outside both ROIs.
      final rowInLeft  = leftArmROI  != null && py >= leftArmROI.top  && py <= leftArmROI.bottom;
      final rowInRight = rightArmROI != null && py >= rightArmROI.top && py <= rightArmROI.bottom;
      if (!rowInLeft && !rowInRight) continue;

      int spanStart = -1;
      for (int mx = 0; mx <= mask.width; mx += step) {
        bool isFg = false;
        if (mx < mask.width) {
          final px = mx * sx;
          final inArm =
              (rowInLeft  && px >= leftArmROI.left  && px <= leftArmROI.right)  ||
              (rowInRight && px >= rightArmROI.left && px <= rightArmROI.right);
          isFg = inArm && mask.isForeground(mx, my, threshold: 100);
        }

        if (isFg && spanStart < 0) {
          spanStart = mx;
        } else if (!isFg && spanStart >= 0) {
          // One rect covers the entire horizontal run
          path.addRect(Rect.fromLTWH(
            spanStart * sx, py,
            (mx - spanStart) * sx, sy * step,
          ));
          spanStart = -1;
        }
      }
    }

    canvas.drawPath(path, erasePaint);
  }

  Rect? _armROI(Landmark? a, Landmark? b, Landmark? c, {required double padding}) {
    if (a == null) return null;
    double minX = a.x, maxX = a.x, minY = a.y, maxY = a.y;
    void expand(Landmark? lm) {
      if (lm == null || !lm.isVisible) return;
      if (lm.x < minX) minX = lm.x;
      if (lm.x > maxX) maxX = lm.x;
      if (lm.y < minY) minY = lm.y;
      if (lm.y > maxY) maxY = lm.y;
    }
    expand(b);
    expand(c);
    return Rect.fromLTRB(minX - padding, minY - padding, maxX + padding, maxY + padding);
  }

  /// Capsule-shaped arm erase with optional soft edges.
  void _eraseArmCapsule(Canvas canvas,
      Landmark? shoulder, Landmark? elbow, Landmark? wrist,
      double thickness, {required bool isSoft}) {

    final points = <Offset>[];
    if (shoulder != null && shoulder.isVisible) points.add(Offset(shoulder.x, shoulder.y));
    if (elbow != null && elbow.isVisible) points.add(Offset(elbow.x, elbow.y));
    if (wrist != null && wrist.isVisible) points.add(Offset(wrist.x, wrist.y));
    if (points.length < 2) return;

    final path = Path();
    bool pathStarted = false;

    for (int i = 0; i < points.length - 1; i++) {
      final a = points[i];
      final b = points[i + 1];
      final seg = b - a;
      final len = seg.distance;
      if (len < 1) continue;

      // Perpendicular direction for capsule width
      final perp = Offset(-seg.dy / len, seg.dx / len) * thickness;

      // Upper and lower edges of the capsule segment
      final tl = a + perp;
      final tr = a - perp;
      final bl = b + perp;
      final br = b - perp;

      if (!pathStarted) {
        path.moveTo(tl.dx, tl.dy);
        pathStarted = true;
      } else {
        path.lineTo(tl.dx, tl.dy);
      }
      path.lineTo(tr.dx, tr.dy);
      path.lineTo(br.dx, br.dy);
      path.lineTo(bl.dx, bl.dy);
      path.close();
    }

    if (!pathStarted) return;

    if (isSoft) {
      // Soft feathered edge: draw two layers
      // Outer layer: blurred (feathered boundary)
      final blurPaint = Paint()
        ..blendMode = BlendMode.dstOut
        ..style = PaintingStyle.fill
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8.0);
      canvas.drawPath(path, blurPaint);

      // Inner layer: hard core (full erase at center of arm)
      final hardPaint = Paint()
        ..blendMode = BlendMode.dstOut
        ..style = PaintingStyle.fill;
      canvas.drawPath(path, hardPaint);
    } else {
      canvas.drawPath(path,
          Paint()..blendMode = BlendMode.dstOut..style = PaintingStyle.fill);
    }
  }

  /// Elliptical erase blob at wrist/hand — ensures fingers don't clip through.
  void _eraseHandBlob(Canvas canvas, Landmark? wrist, double thickness) {
    if (wrist == null || !wrist.isVisible) return;

    final center = Offset(wrist.x, wrist.y);
    final rx = thickness * 1.1;
    final ry = thickness * 0.85;

    // Soft outer ring
    canvas.drawOval(
      Rect.fromCenter(center: center, width: rx * 2.6, height: ry * 2.6),
      Paint()
        ..blendMode = BlendMode.dstOut
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10.0),
    );
    // Hard inner core
    canvas.drawOval(
      Rect.fromCenter(center: center, width: rx * 2.0, height: ry * 2.0),
      Paint()..blendMode = BlendMode.dstOut,
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // FALLBACK: shoulders visible but transform not ready yet
  // ─────────────────────────────────────────────────────────────────────────

  void _drawFallbackGarment(Canvas canvas, Size size) {
    final img = garmentImage!;
    final ls = pose!.leftShoulder!;
    final rs = pose!.rightShoulder!;

    final cx = (ls.x + rs.x) / 2;
    final cy = (ls.y + rs.y) / 2;
    final shoulderW = ls.distanceTo2D(rs);
    if (shoulderW < 10) return;

    final garmentW = shoulderW * 1.5;
    final scale = garmentW / img.width;
    final dstW = img.width * scale;
    final dstH = img.height * scale;

    canvas.saveLayer(
      Rect.fromLTWH(0, 0, size.width, size.height),
      Paint(),
    );

    canvas.save();
    canvas.translate(cx, cy + dstH * 0.35);
    canvas.drawImageRect(
      img,
      Rect.fromLTWH(0, 0, img.width.toDouble(), img.height.toDouble()),
      Rect.fromCenter(center: Offset.zero, width: dstW, height: dstH),
      Paint()
        ..filterQuality = FilterQuality.high
        ..isAntiAlias = true
        ..color = const Color(0xCCFFFFFF),
    );
    canvas.restore();

    if (segmentationImage != null && maskAgeFrames <= 4) {
      _clipToBodySilhouette(canvas, size);
    }

    _occludeArms(canvas, size);

    canvas.restore();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // GHOST: no body detected — semi-transparent preview
  // ─────────────────────────────────────────────────────────────────────────

  void _drawGhostGarment(Canvas canvas, Size size) {
    final img = garmentImage!;
    final scale = (size.width * 0.6) / img.width;
    final dstW = img.width * scale;
    final dstH = img.height * scale;

    canvas.save();
    canvas.translate(size.width / 2, size.height * 0.38);
    canvas.drawImageRect(
      img,
      Rect.fromLTWH(0, 0, img.width.toDouble(), img.height.toDouble()),
      Rect.fromCenter(center: Offset.zero, width: dstW, height: dstH),
      Paint()
        ..filterQuality = FilterQuality.high
        ..isAntiAlias = true
        ..color = const Color(0x55FFFFFF),
    );
    canvas.restore();
  }

  @override
  bool shouldRepaint(GarmentOverlayPainter old) =>
      transform != old.transform ||
      garmentImage != old.garmentImage ||
      pose?.timestampMs != old.pose?.timestampMs ||
      segmentationMask != old.segmentationMask ||
      segmentationImage != old.segmentationImage ||
      maskAgeFrames != old.maskAgeFrames;
}
