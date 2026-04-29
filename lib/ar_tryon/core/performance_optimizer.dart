import 'dart:collection';

/// Performance optimization utilities for maintaining 30 FPS.
///
/// OPTIMIZATION STRATEGIES:
///
/// 1. GPU DELEGATE FOR TFLITE
///    - Android: GpuDelegateV2 (OpenCL) — 3-5x faster than CPU
///    - iOS: GpuDelegate (Metal) — 4-8x faster than CPU
///    - Fallback: NNAPI delegate on Android, CoreML on iOS
///
/// 2. FRAME DOWNSCALING
///    Camera captures at 480p (640×480) but ML models expect 256×256.
///    Downscaling before inference saves significant time.
///    ┌────────────────────────────────────────────────┐
///    │ Resolution   │ Pose Det  │ Segment  │ Total    │
///    ├────────────────────────────────────────────────┤
///    │ 1080p        │ ~45ms     │ ~35ms    │ ~80ms    │
///    │ 720p         │ ~25ms     │ ~20ms    │ ~45ms    │
///    │ 480p         │ ~15ms     │ ~12ms    │ ~27ms ✓  │
///    │ 360p         │ ~10ms     │ ~8ms     │ ~18ms    │
///    └────────────────────────────────────────────────┘
///
/// 3. FRAME DROPPING
///    If pipeline takes > 33ms, skip next frame rather than queuing.
///    This prevents latency accumulation.
///
/// 4. ADAPTIVE QUALITY
///    Monitor FPS and automatically adjust:
///    - Drop segmentation if FPS < 25
///    - Drop occlusion if FPS < 20
///    - Reduce pose model to "base" if FPS < 15
///
/// 5. ISOLATE THREADING
///    Heavy computations (image preprocessing) can run in isolates.
///    However, TFLite and MLKit use their own thread pools, so
///    additional isolates are mainly for image format conversion.
///
/// 6. MEMORY MANAGEMENT
///    - Reuse input/output buffers (avoid allocation per frame)
///    - Release CameraImage immediately after processing
///    - Limit segmentation mask history to 3 frames

/// Tracks frame timings and adapts quality settings.
class PerformanceMonitor {
  final Queue<FrameTiming> _timings = Queue();
  static const int _windowSize = 30; // Sliding window of 30 frames

  // Thresholds for quality adjustment
  static const double _targetFps = 30.0;
  static const double _minAcceptableFps = 20.0;
  static const double _criticalFps = 15.0;

  // Current quality level
  QualityLevel _currentQuality = QualityLevel.high;
  QualityLevel get currentQuality => _currentQuality;

  /// Record a frame's processing time.
  void recordFrame({
    required int poseDetectionMs,
    required int segmentationMs,
    required int fittingMs,
    required int renderingMs,
  }) {
    final timing = FrameTiming(
      poseDetectionMs: poseDetectionMs,
      segmentationMs: segmentationMs,
      fittingMs: fittingMs,
      renderingMs: renderingMs,
      totalMs: poseDetectionMs + segmentationMs + fittingMs + renderingMs,
      timestamp: DateTime.now(),
    );

    _timings.add(timing);
    if (_timings.length > _windowSize) {
      _timings.removeFirst();
    }

    _adaptQuality();
  }

  /// Get current average FPS over the window.
  double get averageFps {
    if (_timings.length < 2) return 30.0;
    final avgMs = _timings.fold<int>(0, (sum, t) => sum + t.totalMs) /
        _timings.length;
    return avgMs > 0 ? 1000.0 / avgMs : 30.0;
  }

  /// Get average processing time breakdown.
  Map<String, double> get averageTimings {
    if (_timings.isEmpty) return {};
    final n = _timings.length;
    return {
      'pose_ms': _timings.fold<int>(0, (s, t) => s + t.poseDetectionMs) / n,
      'segment_ms': _timings.fold<int>(0, (s, t) => s + t.segmentationMs) / n,
      'fitting_ms': _timings.fold<int>(0, (s, t) => s + t.fittingMs) / n,
      'rendering_ms': _timings.fold<int>(0, (s, t) => s + t.renderingMs) / n,
      'total_ms': _timings.fold<int>(0, (s, t) => s + t.totalMs) / n,
    };
  }

  /// Automatically adjust quality based on FPS.
  void _adaptQuality() {
    final fps = averageFps;

    if (fps >= _targetFps) {
      _currentQuality = QualityLevel.high;
    } else if (fps >= _minAcceptableFps) {
      _currentQuality = QualityLevel.medium;
    } else if (fps >= _criticalFps) {
      _currentQuality = QualityLevel.low;
    } else {
      _currentQuality = QualityLevel.minimal;
    }
  }

  /// Get current quality recommendations.
  QualitySettings getRecommendedSettings() {
    switch (_currentQuality) {
      case QualityLevel.high:
        return const QualitySettings(
          enableSegmentation: true,
          enableOcclusion: true,
          poseModelAccurate: true,
          inputResolution: 480,
          smoothingFrames: 5,
        );
      case QualityLevel.medium:
        return const QualitySettings(
          enableSegmentation: true,
          enableOcclusion: false,
          poseModelAccurate: false,
          inputResolution: 480,
          smoothingFrames: 3,
        );
      case QualityLevel.low:
        return const QualitySettings(
          enableSegmentation: false,
          enableOcclusion: false,
          poseModelAccurate: false,
          inputResolution: 360,
          smoothingFrames: 3,
        );
      case QualityLevel.minimal:
        return const QualitySettings(
          enableSegmentation: false,
          enableOcclusion: false,
          poseModelAccurate: false,
          inputResolution: 240,
          smoothingFrames: 2,
        );
    }
  }
}

enum QualityLevel { high, medium, low, minimal }

class QualitySettings {
  final bool enableSegmentation;
  final bool enableOcclusion;
  final bool poseModelAccurate;
  final int inputResolution;
  final int smoothingFrames;

  const QualitySettings({
    required this.enableSegmentation,
    required this.enableOcclusion,
    required this.poseModelAccurate,
    required this.inputResolution,
    required this.smoothingFrames,
  });
}

class FrameTiming {
  final int poseDetectionMs;
  final int segmentationMs;
  final int fittingMs;
  final int renderingMs;
  final int totalMs;
  final DateTime timestamp;

  const FrameTiming({
    required this.poseDetectionMs,
    required this.segmentationMs,
    required this.fittingMs,
    required this.renderingMs,
    required this.totalMs,
    required this.timestamp,
  });
}

/// Frame rate limiter to cap processing at target FPS.
class FrameRateLimiter {
  final int targetFps;
  DateTime _lastFrameTime = DateTime.now();

  FrameRateLimiter({this.targetFps = 30});

  /// Returns true if enough time has passed for the next frame.
  bool shouldProcessFrame() {
    final now = DateTime.now();
    final elapsed = now.difference(_lastFrameTime).inMilliseconds;
    final frameInterval = 1000 ~/ targetFps;

    if (elapsed >= frameInterval) {
      _lastFrameTime = now;
      return true;
    }
    return false;
  }
}
