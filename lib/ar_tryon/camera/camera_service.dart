import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Camera service for real-time frame capture.
///
/// Targets 30 FPS with resolution downscaling for ML inference.
/// Uses front camera by default for try-on.
class CameraService {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  bool _isStreaming = false;

  // Frame processing callback
  void Function(CameraImage image)? _onFrame;

  // Performance tracking
  int _frameCount = 0;
  DateTime? _fpsStartTime;
  double _currentFps = 0;

  // Downscaling
  static const int targetWidth = 480;
  static const int targetHeight = 640;

  CameraController? get controller => _controller;
  bool get isInitialized => _isInitialized;
  bool get isStreaming => _isStreaming;
  double get currentFps => _currentFps;
  int get sensorOrientation => _controller?.description.sensorOrientation ?? 270;

  /// Initialize the camera (front-facing by default).
  Future<void> initialize({bool useFrontCamera = true}) async {
    _cameras = await availableCameras();
    if (_cameras == null || _cameras!.isEmpty) {
      throw CameraException('no_cameras', 'No cameras available on device');
    }

    final camera = _cameras!.firstWhere(
      (c) => c.lensDirection ==
          (useFrontCamera
              ? CameraLensDirection.front
              : CameraLensDirection.back),
      orElse: () => _cameras!.first,
    );

    _controller = CameraController(
      camera,
      // Use medium resolution for balance of quality vs performance.
      // 480p is sufficient for pose detection while keeping FPS high.
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: defaultTargetPlatform == TargetPlatform.iOS
          ? ImageFormatGroup.bgra8888
          : ImageFormatGroup.yuv420,
    );

    await _controller!.initialize();

    // Lock orientation to portrait for consistent landmark mapping
    await _controller!.lockCaptureOrientation(DeviceOrientation.portraitUp);

    _isInitialized = true;
  }

  /// Start streaming camera frames for processing.
  ///
  /// [onFrame] is called for each camera frame. Process quickly
  /// or drop frames to maintain 30 FPS.
  Future<void> startFrameStream(void Function(CameraImage image) onFrame) async {
    if (!_isInitialized || _controller == null) {
      throw StateError('Camera not initialized. Call initialize() first.');
    }
    if (_isStreaming) return;

    _onFrame = onFrame;
    _frameCount = 0;
    _fpsStartTime = DateTime.now();

    await _controller!.startImageStream((CameraImage image) {
      _frameCount++;
      _updateFps();

      // Call the frame processor
      _onFrame?.call(image);
    });

    _isStreaming = true;
  }

  /// Stop the frame stream.
  Future<void> stopFrameStream() async {
    if (!_isStreaming || _controller == null) return;
    await _controller!.stopImageStream();
    _isStreaming = false;
    _onFrame = null;
  }

  /// Toggle between front and back camera.
  Future<void> switchCamera() async {
    if (_cameras == null || _cameras!.length < 2) return;

    final wasStreaming = _isStreaming;
    final callback = _onFrame;

    if (wasStreaming) await stopFrameStream();

    final currentDirection = _controller!.description.lensDirection;
    final newDirection = currentDirection == CameraLensDirection.front
        ? CameraLensDirection.back
        : CameraLensDirection.front;

    await _controller!.dispose();

    final newCamera = _cameras!.firstWhere(
      (c) => c.lensDirection == newDirection,
      orElse: () => _cameras!.first,
    );

    _controller = CameraController(
      newCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: defaultTargetPlatform == TargetPlatform.iOS
          ? ImageFormatGroup.bgra8888
          : ImageFormatGroup.yuv420,
    );

    await _controller!.initialize();
    await _controller!.lockCaptureOrientation(DeviceOrientation.portraitUp);

    if (wasStreaming && callback != null) {
      await startFrameStream(callback);
    }
  }

  void _updateFps() {
    final now = DateTime.now();
    if (_fpsStartTime != null) {
      final elapsed = now.difference(_fpsStartTime!).inMilliseconds;
      if (elapsed >= 1000) {
        _currentFps = _frameCount * 1000.0 / elapsed;
        _frameCount = 0;
        _fpsStartTime = now;
      }
    }
  }

  /// Convert CameraImage to bytes for ML processing.
  /// Returns Y plane (grayscale) for pose detection — fast and sufficient.
  static Uint8List extractYPlane(CameraImage image) {
    if (image.format.group == ImageFormatGroup.yuv420) {
      // YUV420: first plane is Y (luminance)
      return image.planes[0].bytes;
    } else if (image.format.group == ImageFormatGroup.bgra8888) {
      // iOS BGRA: convert to grayscale, respecting bytesPerRow stride
      final bytes = image.planes[0].bytes;
      final bytesPerRow = image.planes[0].bytesPerRow;
      final w = image.width;
      final h = image.height;
      final gray = Uint8List(w * h);
      for (int row = 0; row < h; row++) {
        final rowOffset = row * bytesPerRow;
        for (int col = 0; col < w; col++) {
          final px = rowOffset + col * 4;
          if (px + 2 >= bytes.length) break;
          final b = bytes[px];
          final g = bytes[px + 1];
          final r = bytes[px + 2];
          gray[row * w + col] = ((r + g + g + b) >> 2).clamp(0, 255);
        }
      }
      return gray;
    }
    return image.planes[0].bytes;
  }

  /// Get NV21 bytes from CameraImage for MLKit processing.
  static Uint8List? getNv21Bytes(CameraImage image) {
    if (image.format.group == ImageFormatGroup.yuv420) {
      final yPlane = image.planes[0];
      final uPlane = image.planes[1];
      final vPlane = image.planes[2];

      final ySize = yPlane.bytes.length;
      final uvSize = uPlane.bytes.length;

      final nv21 = Uint8List(ySize + uvSize * 2);

      // Copy Y plane
      nv21.setRange(0, ySize, yPlane.bytes);

      // Interleave V and U planes for NV21
      int offset = ySize;
      for (int i = 0; i < uvSize; i++) {
        nv21[offset++] = vPlane.bytes[i];
        nv21[offset++] = uPlane.bytes[i];
      }

      return nv21;
    }
    return null;
  }

  /// Dispose camera resources.
  Future<void> dispose() async {
    if (_isStreaming) await stopFrameStream();
    await _controller?.dispose();
    _controller = null;
    _isInitialized = false;
  }
}
