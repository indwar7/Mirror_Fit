import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../core/tryon_pipeline.dart';
import '../fitting/garment_fitter.dart';
import '../rendering/garment_renderer.dart';
import '../compositing/photo_compositor.dart';
import '../widgets/garment_selector_bar.dart';
import 'tryon_photo_screen.dart';

/// Real-time AR Try-On — Snapchat-style instant garment overlay.
class ARTryOnScreen extends StatefulWidget {
  final String? initialModelPath;
  final GarmentType garmentType;

  const ARTryOnScreen({
    super.key,
    this.initialModelPath,
    this.garmentType = GarmentType.top,
  });

  @override
  State<ARTryOnScreen> createState() => _ARTryOnScreenState();
}

class _ARTryOnScreenState extends State<ARTryOnScreen>
    with WidgetsBindingObserver {
  late TryOnPipeline _pipeline;
  bool _isInitialized = false;
  bool _isDisposed = false;
  String? _errorMessage;
  ui.Image? _garmentImage;
  bool _garmentLoading = false;
  String _currentGarmentAsset = 'assets/images/garment_jacket_dark.png';
  String _currentGarmentName = 'Jacket';
  bool _isCapturing = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _pipeline = TryOnPipeline();
    _loadThenStart();
  }

  /// Load garment → init camera → start. Garment is ready before first frame.
  Future<void> _loadThenStart() async {
    setState(() => _garmentLoading = true);
    debugPrint('[AR-SCREEN] Loading garment: $_currentGarmentAsset');
    try {
      final data = await rootBundle.load(_currentGarmentAsset);
      debugPrint('[AR-SCREEN] Asset loaded, ${data.lengthInBytes} bytes');
      final codec = await ui.instantiateImageCodec(
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
      );
      final frame = await codec.getNextFrame();
      if (!mounted) return;
      _garmentImage = frame.image;
      debugPrint('[AR-SCREEN] Garment image: ${_garmentImage!.width}x${_garmentImage!.height}');
      setState(() => _garmentLoading = false);
    } catch (e) {
      debugPrint('[AR-SCREEN] FAILED to load garment: $e');
      if (mounted) setState(() { _garmentLoading = false; _errorMessage = 'Failed to load garment: $e'; });
      return;
    }

    try {
      debugPrint('[AR-SCREEN] Initializing pipeline...');
      await _pipeline.initialize();
      debugPrint('[AR-SCREEN] Pipeline initialized, setting garment config');
      if (_garmentImage != null) {
        _pipeline.setGarment(GarmentConfig(
          modelPath: _currentGarmentAsset,
          type: GarmentType.top,
          originalWidth: _garmentImage!.width.toDouble(),
          originalHeight: _garmentImage!.height.toDouble(),
          shoulderPadding: 1.5,
          shoulderLineInImage: 0.30,
        ));
      }
      debugPrint('[AR-SCREEN] Starting pipeline...');
      await _pipeline.start();
      debugPrint('[AR-SCREEN] Pipeline started!');
      if (mounted) setState(() => _isInitialized = true);
    } catch (e) {
      debugPrint('[AR-SCREEN] FAILED to start: $e');
      if (mounted) setState(() => _errorMessage = 'Failed to start camera: $e');
    }
  }

  Future<void> _captureAndTryOn() async {
    if (_isCapturing) return;
    final controller = _pipeline.cameraService.controller;
    if (controller == null || !controller.value.isInitialized) return;
    final transform = _pipeline.currentFrame?.garmentTransform;
    if (transform == null || _garmentImage == null) return;

    setState(() => _isCapturing = true);
    try {
      _pipeline.stop();
      final xFile = await controller.takePicture();
      final personBytes = await xFile.readAsBytes();

      // Get preview dimensions for scaling transform to photo space
      final ps = controller.value.previewSize ?? const Size(480, 640);
      final previewW = ps.shortestSide;
      final previewH = ps.longestSide;

      // Composite locally — instant, no network needed
      final resultBytes = await PhotoCompositor.composite(
        photoBytes: personBytes,
        garmentImage: _garmentImage!,
        transform: transform,
        previewWidth: previewW,
        previewHeight: previewH,
      );

      if (!mounted) return;
      setState(() => _isCapturing = false);

      if (resultBytes != null) {
        await Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => TryOnPhotoScreen(
              photoBytes: resultBytes,
              garmentName: _currentGarmentName,
            ),
          ),
        );
      }
      if (mounted && _isInitialized) await _pipeline.start();
    } catch (e) {
      debugPrint('Capture error: $e');
      if (mounted && !_isDisposed) {
        setState(() => _isCapturing = false);
        if (_isInitialized) await _pipeline.start();
      }
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_isDisposed) return;
    if (state == AppLifecycleState.inactive) {
      _pipeline.stop();
    } else if (state == AppLifecycleState.resumed && _isInitialized) {
      _pipeline.start();
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    WidgetsBinding.instance.removeObserver(this);
    _pipeline.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_errorMessage != null) return _buildErrorScreen();
    if (!_isInitialized) return _buildLoadingScreen();

    return Scaffold(
      backgroundColor: Colors.black,
      body: ListenableBuilder(
        listenable: _pipeline,
        builder: (context, _) {
          return Stack(
            fit: StackFit.expand,
            children: [
              _buildCameraWithOverlay(),
              _buildStatusPill(),
              _buildControls(),
            ],
          );
        },
      ),
    );
  }

  /// Camera + garment overlay in same FittedBox for pixel-perfect alignment.
  Widget _buildCameraWithOverlay() {
    if (_isDisposed) return const SizedBox.shrink();
    final controller = _pipeline.cameraService.controller;
    if (controller == null || !controller.value.isInitialized) {
      return const Center(child: CircularProgressIndicator(color: Colors.white));
    }

    final ps = controller.value.previewSize ?? const Size(480, 640);
    final previewW = ps.shortestSide;  // portrait width (narrow side)
    final previewH = ps.longestSide;   // portrait height (tall side)
    _pipeline.setViewport(previewW, previewH);

    return SizedBox.expand(
      child: FittedBox(
        fit: BoxFit.cover,
        clipBehavior: Clip.hardEdge,
        child: SizedBox(
          width: previewW,
          height: previewH,
          child: Stack(
            children: [
              SizedBox(
                width: previewW,
                height: previewH,
                child: CameraPreview(controller),
              ),
              CustomPaint(
                size: Size(previewW, previewH),
                painter: GarmentOverlayPainter(
                  transform: _pipeline.currentFrame?.garmentTransform,
                  garmentImage: _garmentImage,
                  pose: _pipeline.currentPose,
                  viewportSize: Size(previewW, previewH),
                  segmentationMask: _pipeline.currentFrame?.segmentationMask,
                  maskAgeFrames: _pipeline.currentFrame?.maskAgeFrames ?? 999,
                  segmentationImage: _pipeline.currentSegImage,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// Small status pill — "Tracking body" or "Stand back..."
  Widget _buildStatusPill() {
    if (_garmentLoading) {
      return Positioned(
        top: MediaQuery.of(context).padding.top + 60,
        left: 0, right: 0,
        child: Center(
          child: _pill(Colors.black54, 'Loading garment...'),
        ),
      );
    }
    final tracking = _pipeline.currentPose != null;
    return Positioned(
      top: MediaQuery.of(context).padding.top + 60,
      left: 0, right: 0,
      child: Center(
        child: _pill(
          tracking ? Colors.green.withValues(alpha: 0.7) : Colors.orange.withValues(alpha: 0.7),
          tracking ? 'Tracking body' : 'Stand back so camera sees your body',
        ),
      ),
    );
  }

  Widget _pill(Color color, String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(text, style: const TextStyle(color: Colors.white, fontSize: 12)),
    );
  }

  /// Top bar + bottom garment selector + capture button.
  Widget _buildControls() {
    return SafeArea(
      child: Column(
        children: [
          // Top bar
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              children: [
                _circleBtn(Icons.arrow_back_ios_new, () => Navigator.pop(context)),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${_pipeline.fps.toStringAsFixed(0)} FPS',
                    style: TextStyle(
                      color: _pipeline.fps >= 25 ? Colors.greenAccent : Colors.orangeAccent,
                      fontSize: 12, fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                _circleBtn(Icons.flip_camera_ios_rounded, () => _pipeline.cameraService.switchCamera()),
              ],
            ),
          ),

          const Spacer(),

          // Capture button — tap to generate AI try-on
          Padding(
            padding: const EdgeInsets.only(bottom: 16),
            child: GestureDetector(
              onTap: _captureAndTryOn,
              child: Container(
                width: 72,
                height: 72,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white,
                  border: Border.all(color: Colors.white54, width: 3),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.4),
                      blurRadius: 12,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: _isCapturing
                    ? const Padding(
                        padding: EdgeInsets.all(20),
                        child: CircularProgressIndicator(
                          strokeWidth: 3,
                          color: Colors.black,
                        ),
                      )
                    : const Icon(Icons.camera_alt, color: Colors.black, size: 32),
              ),
            ),
          ),

          // Garment selector
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: GarmentSelectorBar(
              onGarmentSelected: (config, image) {
                _pipeline.setGarment(config);
                setState(() {
                  _garmentImage = image;
                  _currentGarmentAsset = config.modelPath;
                  final fileName = config.modelPath.split('/').last;
                  _currentGarmentName = fileName
                      .replaceAll('garment_', '').replaceAll('.png', '')
                      .replaceAll('_', ' ')
                      .split(' ')
                      .map((w) => w.isNotEmpty ? '${w[0].toUpperCase()}${w.substring(1)}' : '')
                      .join(' ');
                });
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _circleBtn(IconData icon, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 40, height: 40,
        decoration: BoxDecoration(
          color: Colors.black54, shape: BoxShape.circle,
          border: Border.all(color: Colors.white24),
        ),
        child: Icon(icon, color: Colors.white, size: 20),
      ),
    );
  }

  Widget _buildLoadingScreen() {
    return const Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: Colors.white),
            SizedBox(height: 16),
            Text('Starting camera...', style: TextStyle(color: Colors.white70, fontSize: 14)),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorScreen() {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, color: Colors.redAccent, size: 48),
              const SizedBox(height: 16),
              Text(_errorMessage ?? 'Unknown error',
                style: const TextStyle(color: Colors.white54, fontSize: 14),
                textAlign: TextAlign.center),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () { setState(() => _errorMessage = null); _loadThenStart(); },
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
