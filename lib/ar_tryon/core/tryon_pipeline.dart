import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import '../camera/camera_service.dart';
import '../fitting/garment_fitter.dart';
import 'pipeline_types.dart';

// Stubbed — Flutter AR pipeline disabled, native ARKit used instead
class TryOnPipeline extends ChangeNotifier {
  final CameraService cameraService = CameraService();

  PipelineFrame? get currentFrame => null;
  PoseResult? get currentPose => null;
  bool get isRunning => false;
  double get fps => 0;
  double get avgProcessingMs => 0;
  int get processedFrames => 0;
  bool get segmentationAvailable => false;
  ui.Image? get currentSegImage => null;

  Future<void> initialize({bool useFrontCamera = true}) async {}
  void setGarment(GarmentConfig config) {}
  void setViewport(double width, double height) {}
  Future<void> start({void Function(CameraImage)? onFrame}) async {}
  void stop() {}

  @override
  void dispose() {
    cameraService.dispose();
    super.dispose();
  }
}
