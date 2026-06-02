import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import '../core/pipeline_types.dart';

// Stubbed — tflite_flutter removed (crashes on iOS 26 beta)
class BodySegmenter {
  bool get isInitialized => false;
  Future<void> initialize({String modelPath = 'assets/ml_models/selfie_segmentation.tflite'}) async {}
  Future<SegmentationMask?> segment(CameraImage image) async => null;
  Future<ui.Image?> maskToImage(SegmentationMask mask) async => null;
  void dispose() {}
}
