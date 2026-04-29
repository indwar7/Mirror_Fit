import 'package:camera/camera.dart';
import '../core/pipeline_types.dart';

// Stubbed — google_mlkit_pose_detection removed (crashes on iOS 26 beta)
class PoseDetectorService {
  void initialize({int sensorOrientation = 270, bool isFrontCamera = true}) {}
  Future<void> prewarm(CameraImage firstFrame) async {}
  PoseResult? predictPose() => null;
  Future<PoseResult?> detectPose(CameraImage image) async => null;
  Future<void> dispose() async {}
}
