import 'package:flutter/material.dart';
import '../core/pipeline_types.dart';

/// Debug overlay showing real-time pipeline metrics.
class DebugOverlay extends StatelessWidget {
  final double fps;
  final double processingTimeMs;
  final bool poseDetected;
  final bool segmentationActive;
  final int processedFrames;
  final PoseResult? pose;

  const DebugOverlay({
    super.key,
    required this.fps,
    required this.processingTimeMs,
    required this.poseDetected,
    required this.segmentationActive,
    required this.processedFrames,
    this.pose,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 100,
      left: 8,
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.7),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            _header('PIPELINE DEBUG'),
            const SizedBox(height: 6),
            _row('FPS', fps.toStringAsFixed(1),
                color: fps >= 25
                    ? Colors.greenAccent
                    : fps >= 15
                        ? Colors.orangeAccent
                        : Colors.redAccent),
            _row('Process', '${processingTimeMs.toStringAsFixed(1)}ms'),
            _row('Frames', '$processedFrames'),
            _row('Pose', poseDetected ? 'YES' : 'NO',
                color: poseDetected ? Colors.greenAccent : Colors.redAccent),
            _row('Segment', segmentationActive ? 'ON' : 'OFF',
                color: segmentationActive
                    ? Colors.greenAccent
                    : Colors.grey),

            if (pose != null && pose!.isValid) ...[
              const SizedBox(height: 8),
              _header('BODY METRICS'),
              const SizedBox(height: 4),
              if (pose!.leftShoulder != null && pose!.rightShoulder != null)
                _row('Shoulder W',
                    '${pose!.leftShoulder!.distanceTo2D(pose!.rightShoulder!).toStringAsFixed(0)}px'),
              if (pose!.shoulderCenter != null && pose!.hipCenter != null)
                _row('Torso H',
                    '${(pose!.hipCenter! - pose!.shoulderCenter!).distance.toStringAsFixed(0)}px'),
              if (pose!.leftHip != null && pose!.rightHip != null)
                _row('Hip W',
                    '${pose!.leftHip!.distanceTo2D(pose!.rightHip!).toStringAsFixed(0)}px'),
            ],
          ],
        ),
      ),
    );
  }

  Widget _header(String text) {
    return Text(
      text,
      style: const TextStyle(
        color: Colors.white70,
        fontSize: 10,
        fontWeight: FontWeight.bold,
        letterSpacing: 1.5,
      ),
    );
  }

  Widget _row(String label, String value, {Color color = Colors.white}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 2),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 70,
            child: Text(
              label,
              style: const TextStyle(color: Colors.white54, fontSize: 11),
            ),
          ),
          Text(
            value,
            style: TextStyle(
              color: color,
              fontSize: 11,
              fontWeight: FontWeight.bold,
              fontFamily: 'monospace',
            ),
          ),
        ],
      ),
    );
  }
}
