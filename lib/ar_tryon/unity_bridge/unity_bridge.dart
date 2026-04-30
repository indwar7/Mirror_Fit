import 'dart:convert';
import 'package:flutter/foundation.dart';

/// Bridge between Flutter and Unity for 3D garment rendering.
///
/// ARCHITECTURE:
///
/// ┌──────────────────┐         ┌──────────────────┐
/// │     Flutter       │  JSON   │      Unity        │
/// │  (Pose + Fitting) │ ──────▶ │  (3D Rendering)   │
/// │                   │         │                   │
/// │  - Camera capture │         │  - Load .glb mesh │
/// │  - Pose detection │         │  - Apply transform│
/// │  - Body estimation│         │  - Render cloth   │
/// │  - Fitting calc   │         │  - Occlusion mask │
/// └──────────────────┘         └──────────────────┘
///         │                            │
///         └──────────┬─────────────────┘
///                    ▼
///             Final Composite
///              (Screen Output)
///
/// SETUP STEPS:
///
/// 1. Create Unity project with AR Foundation
/// 2. Add flutter_unity_widget package to Flutter
/// 3. Export Unity as Android Library / iOS Framework
/// 4. Unity C# script receives pose data via message passing
/// 5. Unity renders 3D cloth model with received transforms
///
/// MESSAGE FORMAT (Flutter → Unity):
/// {
///   "type": "update_transform",
///   "model": "tshirt_001.glb",
///   "transform": {
///     "position": {"x": 0.5, "y": 0.3, "z": 0},
///     "rotation": {"x": 0, "y": 15, "z": 0},
///     "scale": {"x": 1.2, "y": 1.1, "z": 1.0}
///   },
///   "landmarks": {
///     "left_shoulder": {"x": 0.3, "y": 0.25},
///     "right_shoulder": {"x": 0.7, "y": 0.24},
///     ...
///   }
/// }
///
/// MESSAGE FORMAT (Unity → Flutter):
/// {
///   "type": "render_complete",
///   "frame_id": 42,
///   "render_time_ms": 8
/// }
///
/// NOTE: flutter_unity_widget embeds a Unity view inside Flutter.
/// Install: flutter pub add flutter_unity_widget
/// Requires Unity Editor to build the Unity project first.

/// Represents a message sent to Unity for garment rendering.
class UnityGarmentMessage {
  final String type;
  final String modelPath;
  final UnityTransform transform;
  final Map<String, UnityPoint>? landmarks;

  const UnityGarmentMessage({
    required this.type,
    required this.modelPath,
    required this.transform,
    this.landmarks,
  });

  String toJson() => jsonEncode({
        'type': type,
        'model': modelPath,
        'transform': transform.toMap(),
        if (landmarks != null)
          'landmarks': landmarks!.map((k, v) => MapEntry(k, v.toMap())),
      });
}

class UnityTransform {
  final UnityPoint position;
  final UnityPoint rotation;
  final UnityPoint scale;

  const UnityTransform({
    required this.position,
    required this.rotation,
    required this.scale,
  });

  Map<String, dynamic> toMap() => {
        'position': position.toMap(),
        'rotation': rotation.toMap(),
        'scale': scale.toMap(),
      };
}

class UnityPoint {
  final double x, y, z;
  const UnityPoint(this.x, this.y, this.z);

  Map<String, double> toMap() => {'x': x, 'y': y, 'z': z};
}

/// Unity bridge controller.
///
/// This class manages communication with the Unity rendering engine
/// embedded via flutter_unity_widget.
///
/// USAGE:
/// ```dart
/// // In your widget
/// UnityWidget(
///   onUnityCreated: (controller) {
///     _unityBridge = UnityBridgeController(controller);
///   },
///   onUnityMessage: (message) {
///     _unityBridge.handleMessage(message);
///   },
/// )
///
/// // Each frame:
/// _unityBridge.updateGarmentTransform(
///   modelPath: 'tshirt.glb',
///   x: transform.x, y: transform.y,
///   scaleX: transform.scaleX, scaleY: transform.scaleY,
///   rotationY: transform.rotationDegrees,
/// );
/// ```
class UnityBridgeController {
  // The actual UnityWidgetController from flutter_unity_widget
  // Type is dynamic to avoid hard dependency when Unity isn't set up
  final dynamic _unityController;
  bool _isReady = false;
  // ignore: unused_field
  int _frameId = 0;

  UnityBridgeController(this._unityController);

  bool get isReady => _isReady;

  /// Load a 3D garment model in Unity.
  void loadModel(String modelPath) {
    _sendMessage('load_model', {'model': modelPath});
  }

  /// Update garment transform each frame.
  void updateGarmentTransform({
    required String modelPath,
    required double x,
    required double y,
    required double scaleX,
    required double scaleY,
    required double rotationY,
  }) {
    _frameId++;
    final message = UnityGarmentMessage(
      type: 'update_transform',
      modelPath: modelPath,
      transform: UnityTransform(
        position: UnityPoint(x, y, 0),
        rotation: UnityPoint(0, rotationY, 0),
        scale: UnityPoint(scaleX, scaleY, 1.0),
      ),
    );

    _sendRawMessage(message.toJson());
  }

  /// Update occlusion mask in Unity.
  void updateOcclusionMask(List<int> maskData, int width, int height) {
    _sendMessage('update_occlusion', {
      'mask': base64Encode(maskData),
      'width': width,
      'height': height,
    });
  }

  /// Handle messages from Unity.
  void handleMessage(dynamic message) {
    try {
      final data = message is String ? jsonDecode(message) : message;
      final type = data['type'] as String?;

      switch (type) {
        case 'ready':
          _isReady = true;
          debugPrint('Unity renderer ready');
          break;
        case 'render_complete':
          final renderTime = data['render_time_ms'] as int?;
          if (renderTime != null && renderTime > 16) {
            debugPrint('Unity render slow: ${renderTime}ms');
          }
          break;
        case 'error':
          debugPrint('Unity error: ${data['message']}');
          break;
      }
    } catch (e) {
      debugPrint('Error parsing Unity message: $e');
    }
  }

  void _sendMessage(String method, Map<String, dynamic> data) {
    _sendRawMessage(jsonEncode({'type': method, ...data}));
  }

  void _sendRawMessage(String json) {
    try {
      // flutter_unity_widget's postMessage method
      _unityController?.postMessage(
        'GarmentController',  // Unity GameObject name
        'OnFlutterMessage',   // Unity method name
        json,
      );
    } catch (e) {
      debugPrint('Failed to send message to Unity: $e');
    }
  }

  /// Dispose Unity resources.
  void dispose() {
    _sendMessage('dispose', {});
  }
}

/// ─────────────────────────────────────────────────────────────────────────────
/// UNITY C# SCRIPT (place in Unity project)
/// ─────────────────────────────────────────────────────────────────────────────
///
/// Create a Unity script called GarmentController.cs:
///
/// ```csharp
/// using UnityEngine;
/// using System;
///
/// [Serializable]
/// public class TransformData {
///     public Vector3Data position;
///     public Vector3Data rotation;
///     public Vector3Data scale;
/// }
///
/// [Serializable]
/// public class Vector3Data {
///     public float x, y, z;
/// }
///
/// [Serializable]
/// public class GarmentMessage {
///     public string type;
///     public string model;
///     public TransformData transform;
/// }
///
/// public class GarmentController : MonoBehaviour {
///     private GameObject currentModel;
///     private Camera arCamera;
///
///     void Start() {
///         arCamera = Camera.main;
///         // Notify Flutter that Unity is ready
///         UnityMessageManager.Instance.SendMessageToFlutter(
///             "{\"type\":\"ready\"}"
///         );
///     }
///
///     // Called by Flutter via flutter_unity_widget
///     public void OnFlutterMessage(string message) {
///         try {
///             var data = JsonUtility.FromJson<GarmentMessage>(message);
///
///             switch (data.type) {
///                 case "load_model":
///                     LoadModel(data.model);
///                     break;
///                 case "update_transform":
///                     UpdateTransform(data.transform);
///                     break;
///                 case "dispose":
///                     Cleanup();
///                     break;
///             }
///         } catch (Exception e) {
///             Debug.LogError($"Parse error: {e.Message}");
///         }
///     }
///
///     void LoadModel(string modelPath) {
///         // Load .glb at runtime using GLTFUtility or UnityGLTF
///         // var model = Importer.LoadFromFile(modelPath);
///         // currentModel = model;
///     }
///
///     void UpdateTransform(TransformData t) {
///         if (currentModel == null) return;
///
///         // Convert 2D screen coordinates to 3D world position
///         var screenPos = new Vector3(
///             t.position.x * Screen.width,
///             t.position.y * Screen.height,
///             2f // Distance from camera
///         );
///         var worldPos = arCamera.ScreenToWorldPoint(screenPos);
///
///         currentModel.transform.position = worldPos;
///         currentModel.transform.eulerAngles = new Vector3(
///             t.rotation.x, t.rotation.y, t.rotation.z
///         );
///         currentModel.transform.localScale = new Vector3(
///             t.scale.x, t.scale.y, t.scale.z
///         );
///
///         // Send render timing back
///         var startTime = Time.realtimeSinceStartup;
///         // ... render happens automatically in Unity's loop
///         var renderTime = (Time.realtimeSinceStartup - startTime) * 1000;
///         UnityMessageManager.Instance.SendMessageToFlutter(
///             $"{{\"type\":\"render_complete\",\"render_time_ms\":{renderTime}}}"
///         );
///     }
///
///     void Cleanup() {
///         if (currentModel != null) {
///             Destroy(currentModel);
///         }
///     }
/// }
/// ```
