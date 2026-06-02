import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

/// Handles loading and preparing 3D garment models (.glb) for rendering.
///
/// BLENDER EXPORT SETTINGS FOR MOBILE:
///
/// 1. MODEL PREPARATION IN BLENDER:
///    - Target: < 50,000 polygons per garment
///    - Use Decimate modifier to reduce poly count
///    - Remove interior faces (not visible on mobile)
///    - Merge vertices by distance (0.001m threshold)
///    - Apply all modifiers before export
///
/// 2. EXPORT FORMAT: glTF Binary (.glb)
///    File → Export → glTF 2.0 (.glb/.gltf)
///    Settings:
///      Format:        glTF Binary (.glb) ← single file, faster loading
///      Include:       ☑ Selected Objects only
///      Transform:     ☑ +Y Up (standard for glTF)
///      Mesh:          ☑ Apply Modifiers
///                     ☑ UVs
///                     ☑ Normals
///                     ☐ Tangents (skip for mobile)
///                     ☐ Vertex Colors (unless needed)
///      Material:      ☑ Materials (PBR)
///                     Image format: JPEG (smaller) or PNG (if transparency)
///      Animation:     ☐ (skip unless cloth simulation)
///      Compression:   ☑ Draco mesh compression (reduces file 60-80%)
///
/// 3. TEXTURE OPTIMIZATION:
///    - Max texture size: 1024×1024 (2048 only if critical)
///    - Use JPEG for opaque textures
///    - Use PNG only for transparent textures
///    - Bake complex materials to simple PBR maps
///    - Maps needed: Base Color, Normal (optional), Roughness (optional)
///
/// 4. FILE SIZE TARGETS:
///    - Simple garment (t-shirt): 200KB-500KB
///    - Complex garment (jacket with details): 500KB-2MB
///    - Full outfit with textures: 1MB-3MB
///
/// 5. POLYGON BUDGET:
///    ┌─────────────────┬──────────────┐
///    │ Garment Type     │ Target Polys │
///    ├─────────────────┼──────────────┤
///    │ T-Shirt          │ 5K - 15K     │
///    │ Button Shirt     │ 10K - 25K    │
///    │ Jacket           │ 15K - 35K    │
///    │ Dress            │ 15K - 40K    │
///    │ Full Suit        │ 25K - 50K    │
///    └─────────────────┴──────────────┘
class ModelLoader {
  /// Copy a .glb model from assets to a temporary file for rendering.
  ///
  /// model_viewer_plus and flutter_unity_widget need file:// or http:// URLs,
  /// not asset paths. This copies the asset to temp storage.
  static Future<String> loadFromAsset(String assetPath) async {
    try {
      final data = await rootBundle.load(assetPath);
      final dir = await getApplicationDocumentsDirectory();
      final fileName = assetPath.split('/').last;
      final file = File('${dir.path}/models/$fileName');

      // Create directory if needed
      await file.parent.create(recursive: true);

      // Write only if file doesn't exist or size changed
      if (!await file.exists() || await file.length() != data.lengthInBytes) {
        await file.writeAsBytes(
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
        );
      }

      return file.path;
    } catch (e) {
      debugPrint('Failed to load model from asset: $e');
      rethrow;
    }
  }

  /// Load a .glb model from local file path.
  static Future<String> loadFromFile(String filePath) async {
    final file = File(filePath);
    if (!await file.exists()) {
      throw FileSystemException('Model file not found', filePath);
    }

    // Validate file extension
    if (!filePath.endsWith('.glb') &&
        !filePath.endsWith('.gltf') &&
        !filePath.endsWith('.obj')) {
      throw FormatException('Unsupported model format. Use .glb, .gltf, or .obj');
    }

    // Check file size (warn if > 5MB)
    final size = await file.length();
    if (size > 5 * 1024 * 1024) {
      debugPrint(
        'WARNING: Model file is ${(size / 1024 / 1024).toStringAsFixed(1)}MB. '
        'Consider optimizing for mobile (target < 3MB).',
      );
    }

    return filePath;
  }

  /// Get model info for debugging.
  static Future<Map<String, dynamic>> getModelInfo(String filePath) async {
    final file = File(filePath);
    final stat = await file.stat();

    return {
      'path': filePath,
      'size_bytes': stat.size,
      'size_mb': (stat.size / 1024 / 1024).toStringAsFixed(2),
      'format': filePath.split('.').last.toUpperCase(),
      'modified': stat.modified.toIso8601String(),
    };
  }

  /// List all available garment models in the models directory.
  static Future<List<String>> listAvailableModels() async {
    final dir = await getApplicationDocumentsDirectory();
    final modelsDir = Directory('${dir.path}/models');

    if (!await modelsDir.exists()) return [];

    final files = await modelsDir
        .list()
        .where((f) =>
            f.path.endsWith('.glb') ||
            f.path.endsWith('.gltf') ||
            f.path.endsWith('.obj'))
        .map((f) => f.path)
        .toList();

    return files;
  }
}
