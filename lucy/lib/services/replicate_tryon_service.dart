import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

/// Virtual try-on using Replicate's IDM-VTON model.
/// Reliable, fast (~20 sec), $0.025/image.
class ReplicateTryOnService {
  static const _apiUrl = 'https://api.replicate.com/v1/predictions';
  static const _apiToken = 'r8_6WzyWiVBzsg2pZxV7NjVUnsIAi5tLAA1CwIMd';
  static const _modelVersion =
      '0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985';

  /// Run virtual try-on.
  /// [personImage] — photo of the person (File)
  /// [garmentImage] — photo of the garment (File)
  /// Returns a local File with the result image.
  static Future<File> tryOn({
    required File personImage,
    required File garmentImage,
    String garmentDescription = 'A jacket',
    String category = 'upper_body',
    void Function(double progress, String message)? onProgress,
  }) async {
    onProgress?.call(0.05, 'Uploading photos...');

    // Step 1: Upload images to Replicate's file hosting
    final personUrl = await _uploadFile(personImage);
    final garmentUrl = await _uploadFile(garmentImage);
    debugPrint('[REPLICATE] Person URL: $personUrl');
    debugPrint('[REPLICATE] Garment URL: $garmentUrl');

    onProgress?.call(0.15, 'Starting AI try-on...');

    // Step 2: Create prediction
    final response = await http.post(
      Uri.parse(_apiUrl),
      headers: {
        'Authorization': 'Bearer $_apiToken',
        'Content-Type': 'application/json',
        'Prefer': 'wait',
      },
      body: jsonEncode({
        'version': _modelVersion,
        'input': {
          'human_img': personUrl,
          'garm_img': garmentUrl,
          'garment_des': garmentDescription,
          'category': category,
          'crop': true,
          'steps': 30,
          'seed': 42,
        },
      }),
    ).timeout(
      const Duration(seconds: 30),
      onTimeout: () => throw Exception('Connection timeout'),
    );

    if (response.statusCode != 201) {
      debugPrint('[REPLICATE] Create failed: ${response.statusCode} ${response.body}');
      throw Exception('Failed to start AI (${response.statusCode}): ${response.body}');
    }

    final prediction = jsonDecode(response.body);
    final predictionId = prediction['id'] as String;
    final getUrl = prediction['urls']?['get'] as String? ??
        '$_apiUrl/$predictionId';
    debugPrint('[REPLICATE] Prediction created: $predictionId');

    onProgress?.call(0.3, 'AI is generating your look...');

    // Step 3: Poll for result
    String? outputUrl;
    for (int i = 0; i < 120; i++) {
      await Future.delayed(const Duration(seconds: 2));

      final pollResponse = await http.get(
        Uri.parse(getUrl),
        headers: {'Authorization': 'Bearer $_apiToken'},
      );

      if (pollResponse.statusCode != 200) {
        debugPrint('[REPLICATE] Poll error: ${pollResponse.statusCode}');
        continue;
      }

      final data = jsonDecode(pollResponse.body);
      final status = data['status'] as String;
      debugPrint('[REPLICATE] Status: $status');

      if (status == 'succeeded') {
        final output = data['output'];
        if (output is String) {
          outputUrl = output;
        } else if (output is List && output.isNotEmpty) {
          outputUrl = output[0] as String;
        }
        break;
      }

      if (status == 'failed' || status == 'canceled') {
        final error = data['error'] ?? 'Unknown error';
        throw Exception('AI generation failed: $error');
      }

      // Update progress
      final elapsed = i * 2;
      if (elapsed < 20) {
        onProgress?.call(0.3 + (elapsed / 20) * 0.4, 'AI is working... ${elapsed}s');
      } else {
        onProgress?.call(0.7 + (elapsed / 60) * 0.2, 'Almost done... ${elapsed}s');
      }
    }

    if (outputUrl == null) {
      throw Exception('AI generation timed out');
    }

    onProgress?.call(0.9, 'Downloading result...');

    // Step 4: Download result image
    debugPrint('[REPLICATE] Downloading: $outputUrl');
    final imageResponse = await http.get(Uri.parse(outputUrl)).timeout(
      const Duration(seconds: 30),
    );

    if (imageResponse.statusCode != 200) {
      throw Exception('Failed to download result (${imageResponse.statusCode})');
    }

    final dir = await getTemporaryDirectory();
    final file = File(
        '${dir.path}/lucy_tryon_${DateTime.now().millisecondsSinceEpoch}.jpg');
    await file.writeAsBytes(imageResponse.bodyBytes);

    onProgress?.call(1.0, 'Your look is ready!');
    debugPrint('[REPLICATE] Success! Saved to ${file.path}');
    return file;
  }

  /// Upload a file to Replicate's file hosting using multipart form.
  static Future<String> _uploadFile(File file) async {
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('https://api.replicate.com/v1/files'),
    );
    request.headers['Authorization'] = 'Bearer $_apiToken';
    request.files.add(
      await http.MultipartFile.fromPath('content', file.path),
    );

    final streamedResponse = await request.send().timeout(
      const Duration(seconds: 120),
      onTimeout: () => throw Exception('Upload timeout'),
    );

    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode != 201) {
      debugPrint('[REPLICATE] Upload failed: ${response.statusCode} ${response.body}');
      throw Exception('Upload failed (${response.statusCode}): ${response.body}');
    }

    final data = jsonDecode(response.body);
    final url = data['urls']?['get'] as String?;
    if (url == null) {
      throw Exception('No URL returned from upload');
    }
    debugPrint('[REPLICATE] Uploaded: $url');
    return url;
  }
}
