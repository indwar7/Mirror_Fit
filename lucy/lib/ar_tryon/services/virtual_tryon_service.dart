import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';

/// Calls HuggingFace IDM-VTON API for realistic virtual try-on.
/// FREE — no API key needed.
///
/// Flow:
///   1. Upload person photo + garment photo to HF
///   2. Submit try-on job
///   3. Poll for result
///   4. Return result image URL
class VirtualTryOnService {
  static const _baseUrl = 'https://yisol-idm-vton.hf.space';

  /// Upload an image (bytes) to HuggingFace and get a file path back.
  static Future<String?> _uploadImage(Uint8List imageBytes, String filename) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/upload'),
      );
      request.files.add(http.MultipartFile.fromBytes(
        'files',
        imageBytes,
        filename: filename,
      ));

      final response = await request.send();
      if (response.statusCode == 200) {
        final body = await response.stream.bytesToString();
        final paths = jsonDecode(body) as List;
        if (paths.isNotEmpty) {
          return paths[0] as String;
        }
      }
      debugPrint('[TRYON] Upload failed: ${response.statusCode}');
      return null;
    } catch (e) {
      debugPrint('[TRYON] Upload error: $e');
      return null;
    }
  }

  /// Run virtual try-on: person photo + garment photo → result image URL.
  ///
  /// [personImageBytes] — photo of the person (JPEG/PNG bytes)
  /// [garmentImageBytes] — photo of the garment (JPEG/PNG bytes)
  /// [garmentDescription] — text description of the garment
  ///
  /// Returns the URL of the result image, or null if failed.
  static Future<String?> tryOn({
    required Uint8List personImageBytes,
    required Uint8List garmentImageBytes,
    String garmentDescription = 'Jacket',
    void Function(String status)? onStatus,
  }) async {
    try {
      // Step 1: Upload both images
      onStatus?.call('Uploading photos...');
      debugPrint('[TRYON] Uploading person image (${personImageBytes.length} bytes)...');

      final personPath = await _uploadImage(personImageBytes, 'person.jpg');
      if (personPath == null) {
        onStatus?.call('Upload failed');
        return null;
      }
      debugPrint('[TRYON] Person uploaded: $personPath');

      final garmentPath = await _uploadImage(garmentImageBytes, 'garment.png');
      if (garmentPath == null) {
        onStatus?.call('Upload failed');
        return null;
      }
      debugPrint('[TRYON] Garment uploaded: $garmentPath');

      // Step 2: Submit try-on job
      onStatus?.call('AI generating try-on...');
      debugPrint('[TRYON] Submitting try-on job...');

      final submitResponse = await http.post(
        Uri.parse('$_baseUrl/call/tryon'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'data': [
            {
              'background': {'path': personPath},
              'layers': [],
              'composite': null,
            },
            {'path': garmentPath},
            garmentDescription,
            true,  // auto-mask
            true,  // auto-crop
            30,    // denoise steps
            42,    // seed
          ],
        }),
      );

      if (submitResponse.statusCode != 200) {
        debugPrint('[TRYON] Submit failed: ${submitResponse.statusCode} ${submitResponse.body}');
        onStatus?.call('API error: ${submitResponse.statusCode}');
        return null;
      }

      final eventId = jsonDecode(submitResponse.body)['event_id'] as String;
      debugPrint('[TRYON] Job submitted, event_id: $eventId');

      // Step 3: Poll for result (SSE stream)
      onStatus?.call('Processing... (15-30 sec)');

      // Poll with retries — HF takes time to process
      for (int attempt = 0; attempt < 60; attempt++) {
        await Future.delayed(const Duration(seconds: 2));

        try {
          final resultResponse = await http.get(
            Uri.parse('$_baseUrl/call/tryon/$eventId'),
          ).timeout(const Duration(seconds: 10));

          final body = resultResponse.body;

          // Check for completion
          if (body.contains('event: complete')) {
            final lines = body.split('\n');
            for (int i = 0; i < lines.length; i++) {
              if (lines[i].trim() == 'event: complete' && i + 1 < lines.length) {
                final dataLine = lines[i + 1];
                if (dataLine.startsWith('data: ')) {
                  final jsonData = jsonDecode(dataLine.substring(6)) as List;
                  if (jsonData.isNotEmpty && jsonData[0] is Map) {
                    final resultUrl = jsonData[0]['url'] as String?;
                    if (resultUrl != null) {
                      debugPrint('[TRYON] Result ready: $resultUrl');
                      onStatus?.call('Done!');
                      return resultUrl;
                    }
                  }
                }
              }
            }
          }

          // Check for error
          if (body.contains('event: error')) {
            debugPrint('[TRYON] API error: $body');
            onStatus?.call('AI processing failed, try again');
            return null;
          }

          // Still processing
          if (attempt % 5 == 0) {
            onStatus?.call('Still processing... (${attempt * 2}s)');
          }
        } catch (e) {
          // Timeout or connection error — keep retrying
          debugPrint('[TRYON] Poll attempt $attempt error: $e');
        }
      }

      onStatus?.call('Timeout — try again');
      return null;
    } catch (e) {
      debugPrint('[TRYON] Error: $e');
      onStatus?.call('Error: $e');
      return null;
    }
  }

  /// Download result image as bytes.
  static Future<Uint8List?> downloadImage(String url) async {
    try {
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        return response.bodyBytes;
      }
      return null;
    } catch (e) {
      debugPrint('[TRYON] Download error: $e');
      return null;
    }
  }
}
