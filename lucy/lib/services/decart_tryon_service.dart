import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class DecartTryOnException implements Exception {
  final String message;
  DecartTryOnException(this.message);
  @override
  String toString() => 'DecartTryOnException: $message';
}

/// Virtual try-on using Decart AI's lucy-pro-i2i image-to-image model.
class DecartTryOnService {
  static const _baseUrl = 'https://api.decart.ai/v1/generate/lucy-pro-i2i';
  static const _apiKey =
      'lucy_pHvLsDWaDEDGdcwUJYFRwvfCYotVOpgTrmihYiMcKxgjfbEwVDsSrpWwUioFcCyH';

  static const bool isDemoMode = false;

  /// Run virtual try-on via Decart lucy-pro-i2i.
  /// Returns result image as [Uint8List].
  static Future<Uint8List> tryOn({
    required Uint8List personBytes,
    required Uint8List garmentBytes,
    String garmentName = 'garment',
    String resolution = '720p',
    void Function(double progress, String message)? onProgress,
  }) async {
    onProgress?.call(0.1, 'Preparing images...');

    final prompt =
        'person wearing the exact $garmentName from the reference image, '
        'garment fitted naturally on body, realistic fabric drape and texture, '
        'fashion photography, high quality, photorealistic';

    debugPrint('[DECART] Starting try-on — model: lucy-pro-i2i  res: $resolution');

    final request = http.MultipartRequest('POST', Uri.parse(_baseUrl))
      ..headers['X-API-KEY'] = _apiKey
      ..fields['prompt'] = prompt
      ..fields['resolution'] = resolution;

    request.files.add(
      http.MultipartFile.fromBytes('data', personBytes, filename: 'person.jpg'),
    );
    request.files.add(
      http.MultipartFile.fromBytes('reference_image', garmentBytes, filename: 'garment.jpg'),
    );

    onProgress?.call(0.3, 'Uploading images...');

    final streamed = await request.send().timeout(
      const Duration(seconds: 180),
      onTimeout: () => throw DecartTryOnException('Request timed out — please try again'),
    );

    onProgress?.call(0.75, 'AI is fitting the garment...');

    final response = await http.Response.fromStream(streamed);

    debugPrint('[DECART] Response status: ${response.statusCode}');

    if (response.statusCode != 200) {
      debugPrint('[DECART] Error body: ${response.body}');
      throw DecartTryOnException(
        'Decart error (${response.statusCode}): ${response.body}',
      );
    }

    onProgress?.call(1.0, 'Your look is ready!');
    return response.bodyBytes;
  }
}
