import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'user_preferences.dart';

class FashnApiService {
  static const String _baseUrl = 'https://api.fashn.ai/v1';

  static String get _apiKey => UserPreferences.apiKey;
  static bool get isDemoMode => _apiKey.isEmpty;

  static Map<String, String> get _headers => {
        'Authorization': 'Bearer $_apiKey',
        'Content-Type': 'application/json',
      };

  /// Start a try-on prediction.
  static Future<String> startTryOn({
    required String modelImageBase64,
    required String garmentImageUrl,
    required String category,
  }) async {
    final body = jsonEncode({
      'model_image': 'data:image/jpeg;base64,$modelImageBase64',
      'garment_image': garmentImageUrl,
      'category': category,
      'adjust_hands': true,
      'restore_background': true,
      'restore_clothes': true,
    });

    final response = await http.post(
      Uri.parse('$_baseUrl/run'),
      headers: _headers,
      body: body,
    );

    if (response.statusCode == 200 || response.statusCode == 201) {
      final data = jsonDecode(response.body);
      final id = data['id'] as String?;
      if (id == null) throw FashnApiException('No prediction ID in response');
      return id;
    } else {
      throw FashnApiException(
        'startTryOn failed: ${response.statusCode} ${response.body}',
      );
    }
  }

  /// Poll for result until status == "completed".
  static Future<String> pollForResult(String predictionId) async {
    const maxAttempts = 24;
    const pollInterval = Duration(seconds: 5);

    for (int attempt = 0; attempt < maxAttempts; attempt++) {
      await Future.delayed(pollInterval);

      final response = await http.get(
        Uri.parse('$_baseUrl/status/$predictionId'),
        headers: _headers,
      );

      if (response.statusCode != 200) {
        throw FashnApiException(
          'pollForResult failed: ${response.statusCode} ${response.body}',
        );
      }

      final data = jsonDecode(response.body);
      final status = data['status'] as String? ?? '';

      switch (status) {
        case 'completed':
          final output = data['output'];
          if (output is List && output.isNotEmpty) {
            return output.first as String;
          }
          throw FashnApiException('No output URL in completed response');

        case 'failed':
        case 'canceled':
          throw FashnApiException('Prediction $status: ${data['error']}');

        default:
          continue;
      }
    }

    throw FashnApiException('Timeout: prediction did not complete in 2 minutes');
  }

  /// Encode bytes to base64 string.
  static String bytesToBase64(Uint8List bytes) => base64Encode(bytes);
}

class FashnApiException implements Exception {
  final String message;
  FashnApiException(this.message);

  @override
  String toString() => 'FashnApiException: $message';
}
