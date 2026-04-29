import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class ColabTryOnService {
  static const _prefKey = 'colab_ngrok_url';

  static Future<String?> getSavedUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_prefKey);
  }

  static Future<void> saveUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    final trimmed = url.trim().replaceAll(RegExp(r'/$'), '');
    await prefs.setString(_prefKey, trimmed);
    debugPrint('[COLAB] Saved URL: $trimmed');
  }

  static Future<void> clearUrl() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_prefKey);
    debugPrint('[COLAB] URL cleared');
  }

  /// Run try-on via the Colab Flask API.
  /// Returns result as [Uint8List].
  static Future<Uint8List> tryOn({
    required Uint8List personBytes,
    required Uint8List garmentBytes,
    required String ngrokUrl,
    void Function(double progress, String message)? onProgress,
  }) async {
    onProgress?.call(0.1, 'Connecting to Colab GPU...');
    debugPrint('[COLAB] Calling $ngrokUrl/tryon');

    onProgress?.call(0.25, 'Sending photos to GPU...');

    final http.Response response;
    try {
      response = await http
          .post(
            Uri.parse('$ngrokUrl/tryon'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'person': base64Encode(personBytes),
              'garment': base64Encode(garmentBytes),
            }),
          )
          .timeout(
            const Duration(minutes: 3),
            onTimeout: () => throw ColabTryOnException(
                'GPU timed out — make sure your Colab is still running.'),
          );
    } catch (e) {
      if (e is ColabTryOnException) rethrow;
      throw ColabTryOnException('Could not reach Colab: $e');
    }

    debugPrint('[COLAB] Response: ${response.statusCode}');

    if (response.statusCode != 200) {
      throw ColabTryOnException(
          'Colab error (${response.statusCode}): ${response.body.substring(0, response.body.length.clamp(0, 200))}');
    }

    onProgress?.call(0.85, 'AI done — decoding result...');

    final Map<String, dynamic> data;
    try {
      data = jsonDecode(response.body) as Map<String, dynamic>;
    } catch (_) {
      throw ColabTryOnException('Invalid response from Colab server.');
    }

    final resultBase64 = data['result'] as String?;
    if (resultBase64 == null || resultBase64.isEmpty) {
      throw ColabTryOnException('No result image in Colab response.');
    }

    onProgress?.call(1.0, 'Your look is ready!');
    return base64Decode(resultBase64);
  }
}

class ColabTryOnException implements Exception {
  final String message;
  ColabTryOnException(this.message);

  @override
  String toString() => 'ColabTryOnException: $message';
}
