import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class FaceSwapException implements Exception {
  final String message;
  FaceSwapException(this.message);
  @override
  String toString() => message;
}

/// Face swap via Replicate yan-ops/face-swap — no local server needed.
class FaceSwapService {
  static const _apiToken = 'r8_6WzyWiVBzsg2pZxV7NjVUnsIAi5tLAA1CwIMd';
  static const _predictUrl =
      'https://api.replicate.com/v1/models/codeplugtech/face-swap/predictions';

  static Map<String, String> get _headers => {
        'Authorization': 'Bearer $_apiToken',
        'Content-Type': 'application/json',
      };

  /// Swap the face from [sourceImage] (user's face) into [targetImage].
  static Future<Uint8List> swapFace({
    required Uint8List sourceImage,
    required Uint8List targetImage,
    void Function(String message)? onStatus,
  }) async {
    onStatus?.call('Uploading images…');

    final results = await Future.wait([
      _uploadBytes(sourceImage, 'source.jpg'),
      _uploadBytes(targetImage, 'target.jpg'),
    ]);
    final sourceUrl = results[0];
    final targetUrl = results[1];

    onStatus?.call('Swapping face…');

    final createRes = await http
        .post(
          Uri.parse(_predictUrl),
          headers: _headers,
          body: jsonEncode({
            'input': {
              'swap_image': sourceUrl,
              'input_image': targetUrl,
            },
          }),
        )
        .timeout(const Duration(seconds: 30));

    debugPrint('[FACESWAP] Create → ${createRes.statusCode}: ${createRes.body}');

    if (createRes.statusCode != 201) {
      throw FaceSwapException(
          'Replicate error ${createRes.statusCode}: ${createRes.body}');
    }

    final prediction = jsonDecode(createRes.body) as Map<String, dynamic>;
    final predId = prediction['id'] as String;
    final pollUrl =
        (prediction['urls'] as Map?)?.cast<String, dynamic>()['get'] as String? ??
        'https://api.replicate.com/v1/predictions/$predId';

    for (int i = 0; i < 60; i++) {
      await Future.delayed(const Duration(seconds: 2));

      final pollRes = await http.get(
        Uri.parse(pollUrl),
        headers: {'Authorization': 'Bearer $_apiToken'},
      );
      if (pollRes.statusCode != 200) continue;

      final data = jsonDecode(pollRes.body) as Map<String, dynamic>;
      final status = data['status'] as String;
      debugPrint('[FACESWAP] Status: $status (${i * 2}s)');

      if (status == 'succeeded') {
        final output = data['output'];
        String? outputUrl;
        if (output is String) outputUrl = output;
        if (output is List && output.isNotEmpty) outputUrl = output[0] as String;
        if (outputUrl == null) throw FaceSwapException('No output URL');

        onStatus?.call('Downloading result…');
        final imgRes = await http
            .get(Uri.parse(outputUrl))
            .timeout(const Duration(seconds: 30));
        if (imgRes.statusCode != 200) {
          throw FaceSwapException('Download failed (${imgRes.statusCode})');
        }
        return imgRes.bodyBytes;
      }

      if (status == 'failed' || status == 'canceled') {
        final error = data['error'] ?? 'Unknown error';
        throw FaceSwapException('Swap failed: $error');
      }
    }

    throw FaceSwapException('Timed out after 120s');
  }

  static Future<String> _uploadBytes(Uint8List bytes, String filename) async {
    final req = http.MultipartRequest(
      'POST',
      Uri.parse('https://api.replicate.com/v1/files'),
    )
      ..headers['Authorization'] = 'Bearer $_apiToken'
      ..files.add(http.MultipartFile.fromBytes('content', bytes,
          filename: filename));

    final streamed = await req.send().timeout(const Duration(seconds: 60));
    final res = await http.Response.fromStream(streamed);

    if (res.statusCode != 201) {
      throw FaceSwapException('Upload failed (${res.statusCode})');
    }

    final url = (jsonDecode(res.body) as Map<String, dynamic>)['urls']
        ?.cast<String, dynamic>()['get'] as String?;
    if (url == null) throw FaceSwapException('No upload URL returned');
    debugPrint('[FACESWAP] Uploaded $filename → $url');
    return url;
  }
}
