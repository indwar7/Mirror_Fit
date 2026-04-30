import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

// ── Local backend base URL ────────────────────────────────────────────────────
// Change to your machine's LAN IP when testing on a physical device,
// e.g. "http://192.168.1.5:7860"
const String _kLocalBase = 'http://localhost:7860';

class FaceSwapException implements Exception {
  final String message;
  FaceSwapException(this.message);
  @override
  String toString() => message;
}

// ── Avatar model ──────────────────────────────────────────────────────────────

class AvatarModel {
  final String id;
  final String name;
  final String category;
  final String imageUrl; // absolute URL including host

  const AvatarModel({
    required this.id,
    required this.name,
    required this.category,
    required this.imageUrl,
  });

  factory AvatarModel.fromJson(Map<String, dynamic> j) => AvatarModel(
        id: j['id'] as String,
        name: j['name'] as String,
        category: j['category'] as String,
        imageUrl: '$_kLocalBase${j['image_url']}',
      );
}

// ── Face swap service ─────────────────────────────────────────────────────────

class FaceSwapService {
  // ── Replicate (cloud, no local server needed) ─────────────────────
  static const _apiToken = 'r8_6WzyWiVBzsg2pZxV7NjVUnsIAi5tLAA1CwIMd';
  static const _predictUrl =
      'https://api.replicate.com/v1/models/codeplugtech/face-swap/predictions';
  static Map<String, String> get _headers => {
        'Authorization': 'Bearer $_apiToken',
        'Content-Type': 'application/json',
      };

  // ── Avatar catalogue ──────────────────────────────────────────────

  /// Fetch the full avatar list from the local backend.
  static Future<List<AvatarModel>> getAvatars() async {
    final res = await http
        .get(Uri.parse('$_kLocalBase/avatars'))
        .timeout(const Duration(seconds: 10));
    if (res.statusCode != 200) {
      throw FaceSwapException('Could not load avatars (${res.statusCode})');
    }
    final data = jsonDecode(res.body) as Map<String, dynamic>;
    final list = (data['avatars'] as List).cast<Map<String, dynamic>>();
    return list.map(AvatarModel.fromJson).toList();
  }

  // ── Swap into avatar (local backend) ─────────────────────────────

  /// Upload [sourceImage] (user's face) and swap it into the selected avatar.
  static Future<Uint8List> swapWithAvatar({
    required Uint8List sourceImage,
    required String avatarId,
    void Function(String)? onStatus,
  }) async {
    onStatus?.call('Swapping face…');
    final req = http.MultipartRequest(
      'POST',
      Uri.parse('$_kLocalBase/face-swap/with-avatar'),
    )
      ..fields['avatar_id'] = avatarId
      ..files.add(http.MultipartFile.fromBytes('source', sourceImage,
          filename: 'source.jpg'));

    final streamed = await req.send().timeout(const Duration(seconds: 60));
    final res = await http.Response.fromStream(streamed);
    if (res.statusCode != 200) {
      final detail = _extractDetail(res.body);
      throw FaceSwapException('Swap failed: $detail');
    }
    return res.bodyBytes;
  }

  // ── Swap two custom photos (Replicate cloud) ──────────────────────

  static Future<Uint8List> swapFace({
    required Uint8List sourceImage,
    required Uint8List targetImage,
    void Function(String)? onStatus,
  }) async {
    onStatus?.call('Uploading images…');
    final results = await Future.wait([
      _uploadBytes(sourceImage, 'source.jpg'),
      _uploadBytes(targetImage, 'target.jpg'),
    ]);

    onStatus?.call('Swapping face…');
    final createRes = await http
        .post(
          Uri.parse(_predictUrl),
          headers: _headers,
          body: jsonEncode({
            'input': {
              'swap_image': results[0],
              'input_image': results[1],
            },
          }),
        )
        .timeout(const Duration(seconds: 30));

    if (createRes.statusCode != 201) {
      throw FaceSwapException(
          'Replicate error ${createRes.statusCode}: ${createRes.body}');
    }

    final prediction = jsonDecode(createRes.body) as Map<String, dynamic>;
    final predId = prediction['id'] as String;
    final pollUrl =
        (prediction['urls'] as Map?)?.cast<String, dynamic>()['get']
            as String? ??
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
        throw FaceSwapException('Swap failed: ${data['error'] ?? 'Unknown'}');
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
    return url;
  }

  static String _extractDetail(String body) {
    try {
      return (jsonDecode(body) as Map)['detail']?.toString() ?? body;
    } catch (_) {
      return body;
    }
  }
}
