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

  /// True for avatars the user enrolled from their own photo. Presets are
  /// fixed and cannot be deleted; enrolled ones can.
  final bool enrolled;

  /// Whether a full-body image has been generated for this avatar yet. Only
  /// meaningful for enrolled avatars — try-on needs a torso, and the enrolled
  /// selfie alone does not have one.
  final bool hasBody;

  /// Body measurements captured for this avatar, or null if not provided yet.
  final Map<String, dynamic>? measurements;

  const AvatarModel({
    required this.id,
    required this.name,
    required this.category,
    required this.imageUrl,
    this.enrolled = false,
    this.hasBody = false,
    this.measurements,
  });

  factory AvatarModel.fromJson(Map<String, dynamic> j) => AvatarModel(
        id: j['id'] as String,
        name: j['name'] as String,
        category: j['category'] as String,
        imageUrl: '$_kLocalBase${j['image_url']}',
        enrolled: j['enrolled'] as bool? ?? false,
        hasBody: j['has_body'] as bool? ?? false,
        measurements: j['measurements'] as Map<String, dynamic>?,
      );
}

// ── Face swap service ─────────────────────────────────────────────────────────

class FaceSwapService {
  // ── Replicate (cloud, no local server needed) ─────────────────────
  static const _apiToken = String.fromEnvironment('REPLICATE_API_TOKEN');
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

  // ── Enrol the user's own face as an avatar ───────────────────────

  /// Upload [photo] as a new avatar. The backend validates that a face is
  /// actually detectable before it stores anything, so a 422 here means the
  /// photo is unusable rather than that the request was malformed.
  ///
  /// Returns the created avatar. Its id works everywhere a preset id does.
  static Future<AvatarModel> createAvatar({
    required Uint8List photo,
    String name = 'You',
    String? gender,
    void Function(String)? onStatus,
  }) async {
    onStatus?.call('Creating your avatar…');
    final req = http.MultipartRequest(
      'POST',
      Uri.parse('$_kLocalBase/avatars/create'),
    )
      ..fields['name'] = name
      ..files.add(http.MultipartFile.fromBytes('photo', photo,
          filename: 'selfie.jpg'));
    if (gender != null) req.fields['gender'] = gender;

    final streamed = await req.send().timeout(const Duration(seconds: 60));
    final res = await http.Response.fromStream(streamed);
    if (res.statusCode != 200) {
      throw FaceSwapException('Enrolment failed: ${_extractDetail(res.body)}');
    }
    return AvatarModel.fromJson(
        jsonDecode(res.body) as Map<String, dynamic>);
  }

  /// Put the enrolled face on a curated base body.
  ///
  /// [chestCm]/[waistCm] are optional and refine which of the eight approved
  /// base photographs is used; omit both and the gender's average build is
  /// chosen. Supply one without the other and the server rejects it — one
  /// alone gives no ratio, and half-using it would look like it counted.
  ///
  /// A **501** means that build has no approved photograph yet. The server
  /// does not substitute a neighbouring build, so surface it as "not available
  /// yet" rather than as a malformed request.
  ///
  /// The composite is written over the avatar's own image, so after this
  /// [avatarImageUrl] returns the full-body version.
  static Future<Map<String, dynamic>> createAvatarBody({
    required String avatarId,
    required String gender,
    double? chestCm,
    double? waistCm,
    double? heightCm,
    double? hipsCm,
    void Function(String)? onStatus,
  }) async {
    onStatus?.call('Putting you on a body…');
    final req = http.MultipartRequest(
      'POST',
      Uri.parse('$_kLocalBase/avatars/$avatarId/body'),
    )..fields['gender'] = gender;
    if (chestCm != null) req.fields['chest_cm'] = chestCm.toString();
    if (waistCm != null) req.fields['waist_cm'] = waistCm.toString();
    if (heightCm != null) req.fields['height_cm'] = heightCm.toString();
    if (hipsCm != null) req.fields['hips_cm'] = hipsCm.toString();

    final res = await http.Response.fromStream(
        await req.send().timeout(const Duration(seconds: 90)));
    if (res.statusCode == 501) {
      throw FaceSwapException(
          'No base body for this build yet: ${_extractDetail(res.body)}');
    }
    if (res.statusCode != 200) {
      throw FaceSwapException('Body setup failed: ${_extractDetail(res.body)}');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  /// Put a garment on the avatar. Returns the fitted JPEG bytes.
  ///
  /// [category] is `upper`, `lower` or `dress`. The server pastes back
  /// everything outside the garment mask, so the face comes through
  /// bit-identical — this cannot change what the person looks like.
  ///
  /// A **503** means the try-on models are not loaded on the server; it
  /// deliberately does not return the avatar unchanged, which would look like
  /// a try-on that had worked.
  static Future<Uint8List> tryOnGarment({
    required String avatarId,
    required Uint8List garment,
    String category = 'upper',
    void Function(String)? onStatus,
  }) async {
    onStatus?.call('Fitting the garment…');
    final req = http.MultipartRequest(
      'POST',
      Uri.parse('$_kLocalBase/avatars/$avatarId/tryon'),
    )
      ..fields['category'] = category
      ..files.add(http.MultipartFile.fromBytes('garment_image', garment,
          filename: 'garment.jpg'));

    final streamed = await req.send().timeout(const Duration(seconds: 180));
    final bytes = await streamed.stream.toBytes();
    if (streamed.statusCode != 200) {
      throw FaceSwapException(
          'Try-on failed: ${_extractDetail(utf8.decode(bytes, allowMalformed: true))}');
    }
    return bytes;
  }

  /// Per-bin readiness of the curated base-body set.
  ///
  /// Reported per id rather than as a count, so a client can say which build
  /// is missing instead of "some bodies unavailable".
  static Future<Map<String, dynamic>> baseBodyStatus() async {
    final res = await http.get(Uri.parse('$_kLocalBase/base-bodies'));
    if (res.statusCode != 200) {
      throw FaceSwapException('Could not read base bodies (${res.statusCode})');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  /// URL of an avatar's image — the full-body composite once a body was built.
  static String avatarImageUrl(String avatarId) =>
      '$_kLocalBase/avatars/$avatarId/image';

  /// Delete an enrolled avatar and its images. Presets return 403.
  static Future<void> deleteAvatar(String avatarId) async {
    final res = await http
        .delete(Uri.parse('$_kLocalBase/avatars/$avatarId'))
        .timeout(const Duration(seconds: 10));
    if (res.statusCode != 200) {
      throw FaceSwapException('Delete failed: ${_extractDetail(res.body)}');
    }
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
