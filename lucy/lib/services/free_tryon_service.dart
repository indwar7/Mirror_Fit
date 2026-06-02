import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'colab_tryon_service.dart';

class _SpaceConfig {
  final String name;
  final String url;
  final String apiName;
  final List<dynamic> Function(Map<String, dynamic> person, Map<String, dynamic> garment) buildData;

  const _SpaceConfig({
    required this.name,
    required this.url,
    required this.apiName,
    required this.buildData,
  });
}

class FreeTryOnService {
  static final _rng = Random();

  static Map<String, dynamic> _editor(Map<String, dynamic> img) =>
      {'background': img, 'layers': <Map<String, dynamic>>[], 'composite': null};

  static final List<_SpaceConfig> _spaces = [
    _SpaceConfig(
      name: 'CatVTON',
      url: 'https://zhengchong-catvton.hf.space',
      apiName: 'submit_function_p2p',
      buildData: (p, g) => [_editor(p), g, 50, 2.5, 42],
    ),
    _SpaceConfig(
      name: 'CatVTON-Alt',
      url: 'https://nymbo-catvton.hf.space',
      apiName: 'submit_function_p2p',
      buildData: (p, g) => [_editor(p), g, 50, 2.5, 42],
    ),
    _SpaceConfig(
      name: 'IDM-VTON',
      url: 'https://yisol-idm-vton.hf.space',
      apiName: 'tryon',
      buildData: (p, g) => [_editor(p), g, 'A photo of a person wearing a garment', true, false, 30, 42],
    ),
    _SpaceConfig(
      name: 'IDM-VTON-2025',
      url: 'https://ronniechoyy-idm-vton-20250428.hf.space',
      apiName: 'tryon',
      buildData: (p, g) => [_editor(p), g, 'A photo of a person wearing a garment', true, false, 30, 42, 1],
    ),
    _SpaceConfig(
      name: 'IDM-VTON-Alt',
      url: 'https://themanfrom-virtual-try-on-image.hf.space',
      apiName: 'tryon',
      buildData: (p, g) => [_editor(p), g, 'A photo of a person wearing a garment', true, false, 30, 42],
    ),
    _SpaceConfig(
      name: 'Miragic',
      url: 'https://miragic-ai-miragic-virtual-try-on.hf.space',
      apiName: 'virtual_tryon',
      buildData: (p, g) => [p, g, 'Top'],
    ),
    _SpaceConfig(
      name: 'NikhilJoson',
      url: 'https://nikhiljoson-virtual-try-on.hf.space',
      apiName: 'tryon',
      buildData: (p, g) => [_editor(p), g, 'A garment', null, '', null, '', true, false, 26, 42],
    ),
  ];

  static Future<Uint8List> tryOn({
    required Uint8List personBytes,
    required Uint8List garmentBytes,
    void Function(double progress, String message)? onProgress,
  }) async {
    // Colab GPU first — fastest option
    final colabUrl = await ColabTryOnService.getSavedUrl();
    if (colabUrl != null && colabUrl.isNotEmpty) {
      try {
        return await ColabTryOnService.tryOn(
          personBytes: personBytes,
          garmentBytes: garmentBytes,
          ngrokUrl: colabUrl,
          onProgress: onProgress,
        );
      } catch (e) {
        debugPrint('[TRYON] Colab failed ($e), falling back to HuggingFace...');
        onProgress?.call(0.05, 'GPU server unavailable, trying HuggingFace...');
      }
    }

    // Resize images to 768px max — reduces upload time 3-4x
    onProgress?.call(0.05, 'Preparing images...');
    final resizedPerson  = await _resizeImage(personBytes);
    final resizedGarment = await _resizeImage(garmentBytes);

    // Try CatVTON spaces in parallel — take whichever responds first
    onProgress?.call(0.1, 'Connecting to AI...');
    try {
      return await _raceSpaces(
        _spaces.sublist(0, 2),
        resizedPerson,
        resizedGarment,
        onProgress,
      );
    } catch (_) {
      // All CatVTON spaces failed — fall through to sequential IDM-VTON fallbacks
    }

    // Sequential fallback for remaining spaces
    final errors = <String>[];
    for (int i = 2; i < _spaces.length; i++) {
      final space = _spaces[i];
      try {
        onProgress?.call(0.05, 'Trying ${space.name}...');
        return await _runSpace(space, resizedPerson, resizedGarment, onProgress);
      } catch (e) {
        errors.add('${space.name}: $e');
        debugPrint('[TRYON] ${space.name} failed: $e');
        if (i < _spaces.length - 1) {
          onProgress?.call(0.05, '${space.name} unavailable, trying next...');
        }
      }
    }

    throw FreeTryOnException(
      'All AI servers are currently busy. Please try again in a minute.\n'
      'Details: ${errors.join('; ')}',
    );
  }

  /// Race two spaces — return whichever finishes first.
  static Future<Uint8List> _raceSpaces(
    List<_SpaceConfig> spaces,
    Uint8List personBytes,
    Uint8List garmentBytes,
    void Function(double, String)? onProgress,
  ) {
    final completer = Completer<Uint8List>();
    int failed = 0;
    final errors = <String>[];

    for (final space in spaces) {
      _runSpace(space, personBytes, garmentBytes, onProgress).then((result) {
        if (!completer.isCompleted) completer.complete(result);
      }).catchError((e) {
        errors.add('${space.name}: $e');
        failed++;
        if (failed == spaces.length && !completer.isCompleted) {
          completer.completeError(FreeTryOnException(errors.join('; ')));
        }
      });
    }

    return completer.future;
  }

  static Future<Uint8List> _runSpace(
    _SpaceConfig space,
    Uint8List personBytes,
    Uint8List garmentBytes,
    void Function(double, String)? onProgress,
  ) async {
    onProgress?.call(0.1, 'Uploading to ${space.name}...');
    final paths = await _uploadFiles(space.url, personBytes, garmentBytes);

    onProgress?.call(0.2, 'Starting AI...');
    final personData  = _fileData(paths[0], 'person.jpg',  'image/jpeg');
    final garmentData = _fileData(paths[1], 'garment.png', 'image/png');

    final (eventId, sessionHash) = await _callApi(
      space.url, space.apiName, space.buildData(personData, garmentData),
    );

    onProgress?.call(0.3, 'AI is generating your look...');
    final resultUrl = await _pollResult(space.url, space.apiName, eventId, sessionHash, onProgress);

    onProgress?.call(0.9, 'Downloading result...');
    return _downloadResult(resultUrl);
  }

  static Map<String, dynamic> _fileData(String path, String name, String mime) => {
    'path': path,
    'orig_name': name,
    'mime_type': mime,
    'meta': {'_type': 'gradio.FileData'},
  };

  /// Resize image to max 768px on longest side — optimal for VTON models.
  static Future<Uint8List> _resizeImage(Uint8List bytes, {int maxSize = 768}) async {
    final codec = await ui.instantiateImageCodec(bytes, targetWidth: maxSize);
    final frame = await codec.getNextFrame();
    final data  = await frame.image.toByteData(format: ui.ImageByteFormat.png);
    frame.image.dispose();
    return data!.buffer.asUint8List();
  }

  static Future<List<String>> _uploadFiles(String spaceUrl, Uint8List personBytes, Uint8List garmentBytes) async {
    final request = http.MultipartRequest('POST', Uri.parse('$spaceUrl/upload'))
      ..files.add(http.MultipartFile.fromBytes('files', personBytes, filename: 'person.png'))
      ..files.add(http.MultipartFile.fromBytes('files', garmentBytes, filename: 'garment.png'));

    final streamed  = await request.send().timeout(
      const Duration(seconds: 90),
      onTimeout: () => throw FreeTryOnException('Upload timeout.'),
    );
    final response = await http.Response.fromStream(streamed);
    if (response.statusCode != 200) throw FreeTryOnException('Upload failed (${response.statusCode})');
    return (jsonDecode(response.body) as List).cast<String>();
  }

  static String _generateSessionHash() {
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
    return List.generate(11, (_) => chars[_rng.nextInt(chars.length)]).join();
  }

  static Future<(String, String)> _callApi(String spaceUrl, String apiName, List<dynamic> data) async {
    final sessionHash = _generateSessionHash();
    final response = await http
        .post(
          Uri.parse('$spaceUrl/call/$apiName'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'data': data, 'session_hash': sessionHash}),
        )
        .timeout(
          const Duration(seconds: 60),
          onTimeout: () => throw FreeTryOnException('Connection timeout.'),
        );

    if (response.statusCode == 503) throw FreeTryOnException('Server sleeping (503).');
    if (response.statusCode == 405) throw FreeTryOnException('API not available (405).');
    if (response.statusCode != 200) throw FreeTryOnException('API call failed (${response.statusCode})');

    final eventId = (jsonDecode(response.body) as Map<String, dynamic>)['event_id'] as String?;
    if (eventId == null) throw FreeTryOnException('No event ID returned.');
    return (eventId, sessionHash);
  }

  static Future<String> _pollResult(
    String spaceUrl,
    String apiName,
    String eventId,
    String sessionHash,
    void Function(double, String)? onProgress,
  ) async {
    final client = http.Client();
    try {
      final request = http.Request('GET', Uri.parse('$spaceUrl/call/$apiName/$eventId?session_hash=$sessionHash'))
        ..headers['Accept'] = 'text/event-stream';

      final streamed = await client.send(request).timeout(
        const Duration(minutes: 5),
        onTimeout: () => throw FreeTryOnException('AI generation timed out.'),
      );

      String? lastEvent;
      String resultUrl = '';

      await for (final line in streamed.stream.transform(utf8.decoder).transform(const LineSplitter())) {
        if (line.startsWith('event: ')) {
          lastEvent = line.substring(7).trim();
        } else if (line.startsWith('data: ')) {
          final dataStr = line.substring(6).trim();
          if (lastEvent == 'heartbeat') continue;
          if (lastEvent == 'generating') onProgress?.call(0.5, 'AI is fitting the garment...');
          if (lastEvent == 'error') {
            String msg = dataStr;
            try {
              final j = jsonDecode(dataStr);
              if (j is Map) msg = j['message'] as String? ?? j['error'] as String? ?? dataStr;
            } catch (_) {}
            throw FreeTryOnException('AI error: $msg');
          }
          if (lastEvent == 'complete') {
            try {
              final result = jsonDecode(dataStr);
              if (result is List && result.isNotEmpty && result[0] is Map) {
                final r = result[0] as Map;
                resultUrl = r['url'] as String? ?? '';
                if (resultUrl.isEmpty) {
                  final path = r['path'] as String? ?? '';
                  if (path.isNotEmpty) resultUrl = '$spaceUrl/file=$path';
                }
              }
            } catch (e) {
              throw FreeTryOnException('Failed to parse result: $e');
            }
            break;
          }
        }
      }

      if (resultUrl.isEmpty) throw FreeTryOnException('No result image received.');
      return resultUrl;
    } finally {
      client.close();
    }
  }

  static Future<Uint8List> _downloadResult(String url) async {
    final response = await http.get(Uri.parse(url)).timeout(
      const Duration(seconds: 30),
      onTimeout: () => throw FreeTryOnException('Download timed out.'),
    );
    if (response.statusCode != 200) throw FreeTryOnException('Download failed (${response.statusCode}).');
    return response.bodyBytes;
  }
}

class FreeTryOnException implements Exception {
  final String message;
  FreeTryOnException(this.message);

  @override
  String toString() => 'FreeTryOnException: $message';
}
