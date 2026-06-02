import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// A single fabric definition with PBR material properties and a Polyhaven
/// CDN texture URL that can be applied to a model-viewer element.
class FabricDef {
  final String id;          // Used to build the Polyhaven texture URL
  final String name;
  final String emoji;
  final String description;
  final double roughness;   // PBR roughness factor  0 = mirror-smooth, 1 = matte
  final double metalness;   // PBR metalness factor  0 = non-metal, 1 = full metal
  final Color swatch;       // Chip preview colour

  const FabricDef({
    required this.id,
    required this.name,
    required this.emoji,
    required this.description,
    required this.roughness,
    required this.metalness,
    required this.swatch,
  });

  /// Polyhaven diffuse/colour texture at 1 K resolution (HTTPS, CORS-open).
  String get textureUrl =>
      'https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k'
      '/$id/${id}_diff_1k.jpg';

  /// Roughness-metalness texture (controls surface micro-detail).
  String get roughnessUrl =>
      'https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k'
      '/$id/${id}_rough_1k.jpg';
}

/// Fetches the fabric list from the Polyhaven public API.
/// Falls back to [FabricService.defaults] on any network/parse failure.
class FabricService {
  FabricService._();

  static const _apiUrl =
      'https://api.polyhaven.com/assets?type=textures&categories=fabric&limit=20';

  // ── PBR properties keyed by fabric keyword ─────────────────────────────────
  // These are matched against Polyhaven asset IDs at runtime.
  static const Map<String, _Preset> _presets = {
    'cotton':  _Preset('🧶', 'Breathable & soft',    0.88, 0.00, Color(0xFFE8E0D6)),
    'silk':    _Preset('✨', 'Smooth & lustrous',     0.12, 0.08, Color(0xFFF5EFE0)),
    'denim':   _Preset('👖', 'Tough & classic',       0.82, 0.00, Color(0xFF4A6FA5)),
    'wool':    _Preset('🐑', 'Warm & cosy',           0.95, 0.00, Color(0xFFD6C5A0)),
    'linen':   _Preset('🌿', 'Light & natural',       0.90, 0.00, Color(0xFFD4C4A0)),
    'leather': _Preset('🧳', 'Premium & durable',     0.45, 0.02, Color(0xFF8B5E3C)),
    'satin':   _Preset('💫', 'Glossy & elegant',      0.08, 0.05, Color(0xFFF0E6D8)),
    'velvet':  _Preset('🎭', 'Rich & deep texture',   0.92, 0.00, Color(0xFF6B3FA0)),
  };

  // ── Hardcoded fallbacks (Polyhaven IDs verified to exist) ──────────────────
  static const List<FabricDef> defaults = [
    FabricDef(id: 'fabric_pattern_05', name: 'Cotton',  emoji: '🧶',
              description: 'Breathable & soft',    roughness: 0.88, metalness: 0.00,
              swatch: Color(0xFFE8E0D6)),
    FabricDef(id: 'satin',             name: 'Silk',    emoji: '✨',
              description: 'Smooth & lustrous',    roughness: 0.12, metalness: 0.08,
              swatch: Color(0xFFF5EFE0)),
    FabricDef(id: 'denim_fabric',      name: 'Denim',   emoji: '👖',
              description: 'Tough & classic',      roughness: 0.82, metalness: 0.00,
              swatch: Color(0xFF4A6FA5)),
    FabricDef(id: 'wool_blanket',      name: 'Wool',    emoji: '🐑',
              description: 'Warm & cosy',          roughness: 0.95, metalness: 0.00,
              swatch: Color(0xFFD6C5A0)),
    FabricDef(id: 'linen_fabric_02',   name: 'Linen',   emoji: '🌿',
              description: 'Light & natural',      roughness: 0.90, metalness: 0.00,
              swatch: Color(0xFFD4C4A0)),
    FabricDef(id: 'leather_white',     name: 'Leather', emoji: '🧳',
              description: 'Premium & durable',    roughness: 0.45, metalness: 0.02,
              swatch: Color(0xFF8B5E3C)),
  ];

  /// Fetches fabric list.
  ///
  /// 1. Calls Polyhaven API for fabric textures.
  /// 2. Matches returned asset IDs against [_presets] by keyword.
  /// 3. Returns up to 6 matched fabrics; fills remaining slots from [defaults].
  /// 4. Returns [defaults] immediately on any error.
  static Future<List<FabricDef>> fetchFabrics() async {
    try {
      final resp = await http
          .get(Uri.parse(_apiUrl))
          .timeout(const Duration(seconds: 6));

      if (resp.statusCode != 200) return defaults;

      final body = json.decode(resp.body) as Map<String, dynamic>;
      final matched = <FabricDef>[];
      final usedKeys = <String>{};

      for (final entry in body.entries) {
        if (matched.length >= 6) break;
        final assetId = entry.key.toLowerCase();

        for (final preset in _presets.entries) {
          if (usedKeys.contains(preset.key)) continue;
          if (assetId.contains(preset.key)) {
            final p = preset.value;
            matched.add(FabricDef(
              id: entry.key,
              name: _capitalise(preset.key),
              emoji: p.emoji,
              description: p.description,
              roughness: p.roughness,
              metalness: p.metalness,
              swatch: p.swatch,
            ));
            usedKeys.add(preset.key);
            break;
          }
        }
      }

      if (matched.isEmpty) return defaults;

      // Fill remaining slots from defaults for any preset types not found
      final remaining = defaults
          .where((d) => !usedKeys.contains(d.name.toLowerCase()))
          .take(6 - matched.length);

      return [...matched, ...remaining];
    } catch (e) {
      debugPrint('[FabricService] API error — using defaults. $e');
      return defaults;
    }
  }

  static String _capitalise(String s) =>
      s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);
}

class _Preset {
  final String emoji;
  final String description;
  final double roughness;
  final double metalness;
  final Color swatch;
  const _Preset(this.emoji, this.description, this.roughness, this.metalness,
      this.swatch);
}
