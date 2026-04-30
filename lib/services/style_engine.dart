import 'dart:math';
import '../models/cloth_item.dart';
import '../models/style_score.dart';
import '../models/user_profile.dart';

class StyleEngine {
  StyleEngine._();

  // ── Size Recommendation ─────────────────────────────────────────────────
  static String recommendSize(UserProfile profile, ClothItem item) {
    final sizes = item.sizes;

    if (item.category == ClothCategory.jeans ||
        item.category == ClothCategory.trousers) {
      final rec = profile.recommendedPantSize;
      return sizes.contains(rec) ? rec : _closestSize(rec, sizes);
    }

    final rec = profile.recommendedSize;
    return sizes.contains(rec) ? rec : _closestSize(rec, sizes);
  }

  static String _closestSize(String target, List<String> available) {
    const order = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL'];
    const pantOrder = ['28', '30', '32', '34', '36', '38', '40', '42'];

    final isPant = pantOrder.contains(target);
    final orderList = isPant ? pantOrder : order;

    final targetIdx = orderList.indexOf(target);
    if (targetIdx == -1) return available.first;

    int minDist = 999;
    String closest = available.first;
    for (final s in available) {
      final idx = orderList.indexOf(s);
      if (idx != -1) {
        final dist = (idx - targetIdx).abs();
        if (dist < minDist) {
          minDist = dist;
          closest = s;
        }
      }
    }
    return closest;
  }

  // ── Color Harmony Check ────────────────────────────────────────────────
  static ColorHarmonyResult checkColorHarmony(
      SkinTone skinTone, String colorName) {
    final warm = _warmColors.contains(colorName.toLowerCase());
    final cool = _coolColors.contains(colorName.toLowerCase());
    final neutral = _neutralColors.contains(colorName.toLowerCase());

    final warmSkin = skinTone == SkinTone.olive ||
        skinTone == SkinTone.medium ||
        skinTone == SkinTone.brown;

    if (neutral) {
      return ColorHarmonyResult(
        score: 9.0,
        verdict: 'Perfect neutral — works with every skin tone',
        isGreat: true,
      );
    }

    if ((warmSkin && warm) || (!warmSkin && cool)) {
      return ColorHarmonyResult(
        score: 8.5,
        verdict: 'Excellent match for your skin tone',
        isGreat: true,
      );
    }

    if ((warmSkin && cool) || (!warmSkin && warm)) {
      return ColorHarmonyResult(
        score: 6.5,
        verdict: 'Consider a warmer/cooler shade for better harmony',
        isGreat: false,
      );
    }

    return ColorHarmonyResult(
      score: 7.5,
      verdict: 'Good color choice',
      isGreat: true,
    );
  }

  static const _warmColors = {
    'camel', 'tan', 'cognac', 'brown', 'olive', 'gold', 'maroon',
    'khaki', 'sand', 'beige', 'forest green', 'sage', 'ivory',
  };
  static const _coolColors = {
    'navy', 'royal blue', 'indigo', 'sky blue', 'light blue', 'chambray',
    'midnight blue', 'classic navy', 'slate', 'dark wash', 'raw',
  };
  static const _neutralColors = {
    'white', 'black', 'charcoal', 'dark grey', 'jet', 'pearl', 'ivory',
  };

  // ── Style Score ────────────────────────────────────────────────────────
  static StyleScore generateScore({
    required ClothItem item,
    required UserProfile profile,
    required String selectedColor,
    required String selectedSize,
  }) {
    final rng = Random(item.id.hashCode + profile.heightCm.toInt());

    // Fit score — based on size match
    final recSize = recommendSize(profile, item);
    final fitScore = selectedSize == recSize
        ? 8.5 + rng.nextDouble() * 1.5
        : 6.0 + rng.nextDouble() * 2.0;

    // Color match
    final harmony = checkColorHarmony(profile.skinTone, selectedColor);
    final colorScore = harmony.score + (rng.nextDouble() - 0.5);

    // Trend score
    double trendBase = 7.0;
    if (item.isNew) trendBase += 1.0;
    if (item.isBestseller) trendBase += 0.5;
    if (profile.stylePrefs.any((p) => _matchesStyle(p, item.category))) {
      trendBase += 0.5;
    }
    final trendScore = (trendBase + rng.nextDouble()).clamp(5.0, 10.0);

    final overall =
        ((fitScore * 0.4 + colorScore * 0.3 + trendScore * 0.3) * 10)
                .round() /
            10;

    // Tips
    final tips = <String>[];
    if (selectedSize != recSize) {
      tips.add('We recommend size $recSize for your build');
    }
    if (!harmony.isGreat) {
      tips.add(harmony.verdict);
    }
    if (item.category == ClothCategory.shirt && overall >= 8) {
      tips.add('Try pairing with dark trousers for a sharp contrast');
    }
    if (item.category == ClothCategory.suit) {
      tips.add('Complete the look with a pocket square');
    }
    if (tips.isEmpty) {
      tips.add('This outfit suits you perfectly!');
    }

    // Verdict
    String verdict;
    if (overall >= 9) {
      verdict = 'Showstopper! This look is made for you';
    } else if (overall >= 8) {
      verdict = 'Great choice — you\'ll turn heads';
    } else if (overall >= 7) {
      verdict = 'Solid look with room to elevate';
    } else {
      verdict = 'Consider trying a different size or color';
    }

    return StyleScore(
      overall: overall.clamp(5.0, 10.0),
      fitScore: fitScore.clamp(5.0, 10.0),
      colorMatch: colorScore.clamp(5.0, 10.0),
      trendScore: trendScore.clamp(5.0, 10.0),
      verdict: verdict,
      tips: tips,
    );
  }

  static bool _matchesStyle(StylePreference pref, ClothCategory cat) {
    switch (pref) {
      case StylePreference.formal:
        return cat == ClothCategory.shirt ||
            cat == ClothCategory.suit ||
            cat == ClothCategory.trousers;
      case StylePreference.casual:
        return cat == ClothCategory.tShirt ||
            cat == ClothCategory.jeans;
      case StylePreference.ethnic:
        return cat == ClothCategory.kurta;
      case StylePreference.streetwear:
        return cat == ClothCategory.jacket ||
            cat == ClothCategory.tShirt;
      case StylePreference.minimal:
        return cat == ClothCategory.shirt ||
            cat == ClothCategory.trousers;
    }
  }

  // ── Complete the Look Suggestions ──────────────────────────────────────
  static List<ClothItem> getCompleteLook(ClothItem current) {
    final suggestions = <ClothItem>[];
    final catalog = SampleCatalog.items;

    switch (current.category) {
      case ClothCategory.shirt:
      case ClothCategory.tShirt:
      case ClothCategory.kurta:
      case ClothCategory.sweatshirt:
      case ClothCategory.hoodie:
        // Suggest bottoms
        suggestions.addAll(
          catalog
              .where((i) =>
                  i.category == ClothCategory.jeans ||
                  i.category == ClothCategory.trousers)
              .take(2),
        );
        // Suggest jacket
        suggestions.addAll(
          catalog
              .where((i) => i.category == ClothCategory.jacket)
              .take(1),
        );
        break;
      case ClothCategory.jeans:
      case ClothCategory.trousers:
      case ClothCategory.shorts:
        // Suggest tops
        suggestions.addAll(
          catalog
              .where((i) =>
                  i.category == ClothCategory.shirt ||
                  i.category == ClothCategory.tShirt)
              .take(2),
        );
        break;
      case ClothCategory.jacket:
      case ClothCategory.coat:
        // Suggest shirt + bottoms
        suggestions.addAll(
          catalog
              .where((i) => i.category == ClothCategory.shirt)
              .take(1),
        );
        suggestions.addAll(
          catalog
              .where((i) => i.category == ClothCategory.trousers)
              .take(1),
        );
        break;
      case ClothCategory.suit:
        // Suggest shirt
        suggestions.addAll(
          catalog
              .where((i) => i.category == ClothCategory.shirt)
              .take(2),
        );
        break;
    }

    // Remove self
    suggestions.removeWhere((i) => i.id == current.id);
    return suggestions;
  }
}

class ColorHarmonyResult {
  final double score;
  final String verdict;
  final bool isGreat;

  const ColorHarmonyResult({
    required this.score,
    required this.verdict,
    required this.isGreat,
  });
}
