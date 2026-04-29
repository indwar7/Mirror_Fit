class StyleScore {
  final double overall;
  final double fitScore;
  final double colorMatch;
  final double trendScore;
  final String verdict;
  final List<String> tips;

  const StyleScore({
    required this.overall,
    required this.fitScore,
    required this.colorMatch,
    required this.trendScore,
    required this.verdict,
    required this.tips,
  });

  String get emoji {
    if (overall >= 9) return '🔥';
    if (overall >= 8) return '⭐';
    if (overall >= 7) return '✅';
    if (overall >= 6) return '🤔';
    return '😐';
  }

  String get grade {
    if (overall >= 9) return 'S';
    if (overall >= 8) return 'A';
    if (overall >= 7) return 'B';
    if (overall >= 6) return 'C';
    return 'D';
  }
}
