import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import '../models/style_score.dart';
import '../utils/app_theme.dart';

class StyleScoreCard extends StatelessWidget {
  final StyleScore score;

  const StyleScoreCard({super.key, required this.score});

  @override
  Widget build(BuildContext context) {
    return FadeInUp(
      duration: const Duration(milliseconds: 500),
      child: Container(
        padding: const EdgeInsets.all(AppSpacing.md),
        decoration: AppDecoration.gradientCard,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                Text('STYLE SCORE', style: AppText.labelSmall),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: _gradeColor.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(AppRadius.full),
                    border: Border.all(
                        color: _gradeColor.withValues(alpha: 0.4)),
                  ),
                  child: Text(
                    'Grade ${score.grade}',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: _gradeColor,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Big score
            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0, end: score.overall),
                  duration: const Duration(milliseconds: 1200),
                  curve: Curves.easeOutCubic,
                  builder: (context, val, child) => Text(
                    val.toStringAsFixed(1),
                    style: AppText.displayLarge.copyWith(
                      fontSize: 48,
                      color: _gradeColor,
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(bottom: 10, left: 4),
                  child: Text(
                    '/10',
                    style: AppText.bodyLarge.copyWith(
                        fontSize: 18, color: AppColors.grey),
                  ),
                ),
                const Spacer(),
                // Mini score bars
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    _MiniBar(label: 'Fit', value: score.fitScore),
                    const SizedBox(height: 6),
                    _MiniBar(label: 'Color', value: score.colorMatch),
                    const SizedBox(height: 6),
                    _MiniBar(label: 'Trend', value: score.trendScore),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Verdict
            Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 12, vertical: 10),
              decoration: BoxDecoration(
                color: AppColors.surfaceLight.withValues(alpha: 0.5),
                borderRadius: BorderRadius.circular(AppRadius.md),
              ),
              child: Row(
                children: [
                  Icon(
                    score.overall >= 8
                        ? Icons.local_fire_department
                        : Icons.lightbulb_outline,
                    color: AppColors.gold,
                    size: 18,
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      score.verdict,
                      style: AppText.bodyMedium
                          .copyWith(color: AppColors.textPrimary),
                    ),
                  ),
                ],
              ),
            ),

            // Tips
            if (score.tips.isNotEmpty) ...[
              const SizedBox(height: 12),
              ...score.tips.map((tip) => Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Padding(
                          padding: EdgeInsets.only(top: 4),
                          child: Icon(Icons.arrow_right,
                              size: 14, color: AppColors.gold),
                        ),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(tip,
                              style: AppText.bodyMedium
                                  .copyWith(fontSize: 12)),
                        ),
                      ],
                    ),
                  )),
            ],
          ],
        ),
      ),
    );
  }

  Color get _gradeColor {
    if (score.overall >= 9) return const Color(0xFFFF6B35);
    if (score.overall >= 8) return AppColors.gold;
    if (score.overall >= 7) return AppColors.success;
    return AppColors.greyLight;
  }
}

class _MiniBar extends StatelessWidget {
  final String label;
  final double value;

  const _MiniBar({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(label,
            style: AppText.bodyMedium.copyWith(fontSize: 10)),
        const SizedBox(width: 6),
        SizedBox(
          width: 60,
          height: 4,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(2),
            child: TweenAnimationBuilder<double>(
              tween: Tween(begin: 0, end: value / 10),
              duration: const Duration(milliseconds: 1000),
              curve: Curves.easeOut,
              builder: (context, val, child) => LinearProgressIndicator(
                value: val,
                backgroundColor: AppColors.surfaceLight,
                valueColor:
                    const AlwaysStoppedAnimation<Color>(AppColors.gold),
              ),
            ),
          ),
        ),
        const SizedBox(width: 4),
        SizedBox(
          width: 22,
          child: Text(
            value.toStringAsFixed(1),
            style: AppText.bodyMedium.copyWith(fontSize: 10),
            textAlign: TextAlign.right,
          ),
        ),
      ],
    );
  }
}
