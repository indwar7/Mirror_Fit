import 'package:flutter/material.dart';
import '../utils/app_theme.dart';

class SizeSelector extends StatelessWidget {
  final List<String> sizes;
  final String selected;
  final ValueChanged<String> onSelect;

  const SizeSelector({
    super.key,
    required this.sizes,
    required this.selected,
    required this.onSelect,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('SIZE', style: AppText.labelSmall),
        const SizedBox(height: AppSpacing.sm),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: sizes.map((size) {
            final isActive = size == selected;
            return GestureDetector(
              onTap: () => onSelect(size),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                width: 44,
                height: 44,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: isActive ? AppColors.gold : AppColors.surface,
                  borderRadius: BorderRadius.circular(AppRadius.sm),
                  border: Border.all(
                    color: isActive ? AppColors.gold : AppColors.surfaceLight,
                    width: isActive ? 1.5 : 0.5,
                  ),
                ),
                child: Text(
                  size,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: isActive ? FontWeight.w700 : FontWeight.w400,
                    color: isActive ? AppColors.background : AppColors.greyLight,
                  ),
                ),
              ),
            );
          }).toList(),
        ),
      ],
    );
  }
}

class ColorSelector extends StatelessWidget {
  final List<String> colors;
  final String selected;
  final ValueChanged<String> onSelect;

  const ColorSelector({
    super.key,
    required this.colors,
    required this.selected,
    required this.onSelect,
  });

  static const Map<String, Color> _colorMap = {
    'white': Color(0xFFF5F5F5),
    'ivory': Color(0xFFFFF8E7),
    'pearl': Color(0xFFEAE0C8),
    'navy': Color(0xFF1B2A4A),
    'royal blue': Color(0xFF1E40AF),
    'indigo': Color(0xFF312E81),
    'sky blue': Color(0xFF7DD3FC),
    'light blue': Color(0xFFBAE6FD),
    'chambray': Color(0xFF6B8DB2),
    'black': Color(0xFF1A1A1A),
    'charcoal': Color(0xFF374151),
    'dark grey': Color(0xFF4B5563),
    'slate': Color(0xFF475569),
    'jet': Color(0xFF111111),
    'camel': Color(0xFFC19A6B),
    'tan': Color(0xFFD2B48C),
    'cognac': Color(0xFF9A4E1E),
    'brown': Color(0xFF7B3F00),
    'olive': Color(0xFF6B7B3A),
    'forest green': Color(0xFF228B22),
    'sage': Color(0xFFB2AC88),
    'beige': Color(0xFFF5F0DC),
    'khaki': Color(0xFFC3B091),
    'sand': Color(0xFFC2B280),
    'midnight blue': Color(0xFF191970),
    'classic navy': Color(0xFF1B2A4A),
    'gold': Color(0xFFD4AF37),
    'maroon': Color(0xFF800000),
    'raw': Color(0xFF2C3E50),
    'dark wash': Color(0xFF1E293B),
  };

  Color _getColor(String name) =>
      _colorMap[name.toLowerCase()] ?? AppColors.surfaceBright;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('COLOR', style: AppText.labelSmall),
        const SizedBox(height: AppSpacing.sm),
        Wrap(
          spacing: 10,
          runSpacing: 10,
          children: colors.map((color) {
            final isActive = color == selected;
            return GestureDetector(
              onTap: () => onSelect(color),
              child: Column(
                children: [
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: _getColor(color),
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: isActive ? AppColors.gold : AppColors.surfaceLight,
                        width: isActive ? 2.5 : 1,
                      ),
                      boxShadow: isActive
                          ? [
                              BoxShadow(
                                color: AppColors.gold.withValues(alpha: 0.3),
                                blurRadius: 8,
                                spreadRadius: 1,
                              )
                            ]
                          : null,
                    ),
                    child: isActive
                        ? Icon(
                            Icons.check,
                            size: 16,
                            color: _needsLightIcon(color)
                                ? AppColors.white
                                : AppColors.background,
                          )
                        : null,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    color,
                    style: TextStyle(
                      fontSize: 10,
                      color: isActive ? AppColors.gold : AppColors.grey,
                      fontWeight:
                          isActive ? FontWeight.w600 : FontWeight.w400,
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
        ),
      ],
    );
  }

  bool _needsLightIcon(String color) {
    final c = color.toLowerCase();
    return c.contains('black') ||
        c.contains('navy') ||
        c.contains('dark') ||
        c.contains('charcoal') ||
        c.contains('indigo') ||
        c.contains('midnight') ||
        c.contains('jet') ||
        c.contains('maroon') ||
        c.contains('raw') ||
        c.contains('slate');
  }
}
