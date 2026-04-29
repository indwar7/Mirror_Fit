import 'package:flutter/material.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:animate_do/animate_do.dart';
import 'package:provider/provider.dart';
import '../models/cloth_item.dart';
import '../services/tryon_provider.dart';
import '../services/user_preferences.dart';
import '../services/style_engine.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';
import '../widgets/size_color_selector.dart';
import 'tryon_screen.dart';

class ProductDetailScreen extends StatefulWidget {
  final ClothItem item;
  const ProductDetailScreen({super.key, required this.item});

  @override
  State<ProductDetailScreen> createState() => _ProductDetailScreenState();
}

class _ProductDetailScreenState extends State<ProductDetailScreen> {
  late String _selectedSize;
  late String _selectedColor;
  bool _isWishlisted = false;

  // Single source of truth for wishlist toggle — removes the duplicated
  // setState+UserPreferences calls that existed in both appBar and bottomBar.
  void _toggleWishlist() {
    Haptics.medium();
    setState(() => _isWishlisted = !_isWishlisted);
    UserPreferences.toggleWishlist(widget.item.id);
  }

  @override
  void initState() {
    super.initState();
    final profile = UserPreferences.getProfile();
    _selectedSize = profile != null
        ? StyleEngine.recommendSize(profile, widget.item)
        : widget.item.sizes.first;
    _selectedColor = widget.item.colors.first;
    _isWishlisted = UserPreferences.isWishlisted(widget.item.id);
    UserPreferences.addRecentlyViewed(widget.item.id);
  }

  @override
  Widget build(BuildContext context) {
    final item = widget.item;
    final profile = UserPreferences.getProfile();
    final colorHarmony = profile != null
        ? StyleEngine.checkColorHarmony(profile.skinTone, _selectedColor)
        : null;
    final completeLook = StyleEngine.getCompleteLook(item);

    return Scaffold(
      backgroundColor: AppColors.background,
      body: CustomScrollView(
        physics: const BouncingScrollPhysics(),
        slivers: [
          // ── Hero Image ─────────────────────────────────────────────────────
          SliverAppBar(
            backgroundColor: AppColors.background,
            expandedHeight: MediaQuery.of(context).size.height * 0.5,
            pinned: true,
            leading: const _CircleBackButton(),
            actions: [
              Padding(
                padding: const EdgeInsets.only(right: 8),
                child: GestureDetector(
                  onTap: _toggleWishlist,
                  child: Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: AppColors.surface.withValues(alpha: 0.85),
                      shape: BoxShape.circle,
                    ),
                    child: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 300),
                      child: Icon(
                        _isWishlisted
                            ? Icons.favorite
                            : Icons.favorite_border,
                        key: ValueKey(_isWishlisted),
                        color: _isWishlisted
                            ? AppColors.error
                            : AppColors.white,
                        size: 22,
                      ),
                    ),
                  ),
                ),
              ),
            ],
            flexibleSpace: FlexibleSpaceBar(
              background: Stack(
                fit: StackFit.expand,
                children: [
                  Hero(
                    tag: 'cloth_${item.id}',
                    child: CachedNetworkImage(
                      imageUrl: item.imageUrl,
                      fit: BoxFit.cover,
                    ),
                  ),
                  // Bottom gradient
                  Positioned(
                    bottom: 0,
                    left: 0,
                    right: 0,
                    height: 120,
                    child: Container(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [
                            Colors.transparent,
                            AppColors.background,
                          ],
                        ),
                      ),
                    ),
                  ),
                  // Badges
                  if (item.isNew || item.isBestseller)
                    Positioned(
                      bottom: 16,
                      left: 16,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 12, vertical: 5),
                        decoration: BoxDecoration(
                          color: item.isNew
                              ? AppColors.gold
                              : AppColors.success,
                          borderRadius:
                              BorderRadius.circular(AppRadius.full),
                        ),
                        child: Text(
                          item.isNew ? 'NEW ARRIVAL' : 'BESTSELLER',
                          style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 1.5,
                            color: item.isNew
                                ? AppColors.background
                                : AppColors.white,
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),

          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ── Name & Price ─────────────────────────────────────────────
                  FadeInUp(
                    duration: const Duration(milliseconds: 400),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(item.categoryLabel,
                                  style: AppText.labelSmall),
                              const SizedBox(height: 4),
                              Text(item.name, style: AppText.titleLarge),
                            ],
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 8),
                          decoration: BoxDecoration(
                            color: AppColors.gold.withValues(alpha: 0.1),
                            borderRadius:
                                BorderRadius.circular(AppRadius.full),
                            border: Border.all(
                                color: AppColors.gold.withValues(alpha: 0.3)),
                          ),
                          child: Text(
                            '\u20B9${item.price.toStringAsFixed(0)}',
                            style: AppText.priceText,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),

                  // ── Description ──────────────────────────────────────────────
                  FadeInUp(
                    duration: const Duration(milliseconds: 400),
                    delay: const Duration(milliseconds: 100),
                    child: Text(item.description, style: AppText.bodyLarge),
                  ),
                  const SizedBox(height: 24),

                  // ── Size Selector ────────────────────────────────────────────
                  FadeInUp(
                    duration: const Duration(milliseconds: 400),
                    delay: const Duration(milliseconds: 150),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        SizeSelector(
                          sizes: item.sizes,
                          selected: _selectedSize,
                          onSelect: (s) {
                            Haptics.selection();
                            setState(() => _selectedSize = s);
                          },
                        ),
                        if (profile != null) ...[
                          const SizedBox(height: 8),
                          Row(
                            children: [
                              Icon(Icons.auto_awesome,
                                  size: 14,
                                  color: AppColors.gold.withValues(alpha: 0.7)),
                              const SizedBox(width: 6),
                              Text(
                                'Recommended: ${StyleEngine.recommendSize(profile, item)}',
                                style: AppText.bodyMedium.copyWith(
                                  fontSize: 12,
                                  color: AppColors.gold.withValues(alpha: 0.8),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),

                  // ── Color Selector ───────────────────────────────────────────
                  FadeInUp(
                    duration: const Duration(milliseconds: 400),
                    delay: const Duration(milliseconds: 200),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        ColorSelector(
                          colors: item.colors,
                          selected: _selectedColor,
                          onSelect: (c) {
                            Haptics.selection();
                            setState(() => _selectedColor = c);
                          },
                        ),
                        if (colorHarmony != null) ...[
                          const SizedBox(height: 10),
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 12, vertical: 8),
                            decoration: BoxDecoration(
                              color: (colorHarmony.isGreat
                                      ? AppColors.success
                                      : AppColors.gold)
                                  .withValues(alpha: 0.1),
                              borderRadius:
                                  BorderRadius.circular(AppRadius.md),
                              border: Border.all(
                                color: (colorHarmony.isGreat
                                        ? AppColors.success
                                        : AppColors.gold)
                                    .withValues(alpha: 0.3),
                              ),
                            ),
                            child: Row(
                              children: [
                                Icon(
                                  colorHarmony.isGreat
                                      ? Icons.thumb_up_rounded
                                      : Icons.info_outline,
                                  size: 16,
                                  color: colorHarmony.isGreat
                                      ? AppColors.success
                                      : AppColors.gold,
                                ),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: Text(
                                    colorHarmony.verdict,
                                    style: AppText.bodyMedium
                                        .copyWith(fontSize: 12),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                  const SizedBox(height: 28),

                  // ── Complete the Look ────────────────────────────────────────
                  if (completeLook.isNotEmpty) ...[
                    FadeInUp(
                      duration: const Duration(milliseconds: 400),
                      delay: const Duration(milliseconds: 300),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('COMPLETE THE LOOK',
                              style: AppText.labelSmall),
                          const SizedBox(height: 12),
                          SizedBox(
                            height: 120,
                            child: ListView.separated(
                              scrollDirection: Axis.horizontal,
                              itemCount: completeLook.length,
                              separatorBuilder: (_, i) =>
                                  const SizedBox(width: 10),
                              itemBuilder: (_, i) {
                                final s = completeLook[i];
                                return GestureDetector(
                                  onTap: () {
                                    Haptics.light();
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                        builder: (_) =>
                                            ProductDetailScreen(item: s),
                                      ),
                                    );
                                  },
                                  child: Container(
                                    width: 90,
                                    decoration: AppDecoration.card,
                                    clipBehavior: Clip.antiAlias,
                                    child: Column(
                                      children: [
                                        Expanded(
                                          child: CachedNetworkImage(
                                            imageUrl: s.imageUrl,
                                            fit: BoxFit.cover,
                                            width: 90,
                                          ),
                                        ),
                                        Padding(
                                          padding: const EdgeInsets.all(6),
                                          child: Text(
                                            s.name,
                                            maxLines: 1,
                                            overflow: TextOverflow.ellipsis,
                                            style: AppText.bodyMedium
                                                .copyWith(fontSize: 10),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                );
                              },
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 28),
                  ],

                  const SizedBox(height: 80),
                ],
              ),
            ),
          ),
        ],
      ),

      // ── Bottom CTA ─────────────────────────────────────────────────────────
      bottomNavigationBar: SafeArea(
        child: Container(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          decoration: BoxDecoration(
            color: Colors.white,
            border: Border(
              top: BorderSide(color: AppColors.surfaceLight, width: 0.5),
            ),
          ),
          child: Row(
            children: [
              // Wishlist button
              GestureDetector(
                onTap: _toggleWishlist,
                child: Container(
                  width: 54,
                  height: 54,
                  decoration: BoxDecoration(
                    color: AppColors.surface,
                    borderRadius: BorderRadius.circular(AppRadius.lg),
                    border: Border.all(color: AppColors.surfaceLight),
                  ),
                  child: Icon(
                    _isWishlisted
                        ? Icons.favorite
                        : Icons.favorite_border,
                    color: _isWishlisted
                        ? AppColors.error
                        : AppColors.greyLight,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              // Try On button
              Expanded(
                child: SizedBox(
                  height: 54,
                  child: ElevatedButton.icon(
                    onPressed: () {
                      Haptics.medium();
                      final provider =
                          context.read<TryOnProvider>();
                      provider.selectCloth(item);
                      provider.setSize(_selectedSize);
                      provider.setColor(_selectedColor);
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (_) => const TryOnScreen()),
                      );
                    },
                    icon: const Icon(Icons.checkroom_rounded,
                        color: Colors.white),
                    label: Text('TRY ON', style: AppText.buttonText),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.gold,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(AppRadius.full),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _CircleBackButton extends StatelessWidget {
  const _CircleBackButton();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 8),
      child: GestureDetector(
        onTap: () => Navigator.pop(context),
        child: Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: AppColors.surface.withValues(alpha: 0.85),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.arrow_back_ios_new,
              color: AppColors.textPrimary, size: 18),
        ),
      ),
    );
  }
}
