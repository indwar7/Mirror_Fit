import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import '../models/cloth_item.dart';
import '../services/user_preferences.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';
import '../widgets/cloth_card.dart';
import 'product_detail_screen.dart';

class WishlistScreen extends StatefulWidget {
  const WishlistScreen({super.key});

  @override
  State<WishlistScreen> createState() => _WishlistScreenState();
}

class _WishlistScreenState extends State<WishlistScreen> {
  List<ClothItem> get _wishlistItems {
    final ids = UserPreferences.wishlistIds;
    return SampleCatalog.items.where((i) => ids.contains(i.id)).toList();
  }

  @override
  Widget build(BuildContext context) {
    final items = _wishlistItems;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: AppColors.background,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new,
              color: AppColors.textPrimary, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text('Wishlist', style: AppText.titleLarge),
        centerTitle: true,
        actions: [
          if (items.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: AppColors.gold.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(AppRadius.full),
                  ),
                  child: Text(
                    '${items.length} items',
                    style: AppText.labelSmall.copyWith(fontSize: 10),
                  ),
                ),
              ),
            ),
        ],
      ),
      body: items.isEmpty
          ? _buildEmpty()
          : GridView.builder(
              padding: const EdgeInsets.all(12),
              physics: const BouncingScrollPhysics(),
              gridDelegate:
                  const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                mainAxisSpacing: 12,
                crossAxisSpacing: 12,
                childAspectRatio: 0.65,
              ),
              itemCount: items.length,
              itemBuilder: (_, i) => FadeInUp(
                duration: const Duration(milliseconds: 400),
                delay: Duration(milliseconds: 50 * i),
                child: Dismissible(
                  key: ValueKey(items[i].id),
                  direction: DismissDirection.endToStart,
                  background: Container(
                    alignment: Alignment.centerRight,
                    padding: const EdgeInsets.only(right: 20),
                    decoration: BoxDecoration(
                      color: AppColors.error.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(AppRadius.lg),
                    ),
                    child: const Icon(Icons.delete_outline,
                        color: AppColors.error),
                  ),
                  onDismissed: (_) {
                    Haptics.medium();
                    UserPreferences.toggleWishlist(items[i].id);
                    setState(() {});
                  },
                  child: ClothCard(
                    item: items[i],
                    onTap: () async {
                      await Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) =>
                              ProductDetailScreen(item: items[i]),
                        ),
                      );
                      setState(() {}); // Refresh on return
                    },
                  ),
                ),
              ),
            ),
    );
  }

  Widget _buildEmpty() {
    return Center(
      child: FadeIn(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: AppColors.surface,
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.favorite_border,
                  color: AppColors.grey, size: 36),
            ),
            const SizedBox(height: 20),
            Text('No saved items yet', style: AppText.titleMedium),
            const SizedBox(height: 8),
            Text('Items you love will appear here',
                style: AppText.bodyMedium),
          ],
        ),
      ),
    );
  }
}
