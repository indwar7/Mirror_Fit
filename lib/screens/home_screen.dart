import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:animate_do/animate_do.dart';
import 'package:image_picker/image_picker.dart';
import '../models/cloth_item.dart';
import '../services/tryon_provider.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';
import 'decart_realtime_screen.dart';
import 'tryon_screen.dart';
import 'product_detail_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with TickerProviderStateMixin {
  late AnimationController _pulseCtrl;
  late Animation<double> _pulseAnim;
  String? _selectedCatKey;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1800),
    )..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: 1.0, end: 1.04).animate(
      CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  List<ClothItem> get _filtered {
    if (_selectedCatKey == null) return SampleCatalog.items;
    return SampleCatalog.items.where((i) {
      switch (_selectedCatKey) {
        case 'tops':
          return const {
            ClothCategory.shirt,
            ClothCategory.tShirt,
            ClothCategory.kurta,
            ClothCategory.sweatshirt,
            ClothCategory.hoodie,
          }.contains(i.category);
        case 'bottoms':
          return const {
            ClothCategory.jeans,
            ClothCategory.trousers,
            ClothCategory.shorts,
          }.contains(i.category);
        case 'outerwear':
          return const {
            ClothCategory.jacket,
            ClothCategory.coat,
          }.contains(i.category);
        default:
          return true;
      }
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<TryOnProvider>();

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            // ── Header ────────────────────────────────────────────────
            FadeInDown(
              duration: const Duration(milliseconds: 500),
              child: Padding(
                padding: const EdgeInsets.fromLTRB(24, 20, 24, 0),
                child: Row(
                  children: [
                    Text('LUCY', style: AppText.displayMedium),
                    const Spacer(),
                    _PhotoAvatar(
                      bytes: provider.userPhotoBytes,
                      onTap: () => _pickPhoto(context, provider),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 6),

            FadeInDown(
              duration: const Duration(milliseconds: 500),
              delay: const Duration(milliseconds: 80),
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Row(
                  children: [
                    Text(
                      provider.hasPhoto
                          ? 'Ready to try on — pick a look below'
                          : 'Tap your avatar to add your photo',
                      style: const TextStyle(
                        fontSize: 13,
                        color: AppColors.darkTextTertiary,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // ── Feature Cards ─────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                children: [
                  // Live Try-On — primary pulsing card
                  FadeInUp(
                    duration: const Duration(milliseconds: 600),
                    delay: const Duration(milliseconds: 100),
                    child: ScaleTransition(
                      scale: _pulseAnim,
                      child: _FeatureCard(
                        icon: Icons.videocam_rounded,
                        title: 'Live Try-On',
                        subtitle: 'See clothes on you in real-time via camera',
                        badge: 'LIVE',
                        gradient: const LinearGradient(
                          colors: [Color(0xFF1A1A2E), Color(0xFF16213E)],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        accentColor: AppColors.primary,
                        onTap: () {
                          Haptics.medium();
                          _showGarmentPicker(context, mode: _PickMode.live);
                        },
                      ),
                    ),
                  ),

                  const SizedBox(height: 12),

                  // Generate Look — secondary card
                  FadeInUp(
                    duration: const Duration(milliseconds: 600),
                    delay: const Duration(milliseconds: 200),
                    child: _FeatureCard(
                      icon: Icons.auto_awesome_rounded,
                      title: 'Generate Look',
                      subtitle: 'AI tries the outfit on your photo',
                      gradient: const LinearGradient(
                        colors: [Color(0xFF1C1C1E), Color(0xFF2C2C2E)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      accentColor: const Color(0xFF7C3AED),
                      onTap: () {
                        Haptics.medium();
                        if (!provider.hasPhoto) {
                          _pickPhotoThenProceed(
                              context, provider, _PickMode.generate);
                        } else {
                          _showGarmentPicker(context,
                              mode: _PickMode.generate);
                        }
                      },
                    ),
                  ),

                  const SizedBox(height: 12),

                  // Face Swap on Live Camera
                  FadeInUp(
                    duration: const Duration(milliseconds: 600),
                    delay: const Duration(milliseconds: 300),
                    child: _FeatureCard(
                      icon: Icons.face_retouching_natural,
                      title: 'Face Swap',
                      subtitle: 'Swap your face on live try-on camera',
                      badge: 'NEW',
                      gradient: const LinearGradient(
                        colors: [Color(0xFF1A0D2E), Color(0xFF2D1B4E)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      accentColor: const Color(0xFFD946EF),
                      onTap: () {
                        Haptics.medium();
                        _showGarmentPicker(context, mode: _PickMode.faceSwap);
                      },
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 28),

            // ── Outfit Section ────────────────────────────────────────
            FadeInUp(
              duration: const Duration(milliseconds: 500),
              delay: const Duration(milliseconds: 300),
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Row(
                  children: [
                    const Text(
                      'Choose an Outfit',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: AppColors.darkTextPrimary,
                      ),
                    ),
                    const Spacer(),
                    Text(
                      '${SampleCatalog.items.length} items',
                      style: const TextStyle(
                        fontSize: 12,
                        color: AppColors.darkTextTertiary,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 12),

            // Category chips
            FadeInUp(
              duration: const Duration(milliseconds: 500),
              delay: const Duration(milliseconds: 350),
              child: SizedBox(
                height: 32,
                child: ListView(
                  scrollDirection: Axis.horizontal,
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  children: [
                    _CategoryChip(
                        label: 'All',
                        active: _selectedCatKey == null,
                        onTap: () =>
                            setState(() => _selectedCatKey = null)),
                    _CategoryChip(
                        label: 'Tops',
                        active: _selectedCatKey == 'tops',
                        onTap: () =>
                            setState(() => _selectedCatKey = 'tops')),
                    _CategoryChip(
                        label: 'Bottoms',
                        active: _selectedCatKey == 'bottoms',
                        onTap: () =>
                            setState(() => _selectedCatKey = 'bottoms')),
                    _CategoryChip(
                        label: 'Outerwear',
                        active: _selectedCatKey == 'outerwear',
                        onTap: () =>
                            setState(() => _selectedCatKey = 'outerwear')),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 12),

            // Garment grid
            Expanded(
              child: FadeInUp(
                duration: const Duration(milliseconds: 500),
                delay: const Duration(milliseconds: 400),
                child: _GarmentGrid(
                  items: _filtered,
                  onTap: (item) {
                    Haptics.light();
                    _onGarmentTapped(context, provider, item);
                  },
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _onGarmentTapped(
      BuildContext context, TryOnProvider provider, ClothItem item) {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) => _GarmentActionSheet(
        item: item,
        onLiveTryOn: () {
          Navigator.pop(ctx);
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => DecartRealtimeScreen(
                garmentUrl: item.imageUrl,
                garmentName: item.name,
              ),
            ),
          );
        },
        onGenerate: () {
          Navigator.pop(ctx);
          provider.selectCloth(item);
          if (!provider.hasPhoto) {
            _pickPhotoThenNavigate(context, provider);
          } else {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const TryOnScreen()),
            );
          }
        },
        onDetails: () {
          Navigator.pop(ctx);
          Navigator.push(
            context,
            MaterialPageRoute(
                builder: (_) => ProductDetailScreen(item: item)),
          );
        },
      ),
    );
  }

  void _showGarmentPicker(BuildContext context,
      {required _PickMode mode}) {
    Navigator.push(
      context,
      PageRouteBuilder(
        pageBuilder: (context, anim, secondary) =>
            _GarmentPickerScreen(mode: mode),
        transitionsBuilder: (context, anim, secondary, child) =>
            SlideTransition(
          position: Tween<Offset>(
            begin: const Offset(0, 1),
            end: Offset.zero,
          ).animate(CurvedAnimation(parent: anim, curve: Curves.easeOut)),
          child: child,
        ),
        transitionDuration: const Duration(milliseconds: 380),
      ),
    );
  }

  Future<void> _pickPhoto(
      BuildContext context, TryOnProvider provider) async {
    Haptics.light();
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 36,
                height: 4,
                decoration: BoxDecoration(
                  color: AppColors.darkBorder,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: 20),
              const Text('Your Photo',
                  style: TextStyle(
                      fontSize: 18, fontWeight: FontWeight.w600)),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: _PickerButton(
                      icon: Icons.camera_alt_rounded,
                      label: 'Camera',
                      onTap: () async {
                        Navigator.pop(ctx);
                        await _doPick(ImageSource.camera, provider);
                      },
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _PickerButton(
                      icon: Icons.photo_library_rounded,
                      label: 'Gallery',
                      onTap: () async {
                        Navigator.pop(ctx);
                        await _doPick(ImageSource.gallery, provider);
                      },
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _doPick(
      ImageSource src, TryOnProvider provider) async {
    final picker = ImagePicker();
    final f = await picker.pickImage(
        source: src, maxWidth: 1024, maxHeight: 1024, imageQuality: 85);
    if (f != null) await provider.setUserPhoto(f);
  }

  Future<void> _pickPhotoThenProceed(BuildContext context,
      TryOnProvider provider, _PickMode mode) async {
    await _pickPhoto(context, provider);
    if (!context.mounted) return;
    if (provider.hasPhoto) {
      _showGarmentPicker(context, mode: mode);
    }
  }

  Future<void> _pickPhotoThenNavigate(
      BuildContext context, TryOnProvider provider) async {
    await _pickPhoto(context, provider);
    if (!context.mounted) return;
    if (provider.hasPhoto) {
      Navigator.push(
          context, MaterialPageRoute(builder: (_) => const TryOnScreen()));
    }
  }
}

enum _PickMode { live, generate, faceSwap }

// ── Garment Picker Full-Screen ────────────────────────────────────────────────
class _GarmentPickerScreen extends StatefulWidget {
  final _PickMode mode;
  const _GarmentPickerScreen({required this.mode});

  @override
  State<_GarmentPickerScreen> createState() => _GarmentPickerScreenState();
}

class _GarmentPickerScreenState extends State<_GarmentPickerScreen> {
  String? _catKey;

  List<ClothItem> get _filtered {
    if (_catKey == null) return SampleCatalog.items;
    return SampleCatalog.items.where((i) {
      switch (_catKey) {
        case 'tops':
          return const {
            ClothCategory.shirt, ClothCategory.tShirt,
            ClothCategory.kurta, ClothCategory.sweatshirt,
            ClothCategory.hoodie,
          }.contains(i.category);
        case 'bottoms':
          return const {
            ClothCategory.jeans, ClothCategory.trousers,
            ClothCategory.shorts,
          }.contains(i.category);
        case 'outerwear':
          return const {
            ClothCategory.jacket, ClothCategory.coat,
          }.contains(i.category);
        default:
          return true;
      }
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.read<TryOnProvider>();
    final isLive = widget.mode == _PickMode.live || widget.mode == _PickMode.faceSwap;

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.keyboard_arrow_down_rounded, size: 28),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          isLive ? 'Pick for Live Try-On' : 'Pick for Generate Look',
        ),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
            child: Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: isLive ? AppColors.primary : const Color(0xFF7C3AED),
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  isLive
                      ? 'Camera opens right after you pick'
                      : 'AI will try this on your photo',
                  style: const TextStyle(
                    fontSize: 13,
                    color: AppColors.darkTextSecondary,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: 32,
            child: ListView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 20),
              children: [
                _CategoryChip(
                    label: 'All',
                    active: _catKey == null,
                    onTap: () => setState(() => _catKey = null)),
                _CategoryChip(
                    label: 'Tops',
                    active: _catKey == 'tops',
                    onTap: () => setState(() => _catKey = 'tops')),
                _CategoryChip(
                    label: 'Bottoms',
                    active: _catKey == 'bottoms',
                    onTap: () => setState(() => _catKey = 'bottoms')),
                _CategoryChip(
                    label: 'Outerwear',
                    active: _catKey == 'outerwear',
                    onTap: () => setState(() => _catKey = 'outerwear')),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Expanded(
            child: _GarmentGrid(
              items: _filtered,
              onTap: (item) {
                Haptics.medium();
                if (isLive) {
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(
                      builder: (_) => DecartRealtimeScreen(
                        garmentUrl: item.imageUrl,
                        garmentName: item.name,
                        autoShowFaceSwap: widget.mode == _PickMode.faceSwap,
                      ),
                    ),
                  );
                } else {
                  provider.selectCloth(item);
                  Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(builder: (_) => const TryOnScreen()),
                  );
                }
              },
            ),
          ),
        ],
      ),
    );
  }
}

// ── Feature Card ──────────────────────────────────────────────────────────────
class _FeatureCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final String? badge;
  final LinearGradient gradient;
  final Color accentColor;
  final VoidCallback onTap;

  const _FeatureCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    this.badge,
    required this.gradient,
    required this.accentColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        height: 110,
        decoration: BoxDecoration(
          gradient: gradient,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
              color: accentColor.withValues(alpha: 0.25), width: 1.5),
          boxShadow: [
            BoxShadow(
              color: accentColor.withValues(alpha: 0.15),
              blurRadius: 20,
              offset: const Offset(0, 6),
            ),
          ],
        ),
        child: Stack(
          children: [
            // Glow accent blob
            Positioned(
              right: -20,
              top: -20,
              child: Container(
                width: 100,
                height: 100,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: accentColor.withValues(alpha: 0.08),
                ),
              ),
            ),

            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
              child: Row(
                children: [
                  Container(
                    width: 52,
                    height: 52,
                    decoration: BoxDecoration(
                      color: accentColor.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(
                          color: accentColor.withValues(alpha: 0.3)),
                    ),
                    child: Icon(icon, color: accentColor, size: 26),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Row(
                          children: [
                            Text(
                              title,
                              style: const TextStyle(
                                fontSize: 17,
                                fontWeight: FontWeight.w700,
                                color: AppColors.darkTextPrimary,
                              ),
                            ),
                            if (badge != null) ...[
                              const SizedBox(width: 8),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 7, vertical: 2),
                                decoration: BoxDecoration(
                                  color: accentColor,
                                  borderRadius: BorderRadius.circular(6),
                                ),
                                child: Text(
                                  badge!,
                                  style: const TextStyle(
                                    fontSize: 9,
                                    fontWeight: FontWeight.w800,
                                    color: Colors.white,
                                    letterSpacing: 0.5,
                                  ),
                                ),
                              ),
                            ],
                          ],
                        ),
                        const SizedBox(height: 4),
                        Text(
                          subtitle,
                          style: const TextStyle(
                            fontSize: 12,
                            color: AppColors.darkTextSecondary,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Icon(Icons.arrow_forward_ios_rounded,
                      size: 16, color: accentColor.withValues(alpha: 0.6)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Garment Grid ──────────────────────────────────────────────────────────────
class _GarmentGrid extends StatelessWidget {
  final List<ClothItem> items;
  final void Function(ClothItem) onTap;

  const _GarmentGrid({required this.items, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3,
        childAspectRatio: 0.72,
        crossAxisSpacing: 10,
        mainAxisSpacing: 10,
      ),
      itemCount: items.length,
      itemBuilder: (context, i) {
        final item = items[i];
        return FadeInUp(
          duration: const Duration(milliseconds: 300),
          delay: Duration(milliseconds: (i % 9) * 30),
          child: GestureDetector(
            onTap: () => onTap(item),
            child: Container(
              decoration: BoxDecoration(
                color: AppColors.darkSurface,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.darkBorder),
              ),
              clipBehavior: Clip.antiAlias,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(
                    child: Image.network(
                      item.imageUrl,
                      fit: BoxFit.cover,
                      loadingBuilder: (context, child, event) {
                        if (event == null) return child;
                        return Container(color: AppColors.darkCard);
                      },
                      errorBuilder: (context, url, error) => Container(
                        color: AppColors.darkCard,
                        child: const Icon(Icons.checkroom_rounded,
                            color: AppColors.darkTextTertiary, size: 28),
                      ),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(8, 6, 8, 8),
                    child: Text(
                      item.name,
                      style: const TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: AppColors.darkTextPrimary,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// ── Garment Action Sheet ──────────────────────────────────────────────────────
class _GarmentActionSheet extends StatelessWidget {
  final ClothItem item;
  final VoidCallback onLiveTryOn;
  final VoidCallback onGenerate;
  final VoidCallback onDetails;

  const _GarmentActionSheet({
    required this.item,
    required this.onLiveTryOn,
    required this.onGenerate,
    required this.onDetails,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 36,
              height: 4,
              decoration: BoxDecoration(
                color: AppColors.darkBorder,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: Image.network(item.imageUrl,
                      width: 56,
                      height: 56,
                      fit: BoxFit.cover,
                      errorBuilder: (c, u, e) =>
                          Container(width: 56, height: 56, color: AppColors.darkCard)),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(item.name,
                          style: const TextStyle(
                              fontSize: 15, fontWeight: FontWeight.w600)),
                      Text('₹${item.price.toStringAsFixed(0)}',
                          style: const TextStyle(
                              fontSize: 13, color: AppColors.primary)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            _ActionRow(
              icon: Icons.videocam_rounded,
              label: 'Live Try-On',
              subtitle: 'See it on you in real-time',
              color: AppColors.primary,
              onTap: onLiveTryOn,
            ),
            const SizedBox(height: 10),
            _ActionRow(
              icon: Icons.auto_awesome_rounded,
              label: 'Generate Look',
              subtitle: 'AI composites outfit on your photo',
              color: const Color(0xFF7C3AED),
              onTap: onGenerate,
            ),
            const SizedBox(height: 10),
            _ActionRow(
              icon: Icons.info_outline_rounded,
              label: 'View Details',
              subtitle: 'Sizes, colours, description',
              color: AppColors.darkTextSecondary,
              onTap: onDetails,
            ),
          ],
        ),
      ),
    );
  }
}

class _ActionRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String subtitle;
  final Color color;
  final VoidCallback onTap;

  const _ActionRow({
    required this.icon,
    required this.label,
    required this.subtitle,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          color: AppColors.darkCard,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: color.withValues(alpha: 0.2)),
        ),
        child: Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, color: color, size: 20),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(label,
                      style: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: AppColors.darkTextPrimary)),
                  Text(subtitle,
                      style: const TextStyle(
                          fontSize: 11,
                          color: AppColors.darkTextTertiary)),
                ],
              ),
            ),
            Icon(Icons.arrow_forward_ios_rounded,
                size: 14, color: AppColors.darkTextTertiary),
          ],
        ),
      ),
    );
  }
}

// ── Photo Avatar ──────────────────────────────────────────────────────────────
class _PhotoAvatar extends StatelessWidget {
  final Uint8List? bytes;
  final VoidCallback onTap;

  const _PhotoAvatar({required this.bytes, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(
            color: bytes != null
                ? AppColors.primary.withValues(alpha: 0.6)
                : AppColors.darkBorder,
            width: 2,
          ),
        ),
        clipBehavior: Clip.antiAlias,
        child: bytes != null
            ? Image.memory(bytes!, fit: BoxFit.cover)
            : const Icon(Icons.person_outline_rounded,
                size: 22, color: AppColors.darkTextSecondary),
      ),
    );
  }
}

// ── Category Chip ─────────────────────────────────────────────────────────────
class _CategoryChip extends StatelessWidget {
  final String label;
  final bool active;
  final VoidCallback onTap;

  const _CategoryChip(
      {required this.label, required this.active, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        Haptics.selection();
        onTap();
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        margin: const EdgeInsets.only(right: 8),
        padding:
            const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
        decoration: BoxDecoration(
          color: active ? AppColors.primary : AppColors.darkCard,
          borderRadius: BorderRadius.circular(AppRadius.full),
          border: Border.all(
            color: active ? AppColors.primary : AppColors.darkBorder,
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: active
                ? Colors.white
                : AppColors.darkTextSecondary,
          ),
        ),
      ),
    );
  }
}

// ── Picker Button ─────────────────────────────────────────────────────────────
class _PickerButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _PickerButton(
      {required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 16),
        decoration: BoxDecoration(
          color: AppColors.darkCard,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.darkBorder),
        ),
        child: Column(
          children: [
            Icon(icon, size: 28, color: AppColors.darkTextPrimary),
            const SizedBox(height: 6),
            Text(label,
                style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: AppColors.darkTextSecondary)),
          ],
        ),
      ),
    );
  }
}
