import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../fitting/garment_fitter.dart';

/// A garment available for AR overlay (local transparent PNG).
class _LocalGarment {
  final String id;
  final String name;
  final String assetPath;
  final IconData icon;
  final GarmentType type;
  final double shoulderPadding;
  final double shoulderLineInImage;

  const _LocalGarment({
    required this.id,
    required this.name,
    required this.assetPath,
    required this.icon,
    this.type = GarmentType.top,
    this.shoulderPadding = 1.35,
    this.shoulderLineInImage = 0.15,
  });
}

/// Bottom bar for selecting garments to try on in AR mode.
class GarmentSelectorBar extends StatefulWidget {
  final void Function(GarmentConfig config, ui.Image? image) onGarmentSelected;

  const GarmentSelectorBar({
    super.key,
    required this.onGarmentSelected,
  });

  @override
  State<GarmentSelectorBar> createState() => _GarmentSelectorBarState();
}

class _GarmentSelectorBarState extends State<GarmentSelectorBar> {
  int _selectedIndex = 0; // Default: jacket selected
  final Map<String, ui.Image> _loadedImages = {};
  String? _loadingId;

  static const _garments = [
    _LocalGarment(
      id: 'jacket_dark',
      name: 'Jacket',
      assetPath: 'assets/images/garment_jacket_dark.png',
      icon: Icons.local_fire_department,
      type: GarmentType.top,
      // 3D jacket: collar base ~30% from top; body ~60% of image width
      shoulderPadding: 1.5,
      shoulderLineInImage: 0.30,
    ),
    _LocalGarment(
      id: 'tshirt_navy',
      name: 'T-Shirt',
      assetPath: 'assets/images/garment_tshirt_navy.png',
      icon: Icons.checkroom,
      type: GarmentType.top,
      // Flat 2D: shoulder seam is ~12% from top; shoulders span ~70% of image width
      shoulderPadding: 1.4,
      shoulderLineInImage: 0.12,
    ),
    _LocalGarment(
      id: 'shirt_white',
      name: 'Shirt',
      assetPath: 'assets/images/garment_shirt_white.png',
      icon: Icons.dry_cleaning,
      type: GarmentType.top,
      // Flat 2D: same proportions as t-shirt
      shoulderPadding: 1.4,
      shoulderLineInImage: 0.12,
    ),
    _LocalGarment(
      id: 'kurta_cream',
      name: 'Kurta',
      assetPath: 'assets/images/garment_kurta_cream.png',
      icon: Icons.style,
      type: GarmentType.top,
      // Flat 2D: same proportions as t-shirt
      shoulderPadding: 1.4,
      shoulderLineInImage: 0.12,
    ),
  ];

  Future<void> _selectGarment(int index) async {
    final garment = _garments[index];
    setState(() {
      _selectedIndex = index;
      _loadingId = garment.id;
    });

    // Check cache
    if (_loadedImages.containsKey(garment.id)) {
      _applyGarment(garment, _loadedImages[garment.id]!);
      setState(() => _loadingId = null);
      return;
    }

    // Load from local asset
    try {
      final data = await rootBundle.load(garment.assetPath);
      final codec = await ui.instantiateImageCodec(
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
      );
      final frame = await codec.getNextFrame();
      final image = frame.image;

      _loadedImages[garment.id] = image;

      if (_selectedIndex == index && mounted) {
        _applyGarment(garment, image);
      }
    } catch (e) {
      debugPrint('Failed to load garment: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to load ${garment.name}'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }

    if (mounted) setState(() => _loadingId = null);
  }

  void _applyGarment(_LocalGarment garment, ui.Image image) {
    final config = GarmentConfig(
      modelPath: garment.assetPath,
      type: garment.type,
      originalWidth: image.width.toDouble(),
      originalHeight: image.height.toDouble(),
      shoulderPadding: garment.shoulderPadding,
      shoulderLineInImage: garment.shoulderLineInImage,
    );
    widget.onGarmentSelected(config, image);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 80,
      padding: const EdgeInsets.symmetric(horizontal: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white12),
      ),
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: _garments.length,
        itemBuilder: (context, index) {
          final garment = _garments[index];
          final isSelected = index == _selectedIndex;
          final isLoading = _loadingId == garment.id;

          return GestureDetector(
            onTap: () => _selectGarment(index),
            child: Container(
              width: 70,
              margin: const EdgeInsets.symmetric(horizontal: 4, vertical: 8),
              decoration: BoxDecoration(
                color: isSelected ? Colors.white24 : Colors.transparent,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: isSelected ? Colors.white54 : Colors.white12,
                  width: isSelected ? 2 : 1,
                ),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (isLoading)
                    const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  else
                    Icon(
                      garment.icon,
                      color: isSelected ? Colors.white : Colors.white54,
                      size: 24,
                    ),
                  const SizedBox(height: 4),
                  Text(
                    garment.name,
                    style: TextStyle(
                      color: isSelected ? Colors.white : Colors.white54,
                      fontSize: 10,
                      fontWeight:
                          isSelected ? FontWeight.bold : FontWeight.normal,
                    ),
                    overflow: TextOverflow.ellipsis,
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
