import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/tryon_provider.dart';
import '../utils/app_theme.dart';

class PhotoUploadButton extends StatelessWidget {
  final TryOnProvider provider;

  const PhotoUploadButton({super.key, required this.provider});

  @override
  Widget build(BuildContext context) {
    final hasPhoto = provider.hasPhoto;

    return GestureDetector(
      onTap: () => _showPicker(context),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        padding: const EdgeInsets.all(AppSpacing.md),
        decoration: hasPhoto ? AppDecoration.goldBorderCard : AppDecoration.gradientCard,
        child: Row(
          children: [
            // Photo preview or icon
            Container(
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                color: AppColors.surfaceLight,
                borderRadius: BorderRadius.circular(AppRadius.md),
                border: Border.all(
                  color: hasPhoto ? AppColors.gold : AppColors.surfaceBright,
                  width: hasPhoto ? 1.5 : 0.5,
                ),
                image: hasPhoto && provider.userPhotoBytes != null
                    ? DecorationImage(
                        image: MemoryImage(provider.userPhotoBytes!),
                        fit: BoxFit.cover,
                      )
                    : null,
              ),
              child: hasPhoto
                  ? null
                  : const Icon(
                      Icons.person_add_alt_1_rounded,
                      color: AppColors.gold,
                      size: 26,
                    ),
            ),
            const SizedBox(width: AppSpacing.md),

            // Text
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    hasPhoto ? 'Photo Ready' : 'Upload Your Photo',
                    style: AppText.titleMedium,
                  ),
                  const SizedBox(height: 2),
                  Text(
                    hasPhoto
                        ? 'Tap to change your photo'
                        : 'Take a selfie or pick from gallery',
                    style: AppText.bodyMedium.copyWith(fontSize: 12),
                  ),
                ],
              ),
            ),

            // Action icon
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: hasPhoto
                    ? AppColors.gold.withValues(alpha: 0.15)
                    : AppColors.surfaceLight,
                shape: BoxShape.circle,
              ),
              child: Icon(
                hasPhoto ? Icons.check_circle : Icons.camera_alt_rounded,
                color: AppColors.gold,
                size: 20,
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showPicker(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(AppRadius.xl)),
      ),
      builder: (_) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: AppSpacing.lg),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: AppColors.surfaceBright,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: AppSpacing.lg),
              Text('Choose Photo', style: AppText.titleLarge),
              const SizedBox(height: AppSpacing.lg),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _PickerOption(
                    icon: Icons.camera_alt_rounded,
                    label: 'Camera',
                    onTap: () {
                      Navigator.pop(context);
                      _pickImage(context, ImageSource.camera);
                    },
                  ),
                  _PickerOption(
                    icon: Icons.photo_library_rounded,
                    label: 'Gallery',
                    onTap: () {
                      Navigator.pop(context);
                      _pickImage(context, ImageSource.gallery);
                    },
                  ),
                ],
              ),
              const SizedBox(height: AppSpacing.md),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _pickImage(BuildContext context, ImageSource source) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(
      source: source,
      maxWidth: 1024,
      maxHeight: 1024,
      imageQuality: 85,
    );

    if (picked != null) {
      await provider.setUserPhoto(picked);
    }
  }
}

class _PickerOption extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _PickerOption({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        children: [
          Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              color: AppColors.gold.withValues(alpha: 0.1),
              shape: BoxShape.circle,
              border: Border.all(color: AppColors.gold.withValues(alpha: 0.3)),
            ),
            child: Icon(icon, color: AppColors.gold, size: 28),
          ),
          const SizedBox(height: 8),
          Text(label, style: AppText.labelMedium),
        ],
      ),
    );
  }
}
