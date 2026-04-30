import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:animate_do/animate_do.dart';
import 'package:cached_network_image/cached_network_image.dart';
import '../services/tryon_provider.dart';
import '../services/colab_tryon_service.dart';
import '../utils/app_theme.dart';
import 'decart_realtime_screen.dart';
import 'result_screen.dart';

class TryOnScreen extends StatelessWidget {
  const TryOnScreen({super.key});

  Future<void> _showColabDialog(BuildContext context) async {
    final current = await ColabTryOnService.getSavedUrl();
    if (!context.mounted) return;

    final controller = TextEditingController(text: current ?? '');

    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Colab GPU Server'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Paste your ngrok URL from the Colab notebook.\n'
              'When set, AI runs on your Colab GPU (3–8 s) instead '
              'of HuggingFace (30–60 s).',
              style: TextStyle(
                  fontSize: 13, color: AppColors.darkTextSecondary),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                hintText: 'https://xxxx.ngrok-free.app',
                contentPadding:
                    EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              ),
              keyboardType: TextInputType.url,
              autocorrect: false,
            ),
          ],
        ),
        actions: [
          if (current != null)
            TextButton(
              onPressed: () async {
                await ColabTryOnService.clearUrl();
                if (ctx.mounted) Navigator.pop(ctx);
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Colab URL removed')),
                  );
                }
              },
              child: const Text('Remove',
                  style: TextStyle(color: AppColors.error)),
            ),
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () async {
              final url = controller.text.trim();
              if (url.isNotEmpty) {
                await ColabTryOnService.saveUrl(url);
                if (ctx.mounted) Navigator.pop(ctx);
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                        content: Text('Colab GPU server saved')),
                  );
                }
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
    controller.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<TryOnProvider>();

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text('Try On'),
        actions: [
          IconButton(
            icon: const Icon(Icons.bolt_outlined, size: 22),
            tooltip: 'Colab GPU',
            onPressed: () => _showColabDialog(context),
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Column(
            children: [
              const SizedBox(height: 8),

              // ── Preview: side by side ──────────────────────────────
              Expanded(
                child: FadeInUp(
                  duration: const Duration(milliseconds: 400),
                  child: Row(
                    children: [
                      Expanded(
                        child: _PreviewCard(
                          label: 'You',
                          child: provider.userPhotoBytes != null
                              ? Image.memory(provider.userPhotoBytes!,
                                  fit: BoxFit.cover)
                              : const _EmptyPreview(
                                  icon: Icons.person_outline_rounded,
                                  text: 'No photo'),
                        ),
                      ),
                      Padding(
                        padding:
                            const EdgeInsets.symmetric(horizontal: 10),
                        child: Container(
                          width: 36,
                          height: 36,
                          decoration: BoxDecoration(
                            color: AppColors.darkCard,
                            shape: BoxShape.circle,
                            border: Border.all(
                                color: AppColors.darkBorder),
                          ),
                          child: const Icon(Icons.add_rounded,
                              color: AppColors.darkTextTertiary,
                              size: 20),
                        ),
                      ),
                      Expanded(
                        child: _PreviewCard(
                          label: 'Garment',
                          child: provider.garmentPhotoBytes != null
                              ? Image.memory(
                                  provider.garmentPhotoBytes!,
                                  fit: BoxFit.cover)
                              : provider.selectedCloth != null
                                  ? CachedNetworkImage(
                                      imageUrl: provider
                                          .selectedCloth!.imageUrl,
                                      fit: BoxFit.cover,
                                      placeholder: (context, url) =>
                                          const Center(
                                            child:
                                                CircularProgressIndicator(
                                                    strokeWidth: 2),
                                          ),
                                      errorWidget: (context, url, error) =>
                                          const _EmptyPreview(
                                            icon: Icons
                                                .broken_image_rounded,
                                            text: 'Failed to load',
                                          ),
                                    )
                                  : const _EmptyPreview(
                                      icon: Icons.checkroom_rounded,
                                      text: 'No garment'),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // ── Processing status ──────────────────────────────────
              if (provider.isProcessing)
                FadeIn(
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    margin: const EdgeInsets.only(bottom: 16),
                    decoration: BoxDecoration(
                      color: AppColors.darkSurface,
                      borderRadius: BorderRadius.circular(14),
                      border:
                          Border.all(color: AppColors.darkBorder),
                    ),
                    child: Column(
                      children: [
                        Row(
                          children: [
                            SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: AppColors.primary,
                                value: provider.progress > 0
                                    ? provider.progress
                                    : null,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text(
                                provider.statusMessage,
                                style: const TextStyle(
                                  fontSize: 13,
                                  color: AppColors.darkTextSecondary,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ),
                            Text(
                              '${(provider.progress * 100).toInt()}%',
                              style: const TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w600,
                                color: AppColors.primary,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(100),
                          child: LinearProgressIndicator(
                            value: provider.progress,
                            backgroundColor: AppColors.darkBorder,
                            valueColor:
                                const AlwaysStoppedAnimation<Color>(
                                    AppColors.primary),
                            minHeight: 4,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

              // ── Error ──────────────────────────────────────────────
              if (provider.status == TryOnStatus.error)
                FadeIn(
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(14),
                    margin: const EdgeInsets.only(bottom: 16),
                    decoration: BoxDecoration(
                      color: AppColors.error.withValues(alpha: 0.08),
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(
                          color:
                              AppColors.error.withValues(alpha: 0.25)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Icon(Icons.error_outline,
                                color: AppColors.error, size: 18),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                provider.errorMessage ??
                                    'Something went wrong',
                                style: const TextStyle(
                                  fontSize: 13,
                                  color: AppColors.error,
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        SizedBox(
                          width: double.infinity,
                          child: OutlinedButton(
                            onPressed: () => provider.runTryOn(),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: AppColors.error,
                              side: BorderSide(
                                  color: AppColors.error
                                      .withValues(alpha: 0.4)),
                              shape: RoundedRectangleBorder(
                                borderRadius:
                                    BorderRadius.circular(10),
                              ),
                            ),
                            child: const Text('Retry'),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

              // ── Live Try-On button ─────────────────────────────────
              if (provider.hasGarment && !provider.isProcessing)
                FadeInUp(
                  duration: const Duration(milliseconds: 400),
                  child: Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: SizedBox(
                      width: double.infinity,
                      height: 52,
                      child: OutlinedButton.icon(
                        onPressed: () {
                          final cloth = provider.selectedCloth;
                          final garmentBytes =
                              provider.garmentPhotoBytes;

                          DecartRealtimeScreen screen;
                          if (cloth != null) {
                            screen = DecartRealtimeScreen(
                              garmentUrl: cloth.imageUrl,
                              garmentName: cloth.name,
                            );
                          } else if (garmentBytes != null) {
                            screen = DecartRealtimeScreen.fromBytes(
                              garmentBytes: garmentBytes,
                              garmentName: 'Your Garment',
                            );
                          } else {
                            return;
                          }

                          if (context.mounted) {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                  builder: (_) => screen),
                            );
                          }
                        },
                        icon: const Icon(Icons.videocam_rounded,
                            size: 20),
                        label: const Text(
                          'Live Try-On',
                          style: TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        style: OutlinedButton.styleFrom(
                          foregroundColor:
                              AppColors.darkTextPrimary,
                          side: const BorderSide(
                              color: AppColors.darkBorder,
                              width: 1.5),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(
                                AppRadius.lg),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),

              // ── Generate button ────────────────────────────────────
              FadeInUp(
                duration: const Duration(milliseconds: 400),
                delay: const Duration(milliseconds: 100),
                child: SizedBox(
                  width: double.infinity,
                  height: 56,
                  child: ElevatedButton(
                    onPressed:
                        (provider.isProcessing || !provider.canTryOn)
                            ? null
                            : () async {
                                await provider.runTryOn();
                                if (context.mounted &&
                                    provider.status ==
                                        TryOnStatus.completed) {
                                  Navigator.push(
                                    context,
                                    PageRouteBuilder(
                                      pageBuilder: (context, anim, secondary) =>
                                          const ResultScreen(),
                                      transitionDuration:
                                          const Duration(
                                              milliseconds: 500),
                                      transitionsBuilder:
                                          (context, anim, _, child) =>
                                              FadeTransition(
                                                  opacity: anim,
                                                  child: child),
                                    ),
                                  );
                                }
                              },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.primary,
                      disabledBackgroundColor:
                          AppColors.darkCard,
                      foregroundColor: Colors.white,
                      disabledForegroundColor:
                          AppColors.darkTextTertiary,
                      shape: RoundedRectangleBorder(
                        borderRadius:
                            BorderRadius.circular(AppRadius.lg),
                      ),
                      elevation: 0,
                    ),
                    child: provider.isProcessing
                        ? Row(
                            mainAxisAlignment:
                                MainAxisAlignment.center,
                            children: [
                              const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              ),
                              const SizedBox(width: 12),
                              Text(
                                provider.statusMessage,
                                style: const TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w500,
                                  color: Colors.white,
                                ),
                              ),
                            ],
                          )
                        : const Row(
                            mainAxisAlignment:
                                MainAxisAlignment.center,
                            children: [
                              Icon(Icons.auto_awesome,
                                  size: 20, color: Colors.white),
                              SizedBox(width: 10),
                              Text(
                                'Generate Try-On',
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                  ),
                ),
              ),

              if (!provider.canTryOn && !provider.isProcessing)
                Padding(
                  padding: const EdgeInsets.only(top: 10),
                  child: Text(
                    !provider.hasPhoto
                        ? 'Upload your photo on the home screen first'
                        : 'Select or upload a garment first',
                    style: const TextStyle(
                      fontSize: 12,
                      color: AppColors.darkTextTertiary,
                    ),
                  ),
                ),

              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Preview Card ──────────────────────────────────────────────────────────────
class _PreviewCard extends StatelessWidget {
  final String label;
  final Widget child;

  const _PreviewCard({required this.label, required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.darkSurface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.darkBorder),
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        fit: StackFit.expand,
        children: [
          child,
          Positioned(
            top: 8,
            left: 8,
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.55),
                borderRadius: BorderRadius.circular(100),
              ),
              child: Text(
                label,
                style: const TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _EmptyPreview extends StatelessWidget {
  final IconData icon;
  final String text;

  const _EmptyPreview({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(icon, color: AppColors.darkTextTertiary, size: 36),
        const SizedBox(height: 8),
        Text(
          text,
          style: const TextStyle(
            fontSize: 12,
            color: AppColors.darkTextTertiary,
          ),
        ),
      ],
    );
  }
}
