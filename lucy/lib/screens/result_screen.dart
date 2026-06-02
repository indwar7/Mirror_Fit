import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:animate_do/animate_do.dart';
import 'package:share_plus/share_plus.dart';
import '../services/tryon_provider.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';

class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _fadeCtrl;
  late Animation<double> _fadeAnim;
  bool _showingFaceSwap = false;

  @override
  void initState() {
    super.initState();
    _fadeCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    )..forward();
    _fadeAnim = CurvedAnimation(parent: _fadeCtrl, curve: Curves.easeOut);
  }

  @override
  void dispose() {
    _fadeCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<TryOnProvider>();

    if (provider.faceSwapBytes != null && !_showingFaceSwap) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) setState(() => _showingFaceSwap = true);
      });
    }

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          _showingFaceSwap && provider.faceSwapBytes != null
              ? 'Your Look'
              : 'Result',
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.share_outlined, size: 22),
            onPressed: () => _share(context, provider),
          ),
        ],
      ),
      body: SafeArea(
        child: FadeTransition(
          opacity: _fadeAnim,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Column(
              children: [
                const SizedBox(height: 8),

                // ── Result image ───────────────────────────────────────
                Expanded(
                  child: FadeInUp(
                    duration: const Duration(milliseconds: 500),
                    child: Stack(
                      children: [
                        Container(
                          width: double.infinity,
                          decoration: BoxDecoration(
                            color: AppColors.darkSurface,
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(color: AppColors.darkBorder),
                          ),
                          clipBehavior: Clip.antiAlias,
                          child: _buildResult(provider),
                        ),

                        // Toggle pill — shown only when face swap ready
                        if (provider.faceSwapBytes != null)
                          Positioned(
                            bottom: 12,
                            left: 0,
                            right: 0,
                            child: Center(
                              child: _TogglePill(
                                showingFaceSwap: _showingFaceSwap,
                                onToggle: (val) =>
                                    setState(() => _showingFaceSwap = val),
                              ),
                            ),
                          ),

                        // Face-swap loading overlay
                        if (provider.isFaceSwapping)
                          Positioned.fill(
                            child: Container(
                              decoration: BoxDecoration(
                                color: Colors.black.withValues(alpha: 0.55),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const SizedBox(
                                    width: 32,
                                    height: 32,
                                    child: CircularProgressIndicator(
                                      color: AppColors.primary,
                                      strokeWidth: 2.5,
                                    ),
                                  ),
                                  const SizedBox(height: 12),
                                  Text(
                                    provider.faceSwapMessage,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 13,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 14),

                // ── Face swap button ───────────────────────────────────
                if (provider.resultBytes != null) ...[
                  FadeInUp(
                    duration: const Duration(milliseconds: 400),
                    delay: const Duration(milliseconds: 80),
                    child: _FaceSwapButton(
                      provider: provider,
                      onTap: () {
                        Haptics.medium();
                        setState(() => _showingFaceSwap = false);
                        provider.runFaceSwap();
                      },
                    ),
                  ),
                  const SizedBox(height: 10),
                ],

                // ── Info row ───────────────────────────────────────────
                FadeInUp(
                  duration: const Duration(milliseconds: 500),
                  delay: const Duration(milliseconds: 100),
                  child: Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: AppColors.darkSurface,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: AppColors.darkBorder),
                    ),
                    child: Row(
                      children: [
                        Container(
                          width: 40,
                          height: 40,
                          decoration: BoxDecoration(
                            color: AppColors.success.withValues(alpha: 0.12),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(Icons.check_circle_outline,
                              color: AppColors.success, size: 22),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                provider.garmentDisplayName,
                                style: const TextStyle(
                                  fontSize: 15,
                                  fontWeight: FontWeight.w600,
                                  color: AppColors.darkTextPrimary,
                                ),
                              ),
                              const SizedBox(height: 2),
                              Text(
                                _showingFaceSwap &&
                                        provider.faceSwapBytes != null
                                    ? 'Face-personalised try-on'
                                    : 'AI-generated try-on result',
                                style: const TextStyle(
                                  fontSize: 12,
                                  color: AppColors.darkTextTertiary,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 12),

                // ── Action buttons ─────────────────────────────────────
                FadeInUp(
                  duration: const Duration(milliseconds: 500),
                  delay: const Duration(milliseconds: 200),
                  child: Row(
                    children: [
                      Expanded(
                        child: SizedBox(
                          height: 52,
                          child: OutlinedButton.icon(
                            onPressed: () {
                              Haptics.light();
                              _share(context, provider);
                            },
                            icon: const Icon(Icons.download_rounded,
                                size: 20),
                            label: const Text('Save',
                                style: TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.w600)),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: AppColors.darkTextPrimary,
                              side: const BorderSide(
                                  color: AppColors.darkBorder),
                              shape: RoundedRectangleBorder(
                                borderRadius:
                                    BorderRadius.circular(AppRadius.lg),
                              ),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: SizedBox(
                          height: 52,
                          child: ElevatedButton.icon(
                            onPressed: () {
                              Haptics.light();
                              Navigator.popUntil(
                                  context, (r) => r.isFirst);
                            },
                            icon: const Icon(Icons.refresh_rounded,
                                size: 20, color: Colors.white),
                            label: const Text('Try Another',
                                style: TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.w600)),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppColors.primary,
                              foregroundColor: Colors.white,
                              shape: RoundedRectangleBorder(
                                borderRadius:
                                    BorderRadius.circular(AppRadius.lg),
                              ),
                              elevation: 0,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 16),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildResult(TryOnProvider provider) {
    if (_showingFaceSwap && provider.faceSwapBytes != null) {
      return Image.memory(provider.faceSwapBytes!, fit: BoxFit.cover);
    }
    if (provider.resultBytes != null) {
      return Image.memory(provider.resultBytes!, fit: BoxFit.cover);
    }
    if (provider.resultFrontUrl != null) {
      return CachedNetworkImage(
        imageUrl: provider.resultFrontUrl!,
        fit: BoxFit.cover,
        placeholder: (context, url) => _buildLoading(),
        errorWidget: (context, url, error) => _buildError(),
      );
    }
    return provider.userPhotoBytes != null
        ? Image.memory(provider.userPhotoBytes!, fit: BoxFit.cover)
        : _buildLoading();
  }

  Widget _buildLoading() {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 28,
            height: 28,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              color: AppColors.darkBorder,
            ),
          ),
          SizedBox(height: 12),
          Text('Loading...',
              style: TextStyle(
                  fontSize: 12, color: AppColors.darkTextTertiary)),
        ],
      ),
    );
  }

  Widget _buildError() {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.broken_image_outlined,
              color: AppColors.darkTextTertiary, size: 36),
          SizedBox(height: 8),
          Text('Could not load image',
              style: TextStyle(
                  fontSize: 12, color: AppColors.darkTextTertiary)),
        ],
      ),
    );
  }

  void _share(BuildContext context, TryOnProvider provider) {
    Haptics.light();
    if (provider.resultFrontUrl != null || provider.resultBytes != null) {
      Share.share('Check out this virtual try-on from LUCY!');
    }
  }

}

// ── Face Swap Button ──────────────────────────────────────────────────────────
class _FaceSwapButton extends StatelessWidget {
  final TryOnProvider provider;
  final VoidCallback onTap;

  const _FaceSwapButton({required this.provider, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final hasResult = provider.faceSwapBytes != null;
    final isLoading = provider.isFaceSwapping;

    return GestureDetector(
      onTap: isLoading ? null : onTap,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(vertical: 13, horizontal: 16),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF7C3AED), AppColors.primary],
            begin: Alignment.centerLeft,
            end: Alignment.centerRight,
          ),
          borderRadius: BorderRadius.circular(AppRadius.lg),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              hasResult
                  ? Icons.face_retouching_natural
                  : Icons.face_outlined,
              color: Colors.white,
              size: 20,
            ),
            const SizedBox(width: 8),
            Text(
              hasResult ? 'Redo face swap' : 'Personalise with your face',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Original ↔ My Face toggle pill ───────────────────────────────────────────
class _TogglePill extends StatelessWidget {
  final bool showingFaceSwap;
  final ValueChanged<bool> onToggle;

  const _TogglePill(
      {required this.showingFaceSwap, required this.onToggle});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(30),
      ),
      padding: const EdgeInsets.all(3),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _PillTab(
            label: 'Original',
            selected: !showingFaceSwap,
            onTap: () => onToggle(false),
          ),
          _PillTab(
            label: 'My Face',
            selected: showingFaceSwap,
            onTap: () => onToggle(true),
          ),
        ],
      ),
    );
  }
}

class _PillTab extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _PillTab(
      {required this.label, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding:
            const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
        decoration: BoxDecoration(
          color: selected ? Colors.white : Colors.transparent,
          borderRadius: BorderRadius.circular(26),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: selected ? Colors.black : Colors.white70,
            fontSize: 12,
            fontWeight:
                selected ? FontWeight.w600 : FontWeight.w400,
          ),
        ),
      ),
    );
  }
}
