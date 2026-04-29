import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import 'package:image_picker/image_picker.dart';
import '../services/face_swap_service.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';

class FaceSwapScreen extends StatefulWidget {
  const FaceSwapScreen({super.key});

  @override
  State<FaceSwapScreen> createState() => _FaceSwapScreenState();
}

class _FaceSwapScreenState extends State<FaceSwapScreen> {
  Uint8List? _sourceBytes;  // your face
  Uint8List? _targetBytes;  // photo to put face into
  Uint8List? _resultBytes;
  bool _isSwapping = false;
  String _status = '';

  final _picker = ImagePicker();

  Future<void> _pick(bool isSource, ImageSource src) async {
    final f = await _picker.pickImage(
      source: src, maxWidth: 1024, maxHeight: 1024, imageQuality: 88,
    );
    if (f == null) return;
    final bytes = await f.readAsBytes();
    setState(() {
      if (isSource) {
        _sourceBytes = bytes;
      } else {
        _targetBytes = bytes;
      }
      _resultBytes = null;
    });
  }

  Future<void> _swap() async {
    if (_sourceBytes == null || _targetBytes == null) return;
    Haptics.medium();
    setState(() { _isSwapping = true; _status = ''; _resultBytes = null; });

    try {
      final result = await FaceSwapService.swapFace(
        sourceImage: _sourceBytes!,
        targetImage: _targetBytes!,
        onStatus: (msg) { if (mounted) setState(() => _status = msg); },
      );
      if (mounted) setState(() { _resultBytes = result; _isSwapping = false; _status = ''; });
    } catch (e) {
      if (mounted) {
        setState(() { _isSwapping = false; _status = ''; });
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(e.toString()),
          backgroundColor: AppColors.error,
          behavior: SnackBarBehavior.floating,
        ));
      }
    }
  }

  void _showPickSheet(bool isSource) {
    Haptics.light();
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(width: 36, height: 4,
                decoration: BoxDecoration(color: const Color(0xFF38383A), borderRadius: BorderRadius.circular(2))),
              const SizedBox(height: 18),
              Text(isSource ? 'Pick Your Face' : 'Pick Target Photo',
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
              const SizedBox(height: 18),
              Row(children: [
                if (!kIsWeb) ...[
                  Expanded(child: _PickTile(
                    icon: Icons.camera_alt_rounded, label: 'Camera',
                    onTap: () { Navigator.pop(ctx); _pick(isSource, ImageSource.camera); },
                  )),
                  const SizedBox(width: 12),
                ],
                Expanded(child: _PickTile(
                  icon: Icons.photo_library_rounded, label: 'Gallery',
                  onTap: () { Navigator.pop(ctx); _pick(isSource, ImageSource.gallery); },
                )),
              ]),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final canSwap = _sourceBytes != null && _targetBytes != null && !_isSwapping;

    return Scaffold(
      backgroundColor: const Color(0xFF0D0D0F),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0D0D0F),
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, color: Colors.white, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text('Face Swap',
          style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w700)),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Column(
            children: [
              const SizedBox(height: 8),

              // ── Photo slots ──────────────────────────────────────────
              Expanded(
                child: _resultBytes != null
                    ? _ResultView(
                        bytes: _resultBytes!,
                        onReset: () => setState(() => _resultBytes = null),
                      )
                    : Row(
                        children: [
                          Expanded(
                            child: FadeInLeft(
                              duration: const Duration(milliseconds: 400),
                              child: _PhotoSlot(
                                label: 'Your Face',
                                subtitle: 'Face to apply',
                                icon: Icons.face_rounded,
                                accentColor: AppColors.primary,
                                bytes: _sourceBytes,
                                onTap: () => _showPickSheet(true),
                              ),
                            ),
                          ),
                          const SizedBox(width: 14),
                          Expanded(
                            child: FadeInRight(
                              duration: const Duration(milliseconds: 400),
                              child: _PhotoSlot(
                                label: 'Target Photo',
                                subtitle: 'Face gets replaced',
                                icon: Icons.image_rounded,
                                accentColor: const Color(0xFFD946EF),
                                bytes: _targetBytes,
                                onTap: () => _showPickSheet(false),
                              ),
                            ),
                          ),
                        ],
                      ),
              ),

              const SizedBox(height: 20),

              // ── Status ───────────────────────────────────────────────
              if (_isSwapping) ...[
                FadeIn(
                  child: Column(children: [
                    const SizedBox(
                      width: 28, height: 28,
                      child: CircularProgressIndicator(
                        color: Color(0xFFD946EF), strokeWidth: 2.5),
                    ),
                    const SizedBox(height: 10),
                    Text(_status.isEmpty ? 'Processing…' : _status,
                      style: const TextStyle(color: Color(0xFFAEAEB2), fontSize: 13)),
                  ]),
                ),
                const SizedBox(height: 20),
              ],

              // ── Swap button ──────────────────────────────────────────
              FadeInUp(
                duration: const Duration(milliseconds: 500),
                child: SizedBox(
                  width: double.infinity,
                  height: 56,
                  child: AnimatedOpacity(
                    duration: const Duration(milliseconds: 200),
                    opacity: canSwap ? 1.0 : 0.45,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFF7C3AED), Color(0xFFD946EF)],
                        ),
                        borderRadius: BorderRadius.circular(AppRadius.lg),
                        boxShadow: canSwap ? [BoxShadow(
                          color: const Color(0xFFD946EF).withValues(alpha: 0.35),
                          blurRadius: 18, offset: const Offset(0, 6),
                        )] : [],
                      ),
                      child: ElevatedButton.icon(
                        onPressed: canSwap ? _swap : null,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.transparent,
                          shadowColor: Colors.transparent,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(AppRadius.lg)),
                        ),
                        icon: const Icon(Icons.face_retouching_natural,
                            color: Colors.white, size: 22),
                        label: const Text('Swap Faces',
                          style: TextStyle(color: Colors.white, fontSize: 16,
                              fontWeight: FontWeight.w700)),
                      ),
                    ),
                  ),
                ),
              ),

              const SizedBox(height: 16),

              if (_sourceBytes == null || _targetBytes == null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Text(
                    _sourceBytes == null && _targetBytes == null
                        ? 'Pick both photos to get started'
                        : _sourceBytes == null
                            ? 'Now pick your face photo'
                            : 'Now pick the target photo',
                    style: const TextStyle(
                        color: Color(0xFF636366), fontSize: 12),
                    textAlign: TextAlign.center,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Photo Slot ────────────────────────────────────────────────────────────────
class _PhotoSlot extends StatelessWidget {
  final String label;
  final String subtitle;
  final IconData icon;
  final Color accentColor;
  final Uint8List? bytes;
  final VoidCallback onTap;

  const _PhotoSlot({
    required this.label,
    required this.subtitle,
    required this.icon,
    required this.accentColor,
    required this.bytes,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        decoration: BoxDecoration(
          color: const Color(0xFF1C1C1E),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(
            color: bytes != null
                ? accentColor.withValues(alpha: 0.5)
                : const Color(0xFF38383A),
            width: bytes != null ? 1.5 : 1,
          ),
        ),
        clipBehavior: Clip.antiAlias,
        child: bytes != null
            ? Stack(fit: StackFit.expand, children: [
                Image.memory(bytes!, fit: BoxFit.cover),
                Positioned(
                  bottom: 0, left: 0, right: 0,
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.bottomCenter,
                        end: Alignment.topCenter,
                        colors: [Colors.black.withValues(alpha: 0.7), Colors.transparent],
                      ),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.edit_rounded, color: Colors.white70, size: 12),
                        const SizedBox(width: 4),
                        Text('Change', style: const TextStyle(
                            color: Colors.white70, fontSize: 11, fontWeight: FontWeight.w500)),
                      ],
                    ),
                  ),
                ),
              ])
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 52, height: 52,
                    decoration: BoxDecoration(
                      color: accentColor.withValues(alpha: 0.1),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(icon, color: accentColor, size: 26),
                  ),
                  const SizedBox(height: 12),
                  Text(label, style: const TextStyle(
                      fontSize: 14, fontWeight: FontWeight.w600,
                      color: Colors.white)),
                  const SizedBox(height: 4),
                  Text(subtitle, style: const TextStyle(
                      fontSize: 11, color: Color(0xFF636366)),
                    textAlign: TextAlign.center),
                  const SizedBox(height: 14),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
                    decoration: BoxDecoration(
                      color: accentColor.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text('Pick Photo', style: TextStyle(
                        fontSize: 12, fontWeight: FontWeight.w600, color: accentColor)),
                  ),
                ],
              ),
      ),
    );
  }
}

// ── Result View ───────────────────────────────────────────────────────────────
class _ResultView extends StatelessWidget {
  final Uint8List bytes;
  final VoidCallback onReset;

  const _ResultView({required this.bytes, required this.onReset});

  @override
  Widget build(BuildContext context) {
    return FadeIn(
      duration: const Duration(milliseconds: 400),
      child: Stack(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(18),
            child: Image.memory(bytes,
                width: double.infinity, height: double.infinity, fit: BoxFit.cover),
          ),
          Positioned(
            top: 12, right: 12,
            child: GestureDetector(
              onTap: onReset,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.6),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.refresh_rounded, color: Colors.white, size: 14),
                  SizedBox(width: 5),
                  Text('Redo', style: TextStyle(color: Colors.white,
                      fontSize: 12, fontWeight: FontWeight.w600)),
                ]),
              ),
            ),
          ),
          Positioned(
            top: 12, left: 12,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
              decoration: BoxDecoration(
                color: const Color(0xFFD946EF).withValues(alpha: 0.85),
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.check_circle_rounded, color: Colors.white, size: 14),
                SizedBox(width: 5),
                Text('Swapped!', style: TextStyle(color: Colors.white,
                    fontSize: 12, fontWeight: FontWeight.w700)),
              ]),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Pick Tile ─────────────────────────────────────────────────────────────────
class _PickTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _PickTile({required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 18),
        decoration: BoxDecoration(
          color: const Color(0xFF2C2C2E),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFF38383A)),
        ),
        child: Column(children: [
          Icon(icon, size: 28, color: Colors.white),
          const SizedBox(height: 6),
          Text(label, style: const TextStyle(
              fontSize: 13, fontWeight: FontWeight.w500,
              color: Color(0xFFAEAEB2))),
        ]),
      ),
    );
  }
}
