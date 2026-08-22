import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../services/face_swap_service.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';

/// Turn the person in front of the camera into an avatar they can dress.
///
/// Two stages, because they fail for different reasons and the user should
/// only have to redo the one that failed:
///
///   1. **Face** — a selfie becomes the avatar. Rejected here if no face is
///      detectable, since an avatar with no findable face fails much later,
///      inside the swap, with a far less obvious error.
///   2. **Body** — measurements pick a pre-rendered figure and the enrolled
///      face is swapped onto it. Optional: the avatar already works for face
///      swap and lip-sync without it. It is only needed for try-on, which
///      needs a torso for the garment to sit on.
///
/// Pops the enrolled avatar's id when the user is done, so the caller can
/// refresh its catalogue and select it.
class AvatarEnrollScreen extends StatefulWidget {
  const AvatarEnrollScreen({super.key});

  @override
  State<AvatarEnrollScreen> createState() => _AvatarEnrollScreenState();
}

class _AvatarEnrollScreenState extends State<AvatarEnrollScreen> {
  final _picker = ImagePicker();

  Uint8List? _selfie;
  final _nameCtl = TextEditingController();

  String? _avatarId;
  bool _busy = false;
  String? _error;
  String _status = '';

  // Body stage
  final _chestCtl = TextEditingController();
  final _waistCtl = TextEditingController();
  final _shoulderCtl = TextEditingController();
  final _heightCtl = TextEditingController();
  bool _hasBody = false;
  Map<String, dynamic>? _selection;

  @override
  void dispose() {
    _nameCtl.dispose();
    _chestCtl.dispose();
    _waistCtl.dispose();
    _shoulderCtl.dispose();
    _heightCtl.dispose();
    super.dispose();
  }

  Future<void> _pick(ImageSource src) async {
    final f = await _picker.pickImage(
        source: src, maxWidth: 1024, maxHeight: 1024, imageQuality: 88);
    if (f == null) return;
    final bytes = await f.readAsBytes();
    if (!mounted) return;
    setState(() {
      _selfie = bytes;
      _error = null;
    });
  }

  Future<void> _enrol() async {
    final bytes = _selfie;
    if (bytes == null) return;
    setState(() {
      _busy = true;
      _error = null;
      _status = 'Checking the photo…';
    });
    try {
      final avatar = await FaceSwapService.createAvatar(
        photo: bytes,
        name: _nameCtl.text.trim().isEmpty ? 'You' : _nameCtl.text.trim(),
      );
      if (!mounted) return;
      Haptics.light();
      setState(() {
        _avatarId = avatar.id;
        _status = '';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = _clean(e));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _buildBody() async {
    final id = _avatarId;
    if (id == null) return;

    final chest = double.tryParse(_chestCtl.text.trim());
    final waist = double.tryParse(_waistCtl.text.trim());
    if (chest == null || waist == null) {
      setState(() => _error = 'Chest and waist are needed, in centimetres.');
      return;
    }

    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final res = await FaceSwapService.createAvatarBody(
        avatarId: id,
        chestCm: chest,
        waistCm: waist,
        shoulderCm: double.tryParse(_shoulderCtl.text.trim()),
        heightCm: double.tryParse(_heightCtl.text.trim()),
        onStatus: (s) => setState(() => _status = s),
      );
      if (!mounted) return;
      Haptics.light();
      setState(() {
        _hasBody = true;
        _selection = res['selection'] as Map<String, dynamic>?;
        _status = '';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = _clean(e));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _clean(Object e) =>
      e.toString().replaceFirst('FaceSwapException: ', '');

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.darkBg,
      appBar: AppBar(
        backgroundColor: AppColors.darkBg,
        foregroundColor: AppColors.darkTextPrimary,
        elevation: 0,
        title: const Text('Create my avatar'),
        actions: [
          if (_avatarId != null)
            TextButton(
              onPressed: () => Navigator.pop(context, _avatarId),
              child: const Text('Done',
                  style: TextStyle(color: AppColors.primary, fontWeight: FontWeight.w600)),
            ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(20, 8, 20, 40),
        children: [
          _stepLabel('1', 'Your face', done: _avatarId != null),
          const SizedBox(height: 12),
          _facePane(),
          if (_avatarId == null) ...[
            const SizedBox(height: 16),
            _nameField(),
            const SizedBox(height: 16),
            _primaryButton(
              label: 'Create avatar',
              onTap: _selfie == null || _busy ? null : _enrol,
            ),
          ],
          if (_error != null) ...[
            const SizedBox(height: 14),
            _errorBox(_error!),
          ],
          if (_avatarId != null) ...[
            const SizedBox(height: 32),
            _stepLabel('2', 'Your measurements', done: _hasBody, optional: true),
            const SizedBox(height: 6),
            const Text(
              'Only needed to try clothes on. The selfie is head-and-shoulders, '
              'and a garment needs a torso to sit on.',
              style: TextStyle(color: AppColors.darkTextSecondary, fontSize: 13, height: 1.45),
            ),
            const SizedBox(height: 16),
            Row(children: [
              Expanded(child: _numField(_chestCtl, 'Chest cm', '98')),
              const SizedBox(width: 10),
              Expanded(child: _numField(_waistCtl, 'Waist cm', '86')),
            ]),
            const SizedBox(height: 10),
            Row(children: [
              Expanded(child: _numField(_shoulderCtl, 'Shoulder cm', '45')),
              const SizedBox(width: 10),
              Expanded(child: _numField(_heightCtl, 'Height cm', '175')),
            ]),
            const SizedBox(height: 8),
            const Text(
              'Chest and waist are measured around the body. Shoulder is the '
              'width across the back. Centimetres, not inches.',
              style: TextStyle(color: AppColors.darkTextTertiary, fontSize: 11.5, height: 1.4),
            ),
            const SizedBox(height: 16),
            _primaryButton(
              label: _hasBody ? 'Rebuild body' : 'Build my body',
              onTap: _busy ? null : _buildBody,
            ),
            if (_hasBody) ...[
              const SizedBox(height: 20),
              _bodyPreview(),
            ],
          ],
        ],
      ),
    );
  }

  // ── Pieces ────────────────────────────────────────────────────────────────

  Widget _stepLabel(String n, String title, {bool done = false, bool optional = false}) {
    return Row(children: [
      Container(
        width: 22,
        height: 22,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: done ? AppColors.success : AppColors.darkCard,
          shape: BoxShape.circle,
        ),
        child: done
            ? const Icon(Icons.check, size: 14, color: Colors.white)
            : Text(n, style: const TextStyle(fontSize: 11, color: AppColors.darkTextSecondary)),
      ),
      const SizedBox(width: 10),
      Text(title,
          style: const TextStyle(
              color: AppColors.darkTextPrimary, fontSize: 16, fontWeight: FontWeight.w600)),
      if (optional) ...[
        const SizedBox(width: 8),
        const Text('optional',
            style: TextStyle(color: AppColors.darkTextTertiary, fontSize: 11.5)),
      ],
    ]);
  }

  Widget _facePane() {
    return AspectRatio(
      aspectRatio: 1,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Container(
          color: AppColors.darkSurface,
          child: _selfie == null
              ? Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.person_outline,
                        size: 44, color: AppColors.darkTextTertiary),
                    const SizedBox(height: 14),
                    const Text('A clear, front-facing photo',
                        style: TextStyle(color: AppColors.darkTextSecondary, fontSize: 13)),
                    const SizedBox(height: 18),
                    Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                      _ghostButton(Icons.camera_alt_outlined, 'Camera',
                          () => _pick(ImageSource.camera)),
                      const SizedBox(width: 10),
                      _ghostButton(Icons.photo_library_outlined, 'Gallery',
                          () => _pick(ImageSource.gallery)),
                    ]),
                  ],
                )
              : Stack(fit: StackFit.expand, children: [
                  Image.memory(_selfie!, fit: BoxFit.cover),
                  if (_avatarId == null)
                    Positioned(
                      right: 10,
                      top: 10,
                      child: _ghostButton(Icons.refresh, 'Change',
                          () => setState(() => _selfie = null)),
                    ),
                  if (_busy && _avatarId == null)
                    Container(
                      color: Colors.black.withValues(alpha: 0.55),
                      alignment: Alignment.center,
                      child: Column(mainAxisSize: MainAxisSize.min, children: [
                        const CircularProgressIndicator(color: AppColors.primary),
                        const SizedBox(height: 14),
                        Text(_status,
                            style: const TextStyle(color: Colors.white, fontSize: 13)),
                      ]),
                    ),
                ]),
        ),
      ),
    );
  }

  Widget _bodyPreview() {
    final id = _avatarId!;
    final sel = _selection;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Image.network(
          // Cache-bust so a rebuild after changing measurements is actually
          // shown rather than served from the widget's image cache.
          '${FaceSwapService.bodyImageUrl(id)}?v=${DateTime.now().millisecondsSinceEpoch}',
          height: 340,
          fit: BoxFit.contain,
          errorBuilder: (_, _, _) => Container(
            height: 140,
            color: AppColors.darkSurface,
            alignment: Alignment.center,
            child: const Text('Body image not available',
                style: TextStyle(color: AppColors.darkTextTertiary, fontSize: 13)),
          ),
        ),
      ),
      if (sel != null) ...[
        const SizedBox(height: 10),
        // Named plainly, because it IS an approximation: a fixed set of
        // figures, not a body measured from this person. Saying so is the
        // difference between a helpful preview and a false claim.
        Text(
          'Closest build: ${sel['size']} · ${sel['taper']}. '
          'Your measurements are stored exactly as entered and are what '
          'sizing advice uses — this figure is the nearest of a fixed set.',
          style: const TextStyle(
              color: AppColors.darkTextTertiary, fontSize: 11.5, height: 1.45),
        ),
      ],
    ]);
  }

  Widget _nameField() => TextField(
        controller: _nameCtl,
        style: const TextStyle(color: AppColors.darkTextPrimary),
        decoration: _decoration('Name (optional)', 'You'),
      );

  Widget _numField(TextEditingController c, String label, String hint) => TextField(
        controller: c,
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        style: const TextStyle(color: AppColors.darkTextPrimary),
        decoration: _decoration(label, hint),
      );

  InputDecoration _decoration(String label, String hint) => InputDecoration(
        labelText: label,
        hintText: hint,
        labelStyle: const TextStyle(color: AppColors.darkTextSecondary, fontSize: 13),
        hintStyle: const TextStyle(color: AppColors.darkTextTertiary),
        filled: true,
        fillColor: AppColors.darkSurface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide.none,
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.primary),
        ),
      );

  Widget _primaryButton({required String label, VoidCallback? onTap}) => SizedBox(
        height: 50,
        child: ElevatedButton(
          onPressed: onTap,
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.primary,
            disabledBackgroundColor: AppColors.darkCard,
            foregroundColor: Colors.white,
            disabledForegroundColor: AppColors.darkTextTertiary,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
          child: _busy
              ? const SizedBox(
                  width: 18, height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
              : Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
        ),
      );

  Widget _ghostButton(IconData icon, String label, VoidCallback onTap) => TextButton.icon(
        onPressed: onTap,
        icon: Icon(icon, size: 17),
        label: Text(label, style: const TextStyle(fontSize: 13)),
        style: TextButton.styleFrom(
          foregroundColor: AppColors.darkTextPrimary,
          backgroundColor: AppColors.darkCard,
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        ),
      );

  Widget _errorBox(String msg) => Container(
        padding: const EdgeInsets.all(13),
        decoration: BoxDecoration(
          color: AppColors.error.withValues(alpha: 0.12),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.error.withValues(alpha: 0.35)),
        ),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Icon(Icons.error_outline, color: AppColors.error, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(msg,
                style: const TextStyle(color: AppColors.error, fontSize: 13, height: 1.4)),
          ),
        ]),
      );
}
