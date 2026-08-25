import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../services/face_swap_service.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';

/// Turn the person in front of the camera into an avatar, then dress it.
///
/// Parity with the web demo's **My Avatar** tab — same endpoints, same steps,
/// same wording. Deliberately distinct from the "Face Filter (3D)" feature,
/// which tracks a cartoon head to a live camera and has nothing to do with this.
///
/// Photo, male/female, Create. Measurements are an optional refinement: they
/// choose a closer one of the eight curated base bodies, and left blank the
/// gender's average build is used. Seeing yourself on a body should not
/// require a tape measure.
///
/// A **501** from the body step means that build has no approved photograph
/// yet. The server does not substitute a neighbouring build, so this screen
/// says exactly that rather than implying the photo was at fault.
class AvatarEnrollScreen extends StatefulWidget {
  const AvatarEnrollScreen({super.key});

  @override
  State<AvatarEnrollScreen> createState() => _AvatarEnrollScreenState();
}

class _AvatarEnrollScreenState extends State<AvatarEnrollScreen> {
  final _picker = ImagePicker();

  Uint8List? _selfie;
  String _gender = 'male';
  final _chestCtl = TextEditingController();
  final _waistCtl = TextEditingController();

  String? _avatarId;
  bool _hasBody = false;
  bool _busy = false;
  String _status = '';
  String? _error;
  String? _provenance;

  /// The fitted image, once a garment has been tried. Held in memory rather
  /// than refetched: the try-on endpoint returns the picture itself.
  Uint8List? _fitted;
  String _category = 'upper';

  @override
  void dispose() {
    _chestCtl.dispose();
    _waistCtl.dispose();
    super.dispose();
  }

  double? _num(TextEditingController c) => double.tryParse(c.text.trim());

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

  Future<void> _create() async {
    final bytes = _selfie;
    if (bytes == null) return;

    final chest = _num(_chestCtl);
    final waist = _num(_waistCtl);
    if ((chest == null) != (waist == null)) {
      // The server enforces this too; catching it here saves a round trip and
      // explains why rather than returning a bare 422.
      setState(() => _error =
          'Give both chest and waist, or neither — one alone gives no ratio.');
      return;
    }

    setState(() {
      _busy = true;
      _error = null;
      _status = 'Checking the photo…';
    });
    try {
      final avatar = await FaceSwapService.createAvatar(
        photo: bytes, name: 'You', gender: _gender,
      );
      if (!mounted) return;
      setState(() => _status = 'Putting you on a body…');

      final body = await FaceSwapService.createAvatarBody(
        avatarId: avatar.id,
        gender: _gender,
        chestCm: chest,
        waistCm: waist,
        onStatus: (s) => setState(() => _status = s),
      );
      if (!mounted) return;

      final sel = body['selection'] as Map<String, dynamic>? ?? {};
      final base = body['base_body'] as Map<String, dynamic>? ?? {};
      Haptics.light();
      setState(() {
        _avatarId = avatar.id;
        _hasBody = true;
        _fitted = null;
        _status = '';
        _provenance =
            '${sel['from_defaults'] == true ? 'Standard ' : ''}'
            '${sel['gender']} · ${sel['build']} build. '
            'Base photo ${base['id']} (${base['license'] ?? 'licence unrecorded'}). '
            'Skin tone: ${body['skin_tone_match']}.';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = _clean(e));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _tryGarment() async {
    final id = _avatarId;
    if (id == null) return;
    final f = await _picker.pickImage(
        source: ImageSource.gallery, maxWidth: 1024, imageQuality: 90);
    if (f == null) return;

    setState(() {
      _busy = true;
      _error = null;
      _status = 'Fitting the garment…';
    });
    try {
      final bytes = await FaceSwapService.tryOnGarment(
        avatarId: id,
        garment: await f.readAsBytes(),
        category: _category,
      );
      if (!mounted) return;
      Haptics.light();
      setState(() {
        _fitted = bytes;
        _status = '';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = _clean(e));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _clean(Object e) => e.toString().replaceFirst('FaceSwapException: ', '');

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.darkBg,
      appBar: AppBar(
        backgroundColor: AppColors.darkBg,
        foregroundColor: AppColors.darkTextPrimary,
        elevation: 0,
        title: const Text('My Avatar'),
        actions: [
          if (_avatarId != null)
            TextButton(
              onPressed: () => Navigator.pop(context, _avatarId),
              child: const Text('Done',
                  style: TextStyle(
                      color: AppColors.primary, fontWeight: FontWeight.w600)),
            ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(20, 8, 20, 40),
        children: [
          _step('1', 'Your photo', done: _selfie != null),
          const SizedBox(height: 12),
          _photoPane(),

          const SizedBox(height: 28),
          _step('2', 'Body', done: _hasBody),
          const SizedBox(height: 12),
          Row(children: [
            Expanded(child: _choice('Male', _gender == 'male',
                () => setState(() => _gender = 'male'))),
            const SizedBox(width: 10),
            Expanded(child: _choice('Female', _gender == 'female',
                () => setState(() => _gender = 'female'))),
          ]),
          const SizedBox(height: 12),
          ExpansionTile(
            tilePadding: EdgeInsets.zero,
            title: const Text('Measurements (optional — picks a closer build)',
                style: TextStyle(
                    color: AppColors.darkTextSecondary, fontSize: 13)),
            iconColor: AppColors.darkTextSecondary,
            collapsedIconColor: AppColors.darkTextSecondary,
            children: [
              Row(children: [
                Expanded(child: _numField(_chestCtl, 'Chest cm', '102')),
                const SizedBox(width: 10),
                Expanded(child: _numField(_waistCtl, 'Waist cm', '86')),
              ]),
              const SizedBox(height: 8),
              const Align(
                alignment: Alignment.centerLeft,
                child: Text('Both or neither — one alone gives no ratio.',
                    style: TextStyle(
                        color: AppColors.darkTextTertiary, fontSize: 11.5)),
              ),
            ],
          ),
          const SizedBox(height: 14),
          _primary(
            label: _hasBody ? 'Rebuild avatar' : 'Create my avatar',
            onTap: _selfie == null || _busy ? null : _create,
          ),

          if (_error != null) ...[
            const SizedBox(height: 14),
            _errorBox(_error!),
          ],

          if (_hasBody) ...[
            const SizedBox(height: 28),
            _step('3', 'Try a garment'),
            const SizedBox(height: 12),
            Row(children: [
              for (final c in const ['upper', 'lower', 'dress']) ...[
                Expanded(
                  child: _choice(
                    const {'upper': 'Top', 'lower': 'Bottom', 'dress': 'Dress'}[c]!,
                    _category == c,
                    () => setState(() => _category = c),
                  ),
                ),
                if (c != 'dress') const SizedBox(width: 8),
              ],
            ]),
            const SizedBox(height: 12),
            _primary(
              label: 'Upload a garment',
              onTap: _busy ? null : _tryGarment,
            ),
            const SizedBox(height: 20),
            _resultPane(),
          ],
        ],
      ),
    );
  }

  // ── Pieces ────────────────────────────────────────────────────────────────

  Widget _step(String n, String title, {bool done = false}) => Row(children: [
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
              : Text(n,
                  style: const TextStyle(
                      fontSize: 11, color: AppColors.darkTextSecondary)),
        ),
        const SizedBox(width: 10),
        Text(title,
            style: const TextStyle(
                color: AppColors.darkTextPrimary,
                fontSize: 16,
                fontWeight: FontWeight.w600)),
      ]);

  Widget _photoPane() => AspectRatio(
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
                          style: TextStyle(
                              color: AppColors.darkTextSecondary, fontSize: 13)),
                      const SizedBox(height: 18),
                      Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                        _ghost(Icons.camera_alt_outlined, 'Camera',
                            () => _pick(ImageSource.camera)),
                        const SizedBox(width: 10),
                        _ghost(Icons.photo_library_outlined, 'Gallery',
                            () => _pick(ImageSource.gallery)),
                      ]),
                    ],
                  )
                : Stack(fit: StackFit.expand, children: [
                    Image.memory(_selfie!, fit: BoxFit.cover),
                    Positioned(
                      right: 10,
                      top: 10,
                      child: _ghost(Icons.refresh, 'Change',
                          () => setState(() => _selfie = null)),
                    ),
                  ]),
          ),
        ),
      );

  Widget _resultPane() {
    final id = _avatarId!;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: _fitted != null
            // The try-on endpoint returns the image itself, so it is shown from
            // memory rather than refetched.
            ? Image.memory(_fitted!, height: 420, fit: BoxFit.contain)
            : Image.network(
                // Cache-bust: the composite is written over the avatar's own
                // URL, so without this a rebuild shows the previous body.
                '${FaceSwapService.avatarImageUrl(id)}'
                '?v=${DateTime.now().millisecondsSinceEpoch}',
                height: 420,
                fit: BoxFit.contain,
                errorBuilder: (_, _, _) => Container(
                  height: 140,
                  color: AppColors.darkSurface,
                  alignment: Alignment.center,
                  child: const Text('Avatar image not available',
                      style: TextStyle(
                          color: AppColors.darkTextTertiary, fontSize: 13)),
                ),
              ),
      ),
      if (_provenance != null) ...[
        const SizedBox(height: 10),
        Text(_provenance!,
            style: const TextStyle(
                color: AppColors.darkTextTertiary, fontSize: 11.5, height: 1.45)),
      ],
      if (_fitted != null) ...[
        const SizedBox(height: 6),
        const Text('Your face is untouched — everything outside the garment '
            'is copied from the original, pixel for pixel.',
            style: TextStyle(
                color: AppColors.darkTextTertiary, fontSize: 11.5, height: 1.45)),
      ],
    ]);
  }

  Widget _choice(String label, bool active, VoidCallback onTap) => GestureDetector(
        onTap: () {
          Haptics.light();
          onTap();
        },
        child: Container(
          height: 46,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: active ? AppColors.darkTextPrimary : AppColors.darkSurface,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
                color: active ? AppColors.darkTextPrimary : AppColors.darkBorder),
          ),
          child: Text(label,
              style: TextStyle(
                  color: active ? AppColors.darkBg : AppColors.darkTextPrimary,
                  fontSize: 13.5,
                  fontWeight: FontWeight.w600)),
        ),
      );

  Widget _numField(TextEditingController c, String label, String hint) => TextField(
        controller: c,
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        style: const TextStyle(color: AppColors.darkTextPrimary),
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          labelStyle: const TextStyle(
              color: AppColors.darkTextSecondary, fontSize: 13),
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
        ),
      );

  Widget _primary({required String label, VoidCallback? onTap}) => SizedBox(
        height: 50,
        child: ElevatedButton(
          onPressed: onTap,
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.primary,
            disabledBackgroundColor: AppColors.darkCard,
            foregroundColor: Colors.white,
            disabledForegroundColor: AppColors.darkTextTertiary,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
          child: _busy
              ? Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                  const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white)),
                  if (_status.isNotEmpty) ...[
                    const SizedBox(width: 10),
                    Text(_status, style: const TextStyle(fontSize: 13)),
                  ],
                ])
              : Text(label,
                  style: const TextStyle(fontWeight: FontWeight.w600)),
        ),
      );

  Widget _ghost(IconData icon, String label, VoidCallback onTap) =>
      TextButton.icon(
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
                style: const TextStyle(
                    color: AppColors.error, fontSize: 13, height: 1.4)),
          ),
        ]),
      );
}
