import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import 'package:cached_network_image/cached_network_image.dart';
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
  Uint8List? _sourceBytes;   // user's face
  Uint8List? _targetBytes;   // custom photo (when not using avatar)
  AvatarModel? _selectedAvatar;
  Uint8List? _resultBytes;
  bool _isSwapping = false;
  String _status = '';

  // Avatar picker state
  List<AvatarModel> _avatars = [];
  bool _avatarsLoading = false;
  String _selectedCategory = 'All';

  final _picker = ImagePicker();

  // ── Categories ──────────────────────────────────────────────────────────
  List<String> get _categories {
    final cats = _avatars.map((a) => a.category).toSet().toList()..sort();
    return ['All', ...cats];
  }

  List<AvatarModel> get _filteredAvatars => _selectedCategory == 'All'
      ? _avatars
      : _avatars.where((a) => a.category == _selectedCategory).toList();

  // ── Helpers ──────────────────────────────────────────────────────────────
  bool get _hasTarget => _selectedAvatar != null || _targetBytes != null;
  bool get _canSwap => _sourceBytes != null && _hasTarget && !_isSwapping;

  Future<void> _pickSource(ImageSource src) async {
    final f = await _picker.pickImage(
        source: src, maxWidth: 1024, maxHeight: 1024, imageQuality: 88);
    if (f == null) return;
    setState(() {
      _sourceBytes = null;
      _resultBytes = null;
    });
    final bytes = await f.readAsBytes();
    setState(() => _sourceBytes = bytes);
  }

  Future<void> _pickTarget(ImageSource src) async {
    final f = await _picker.pickImage(
        source: src, maxWidth: 1024, maxHeight: 1024, imageQuality: 88);
    if (f == null) return;
    final bytes = await f.readAsBytes();
    setState(() {
      _targetBytes = bytes;
      _selectedAvatar = null;
      _resultBytes = null;
    });
  }

  Future<void> _loadAvatars() async {
    if (_avatars.isNotEmpty || _avatarsLoading) return;
    setState(() => _avatarsLoading = true);
    try {
      final list = await FaceSwapService.getAvatars();
      if (mounted) setState(() => _avatars = list);
    } catch (_) {
      // ignore — user can still upload custom target
    } finally {
      if (mounted) setState(() => _avatarsLoading = false);
    }
  }

  Future<void> _swap() async {
    if (_sourceBytes == null) return;
    Haptics.medium();
    setState(() {
      _isSwapping = true;
      _status = '';
      _resultBytes = null;
    });

    try {
      Uint8List result;
      if (_selectedAvatar != null) {
        result = await FaceSwapService.swapWithAvatar(
          sourceImage: _sourceBytes!,
          avatarId: _selectedAvatar!.id,
          onStatus: (m) { if (mounted) setState(() => _status = m); },
        );
      } else {
        result = await FaceSwapService.swapFace(
          sourceImage: _sourceBytes!,
          targetImage: _targetBytes!,
          onStatus: (m) { if (mounted) setState(() => _status = m); },
        );
      }
      if (mounted) setState(() { _resultBytes = result; _isSwapping = false; });
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

  // ── Source photo picker sheet ─────────────────────────────────────────────
  void _showSourceSheet() {
    Haptics.light();
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF1C1C1E),
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            _sheetHandle(),
            const SizedBox(height: 18),
            const Text('Pick Your Face',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600,
                    color: Colors.white)),
            const SizedBox(height: 18),
            Row(children: [
              if (!kIsWeb) ...[
                Expanded(child: _PickTile(
                    icon: Icons.camera_alt_rounded, label: 'Camera',
                    onTap: () {
                      Navigator.pop(ctx);
                      _pickSource(ImageSource.camera);
                    })),
                const SizedBox(width: 12),
              ],
              Expanded(child: _PickTile(
                  icon: Icons.photo_library_rounded, label: 'Gallery',
                  onTap: () {
                    Navigator.pop(ctx);
                    _pickSource(ImageSource.gallery);
                  })),
            ]),
          ]),
        ),
      ),
    );
  }

  // ── Target picker sheet (custom photo OR avatar) ──────────────────────────
  void _showTargetSheet() {
    Haptics.light();
    _loadAvatars();
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: const Color(0xFF1C1C1E),
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setSheet) => DraggableScrollableSheet(
          expand: false,
          initialChildSize: 0.82,
          minChildSize: 0.5,
          maxChildSize: 0.95,
          builder: (_, scroll) => Column(children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
              child: Column(children: [
                _sheetHandle(),
                const SizedBox(height: 16),
                const Text('Choose Target',
                    style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700,
                        color: Colors.white)),
                const SizedBox(height: 16),
                // ── Upload custom photo buttons ─────────────────────
                Row(children: [
                  if (!kIsWeb) ...[
                    Expanded(child: _PickTile(
                        icon: Icons.camera_alt_rounded, label: 'Camera',
                        onTap: () {
                          Navigator.pop(ctx);
                          _pickTarget(ImageSource.camera);
                        })),
                    const SizedBox(width: 10),
                  ],
                  Expanded(child: _PickTile(
                      icon: Icons.photo_library_rounded, label: 'Gallery',
                      onTap: () {
                        Navigator.pop(ctx);
                        _pickTarget(ImageSource.gallery);
                      })),
                ]),
                const SizedBox(height: 20),
                // ── Divider ─────────────────────────────────────────
                Row(children: [
                  const Expanded(child: Divider(color: Color(0xFF38383A))),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    child: Text('OR CHOOSE AN AVATAR',
                        style: TextStyle(fontSize: 11,
                            color: Colors.white.withValues(alpha: 0.4),
                            fontWeight: FontWeight.w600,
                            letterSpacing: 0.8)),
                  ),
                  const Expanded(child: Divider(color: Color(0xFF38383A))),
                ]),
                const SizedBox(height: 14),
              ]),
            ),
            // ── Category pills ──────────────────────────────────────
            SizedBox(
              height: 36,
              child: ListView.separated(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 20),
                itemCount: _categories.length,
                separatorBuilder: (_, __) => const SizedBox(width: 8),
                itemBuilder: (_, i) {
                  final cat = _categories[i];
                  final active = cat == _selectedCategory;
                  return GestureDetector(
                    onTap: () => setSheet(() => _selectedCategory = cat),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 180),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 8),
                      decoration: BoxDecoration(
                        color: active
                            ? Colors.white
                            : const Color(0xFF2C2C2E),
                        borderRadius: BorderRadius.circular(20),
                        border: active
                            ? null
                            : Border.all(color: const Color(0xFF38383A)),
                      ),
                      child: Text(cat,
                          style: TextStyle(
                              fontSize: 13,
                              fontWeight: active
                                  ? FontWeight.w700
                                  : FontWeight.w500,
                              color: active
                                  ? Colors.black
                                  : const Color(0xFFAEAEB2))),
                    ),
                  );
                },
              ),
            ),
            const SizedBox(height: 14),
            // ── Avatar grid ─────────────────────────────────────────
            Expanded(
              child: _avatarsLoading
                  ? const Center(
                      child: CircularProgressIndicator(
                          color: Colors.white, strokeWidth: 2))
                  : _avatars.isEmpty
                      ? Center(
                          child: Text('Start the local backend to see avatars',
                              style: TextStyle(
                                  color: Colors.white.withValues(alpha: 0.4),
                                  fontSize: 13)))
                      : GridView.builder(
                          controller: scroll,
                          padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
                          gridDelegate:
                              const SliverGridDelegateWithFixedCrossAxisCount(
                            crossAxisCount: 3,
                            crossAxisSpacing: 10,
                            mainAxisSpacing: 10,
                            childAspectRatio: 0.78,
                          ),
                          itemCount: _filteredAvatars.length,
                          itemBuilder: (_, i) {
                            final av = _filteredAvatars[i];
                            final selected = _selectedAvatar?.id == av.id;
                            return GestureDetector(
                              onTap: () {
                                Haptics.light();
                                setState(() {
                                  _selectedAvatar = av;
                                  _targetBytes = null;
                                  _resultBytes = null;
                                });
                                Navigator.pop(ctx);
                              },
                              child: AnimatedContainer(
                                duration: const Duration(milliseconds: 180),
                                decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(14),
                                  border: Border.all(
                                    color: selected
                                        ? Colors.white
                                        : Colors.transparent,
                                    width: 2.5,
                                  ),
                                ),
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(12),
                                  child: Stack(
                                    fit: StackFit.expand,
                                    children: [
                                      CachedNetworkImage(
                                        imageUrl: av.imageUrl,
                                        fit: BoxFit.cover,
                                        placeholder: (_, _) => Container(
                                            color: const Color(0xFF2C2C2E)),
                                        errorWidget: (_, _, _) =>
                                            const Icon(Icons.person,
                                                color: Color(0xFF636366)),
                                      ),
                                      Positioned(
                                        bottom: 0,
                                        left: 0,
                                        right: 0,
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(
                                              vertical: 6),
                                          decoration: BoxDecoration(
                                            gradient: LinearGradient(
                                              begin: Alignment.bottomCenter,
                                              end: Alignment.topCenter,
                                              colors: [
                                                Colors.black
                                                    .withValues(alpha: 0.8),
                                                Colors.transparent
                                              ],
                                            ),
                                          ),
                                          child: Text(av.name,
                                              textAlign: TextAlign.center,
                                              style: const TextStyle(
                                                  color: Colors.white,
                                                  fontSize: 12,
                                                  fontWeight:
                                                      FontWeight.w600)),
                                        ),
                                      ),
                                      if (selected)
                                        Positioned(
                                          top: 6,
                                          right: 6,
                                          child: Container(
                                            padding: const EdgeInsets.all(3),
                                            decoration: const BoxDecoration(
                                              color: Colors.white,
                                              shape: BoxShape.circle,
                                            ),
                                            child: const Icon(Icons.check,
                                                color: Colors.black,
                                                size: 12),
                                          ),
                                        ),
                                    ],
                                  ),
                                ),
                              ),
                            );
                          },
                        ),
            ),
          ]),
        ),
      ),
    );
  }

  Widget _sheetHandle() => Container(
        width: 36,
        height: 4,
        decoration: BoxDecoration(
            color: const Color(0xFF38383A),
            borderRadius: BorderRadius.circular(2)));

  // ── Build ─────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new,
              color: Colors.white, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text('Face Swap',
            style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700)),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Column(children: [
            const SizedBox(height: 12),

            // ── Swap button (now on top) ─────────────────────────────
            FadeInDown(
              duration: const Duration(milliseconds: 450),
              child: SizedBox(
                width: double.infinity,
                height: 58,
                child: AnimatedOpacity(
                  duration: const Duration(milliseconds: 200),
                  opacity: _canSwap ? 1.0 : 0.5,
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(AppRadius.lg),
                      boxShadow: _canSwap
                          ? [
                              BoxShadow(
                                  color: Colors.white.withValues(alpha: 0.25),
                                  blurRadius: 22,
                                  offset: const Offset(0, 8))
                            ]
                          : [],
                    ),
                    child: ElevatedButton.icon(
                      onPressed: _canSwap ? _swap : null,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.transparent,
                        shadowColor: Colors.transparent,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(AppRadius.lg)),
                      ),
                      icon: _isSwapping
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                  color: Colors.black, strokeWidth: 2.4))
                          : const Icon(Icons.face_retouching_natural,
                              color: Colors.black, size: 22),
                      label: Text(
                        _isSwapping
                            ? (_status.isEmpty ? 'Swapping…' : _status)
                            : (_selectedAvatar != null
                                ? 'Swap into ${_selectedAvatar!.name}'
                                : 'Swap Faces'),
                        style: const TextStyle(
                            color: Colors.black,
                            fontSize: 16,
                            fontWeight: FontWeight.w700),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ),
                ),
              ),
            ),

            const SizedBox(height: 10),

            // ── Helper hint (under button) ───────────────────────────
            SizedBox(
              height: 18,
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 200),
                child: Text(
                  _resultBytes != null
                      ? 'Tap Redo to start over'
                      : (_sourceBytes == null && !_hasTarget
                          ? 'Pick your face, then choose an avatar or photo'
                          : _sourceBytes == null
                              ? 'Pick your face photo'
                              : !_hasTarget
                                  ? 'Choose an avatar or upload a target photo'
                                  : 'Ready — tap Swap Faces above'),
                  key: ValueKey(
                      '${_sourceBytes != null}-$_hasTarget-${_resultBytes != null}'),
                  style: TextStyle(
                      color: _canSwap
                          ? Colors.white
                          : const Color(0xFF8E8E93),
                      fontSize: 12,
                      fontWeight:
                          _canSwap ? FontWeight.w600 : FontWeight.w400),
                  textAlign: TextAlign.center,
                ),
              ),
            ),

            const SizedBox(height: 14),

            // ── Photo slots ──────────────────────────────────────────
            Expanded(
              child: _resultBytes != null
                  ? _ResultView(
                      bytes: _resultBytes!,
                      onReset: () => setState(() => _resultBytes = null),
                    )
                  : Row(children: [
                      // Source slot (your face)
                      Expanded(
                        child: FadeInLeft(
                          duration: const Duration(milliseconds: 400),
                          child: _PhotoSlot(
                            label: 'Your Face',
                            subtitle: 'Face to apply',
                            icon: Icons.face_rounded,
                            accentColor: Colors.white,
                            bytes: _sourceBytes,
                            onTap: _showSourceSheet,
                          ),
                        ),
                      ),
                      const SizedBox(width: 14),
                      // Target slot (avatar or custom photo)
                      Expanded(
                        child: FadeInRight(
                          duration: const Duration(milliseconds: 400),
                          child: _selectedAvatar != null
                              ? _AvatarSlot(
                                  avatar: _selectedAvatar!,
                                  onTap: _showTargetSheet,
                                )
                              : _PhotoSlot(
                                  label: 'Target',
                                  subtitle: 'Avatar or photo',
                                  icon: Icons.auto_awesome_rounded,
                                  accentColor: const Color(0xFFE5E5EA),
                                  bytes: _targetBytes,
                                  onTap: _showTargetSheet,
                                ),
                        ),
                      ),
                    ]),
            ),

            const SizedBox(height: 12),
          ]),
        ),
      ),
    );
  }
}

// ── Avatar Slot (shows selected avatar as target) ─────────────────────────────
class _AvatarSlot extends StatelessWidget {
  final AvatarModel avatar;
  final VoidCallback onTap;
  const _AvatarSlot({required this.avatar, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(18),
          border: Border.all(
              color: Colors.white.withValues(alpha: 0.6), width: 1.5),
        ),
        clipBehavior: Clip.antiAlias,
        child: Stack(fit: StackFit.expand, children: [
          CachedNetworkImage(
            imageUrl: avatar.imageUrl,
            fit: BoxFit.cover,
            placeholder: (_, _) =>
                Container(color: const Color(0xFF2C2C2E)),
          ),
          // Name + category badge
          Positioned(
            bottom: 0, left: 0, right: 0,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.bottomCenter,
                  end: Alignment.topCenter,
                  colors: [
                    Colors.black.withValues(alpha: 0.85),
                    Colors.transparent
                  ],
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(avatar.name,
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 13,
                          fontWeight: FontWeight.w700)),
                  const SizedBox(height: 2),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.9),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(avatar.category,
                        style: const TextStyle(
                            color: Colors.black,
                            fontSize: 10,
                            fontWeight: FontWeight.w600)),
                  ),
                ],
              ),
            ),
          ),
          // Change badge
          Positioned(
            top: 8, right: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.6),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.swap_horiz_rounded, color: Colors.white70, size: 12),
                SizedBox(width: 3),
                Text('Change',
                    style: TextStyle(
                        color: Colors.white70,
                        fontSize: 10,
                        fontWeight: FontWeight.w500)),
              ]),
            ),
          ),
        ]),
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
                        colors: [
                          Colors.black.withValues(alpha: 0.7),
                          Colors.transparent
                        ],
                      ),
                    ),
                    child: const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.edit_rounded,
                            color: Colors.white70, size: 12),
                        SizedBox(width: 4),
                        Text('Change',
                            style: TextStyle(
                                color: Colors.white70,
                                fontSize: 11,
                                fontWeight: FontWeight.w500)),
                      ],
                    ),
                  ),
                ),
              ])
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 52,
                    height: 52,
                    decoration: BoxDecoration(
                      color: accentColor.withValues(alpha: 0.1),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(icon, color: accentColor, size: 26),
                  ),
                  const SizedBox(height: 12),
                  Text(label,
                      style: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: Colors.white)),
                  const SizedBox(height: 4),
                  Text(subtitle,
                      style: const TextStyle(
                          fontSize: 11, color: Color(0xFF636366)),
                      textAlign: TextAlign.center),
                  const SizedBox(height: 14),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 14, vertical: 7),
                    decoration: BoxDecoration(
                      color: accentColor.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text('Pick Photo',
                        style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: accentColor)),
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
      child: Stack(children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(18),
          child: Image.memory(bytes,
              width: double.infinity,
              height: double.infinity,
              fit: BoxFit.cover),
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
                Text('Redo',
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.w600)),
              ]),
            ),
          ),
        ),
        Positioned(
          top: 12, left: 12,
          child: Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.95),
              borderRadius: BorderRadius.circular(20),
            ),
            child: const Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(Icons.check_circle_rounded, color: Colors.black, size: 14),
              SizedBox(width: 5),
              Text('Swapped!',
                  style: TextStyle(
                      color: Colors.black,
                      fontSize: 12,
                      fontWeight: FontWeight.w700)),
            ]),
          ),
        ),
      ]),
    );
  }
}

// ── Pick Tile ─────────────────────────────────────────────────────────────────
class _PickTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  const _PickTile(
      {required this.icon, required this.label, required this.onTap});

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
          Text(label,
              style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                  color: Color(0xFFAEAEB2))),
        ]),
      ),
    );
  }
}
