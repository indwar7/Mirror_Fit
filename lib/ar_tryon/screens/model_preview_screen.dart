import 'package:flutter/material.dart';
import 'package:model_viewer_plus/model_viewer_plus.dart';
import 'package:cached_network_image/cached_network_image.dart';
import '../../services/fabric_service.dart';

/// Wardrobe preview screen.
///
/// Jacket tab  → interactive 3D GLB model with:
///               • Fabric type selector (fetched from Polyhaven API)
///               • Colour picker (18 swatches)
///               Both affect the model in real-time via model-viewer JS API.
///
/// Shirt / T-Shirt tabs → catalogue image grid.
class ModelPreviewScreen extends StatefulWidget {
  final String modelPath;
  final String modelName;

  const ModelPreviewScreen({
    super.key,
    required this.modelPath,
    this.modelName = '3D Preview',
  });

  @override
  State<ModelPreviewScreen> createState() => _ModelPreviewScreenState();
}

class _ModelPreviewScreenState extends State<ModelPreviewScreen> {
  // ── State ─────────────────────────────────────────────────────────────────
  int _typeIndex = 0;
  int _colorIndex = 0;
  int _fabricIndex = 0;
  bool _isApplyingFabric = false;
  bool _fabricsLoading = true;
  List<FabricDef> _fabrics = FabricService.defaults;
  dynamic _webController; // WebViewController (model_viewer_plus)

  // ── Static data ───────────────────────────────────────────────────────────
  static const _types = [
    (icon: Icons.checkroom_rounded,    label: 'Jacket'),
    (icon: Icons.dry_cleaning_rounded, label: 'Shirt'),
    (icon: Icons.style_rounded,        label: 'T-Shirt'),
  ];

  static const _palette = [
    (name: 'Charcoal',   hex: Color(0xFF374151)),
    (name: 'Black',      hex: Color(0xFF111111)),
    (name: 'White',      hex: Color(0xFFF5F5F5)),
    (name: 'Navy',       hex: Color(0xFF1B2A4A)),
    (name: 'Red',        hex: Color(0xFFDC2626)),
    (name: 'Royal Blue', hex: Color(0xFF1E40AF)),
    (name: 'Green',      hex: Color(0xFF166534)),
    (name: 'Purple',     hex: Color(0xFF7C3AED)),
    (name: 'Orange',     hex: Color(0xFFEA580C)),
    (name: 'Teal',       hex: Color(0xFF0F766E)),
    (name: 'Pink',       hex: Color(0xFFDB2777)),
    (name: 'Camel',      hex: Color(0xFFC19A6B)),
    (name: 'Brown',      hex: Color(0xFF7B3F00)),
    (name: 'Olive',      hex: Color(0xFF6B7B3A)),
    (name: 'Maroon',     hex: Color(0xFF800000)),
    (name: 'Slate',      hex: Color(0xFF475569)),
    (name: 'Sky Blue',   hex: Color(0xFF7DD3FC)),
    (name: 'Gold',       hex: Color(0xFFD4AF37)),
  ];

  static const _shirts = [
    (name: 'Classic White Shirt', url: 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400', price: '₹2,499'),
    (name: 'Navy Slim Fit',       url: 'https://images.unsplash.com/photo-1591944489410-16ec1074a18e?w=400', price: '₹2,799'),
    (name: 'Sky Blue Oxford',     url: 'https://images.unsplash.com/photo-1732605559386-bc59426d1b16?w=400', price: '₹3,199'),
  ];

  static const _tshirts = [
    (name: 'Black Crew Neck', url: 'https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=400', price: '₹1,499'),
    (name: 'Olive Polo',      url: 'https://images.unsplash.com/photo-1586363129094-d7a38564fae1?w=400', price: '₹1,999'),
  ];

  // ── Lifecycle ─────────────────────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _loadFabrics();
  }

  Future<void> _loadFabrics() async {
    final result = await FabricService.fetchFabrics();
    if (!mounted) return;
    setState(() {
      _fabrics = result;
      _fabricsLoading = false;
    });
  }

  // ── JS helpers ────────────────────────────────────────────────────────────

  /// Applies a colour via model-viewer's PBR material API.
  /// Self-retrying IIFE polls every 300 ms until the model is ready.
  Future<void> _applyColor(Color color) async {
    if (!mounted || _webController == null) return;
    final r = color.r.toStringAsFixed(4);
    final g = color.g.toStringAsFixed(4);
    final b = color.b.toStringAsFixed(4);
    try {
      await _webController.runJavaScript('''
        (function tryColor() {
          var mv = document.querySelector('model-viewer');
          if (!mv || !mv.model) { setTimeout(tryColor, 300); return; }
          for (var i = 0; i < mv.model.materials.length; i++) {
            mv.model.materials[i].pbrMetallicRoughness
              .setBaseColorFactor([$r, $g, $b, 1.0]);
          }
        })();
      ''');
    } catch (_) {}
  }

  /// Applies a fabric by:
  ///   1. Setting PBR roughness + metalness factors.
  ///   2. Loading a diffuse texture from the Polyhaven CDN via model-viewer's
  ///      `createTexture()` API (the WebView fetches the texture URL directly).
  ///   3. Keeping the current colour tint (baseColorFactor stays as set).
  Future<void> _applyFabric(FabricDef fabric, Color color) async {
    if (!mounted || _webController == null) return;

    setState(() => _isApplyingFabric = true);

    final r = color.r.toStringAsFixed(4);
    final g = color.g.toStringAsFixed(4);
    final b = color.b.toStringAsFixed(4);
    final rough  = fabric.roughness.toStringAsFixed(3);
    final metal  = fabric.metalness.toStringAsFixed(3);
    final texUrl = fabric.textureUrl;

    try {
      await _webController.runJavaScript('''
        (function tryFabric() {
          var mv = document.querySelector('model-viewer');
          if (!mv || !mv.model) { setTimeout(tryFabric, 300); return; }

          var mats = mv.model.materials;
          for (var i = 0; i < mats.length; i++) {
            var pbr = mats[i].pbrMetallicRoughness;

            // ── Physical properties (immediate) ───────────────────────
            pbr.setRoughnessFactor($rough);
            pbr.setMetallicFactor($metal);

            // ── Colour tint stays as selected ─────────────────────────
            pbr.setBaseColorFactor([$r, $g, $b, 1.0]);

            // ── Texture from Polyhaven CDN (async inside WebView) ─────
            // Capture pbr per-iteration to avoid the var-closure bug
            (function(currentPbr) {
              (async function() {
                try {
                  var tex = await mv.createTexture('$texUrl');
                  var texInfo = currentPbr.baseColorTexture;
                  if (texInfo) {
                    texInfo.setTexture(tex);
                  }
                } catch(e) {
                  // Texture load failed — PBR properties still applied,
                  // solid colour remains. No crash.
                  console.warn('[Lucy] Fabric texture unavailable:', e);
                }
              })();
            })(pbr);
          }
        })();
      ''');
    } catch (_) {}

    // Clear loading state — 2.5 s is enough for the texture to load on most
    // connections. If it takes longer the model still renders with PBR props.
    await Future.delayed(const Duration(milliseconds: 2500));
    if (mounted) setState(() => _isApplyingFabric = false);
  }

  // ── Build ─────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final chip   = _palette[_colorIndex];
    final fabric = _fabrics[_fabricIndex];

    return Scaffold(
      backgroundColor: const Color(0xFFF5F5F6),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new,
              color: Color(0xFF111111), size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Wardrobe Preview',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600,
              color: Color(0xFF111111)),
        ),
        centerTitle: true,
      ),
      body: Column(
        children: [
          _TypeTabBar(
            types: _types,
            selectedIndex: _typeIndex,
            onSelect: (i) => setState(() => _typeIndex = i),
          ),
          Expanded(
            child: _typeIndex == 0
                ? ModelViewer(
                    src: widget.modelPath,
                    alt: 'A 3D jacket model',
                    autoRotate: true,
                    autoRotateDelay: 0,
                    rotationPerSecond: '30deg',
                    cameraControls: true,
                    disableZoom: false,
                    backgroundColor: const Color(0xFFF5F5F6),
                    ar: false,
                    shadowIntensity: 1,
                    shadowSoftness: 1,
                    exposure: 1.0,
                    onWebViewCreated: (controller) {
                      _webController = controller;
                      // 500 ms: WebView HTML loads. JS retry handles model load.
                      Future.delayed(const Duration(milliseconds: 500), () {
                        if (!mounted) return;
                        _applyFabric(_fabrics[_fabricIndex],
                            _palette[_colorIndex].hex);
                      });
                    },
                  )
                : _CatalogueGrid(
                    items: _typeIndex == 1 ? _shirts : _tshirts),
          ),
          _BottomPanel(
            typeIndex: _typeIndex,
            fabrics: _fabrics,
            fabricsLoading: _fabricsLoading,
            fabricIndex: _fabricIndex,
            isApplyingFabric: _isApplyingFabric,
            palette: _palette,
            colorIndex: _colorIndex,
            chip: chip,
            currentFabric: fabric,
            onFabricSelect: (i) {
              if (!mounted || _isApplyingFabric) return;
              setState(() => _fabricIndex = i);
              _applyFabric(_fabrics[i], _palette[_colorIndex].hex);
            },
            onColorSelect: (i) {
              if (!mounted) return;
              setState(() => _colorIndex = i);
              _applyColor(_palette[i].hex);
            },
            onDone: () => Navigator.pop(context),
          ),
        ],
      ),
    );
  }
}

// ── _TypeTabBar ───────────────────────────────────────────────────────────────
class _TypeTabBar extends StatelessWidget {
  final List<({IconData icon, String label})> types;
  final int selectedIndex;
  final ValueChanged<int> onSelect;

  const _TypeTabBar({
    required this.types,
    required this.selectedIndex,
    required this.onSelect,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
      child: Row(
        children: List.generate(
          types.length,
          (i) => Expanded(
            child: _TypeTabItem(
              icon: types[i].icon,
              label: types[i].label,
              selected: i == selectedIndex,
              margin: i > 0 ? const EdgeInsets.only(left: 8) : EdgeInsets.zero,
              onTap: () => onSelect(i),
            ),
          ),
        ),
      ),
    );
  }
}

// ── _TypeTabItem ──────────────────────────────────────────────────────────────
class _TypeTabItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool selected;
  final EdgeInsets margin;
  final VoidCallback onTap;

  const _TypeTabItem({
    required this.icon,
    required this.label,
    required this.selected,
    required this.margin,
    required this.onTap,
  });

  static const _dark    = Color(0xFF111111);
  static const _mid     = Color(0xFF6B7280);
  static const _surface = Color(0xFFF8F8FA);
  static const _border  = Color(0xFFE8E8EC);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        margin: margin,
        padding: const EdgeInsets.symmetric(vertical: 10),
        decoration: BoxDecoration(
          color: selected ? _dark : _surface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: selected ? _dark : _border),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 20, color: selected ? Colors.white : _mid),
            const SizedBox(height: 4),
            Text(label,
                style: TextStyle(
                  fontSize: 12, fontWeight: FontWeight.w600,
                  color: selected ? Colors.white : _mid,
                )),
          ],
        ),
      ),
    );
  }
}

// ── _CatalogueGrid ────────────────────────────────────────────────────────────
class _CatalogueGrid extends StatelessWidget {
  final List<({String name, String url, String price})> items;
  const _CatalogueGrid({required this.items});

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      padding: const EdgeInsets.all(16),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2, childAspectRatio: 0.72,
        crossAxisSpacing: 12, mainAxisSpacing: 12,
      ),
      itemCount: items.length,
      itemBuilder: (ctx, i) => _CatalogueCard(
        key: ValueKey(items[i].url),
        name: items[i].name, url: items[i].url, price: items[i].price,
      ),
    );
  }
}

// ── _CatalogueCard ────────────────────────────────────────────────────────────
class _CatalogueCard extends StatelessWidget {
  final String name, url, price;
  const _CatalogueCard({super.key, required this.name, required this.url, required this.price});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE8E8EC)),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: CachedNetworkImage(
              imageUrl: url, fit: BoxFit.cover, width: double.infinity,
              placeholder: (ctx, p) => const Center(
                  child: CircularProgressIndicator(strokeWidth: 2)),
              errorWidget: (ctx, p, e) => const _ImgError(),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(10, 8, 10, 10),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(name,
                  style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600,
                      color: Color(0xFF111111)),
                  maxLines: 1, overflow: TextOverflow.ellipsis),
              const SizedBox(height: 2),
              Text(price, style: const TextStyle(fontSize: 11,
                  fontWeight: FontWeight.w500, color: Color(0xFF6B7280))),
            ]),
          ),
        ],
      ),
    );
  }
}

class _ImgError extends StatelessWidget {
  const _ImgError();
  @override
  Widget build(BuildContext context) => Container(
    color: const Color(0xFFF8F8FA),
    child: const Icon(Icons.image_not_supported_outlined,
        color: Color(0xFFD1D5DB), size: 36),
  );
}

// ── _BottomPanel ──────────────────────────────────────────────────────────────
/// Separated from the 3D viewer so colour / fabric changes never trigger a
/// ModelViewer rebuild.
class _BottomPanel extends StatelessWidget {
  final int typeIndex;
  final List<FabricDef> fabrics;
  final bool fabricsLoading;
  final int fabricIndex;
  final bool isApplyingFabric;
  final List<({String name, Color hex})> palette;
  final int colorIndex;
  final ({String name, Color hex}) chip;
  final FabricDef currentFabric;
  final ValueChanged<int> onFabricSelect;
  final ValueChanged<int> onColorSelect;
  final VoidCallback onDone;

  const _BottomPanel({
    required this.typeIndex,
    required this.fabrics,
    required this.fabricsLoading,
    required this.fabricIndex,
    required this.isApplyingFabric,
    required this.palette,
    required this.colorIndex,
    required this.chip,
    required this.currentFabric,
    required this.onFabricSelect,
    required this.onColorSelect,
    required this.onDone,
  });

  static const _gold   = Color(0xFFD4AF37);
  static const _border = Color(0xFFE8E8EC);
  static const _grey   = Color(0xFF6B7280);

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
        boxShadow: [BoxShadow(color: Color(0x12000000),
            blurRadius: 12, offset: Offset(0, -3))],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (typeIndex == 0) ...[
            // ── Fabric selector ──────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 14, 20, 6),
              child: Row(
                children: [
                  const Text('FABRIC',
                      style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700,
                          color: _grey, letterSpacing: 1.2)),
                  const Spacer(),
                  if (isApplyingFabric)
                    const Row(children: [
                      SizedBox(width: 12, height: 12,
                          child: CircularProgressIndicator(
                              strokeWidth: 1.8,
                              valueColor: AlwaysStoppedAnimation<Color>(_gold))),
                      SizedBox(width: 6),
                      Text('Applying…',
                          style: TextStyle(fontSize: 11, color: _gold,
                              fontWeight: FontWeight.w600)),
                    ])
                  else
                    Text('${currentFabric.emoji} ${currentFabric.name}',
                        style: const TextStyle(fontSize: 12,
                            fontWeight: FontWeight.w600, color: Color(0xFF374151))),
                ],
              ),
            ),
            if (fabricsLoading)
              const Padding(
                padding: EdgeInsets.fromLTRB(20, 4, 20, 8),
                child: LinearProgressIndicator(
                  backgroundColor: Color(0xFFE8E8EC),
                  valueColor: AlwaysStoppedAnimation<Color>(_gold),
                  minHeight: 2,
                ),
              )
            else
              SizedBox(
                height: 80,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  itemCount: fabrics.length,
                  separatorBuilder: (ctx, i) => const SizedBox(width: 10),
                  itemBuilder: (ctx, i) => _FabricChip(
                    key: ValueKey(fabrics[i].id),
                    fabric: fabrics[i],
                    active: i == fabricIndex,
                    disabled: isApplyingFabric,
                    onTap: () => onFabricSelect(i),
                  ),
                ),
              ),

            // ── Colour selector ──────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 6, 20, 6),
              child: Row(children: [
                const Text('COLOR',
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700,
                        color: _grey, letterSpacing: 1.2)),
                const Spacer(),
                Container(
                  width: 14, height: 14,
                  decoration: BoxDecoration(
                    color: chip.hex, shape: BoxShape.circle,
                    border: Border.all(color: _border),
                  ),
                ),
                const SizedBox(width: 5),
                Text(chip.name, style: const TextStyle(
                    fontSize: 12, fontWeight: FontWeight.w600,
                    color: Color(0xFF374151))),
              ]),
            ),
            SizedBox(
              height: 68,
              child: ListView.separated(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 20),
                itemCount: palette.length,
                separatorBuilder: (ctx, i) => const SizedBox(width: 8),
                itemBuilder: (ctx, i) => _ColorSwatch(
                  key: ValueKey(palette[i].name),
                  name: palette[i].name, color: palette[i].hex,
                  active: i == colorIndex,
                  onTap: () => onColorSelect(i),
                ),
              ),
            ),
            const SizedBox(height: 4),
          ],

          // ── Controls ─────────────────────────────────────────────────
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (typeIndex == 0)
                  const Text(
                    'Pinch to zoom  •  Drag to rotate  •  Two fingers to pan',
                    style: TextStyle(fontSize: 12, color: _grey),
                  ),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity, height: 52,
                  child: ElevatedButton(
                    onPressed: onDone,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF111111),
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14)),
                      elevation: 0,
                    ),
                    child: const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.check_rounded, size: 20),
                        SizedBox(width: 8),
                        Text('Done', style: TextStyle(
                            fontSize: 16, fontWeight: FontWeight.w600)),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ── _FabricChip ───────────────────────────────────────────────────────────────
class _FabricChip extends StatelessWidget {
  final FabricDef fabric;
  final bool active;
  final bool disabled;
  final VoidCallback onTap;

  const _FabricChip({
    super.key,
    required this.fabric,
    required this.active,
    required this.disabled,
    required this.onTap,
  });

  static const _gold   = Color(0xFFD4AF37);
  static const _border = Color(0xFFE8E8EC);
  static const _grey   = Color(0xFF6B7280);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: disabled ? null : onTap,
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 200),
        opacity: disabled && !active ? 0.5 : 1.0,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          width: 68,
          decoration: BoxDecoration(
            color: active ? fabric.swatch.withValues(alpha: 0.15) : const Color(0xFFF8F8FA),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: active ? _gold : _border,
              width: active ? 2 : 1,
            ),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Swatch dot
              Container(
                width: 28, height: 28,
                decoration: BoxDecoration(
                  color: fabric.swatch,
                  shape: BoxShape.circle,
                  border: Border.all(color: _border, width: 1),
                ),
                child: active
                    ? const Icon(Icons.check, size: 14, color: Colors.white)
                    : null,
              ),
              const SizedBox(height: 5),
              Text(fabric.emoji, style: const TextStyle(fontSize: 12)),
              const SizedBox(height: 2),
              Text(fabric.name,
                  style: TextStyle(
                    fontSize: 9, fontWeight: FontWeight.w600,
                    color: active ? _gold : _grey,
                  )),
            ],
          ),
        ),
      ),
    );
  }
}

// ── _ColorSwatch ──────────────────────────────────────────────────────────────
class _ColorSwatch extends StatelessWidget {
  final String name;
  final Color color;
  final bool active;
  final VoidCallback onTap;

  const _ColorSwatch({
    super.key,
    required this.name, required this.color,
    required this.active, required this.onTap,
  });

  static const _gold   = Color(0xFFD4AF37);
  static const _border = Color(0xFFE8E8EC);
  static const _grey   = Color(0xFF6B7280);

  @override
  Widget build(BuildContext context) {
    final lum = color.computeLuminance();
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            width: 36, height: 36,
            decoration: BoxDecoration(
              color: color, shape: BoxShape.circle,
              border: Border.all(
                color: active ? _gold : _border,
                width: active ? 2.5 : 1.5,
              ),
              boxShadow: active
                  ? [BoxShadow(color: _gold.withValues(alpha: 0.4),
                      blurRadius: 8, spreadRadius: 1)]
                  : null,
            ),
            child: active
                ? Icon(Icons.check, size: 14,
                    color: lum < 0.5 ? Colors.white : Colors.black)
                : null,
          ),
          const SizedBox(height: 3),
          Text(name, style: TextStyle(
            fontSize: 8, fontWeight: active ? FontWeight.w700 : FontWeight.w400,
            color: active ? _gold : _grey,
          )),
        ],
      ),
    );
  }
}
