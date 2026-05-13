import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';
import 'package:image_picker/image_picker.dart';
import 'package:animate_do/animate_do.dart';
import '../services/colab_tryon_service.dart';
import '../services/face_swap_service.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';
import 'web_ar_view_stub.dart'
    if (dart.library.html) 'web_ar_view_web.dart';

class DecartRealtimeScreen extends StatefulWidget {
  final String garmentUrl;
  final String garmentName;
  final bool autoShowFaceSwap;

  const DecartRealtimeScreen({
    super.key,
    required this.garmentUrl,
    required this.garmentName,
    this.autoShowFaceSwap = false,
  });

  static DecartRealtimeScreen fromBytes({
    required Uint8List garmentBytes,
    required String garmentName,
  }) {
    final b64 = base64Encode(garmentBytes);
    return DecartRealtimeScreen(
      garmentUrl: 'data:image/jpeg;base64,$b64',
      garmentName: garmentName,
    );
  }

  @override
  State<DecartRealtimeScreen> createState() => _DecartRealtimeScreenState();
}

class _DecartRealtimeScreenState extends State<DecartRealtimeScreen>
    with TickerProviderStateMixin {
  InAppWebViewController? _webCtrl;
  late final String _viewId;

  // Resolved WebSocket URL for the AI backend (set in initState).
  // Falls back to a clear error string if the user hasn't configured a server.
  String _backendWsUrl = '';
  String? _htmlToLoad;

  // Face-swap state
  Uint8List? _sourceFaceBytes;
  String? _sourceFaceLabel;
  Uint8List? _faceSwapResult;
  bool _isFaceSwapping = false;
  String _faceSwapStatus = '';
  bool _showFaceSwapOverlay = false;

  // Live indicator pulse
  late AnimationController _liveCtrl;
  late Animation<double> _liveAnim;

  @override
  void initState() {
    super.initState();
    _viewId = 'decart-ar-${widget.garmentUrl.hashCode}';

    _liveCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);
    _liveAnim = Tween<double>(begin: 0.4, end: 1.0)
        .animate(CurvedAnimation(parent: _liveCtrl, curve: Curves.easeInOut));

    _initBackendThenStart();

    if (widget.autoShowFaceSwap) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        Future.delayed(const Duration(milliseconds: 800), () {
          if (mounted) _showPickerSheet();
        });
      });
    }
  }

  /// Default backend (EC2 g5.xlarge). Overridden by whatever the user
  /// saves through the ⚡ bolt-icon dialog on the Try-On screen.
  static const String _defaultBackendUrl = 'http://13.203.75.149:8000';

  /// Resolve backend URL: prefer saved override, otherwise the default EC2
  /// host. Converts to a ws:// or wss:// URL pointing at /ws/tryon, then
  /// hands the HTML to the webview (or web iframe).
  Future<void> _initBackendThenStart() async {
    final saved = await ColabTryOnService.getSavedUrl();
    final base = (saved != null && saved.trim().isNotEmpty)
        ? saved
        : _defaultBackendUrl;
    _backendWsUrl = _toWsUrl(base);

    final html = _buildHtml(widget.garmentUrl, widget.garmentName, _backendWsUrl);

    if (kIsWeb) {
      registerWebArView(html, _viewId);
    } else {
      _htmlToLoad = html;
      _loadTryonPage();
    }
  }

  /// Convert https://host[:port][/path] → wss://host[:port]/ws/tryon
  /// Returns an empty string when no URL is configured — the HTML treats
  /// that as a clear "configure server" status message instead of failing
  /// silently against the old hard-coded placeholder.
  static String _toWsUrl(String? base) {
    if (base == null || base.trim().isEmpty) return '';
    var u = base.trim().replaceAll(RegExp(r'/$'), '');
    if (u.startsWith('https://')) {
      u = 'wss://${u.substring('https://'.length)}';
    } else if (u.startsWith('http://')) {
      u = 'ws://${u.substring('http://'.length)}';
    } else if (!u.startsWith('ws://') && !u.startsWith('wss://')) {
      // Bare host → assume secure
      u = 'wss://$u';
    }
    return '$u/ws/tryon';
  }

  void _loadTryonPage() {
    final html = _htmlToLoad;
    final ctrl = _webCtrl;
    if (html == null || ctrl == null) return;
    // baseUrl=http://localhost: localhost counts as a secure context for
    // getUserMedia in modern Chromium, AND because the page itself is
    // http://, the browser doesn't flag ws://13.203.75.149 as mixed
    // content. (https://localhost would force the page to be "secure"
    // and would then block any ws:// upgrade attempts.)
    ctrl.loadData(
      data: html,
      baseUrl: WebUri('http://localhost'),
      mimeType: 'text/html',
      encoding: 'utf-8',
    );
  }

  @override
  void dispose() {
    try {
      _webCtrl?.evaluateJavascript(source: 'window.decartStop?.()');
    } catch (_) {/* webview already torn down */}
    _liveCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickSourceFace(ImageSource source) async {
    Navigator.pop(context);
    final f = await ImagePicker().pickImage(
        source: source, maxWidth: 512, maxHeight: 512, imageQuality: 90);
    if (f == null) return;
    final bytes = await f.readAsBytes();
    setState(() {
      _sourceFaceBytes = bytes;
      _sourceFaceLabel = source == ImageSource.camera ? 'Selfie' : 'Gallery';
      _faceSwapResult = null;
      _showFaceSwapOverlay = false;
    });
    await _runFaceSwap();
  }

  Future<void> _runFaceSwap() async {
    if (_sourceFaceBytes == null) return;

    setState(() {
      _isFaceSwapping = true;
      _faceSwapStatus = 'Capturing frame…';
    });

    Uint8List? frameBytes;
    try {
      final raw = await _webCtrl
          ?.evaluateJavascript(source: 'window.captureCurrentFrame()')
          .timeout(const Duration(seconds: 8));
      debugPrint('[FACESWAP] captureFrame raw length: ${raw?.length}');
      if (raw != null && raw != 'null' && raw.length > 10) {
        frameBytes = base64Decode(
            raw.replaceAll('"', '').replaceAll("'", '').trim());
      }
    } catch (e) {
      debugPrint('[FACESWAP] captureFrame error: $e');
    }

    if (frameBytes == null || frameBytes.isEmpty) {
      frameBytes = _sourceFaceBytes!;
    }

    try {
      final result = await FaceSwapService.swapFace(
        sourceImage: _sourceFaceBytes!,
        targetImage: frameBytes,
        onStatus: (msg) {
          if (mounted) setState(() => _faceSwapStatus = msg);
        },
      );
      if (mounted) {
        setState(() {
          _faceSwapResult = result;
          _isFaceSwapping = false;
          _showFaceSwapOverlay = result.isNotEmpty;
          _faceSwapStatus = '';
        });
      }
    } catch (e) {
      debugPrint('[FACESWAP] swapFace error: $e');
      if (mounted) {
        setState(() {
          _isFaceSwapping = false;
          _faceSwapStatus = '';
        });
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(e.toString()),
          backgroundColor: AppColors.error,
          duration: const Duration(seconds: 6),
          behavior: SnackBarBehavior.floating,
        ));
      }
    }
  }

  void _showPickerSheet() {
    Haptics.medium();
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(width: 36, height: 4,
                decoration: BoxDecoration(
                  color: const Color(0xFF38383A),
                  borderRadius: BorderRadius.circular(2),
                )),
              const SizedBox(height: 20),
              const Text('Choose Source Face',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
              const SizedBox(height: 6),
              const Text('Pick whose face to place on the live view',
                style: TextStyle(fontSize: 12, color: Color(0xFF636366))),
              const SizedBox(height: 20),
              Row(children: [
                Expanded(child: _FaceOption(
                  icon: Icons.camera_alt_rounded, label: 'Take Selfie',
                  subtitle: 'Use your camera', color: AppColors.primary,
                  onTap: () => _pickSourceFace(ImageSource.camera),
                )),
                const SizedBox(width: 12),
                Expanded(child: _FaceOption(
                  icon: Icons.photo_library_rounded, label: 'Upload Photo',
                  subtitle: 'From gallery', color: const Color(0xFF7C3AED),
                  onTap: () => _pickSourceFace(ImageSource.gallery),
                )),
              ]),
              if (_sourceFaceBytes != null) ...[
                const SizedBox(height: 12),
                _FaceOptionWide(
                  icon: Icons.face_retouching_natural,
                  label: 'Re-apply ${_sourceFaceLabel ?? "saved"} face',
                  subtitle: 'Capture new frame with current face',
                  color: AppColors.success,
                  onTap: () { Navigator.pop(context); _runFaceSwap(); },
                ),
                const SizedBox(height: 8),
                _FaceOptionWide(
                  icon: Icons.close_rounded, label: 'Clear Face',
                  subtitle: 'Remove face swap overlay',
                  color: AppColors.error,
                  onTap: () {
                    Navigator.pop(context);
                    setState(() {
                      _sourceFaceBytes = null;
                      _sourceFaceLabel = null;
                      _faceSwapResult = null;
                      _showFaceSwapOverlay = false;
                    });
                  },
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.home_rounded, color: Colors.white, size: 28),
          onPressed: () =>
              Navigator.popUntil(context, (route) => route.isFirst),
        ),
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            AnimatedBuilder(
              animation: _liveAnim,
              builder: (ctx, child) => Opacity(
                opacity: _liveAnim.value,
                child: Container(
                  width: 8, height: 8,
                  decoration: const BoxDecoration(
                    color: AppColors.primary, shape: BoxShape.circle),
                ),
              ),
            ),
            const SizedBox(width: 8),
            Text(widget.garmentName,
              style: const TextStyle(
                  color: Colors.white, fontSize: 15, fontWeight: FontWeight.w600)),
          ],
        ),
        centerTitle: true,
      ),
      body: Stack(
        children: [
          // ── Camera stream (web: iframe, mobile: InAppWebView) ─────
          Positioned.fill(
            child: kIsWeb
                ? buildWebArView(_viewId)
                : InAppWebView(
                    initialUrlRequest: URLRequest(
                      url: WebUri('about:blank'),
                    ),
                    initialSettings: InAppWebViewSettings(
                      javaScriptEnabled: true,
                      mediaPlaybackRequiresUserGesture: false,
                      allowsInlineMediaPlayback: true,
                      transparentBackground: true,
                      useHybridComposition: true,
                    ),
                    onWebViewCreated: (ctrl) {
                      _webCtrl = ctrl;
                      _loadTryonPage();
                    },
                    onPermissionRequest: (ctrl, request) async =>
                        PermissionResponse(
                          resources: request.resources,
                          action: PermissionResponseAction.GRANT,
                        ),
                    onConsoleMessage: (ctrl, msg) =>
                        debugPrint('[Decart] ${msg.message}'),
                  ),
          ),

          // ── Face-swap result overlay ───────────────────────────────
          if (_showFaceSwapOverlay && _faceSwapResult != null)
            Positioned.fill(
              child: FadeIn(
                duration: const Duration(milliseconds: 400),
                child: Stack(children: [
                  Image.memory(_faceSwapResult!,
                      fit: BoxFit.cover,
                      width: double.infinity,
                      height: double.infinity),
                  Positioned(
                    top: 100, left: 16,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.6),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Row(mainAxisSize: MainAxisSize.min, children: [
                        const Icon(Icons.face_retouching_natural,
                            color: AppColors.primary, size: 14),
                        const SizedBox(width: 6),
                        const Text('Face Swapped',
                            style: TextStyle(color: Colors.white,
                                fontSize: 12, fontWeight: FontWeight.w600)),
                        const SizedBox(width: 10),
                        GestureDetector(
                          onTap: () =>
                              setState(() => _showFaceSwapOverlay = false),
                          child: const Icon(Icons.close,
                              color: Colors.white70, size: 14),
                        ),
                      ]),
                    ),
                  ),
                ]),
              ),
            ),

          // ── Processing spinner ─────────────────────────────────────
          if (_isFaceSwapping)
            Positioned.fill(
              child: Container(
                color: Colors.black.withValues(alpha: 0.5),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const CircularProgressIndicator(
                        color: AppColors.primary, strokeWidth: 2.5),
                    const SizedBox(height: 14),
                    Text(_faceSwapStatus,
                      style: const TextStyle(color: Colors.white,
                          fontSize: 13, fontWeight: FontWeight.w500)),
                  ],
                ),
              ),
            ),

          // ── Bottom overlay ─────────────────────────────────────────
          Positioned(
            bottom: 0, left: 0, right: 0,
            child: _BottomOverlay(
              sourceFaceBytes: _sourceFaceBytes,
              faceSwapResult: _faceSwapResult,
              isFaceSwapping: _isFaceSwapping,
              faceSwapStatus: _faceSwapStatus,
              showOverlay: _showFaceSwapOverlay,
              onFaceChangeTap: _showPickerSheet,
              onToggleOverlay: () =>
                  setState(() => _showFaceSwapOverlay = !_showFaceSwapOverlay),
            ),
          ),
        ],
      ),
    );
  }

  static String _buildHtml(String garmentUrl, String garmentName,
      String backendWsUrl) {
    final garmentUrlJs   = jsonEncode(garmentUrl);
    final backendWsUrlJs = jsonEncode(backendWsUrl);

    return '''
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #000; overflow: hidden; width: 100vw; height: 100vh; }
    #live    { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: center top; background:#000; transform: scaleX(-1); }
    #result  { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: center top; opacity: 0; transition: opacity 0.6s; }
    #result.show { opacity: 1; }
    #status {
      position: fixed; bottom: 140px; left: 50%;
      transform: translateX(-50%);
      background: rgba(0,0,0,0.65); backdrop-filter: blur(10px);
      color: #fff; padding: 10px 22px; border-radius: 28px;
      font-family: -apple-system, sans-serif;
      font-size: 14px; font-weight: 500; white-space: nowrap;
      z-index: 10; transition: opacity 0.4s;
    }
    #status.hidden { opacity: 0; pointer-events: none; }
  </style>
</head>
<body>
  <video id="live" autoplay playsinline muted></video>
  <img   id="result" alt="try-on result" />
  <div   id="status">Connecting…</div>

  <script>
    const GARMENT_URL  = $garmentUrlJs;
    const BACKEND_URL  = $backendWsUrlJs;

    const liveEl   = document.getElementById('live');
    const resultEl = document.getElementById('result');
    const statusEl = document.getElementById('status');

    const setStatus  = (msg) => { statusEl.textContent = msg; statusEl.classList.remove('hidden'); };
    const hideStatus = ()    => statusEl.classList.add('hidden');

    let ws        = null;
    let stream    = null;
    let capturing = false;
    let wsReady   = false;

    // ── Capture a JPEG frame from the live video ──────────────────
    function captureFrame() {
      const canvas = document.createElement('canvas');
      canvas.width  = liveEl.videoWidth  || 512;
      canvas.height = liveEl.videoHeight || 768;
      const ctx = canvas.getContext('2d');
      // Mirror to match what user sees
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(liveEl, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.82).split(',')[1];
    }

    // ── Convert garment URL to base64 ─────────────────────────────
    async function urlToBase64(url) {
      if (url.startsWith('data:')) return url.split(',')[1];
      const res  = await fetch(url);
      const blob = await res.blob();
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(blob);
      });
    }

    // ── Send frames every 3 seconds while connected ───────────────
    function startCapturing() {
      if (capturing) return;
      capturing = true;
      (async function loop() {
        while (capturing && wsReady) {
          try {
            const frame = captureFrame();
            ws.send(JSON.stringify({ type: 'frame', image: frame }));
          } catch (e) { console.error('[TryOn] frame error', e); }
          await new Promise(r => setTimeout(r, 3000));
        }
      })();
    }

    // ── WebSocket setup ───────────────────────────────────────────
    async function connect() {
      setStatus('Starting camera…');
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width:  { ideal: 720,  max: 1280 },
          height: { ideal: 1280, max: 1920 },
          aspectRatio: { ideal: 9/16 },
        },
        audio: false,
      });
      liveEl.srcObject = stream;

      if (!BACKEND_URL) {
        setStatus('⚠ No server set — tap the ⚡ on the Try-On screen and paste your server URL');
        return;
      }

      setStatus('Connecting to AI server…');
      ws = new WebSocket(BACKEND_URL);

      ws.onopen = async () => {
        setStatus('Sending garment…');
        const garmentB64 = await urlToBase64(GARMENT_URL);
        ws.send(JSON.stringify({ type: 'init', garment_image: garmentB64 }));
      };

      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);

        if (msg.type === 'ready') {
          wsReady = true;
          hideStatus();
          startCapturing();

        } else if (msg.type === 'status') {
          setStatus(msg.message);

        } else if (msg.type === 'result') {
          resultEl.src = 'data:image/jpeg;base64,' + msg.image;
          resultEl.classList.add('show');
          hideStatus();

        } else if (msg.type === 'error') {
          setStatus('⚠ ' + msg.message);
          console.error('[TryOn]', msg.message);
        }
      };

      ws.onclose = () => {
        wsReady   = false;
        capturing = false;
        setStatus('Disconnected — retrying…');
        setTimeout(connect, 3000);
      };

      ws.onerror = (err) => {
        console.error('[TryOn] WS error', err);
        setStatus('⚠ Connection error');
      };
    }

    window.captureCurrentFrame = function() {
      try { return captureFrame(); } catch(e) { return null; }
    };

    window.decartStop = function() {
      capturing = false;
      wsReady   = false;
      ws?.close();
      stream?.getTracks().forEach(t => t.stop());
    };

    connect().catch(err => setStatus('⚠ ' + err.message));
  </script>
</body>
</html>
''';
  }
}

// ── Bottom Overlay ────────────────────────────────────────────────────────────
class _BottomOverlay extends StatelessWidget {
  final Uint8List? sourceFaceBytes;
  final Uint8List? faceSwapResult;
  final bool isFaceSwapping;
  final String faceSwapStatus;
  final bool showOverlay;
  final VoidCallback onFaceChangeTap;
  final VoidCallback onToggleOverlay;

  const _BottomOverlay({
    required this.sourceFaceBytes,
    required this.faceSwapResult,
    required this.isFaceSwapping,
    required this.faceSwapStatus,
    required this.showOverlay,
    required this.onFaceChangeTap,
    required this.onToggleOverlay,
  });

  @override
  Widget build(BuildContext context) {
    final hasFace = sourceFaceBytes != null;
    final hasResult = faceSwapResult != null;

    return Container(
      padding: const EdgeInsets.fromLTRB(20, 12, 20, 36),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter,
          end: Alignment.topCenter,
          colors: [Colors.black.withValues(alpha: 0.85), Colors.transparent],
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (faceSwapStatus.isNotEmpty && !isFaceSwapping)
            FadeInUp(
              duration: const Duration(milliseconds: 300),
              child: Container(
                margin: const EdgeInsets.only(bottom: 10),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.7),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                      color: AppColors.error.withValues(alpha: 0.4)),
                ),
                child: Text(faceSwapStatus,
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                  textAlign: TextAlign.center),
              ),
            ),

          Row(children: [
            if (hasResult) ...[
              FadeIn(
                child: GestureDetector(
                  onTap: onToggleOverlay,
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 10),
                    decoration: BoxDecoration(
                      color: showOverlay
                          ? AppColors.primary
                          : Colors.white.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(
                          color: showOverlay ? AppColors.primary : Colors.white30),
                    ),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      Icon(
                        showOverlay
                            ? Icons.face_retouching_natural
                            : Icons.face_outlined,
                        color: Colors.white, size: 16),
                      const SizedBox(width: 6),
                      Text(showOverlay ? 'My Face ON' : 'My Face OFF',
                        style: const TextStyle(color: Colors.white,
                            fontSize: 12, fontWeight: FontWeight.w600)),
                    ]),
                  ),
                ),
              ),
              const SizedBox(width: 10),
            ],

            const Spacer(),

            FadeInUp(
              duration: const Duration(milliseconds: 400),
              child: GestureDetector(
                onTap: onFaceChangeTap,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 20, vertical: 13),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [Color(0xFF7C3AED), AppColors.primary],
                      begin: Alignment.centerLeft,
                      end: Alignment.centerRight,
                    ),
                    borderRadius: BorderRadius.circular(28),
                    boxShadow: [BoxShadow(
                      color: AppColors.primary.withValues(alpha: 0.35),
                      blurRadius: 16, offset: const Offset(0, 4),
                    )],
                  ),
                  child: Row(mainAxisSize: MainAxisSize.min, children: [
                    if (hasFace)
                      Container(
                        width: 26, height: 26,
                        margin: const EdgeInsets.only(right: 8),
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 1.5),
                        ),
                        clipBehavior: Clip.antiAlias,
                        child: Image.memory(sourceFaceBytes!, fit: BoxFit.cover),
                      )
                    else
                      const Icon(Icons.face_outlined,
                          color: Colors.white, size: 20),
                    if (!hasFace) const SizedBox(width: 8),
                    Text(hasFace ? 'Change Face' : 'Face Swap',
                      style: const TextStyle(color: Colors.white,
                          fontSize: 14, fontWeight: FontWeight.w700)),
                  ]),
                ),
              ),
            ),
          ]),
        ],
      ),
    );
  }
}

// ── Face Option Widgets ───────────────────────────────────────────────────────
class _FaceOption extends StatelessWidget {
  final IconData icon;
  final String label;
  final String subtitle;
  final Color color;
  final VoidCallback onTap;

  const _FaceOption({required this.icon, required this.label,
    required this.subtitle, required this.color, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 12),
        decoration: BoxDecoration(
          color: const Color(0xFF2C2C2E),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color.withValues(alpha: 0.25)),
        ),
        child: Column(children: [
          Container(
            width: 48, height: 48,
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.12), shape: BoxShape.circle),
            child: Icon(icon, color: color, size: 24),
          ),
          const SizedBox(height: 10),
          Text(label, style: const TextStyle(
              fontSize: 13, fontWeight: FontWeight.w600, color: Colors.white)),
          const SizedBox(height: 2),
          Text(subtitle, style: const TextStyle(
              fontSize: 11, color: Color(0xFF636366))),
        ]),
      ),
    );
  }
}

class _FaceOptionWide extends StatelessWidget {
  final IconData icon;
  final String label;
  final String subtitle;
  final Color color;
  final VoidCallback onTap;

  const _FaceOptionWide({required this.icon, required this.label,
    required this.subtitle, required this.color, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(vertical: 13, horizontal: 16),
        decoration: BoxDecoration(
          color: const Color(0xFF2C2C2E),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: color.withValues(alpha: 0.25)),
        ),
        child: Row(children: [
          Container(
            width: 36, height: 36,
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: color, size: 18),
          ),
          const SizedBox(width: 12),
          Expanded(child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label, style: const TextStyle(
                  fontSize: 13, fontWeight: FontWeight.w600, color: Colors.white)),
              Text(subtitle, style: const TextStyle(
                  fontSize: 11, color: Color(0xFF636366))),
            ],
          )),
        ]),
      ),
    );
  }
}
