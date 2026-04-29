import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import '../models/cloth_item.dart';
import '../models/style_score.dart';
import '../services/style_engine.dart';
import '../services/user_preferences.dart';
import 'face_swap_service.dart';
import 'free_tryon_service.dart';

enum TryOnStatus { idle, uploading, processing, completed, error }

class TryOnProvider extends ChangeNotifier {
  // ── State ───────────────────────────────────────────────────────────────────
  Uint8List? _userPhotoBytes;
  Uint8List? _garmentPhotoBytes;
  ClothItem? _selectedCloth;
  String? _selectedSize;
  String? _selectedColor;

  TryOnStatus _status = TryOnStatus.idle;
  String? _resultFrontUrl;
  Uint8List? _resultBytes;
  String? _errorMessage;
  double _progress = 0;
  String _statusMessage = '';
  StyleScore? _styleScore;

  bool _isDarkMode = true;

  // ── Face Swap State ─────────────────────────────────────────────────────────
  Uint8List? _faceSwapBytes;
  bool _isFaceSwapping = false;
  String? _faceSwapError;
  String _faceSwapMessage = '';

  // ── Getters ─────────────────────────────────────────────────────────────────
  Uint8List? get userPhotoBytes => _userPhotoBytes;
  Uint8List? get garmentPhotoBytes => _garmentPhotoBytes;
  ClothItem? get selectedCloth => _selectedCloth;
  String? get selectedSize => _selectedSize;
  String? get selectedColor => _selectedColor;

  TryOnStatus get status => _status;
  String? get resultFrontUrl => _resultFrontUrl;
  Uint8List? get resultBytes => _resultBytes;
  String? get errorMessage => _errorMessage;
  double get progress => _progress;
  String get statusMessage => _statusMessage;
  StyleScore? get styleScore => _styleScore;

  bool get hasPhoto => _userPhotoBytes != null;
  bool get hasGarment => _selectedCloth != null || _garmentPhotoBytes != null;
  bool get hasCloth => _selectedCloth != null;
  bool get canTryOn => hasPhoto && hasGarment;
  bool get isProcessing =>
      _status == TryOnStatus.uploading || _status == TryOnStatus.processing;
  bool get isUsingGarmentPhoto =>
      _garmentPhotoBytes != null && _selectedCloth == null;

  bool get isDarkMode => _isDarkMode;

  Uint8List? get faceSwapBytes => _faceSwapBytes;
  bool get isFaceSwapping => _isFaceSwapping;
  String? get faceSwapError => _faceSwapError;
  String get faceSwapMessage => _faceSwapMessage;

  String get garmentDisplayName => _selectedCloth?.name ?? 'Your Garment';

  // ── Init ────────────────────────────────────────────────────────────────────
  void loadPreferences() {
    _isDarkMode = UserPreferences.isDarkMode;
    notifyListeners();
  }

  // ── Theme ───────────────────────────────────────────────────────────────────
  void toggleTheme() {
    _isDarkMode = !_isDarkMode;
    UserPreferences.setDarkMode(_isDarkMode);
    notifyListeners();
  }

  // ── Actions ─────────────────────────────────────────────────────────────────
  Future<void> setUserPhoto(XFile xFile) async {
    _userPhotoBytes = await xFile.readAsBytes();
    _resetResult();
    notifyListeners();
  }

  Future<void> setGarmentPhoto(XFile xFile) async {
    _garmentPhotoBytes = await xFile.readAsBytes();
    _selectedCloth = null;
    _selectedSize = null;
    _selectedColor = null;
    _resetResult();
    notifyListeners();
  }

  void selectCloth(ClothItem cloth) {
    _selectedCloth = cloth;
    _garmentPhotoBytes = null;
    final profile = UserPreferences.getProfile();
    if (profile != null) {
      _selectedSize = StyleEngine.recommendSize(profile, cloth);
    } else {
      _selectedSize = cloth.sizes.first;
    }
    _selectedColor = cloth.colors.first;
    _resetResult();
    notifyListeners();
  }

  void setSize(String size) {
    _selectedSize = size;
    notifyListeners();
  }

  void setColor(String color) {
    _selectedColor = color;
    notifyListeners();
  }

  void _resetResult() {
    _resultFrontUrl = null;
    _resultBytes = null;
    _errorMessage = null;
    _status = TryOnStatus.idle;
    _progress = 0;
    _statusMessage = '';
    _styleScore = null;
    _faceSwapBytes = null;
    _faceSwapError = null;
    _faceSwapMessage = '';
    _isFaceSwapping = false;
  }

  void clearAll() {
    _userPhotoBytes = null;
    _garmentPhotoBytes = null;
    _selectedCloth = null;
    _selectedSize = null;
    _selectedColor = null;
    _resetResult();
    notifyListeners();
  }

  // ── Try-On Flow ─────────────────────────────────────────────────────────────
  Future<void> runTryOn() async {
    if (!canTryOn) return;
    await _runFreeTryOn();
  }

  Future<void> _runFreeTryOn() async {
    try {
      _status = TryOnStatus.uploading;
      _progress = 0.05;
      _statusMessage = 'Connecting to AI...';
      notifyListeners();

      Uint8List garmentBytes;
      if (_garmentPhotoBytes != null) {
        garmentBytes = _garmentPhotoBytes!;
      } else if (_selectedCloth != null) {
        _statusMessage = 'Downloading garment image...';
        notifyListeners();
        garmentBytes = await _downloadBytes(_selectedCloth!.imageUrl);
      } else {
        throw FreeTryOnException('No garment selected');
      }

      _status = TryOnStatus.processing;
      notifyListeners();

      final resultBytes = await FreeTryOnService.tryOn(
        personBytes: _userPhotoBytes!,
        garmentBytes: garmentBytes,
        onProgress: (progress, message) {
          _progress = progress;
          _statusMessage = message;
          notifyListeners();
        },
      );

      _generateStyleScore();
      _resultBytes = resultBytes;
      _resultFrontUrl = null;
      _completeResult();
    } on FreeTryOnException catch (e) {
      _status = TryOnStatus.error;
      _errorMessage = e.message;
      _statusMessage = 'Something went wrong';
      notifyListeners();
    } catch (e) {
      _status = TryOnStatus.error;
      _errorMessage = e.toString();
      _statusMessage = 'Something went wrong';
      notifyListeners();
    }
  }


  Future<Uint8List> _downloadBytes(String url) async {
    final response = await http.get(Uri.parse(url)).timeout(
      const Duration(seconds: 30),
    );
    if (response.statusCode != 200) {
      throw FreeTryOnException('Failed to download garment image');
    }
    return response.bodyBytes;
  }

  void _generateStyleScore() {
    final profile = UserPreferences.getProfile();
    if (profile != null && _selectedCloth != null) {
      _styleScore = StyleEngine.generateScore(
        item: _selectedCloth!,
        profile: profile,
        selectedColor: _selectedColor ?? _selectedCloth!.colors.first,
        selectedSize: _selectedSize ?? _selectedCloth!.sizes.first,
      );
    }
  }

  void _completeResult() {
    _status = TryOnStatus.completed;
    _progress = 1.0;
    _statusMessage = 'Your look is ready!';
    UserPreferences.incrementTryOnCount();
    UserPreferences.addStylePoints(10);
    notifyListeners();
  }

  // ── Face Swap ───────────────────────────────────────────────────────────────
  Future<void> runFaceSwap() async {
    final targetBytes = _resultBytes;
    final sourceBytes = _userPhotoBytes;
    if (targetBytes == null || sourceBytes == null) return;

    _isFaceSwapping = true;
    _faceSwapBytes = null;
    _faceSwapError = null;
    _faceSwapMessage = 'Detecting faces…';
    notifyListeners();

    try {
      final result = await FaceSwapService.swapFace(
        sourceImage: sourceBytes,
        targetImage: targetBytes,
        onStatus: (msg) {
          _faceSwapMessage = msg;
          notifyListeners();
        },
      );
      _faceSwapBytes = result;
    } on FaceSwapException catch (e) {
      _faceSwapError = e.message;
    } catch (e) {
      _faceSwapError = e.toString();
    } finally {
      _isFaceSwapping = false;
      notifyListeners();
    }
  }

  void clearFaceSwap() {
    _faceSwapBytes = null;
    _faceSwapError = null;
    _faceSwapMessage = '';
    _isFaceSwapping = false;
    notifyListeners();
  }

}
