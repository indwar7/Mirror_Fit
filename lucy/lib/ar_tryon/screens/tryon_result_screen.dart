import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../../services/free_tryon_service.dart';
import '../../utils/app_theme.dart';

class TryOnResultScreen extends StatefulWidget {
  final Uint8List personPhoto;
  final Uint8List garmentPhoto;
  final String garmentName;

  const TryOnResultScreen({
    super.key,
    required this.personPhoto,
    required this.garmentPhoto,
    this.garmentName = 'Jacket',
  });

  @override
  State<TryOnResultScreen> createState() => _TryOnResultScreenState();
}

class _TryOnResultScreenState extends State<TryOnResultScreen> {
  String _status = 'Starting...';
  double _progress = 0.0;
  Uint8List? _resultImage;
  bool _isProcessing = true;
  bool _failed = false;

  @override
  void initState() {
    super.initState();
    _runTryOn();
  }

  Future<void> _runTryOn() async {
    setState(() {
      _isProcessing = true;
      _failed = false;
      _status = 'Preparing photos...';
      _progress = 0.0;
    });

    try {
      final resultBytes = await FreeTryOnService.tryOn(
        personBytes: widget.personPhoto,
        garmentBytes: widget.garmentPhoto,
        onProgress: (progress, message) {
          if (mounted) {
            setState(() {
              _progress = progress;
              _status = message;
            });
          }
        },
      );

      if (mounted) {
        setState(() {
          _resultImage = resultBytes;
          _isProcessing = false;
          _status = 'Your look is ready!';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isProcessing = false;
          _failed = true;
          _status = 'Something went wrong: $e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('${widget.garmentName} Try-On'),
        centerTitle: true,
      ),
      body: _isProcessing
          ? _buildProcessingView()
          : _failed
              ? _buildErrorView()
              : _buildResultView(),
    );
  }

  Widget _buildProcessingView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Image.memory(widget.personPhoto,
                  height: 300, fit: BoxFit.cover),
            ),
            const SizedBox(height: 24),
            LinearProgressIndicator(
              value: _progress > 0 ? _progress : null,
              backgroundColor: AppColors.darkBorder,
              valueColor: const AlwaysStoppedAnimation<Color>(
                  AppColors.primary),
            ),
            const SizedBox(height: 16),
            Text(_status,
                style: const TextStyle(
                    color: AppColors.darkTextPrimary, fontSize: 16),
                textAlign: TextAlign.center),
            const SizedBox(height: 8),
            Text(
              _progress > 0
                  ? '${(_progress * 100).toInt()}% complete'
                  : 'Connecting to AI server...',
              style: const TextStyle(
                  color: AppColors.darkTextTertiary, fontSize: 13),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline,
                color: AppColors.error, size: 48),
            const SizedBox(height: 16),
            Text(_status,
                style: const TextStyle(
                    color: AppColors.darkTextPrimary, fontSize: 14),
                textAlign: TextAlign.center),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: _runTryOn,
              icon: const Icon(Icons.refresh),
              label: const Text('Try Again'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultView() {
    if (_resultImage == null) return _buildErrorView();

    return Column(
      children: [
        Expanded(
          child: InteractiveViewer(
            child: Center(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Image.memory(_resultImage!, fit: BoxFit.contain),
              ),
            ),
          ),
        ),
        Container(
          padding: const EdgeInsets.all(16),
          color: AppColors.darkSurface,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _actionButton(
                icon: Icons.refresh,
                label: 'Retry',
                onTap: _runTryOn,
              ),
              _actionButton(
                icon: Icons.check_circle,
                label: 'Looks Great!',
                onTap: () => Navigator.pop(context, _resultImage),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _actionButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: AppColors.darkTextPrimary, size: 28),
          const SizedBox(height: 4),
          Text(label,
              style: const TextStyle(
                  color: AppColors.darkTextSecondary, fontSize: 12)),
        ],
      ),
    );
  }
}
