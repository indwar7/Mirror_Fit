import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';

class ARBodyTryOnScreen extends StatefulWidget {
  const ARBodyTryOnScreen({super.key});

  @override
  State<ARBodyTryOnScreen> createState() => _ARBodyTryOnScreenState();
}

class _ARBodyTryOnScreenState extends State<ARBodyTryOnScreen> {
  static const _channel = MethodChannel('com.lucy.artryon');
  bool _isSupported = true;
  bool _isLoading = false;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _checkSupport();
  }

  Future<void> _checkSupport() async {
    try {
      final supported =
          await _channel.invokeMethod<bool>('isBodyTrackingSupported') ??
              false;
      if (mounted && supported != _isSupported) {
        setState(() => _isSupported = supported);
      }
    } catch (_) {
      // Channel not ready — _isSupported already defaults to true.
    }
  }

  Future<void> _launchAR() async {
    Haptics.medium();
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });
    try {
      await _channel.invokeMethod('launchARTryOn');
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage =
              'Failed to launch AR: ${e.toString().split('\n').first}';
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
                'AR error: ${e.toString().split('\n').first}'),
            backgroundColor: AppColors.error,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AR Body Try-On'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            children: [
              const Spacer(flex: 2),

              Container(
                width: 100,
                height: 100,
                decoration: BoxDecoration(
                  color: AppColors.primary.withValues(alpha: 0.1),
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.accessibility_new_rounded,
                  size: 48,
                  color: AppColors.primary,
                ),
              ),
              const SizedBox(height: 28),

              const Text(
                '3D Body Try-On',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: AppColors.darkTextPrimary,
                ),
              ),
              const SizedBox(height: 10),

              Text(
                _isSupported
                    ? 'The jacket will appear on your body in real-time using ARKit body tracking.'
                    : 'Body tracking requires iPhone XS or newer (A12+ chip) with iOS 14+.',
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 15,
                  color: AppColors.darkTextSecondary,
                  height: 1.5,
                ),
              ),
              const SizedBox(height: 28),

              if (_isSupported) ...[
                _InstructionTile(
                  icon: Icons.person_outline_rounded,
                  text: 'Stand 2–3 meters away',
                  subtitle: 'Full body should be visible',
                ),
                _InstructionTile(
                  icon: Icons.wb_sunny_outlined,
                  text: 'Good lighting',
                  subtitle: 'Avoid strong backlight',
                ),
                _InstructionTile(
                  icon: Icons.rotate_left_rounded,
                  text: 'Turn around',
                  subtitle: 'See front and back view',
                ),
              ],

              if (_errorMessage != null) ...[
                const SizedBox(height: 16),
                Text(
                  _errorMessage!,
                  style: const TextStyle(
                      color: AppColors.error, fontSize: 13),
                  textAlign: TextAlign.center,
                ),
              ],

              const Spacer(flex: 3),

              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  onPressed: _isLoading ? null : _launchAR,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    disabledBackgroundColor:
                        AppColors.primary.withValues(alpha: 0.4),
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius:
                          BorderRadius.circular(AppRadius.lg),
                    ),
                    elevation: 0,
                  ),
                  child: _isLoading
                      ? const SizedBox(
                          width: 24,
                          height: 24,
                          child: CircularProgressIndicator(
                              color: Colors.white, strokeWidth: 2),
                        )
                      : const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.play_arrow_rounded, size: 24),
                            SizedBox(width: 8),
                            Text(
                              'Start AR Try-On',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                                letterSpacing: 0.3,
                              ),
                            ),
                          ],
                        ),
                ),
              ),
              const SizedBox(height: 32),
            ],
          ),
        ),
      ),
    );
  }
}

class _InstructionTile extends StatelessWidget {
  final IconData icon;
  final String text;
  final String subtitle;

  const _InstructionTile({
    required this.icon,
    required this.text,
    required this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: AppColors.darkCard,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: AppColors.darkBorder),
            ),
            child: Icon(icon,
                size: 20, color: AppColors.darkTextSecondary),
          ),
          const SizedBox(width: 14),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                text,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: AppColors.darkTextPrimary,
                ),
              ),
              Text(
                subtitle,
                style: const TextStyle(
                    fontSize: 12,
                    color: AppColors.darkTextTertiary),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
