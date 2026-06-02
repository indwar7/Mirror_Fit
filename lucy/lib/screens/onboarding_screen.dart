import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import 'package:smooth_page_indicator/smooth_page_indicator.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';
import '../services/user_preferences.dart';
import 'style_quiz_screen.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final _controller = PageController();
  int _currentPage = 0;

  static const _pages = [
    _OnboardingPage(
      icon: Icons.camera_alt_rounded,
      title: 'Snap Any Garment',
      subtitle:
          'See a shirt you love? Just take a photo of it — from your closet, a store, or anywhere',
      color: Color(0xFFFF3F6C),
      bgColor: Color(0xFFFFF0F3),
    ),
    _OnboardingPage(
      icon: Icons.person_outline_rounded,
      title: 'Upload Your Photo',
      subtitle:
          'Take a quick selfie or choose from gallery. LUCY adapts the garment to your body',
      color: Color(0xFF3B82F6),
      bgColor: Color(0xFFF0F4FF),
    ),
    _OnboardingPage(
      icon: Icons.auto_awesome,
      title: 'See It On You',
      subtitle:
          'AI generates a realistic try-on in seconds. No more guessing — see how it actually looks on you',
      color: Color(0xFF03A685),
      bgColor: Color(0xFFF0FFF8),
    ),
  ];

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isLast = _currentPage == _pages.length - 1;

    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: Column(
          children: [
            Align(
              alignment: Alignment.topRight,
              child: TextButton(
                onPressed: _finish,
                child: Text(
                  'Skip',
                  style: AppText.bodyMedium.copyWith(color: AppColors.textSecondary),
                ),
              ),
            ),
            Expanded(
              child: PageView.builder(
                controller: _controller,
                itemCount: _pages.length,
                onPageChanged: (i) {
                  setState(() => _currentPage = i);
                  Haptics.selection();
                },
                itemBuilder: (_, i) => _buildPage(_pages[i]),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(bottom: 24),
              child: SmoothPageIndicator(
                controller: _controller,
                count: _pages.length,
                effect: WormEffect(
                  dotWidth: 8,
                  dotHeight: 8,
                  spacing: 12,
                  activeDotColor: AppColors.primary,
                  dotColor: AppColors.surfaceLight,
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 0, 24, 32),
              child: SizedBox(
                width: double.infinity,
                height: 54,
                child: AnimatedSwitcher(
                  duration: const Duration(milliseconds: 300),
                  child: isLast
                      ? ElevatedButton(
                          key: const ValueKey('start'),
                          onPressed: _finish,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppColors.primary,
                            shape: RoundedRectangleBorder(
                              borderRadius:
                                  BorderRadius.circular(AppRadius.full),
                            ),
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text('GET STARTED', style: AppText.buttonText),
                              const SizedBox(width: 8),
                              const Icon(Icons.arrow_forward_rounded,
                                  color: Colors.white, size: 20),
                            ],
                          ),
                        )
                      : OutlinedButton(
                          key: const ValueKey('next'),
                          onPressed: () {
                            _controller.nextPage(
                              duration: const Duration(milliseconds: 400),
                              curve: Curves.easeOut,
                            );
                          },
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: AppColors.primary),
                            shape: RoundedRectangleBorder(
                              borderRadius:
                                  BorderRadius.circular(AppRadius.full),
                            ),
                          ),
                          child: Text(
                            'NEXT',
                            style: AppText.buttonText.copyWith(
                                color: AppColors.primary),
                          ),
                        ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPage(_OnboardingPage page) {
    return Container(
      color: page.bgColor,
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          FadeInDown(
            duration: const Duration(milliseconds: 600),
            child: Container(
              width: 120,
              height: 120,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: page.color.withValues(alpha: 0.1),
                border: Border.all(
                    color: page.color.withValues(alpha: 0.25), width: 2),
              ),
              child: Icon(page.icon, size: 52, color: page.color),
            ),
          ),
          const SizedBox(height: 48),
          FadeIn(
            duration: const Duration(milliseconds: 800),
            child: Text(
              page.title,
              style: AppText.titleLarge.copyWith(
                fontSize: 26,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          const SizedBox(height: 16),
          FadeInUp(
            duration: const Duration(milliseconds: 800),
            delay: const Duration(milliseconds: 200),
            child: Text(
              page.subtitle,
              style: AppText.bodyLarge.copyWith(
                height: 1.6,
                color: AppColors.textSecondary,
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }

  void _finish() {
    Haptics.medium();
    UserPreferences.setOnboardingDone();
    Navigator.pushReplacement(
      context,
      PageRouteBuilder(
        pageBuilder: (context, anim, secondary) => const StyleQuizScreen(),
        transitionDuration: const Duration(milliseconds: 600),
        transitionsBuilder: (context, anim, secondary, child) =>
            FadeTransition(opacity: anim, child: child),
      ),
    );
  }
}

class _OnboardingPage {
  final IconData icon;
  final String title;
  final String subtitle;
  final Color color;
  final Color bgColor;

  const _OnboardingPage({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.color,
    required this.bgColor,
  });
}
