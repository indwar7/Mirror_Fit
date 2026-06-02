import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import '../utils/app_theme.dart';
import '../services/user_preferences.dart';
import 'home_screen.dart';
import 'onboarding_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  late AnimationController _shimmerCtrl;
  late Animation<double> _shimmerAnim;

  @override
  void initState() {
    super.initState();

    _shimmerCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _shimmerAnim = Tween<double>(begin: 0.3, end: 1.0).animate(
      CurvedAnimation(parent: _shimmerCtrl, curve: Curves.easeInOut),
    );

    Future.delayed(const Duration(milliseconds: 3000), () {
      if (mounted) {
        final destination = UserPreferences.isOnboardingDone
            ? const HomeScreen()
            : const OnboardingScreen();

        Navigator.pushReplacement(
          context,
          PageRouteBuilder(
            pageBuilder: (context, anim, secondary) => destination,
            transitionDuration: const Duration(milliseconds: 800),
            transitionsBuilder: (context, anim, secondary, child) {
              return FadeTransition(opacity: anim, child: child);
            },
          ),
        );
      }
    });
  }

  @override
  void dispose() {
    _shimmerCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Color(0xFF1A0A10),
              AppColors.darkBg,
              AppColors.darkBg,
            ],
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Spacer(flex: 3),

            // Accent line
            FadeInDown(
              duration: const Duration(milliseconds: 800),
              child: AnimatedBuilder(
                animation: _shimmerAnim,
                builder: (context, child) => Opacity(
                  opacity: _shimmerAnim.value,
                  child: Container(
                    width: 2,
                    height: 60,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          Colors.transparent,
                          AppColors.primary.withValues(alpha: 0.6),
                          Colors.transparent,
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Brand name
            FadeIn(
              duration: const Duration(milliseconds: 1000),
              delay: const Duration(milliseconds: 400),
              child: Text('LUCY', style: AppText.displayLarge),
            ),
            const SizedBox(height: 8),

            // Tagline
            FadeIn(
              duration: const Duration(milliseconds: 1000),
              delay: const Duration(milliseconds: 800),
              child: Text(
                'VIRTUAL TRY-ON',
                style: AppText.labelSmall.copyWith(
                  fontSize: 13,
                  letterSpacing: 6,
                  color: AppColors.textSecondary,
                ),
              ),
            ),
            const SizedBox(height: 32),

            // Decorative line
            FadeIn(
              duration: const Duration(milliseconds: 1000),
              delay: const Duration(milliseconds: 1200),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 40,
                    height: 0.5,
                    color: AppColors.primary.withValues(alpha: 0.3),
                  ),
                  const SizedBox(width: 12),
                  Container(
                    width: 6,
                    height: 6,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: AppColors.primary.withValues(alpha: 0.5),
                        width: 0.5,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Container(
                    width: 40,
                    height: 0.5,
                    color: AppColors.primary.withValues(alpha: 0.3),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Subtitle
            FadeInUp(
              duration: const Duration(milliseconds: 800),
              delay: const Duration(milliseconds: 1400),
              child: Text(
                'See It On You',
                style: AppText.bodyLarge.copyWith(
                  fontStyle: FontStyle.italic,
                  color: AppColors.greyLight,
                  letterSpacing: 2,
                ),
              ),
            ),

            const Spacer(flex: 3),

            // Loading indicator
            FadeIn(
              duration: const Duration(milliseconds: 600),
              delay: const Duration(milliseconds: 1800),
              child: SizedBox(
                width: 24,
                height: 24,
                child: CircularProgressIndicator(
                  strokeWidth: 1.5,
                  color: AppColors.primary.withValues(alpha: 0.5),
                ),
              ),
            ),
            const SizedBox(height: 48),

            // Powered by
            FadeIn(
              duration: const Duration(milliseconds: 600),
              delay: const Duration(milliseconds: 2000),
              child: Text(
                'Powered by AI',
                style: AppText.bodyMedium.copyWith(
                  fontSize: 10,
                  color: AppColors.grey,
                  letterSpacing: 3,
                ),
              ),
            ),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }
}
