import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import '../models/user_profile.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';
import '../services/user_preferences.dart';
import 'home_screen.dart';

class StyleQuizScreen extends StatefulWidget {
  const StyleQuizScreen({super.key});

  @override
  State<StyleQuizScreen> createState() => _StyleQuizScreenState();
}

class _StyleQuizScreenState extends State<StyleQuizScreen> {
  final _pageCtrl = PageController();
  int _step = 0;

  // Quiz state
  String _name = '';
  double _height = 170;
  double _weight = 70;
  BodyType _bodyType = BodyType.average;
  SkinTone _skinTone = SkinTone.medium;
  final Set<StylePreference> _stylePrefs = {StylePreference.casual};
  Occasion _occasion = Occasion.everyday;

  static const _totalSteps = 4;

  @override
  void dispose() {
    _pageCtrl.dispose();
    super.dispose();
  }

  void _next() {
    if (_step < _totalSteps - 1) {
      Haptics.light();
      _pageCtrl.nextPage(
        duration: const Duration(milliseconds: 400),
        curve: Curves.easeOut,
      );
      setState(() => _step++);
    } else {
      _finish();
    }
  }

  void _finish() {
    Haptics.success();
    final profile = UserProfile(
      name: _name,
      heightCm: _height,
      weightKg: _weight,
      bodyType: _bodyType,
      skinTone: _skinTone,
      stylePrefs: _stylePrefs.toList(),
      primaryOccasion: _occasion,
    );
    UserPreferences.saveProfile(profile);
    UserPreferences.addBadge('style_profiled');

    Navigator.pushReplacement(
      context,
      PageRouteBuilder(
        pageBuilder: (context, anim, secondary) => const HomeScreen(),
        transitionDuration: const Duration(milliseconds: 600),
        transitionsBuilder: (context, anim, secondary, child) =>
            FadeTransition(opacity: anim, child: child),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Column(
          children: [
            // Header
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
              child: Row(
                children: [
                  if (_step > 0)
                    GestureDetector(
                      onTap: () {
                        _pageCtrl.previousPage(
                          duration: const Duration(milliseconds: 300),
                          curve: Curves.easeOut,
                        );
                        setState(() => _step--);
                      },
                      child: const Icon(Icons.arrow_back_ios,
                          color: AppColors.textPrimary, size: 20),
                    ),
                  const Spacer(),
                  TextButton(
                    onPressed: _finish,
                    child: Text('Skip',
                        style: AppText.bodyMedium
                            .copyWith(color: AppColors.grey)),
                  ),
                ],
              ),
            ),

            // Progress bar
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(AppRadius.full),
                child: LinearProgressIndicator(
                  value: (_step + 1) / _totalSteps,
                  backgroundColor: AppColors.surfaceLight,
                  valueColor:
                      const AlwaysStoppedAnimation<Color>(AppColors.gold),
                  minHeight: 3,
                ),
              ),
            ),

            // Pages
            Expanded(
              child: PageView(
                controller: _pageCtrl,
                physics: const NeverScrollableScrollPhysics(),
                children: [
                  _buildBodyStep(),
                  _buildSkinToneStep(),
                  _buildStyleStep(),
                  _buildOccasionStep(),
                ],
              ),
            ),

            // Next button
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 0, 24, 32),
              child: SizedBox(
                width: double.infinity,
                height: 54,
                child: ElevatedButton(
                  onPressed: _next,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.gold,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(AppRadius.full),
                    ),
                  ),
                  child: Text(
                    _step == _totalSteps - 1 ? 'FINISH' : 'CONTINUE',
                    style: AppText.buttonText,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Step 1: Body Measurements ──────────────────────────────────────────────
  Widget _buildBodyStep() {
    return FadeInUp(
      duration: const Duration(milliseconds: 500),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 16),
              Text('Your Body', style: AppText.displayMedium),
              const SizedBox(height: 8),
              Text('Helps us recommend the perfect size',
                  style: AppText.bodyLarge),
              const SizedBox(height: 32),

              // Name
              Text('NAME (optional)', style: AppText.labelSmall),
              const SizedBox(height: 8),
              TextField(
                onChanged: (v) => _name = v,
                style: AppText.bodyLarge.copyWith(color: AppColors.textPrimary),
                decoration: InputDecoration(
                  hintText: 'What should we call you?',
                  hintStyle: AppText.bodyMedium,
                  filled: true,
                  fillColor: AppColors.surface,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(AppRadius.md),
                    borderSide: BorderSide.none,
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(AppRadius.md),
                    borderSide: const BorderSide(color: AppColors.gold),
                  ),
                ),
              ),
              const SizedBox(height: 24),

              // Height
              Text('HEIGHT', style: AppText.labelSmall),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: Slider(
                      value: _height,
                      min: 140,
                      max: 210,
                      divisions: 70,
                      activeColor: AppColors.gold,
                      inactiveColor: AppColors.surfaceLight,
                      onChanged: (v) => setState(() => _height = v),
                    ),
                  ),
                  Container(
                    width: 60,
                    alignment: Alignment.center,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: AppColors.surface,
                      borderRadius: BorderRadius.circular(AppRadius.sm),
                    ),
                    child: Text(
                      '${_height.round()} cm',
                      style: AppText.titleMedium.copyWith(fontSize: 14),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),

              // Weight
              Text('WEIGHT', style: AppText.labelSmall),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: Slider(
                      value: _weight,
                      min: 40,
                      max: 130,
                      divisions: 90,
                      activeColor: AppColors.gold,
                      inactiveColor: AppColors.surfaceLight,
                      onChanged: (v) => setState(() => _weight = v),
                    ),
                  ),
                  Container(
                    width: 60,
                    alignment: Alignment.center,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: AppColors.surface,
                      borderRadius: BorderRadius.circular(AppRadius.sm),
                    ),
                    child: Text(
                      '${_weight.round()} kg',
                      style: AppText.titleMedium.copyWith(fontSize: 14),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 24),

              // Body type
              Text('BODY TYPE', style: AppText.labelSmall),
              const SizedBox(height: 12),
              Wrap(
                spacing: 10,
                runSpacing: 10,
                children: BodyType.values.map((bt) {
                  final isActive = bt == _bodyType;
                  return GestureDetector(
                    onTap: () {
                      Haptics.selection();
                      setState(() => _bodyType = bt);
                    },
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 10),
                      decoration: BoxDecoration(
                        color: isActive ? AppColors.gold : AppColors.surface,
                        borderRadius: BorderRadius.circular(AppRadius.full),
                        border: Border.all(
                          color: isActive
                              ? AppColors.gold
                              : AppColors.surfaceLight,
                        ),
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            bt.label,
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: isActive
                                  ? FontWeight.w700
                                  : FontWeight.w400,
                              color: isActive
                                  ? AppColors.background
                                  : AppColors.greyLight,
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                }).toList(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Step 2: Skin Tone ──────────────────────────────────────────────────────
  Widget _buildSkinToneStep() {
    return FadeInUp(
      duration: const Duration(milliseconds: 500),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 16),
            Text('Your Skin Tone', style: AppText.displayMedium),
            const SizedBox(height: 8),
            Text('We\'ll suggest colors that complement you best',
                style: AppText.bodyLarge),
            const SizedBox(height: 40),
            Center(
              child: Wrap(
                spacing: 16,
                runSpacing: 20,
                alignment: WrapAlignment.center,
                children: SkinTone.values.map((st) {
                  final isActive = st == _skinTone;
                  return GestureDetector(
                    onTap: () {
                      Haptics.selection();
                      setState(() => _skinTone = st);
                    },
                    child: Column(
                      children: [
                        AnimatedContainer(
                          duration: const Duration(milliseconds: 200),
                          width: 60,
                          height: 60,
                          decoration: BoxDecoration(
                            color: Color(st.colorValue),
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: isActive
                                  ? AppColors.gold
                                  : Colors.transparent,
                              width: 3,
                            ),
                            boxShadow: isActive
                                ? [
                                    BoxShadow(
                                      color:
                                          AppColors.gold.withValues(alpha: 0.3),
                                      blurRadius: 12,
                                      spreadRadius: 2,
                                    )
                                  ]
                                : null,
                          ),
                          child: isActive
                              ? const Icon(Icons.check,
                                  color: AppColors.textPrimary, size: 24)
                              : null,
                        ),
                        const SizedBox(height: 6),
                        Text(
                          st.label,
                          style: TextStyle(
                            fontSize: 12,
                            color: isActive ? AppColors.gold : AppColors.grey,
                            fontWeight: isActive
                                ? FontWeight.w600
                                : FontWeight.w400,
                          ),
                        ),
                      ],
                    ),
                  );
                }).toList(),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Step 3: Style Preferences ──────────────────────────────────────────────
  Widget _buildStyleStep() {
    return FadeInUp(
      duration: const Duration(milliseconds: 500),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 16),
            Text('Your Style', style: AppText.displayMedium),
            const SizedBox(height: 8),
            Text('Pick all that describe you (multiple ok)',
                style: AppText.bodyLarge),
            const SizedBox(height: 32),
            ...StylePreference.values.map((sp) {
              final isActive = _stylePrefs.contains(sp);
              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: GestureDetector(
                  onTap: () {
                    Haptics.selection();
                    setState(() {
                      if (isActive) {
                        _stylePrefs.remove(sp);
                      } else {
                        _stylePrefs.add(sp);
                      }
                    });
                  },
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: isActive
                          ? AppColors.gold.withValues(alpha: 0.1)
                          : AppColors.surface,
                      borderRadius: BorderRadius.circular(AppRadius.lg),
                      border: Border.all(
                        color:
                            isActive ? AppColors.gold : AppColors.surfaceLight,
                        width: isActive ? 1.5 : 0.5,
                      ),
                    ),
                    child: Row(
                      children: [
                        AnimatedContainer(
                          duration: const Duration(milliseconds: 200),
                          width: 24,
                          height: 24,
                          decoration: BoxDecoration(
                            color: isActive
                                ? AppColors.gold
                                : Colors.transparent,
                            borderRadius:
                                BorderRadius.circular(6),
                            border: Border.all(
                              color: isActive
                                  ? AppColors.gold
                                  : AppColors.grey,
                              width: 1.5,
                            ),
                          ),
                          child: isActive
                              ? const Icon(Icons.check,
                                  size: 16, color: AppColors.background)
                              : null,
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                sp.label,
                                style: AppText.titleMedium.copyWith(
                                  color: isActive
                                      ? AppColors.white
                                      : AppColors.greyLight,
                                ),
                              ),
                              Text(sp.description,
                                  style: AppText.bodyMedium
                                      .copyWith(fontSize: 12)),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              );
            }),
          ],
        ),
      ),
    );
  }

  // ── Step 4: Occasion ───────────────────────────────────────────────────────
  Widget _buildOccasionStep() {
    final icons = [
      Icons.wb_sunny_rounded,
      Icons.business_center_rounded,
      Icons.favorite_rounded,
      Icons.celebration_rounded,
      Icons.auto_awesome_rounded,
    ];

    return FadeInUp(
      duration: const Duration(milliseconds: 500),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 16),
            Text('Primary Occasion', style: AppText.displayMedium),
            const SizedBox(height: 8),
            Text('What do you mostly dress for?',
                style: AppText.bodyLarge),
            const SizedBox(height: 32),
            ...List.generate(Occasion.values.length, (i) {
              final occ = Occasion.values[i];
              final isActive = occ == _occasion;
              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: GestureDetector(
                  onTap: () {
                    Haptics.selection();
                    setState(() => _occasion = occ);
                  },
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: isActive
                          ? AppColors.gold.withValues(alpha: 0.1)
                          : AppColors.surface,
                      borderRadius: BorderRadius.circular(AppRadius.lg),
                      border: Border.all(
                        color:
                            isActive ? AppColors.gold : AppColors.surfaceLight,
                        width: isActive ? 1.5 : 0.5,
                      ),
                    ),
                    child: Row(
                      children: [
                        Container(
                          width: 44,
                          height: 44,
                          decoration: BoxDecoration(
                            color: isActive
                                ? AppColors.gold
                                : AppColors.surfaceLight,
                            borderRadius:
                                BorderRadius.circular(AppRadius.md),
                          ),
                          child: Icon(
                            icons[i],
                            color: isActive
                                ? AppColors.background
                                : AppColors.grey,
                            size: 22,
                          ),
                        ),
                        const SizedBox(width: 14),
                        Text(
                          occ.label,
                          style: AppText.titleMedium.copyWith(
                            color: isActive
                                ? AppColors.white
                                : AppColors.greyLight,
                          ),
                        ),
                        const Spacer(),
                        if (isActive)
                          const Icon(Icons.check_circle,
                              color: AppColors.gold, size: 22),
                      ],
                    ),
                  ),
                ),
              );
            }),
          ],
        ),
      ),
    );
  }
}
