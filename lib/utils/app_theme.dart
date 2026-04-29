import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

// ── Brand Colors ──────────────────────────────────────────────────────────────
class AppColors {
  AppColors._();

  // Primary accent — coral/pink
  static const Color primary = Color(0xFFFF3F6C);
  static const Color primaryLight = Color(0xFFFF6B8A);
  static const Color primaryDark = Color(0xFFE5335E);

  // Secondary
  static const Color accent = Color(0xFF3B82F6);

  // Semantic
  static const Color success = Color(0xFF03A685);
  static const Color error = Color(0xFFE53935);
  static const Color info = Color(0xFF42A5F5);
  static const Color orange = Color(0xFFFF8F1C);

  // Backward-compat aliases
  static const Color gold = primary;
  static const Color goldLight = primaryLight;
  static const Color goldDark = primaryDark;
  static const Color white = Color(0xFFFFFFFF);

  // ── Dark palette ───────────────────────────────────────────────────────────
  static const Color darkBg = Color(0xFF0D0D0F);
  static const Color darkSurface = Color(0xFF1C1C1E);
  static const Color darkCard = Color(0xFF2C2C2E);
  static const Color darkBorder = Color(0xFF38383A);
  static const Color darkTextPrimary = Color(0xFFFFFFFF);
  static const Color darkTextSecondary = Color(0xFF9E9EA3);
  static const Color darkTextTertiary = Color(0xFF636366);

  // ── Light palette ──────────────────────────────────────────────────────────
  static const Color background = Color(0xFFF5F5F6);
  static const Color surface = Color(0xFFFFFFFF);
  static const Color surfaceLight = Color(0xFFE8E8E8);
  static const Color surfaceBright = Color(0xFFD5D5D5);
  static const Color textPrimary = Color(0xFF282C3F);
  static const Color textSecondary = Color(0xFF535766);
  static const Color greyLight = Color(0xFF7E818C);
  static const Color grey = Color(0xFF94969F);
  static const Color greyDark = Color(0xFF696B79);

  // Light compat aliases
  static const Color lightBackground = background;
  static const Color lightSurface = surface;
  static const Color lightSurfaceLight = surfaceLight;
  static const Color lightSurfaceBright = surfaceBright;
  static const Color lightCardGradient = surface;

  // ── Gradients ──────────────────────────────────────────────────────────────
  static const LinearGradient goldGradient = LinearGradient(
    colors: [primaryDark, primary, primaryLight],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient darkGradient = LinearGradient(
    colors: [Color(0xFF1A0A10), darkBg],
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
  );

  static const LinearGradient cardGradient = LinearGradient(
    colors: [darkSurface, darkCard],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );
}

// ── Spacing & Radius ──────────────────────────────────────────────────────────
class AppSpacing {
  AppSpacing._();
  static const double xs = 4;
  static const double sm = 8;
  static const double md = 16;
  static const double lg = 24;
  static const double xl = 32;
  static const double xxl = 48;
}

class AppRadius {
  AppRadius._();
  static const double sm = 8;
  static const double md = 12;
  static const double lg = 16;
  static const double xl = 20;
  static const double xxl = 28;
  static const double full = 100;
}

// ── Text Styles ───────────────────────────────────────────────────────────────
class AppText {
  AppText._();

  static TextStyle get _base =>
      GoogleFonts.inter(color: AppColors.darkTextPrimary);

  static TextStyle get displayLarge => GoogleFonts.poppins(
        fontSize: 36,
        fontWeight: FontWeight.w700,
        color: AppColors.primary,
        letterSpacing: 4,
      );

  static TextStyle get displayMedium => GoogleFonts.poppins(
        fontSize: 22,
        fontWeight: FontWeight.w700,
        color: AppColors.primary,
        letterSpacing: 2,
      );

  static TextStyle get titleLarge => _base.copyWith(
        fontSize: 20,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.3,
        color: AppColors.darkTextPrimary,
      );

  static TextStyle get titleMedium => _base.copyWith(
        fontSize: 16,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.2,
        color: AppColors.darkTextPrimary,
      );

  static TextStyle get bodyLarge => _base.copyWith(
        fontSize: 16,
        fontWeight: FontWeight.w400,
        color: AppColors.darkTextSecondary,
      );

  static TextStyle get bodyMedium => _base.copyWith(
        fontSize: 14,
        fontWeight: FontWeight.w400,
        color: AppColors.darkTextTertiary,
      );

  static TextStyle get labelSmall => _base.copyWith(
        fontSize: 11,
        fontWeight: FontWeight.w600,
        letterSpacing: 1.5,
        color: AppColors.primary,
      );

  static TextStyle get labelMedium => _base.copyWith(
        fontSize: 13,
        fontWeight: FontWeight.w500,
        letterSpacing: 0.8,
        color: AppColors.darkTextSecondary,
      );

  static TextStyle get priceText => _base.copyWith(
        fontSize: 18,
        fontWeight: FontWeight.w700,
        color: AppColors.darkTextPrimary,
      );

  static TextStyle get buttonText => _base.copyWith(
        fontSize: 14,
        fontWeight: FontWeight.w700,
        letterSpacing: 1,
        color: AppColors.white,
      );
}

// ── Common Decorations ────────────────────────────────────────────────────────
class AppDecoration {
  AppDecoration._();

  static BoxDecoration get card => BoxDecoration(
        color: AppColors.darkSurface,
        borderRadius: BorderRadius.circular(AppRadius.lg),
        border: Border.all(color: AppColors.darkBorder),
      );

  static BoxDecoration get gradientCard => BoxDecoration(
        color: AppColors.darkSurface,
        borderRadius: BorderRadius.circular(AppRadius.lg),
        border: Border.all(color: AppColors.darkBorder),
      );

  static BoxDecoration get goldBorderCard => BoxDecoration(
        color: AppColors.darkSurface,
        borderRadius: BorderRadius.circular(AppRadius.lg),
        border: Border.all(
            color: AppColors.primary.withValues(alpha: 0.5), width: 1.5),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withValues(alpha: 0.12),
            blurRadius: 12,
            offset: const Offset(0, 2),
          ),
        ],
      );

  static BoxDecoration get selectedCard => BoxDecoration(
        color: AppColors.darkSurface,
        borderRadius: BorderRadius.circular(AppRadius.lg),
        border: Border.all(color: AppColors.primary, width: 2),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withValues(alpha: 0.2),
            blurRadius: 12,
            spreadRadius: 2,
          ),
        ],
      );

  static BoxDecoration get imageContainer => BoxDecoration(
        color: AppColors.darkCard,
        borderRadius: BorderRadius.circular(AppRadius.xl),
      );
}

// ── Theme Data ────────────────────────────────────────────────────────────────
class AppTheme {
  AppTheme._();

  static ThemeData get dark => ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: AppColors.darkBg,
        colorScheme: const ColorScheme.dark(
          primary: AppColors.primary,
          secondary: AppColors.accent,
          surface: AppColors.darkSurface,
          error: AppColors.error,
          onPrimary: AppColors.white,
          onSurface: AppColors.darkTextPrimary,
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: AppColors.darkBg,
          elevation: 0,
          centerTitle: true,
          titleTextStyle: AppText.titleLarge,
          iconTheme:
              const IconThemeData(color: AppColors.darkTextPrimary),
          surfaceTintColor: Colors.transparent,
        ),
        cardColor: AppColors.darkSurface,
        dialogTheme: const DialogThemeData(
          backgroundColor: AppColors.darkSurface,
          surfaceTintColor: Colors.transparent,
        ),
        bottomSheetTheme: const BottomSheetThemeData(
          backgroundColor: AppColors.darkSurface,
          surfaceTintColor: Colors.transparent,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.primary,
            foregroundColor: AppColors.white,
            elevation: 0,
            padding:
                const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(AppRadius.full),
            ),
            textStyle: AppText.buttonText,
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: AppColors.darkTextPrimary,
            side: const BorderSide(color: AppColors.darkBorder),
            padding:
                const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(AppRadius.full),
            ),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: AppColors.darkCard,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(AppRadius.md),
            borderSide: const BorderSide(color: AppColors.darkBorder),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(AppRadius.md),
            borderSide: const BorderSide(color: AppColors.darkBorder),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(AppRadius.md),
            borderSide:
                const BorderSide(color: AppColors.primary, width: 1.5),
          ),
          hintStyle:
              const TextStyle(color: AppColors.darkTextTertiary),
          labelStyle:
              const TextStyle(color: AppColors.darkTextSecondary),
        ),
        snackBarTheme: SnackBarThemeData(
          backgroundColor: AppColors.darkCard,
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppRadius.md),
          ),
          contentTextStyle:
              const TextStyle(color: AppColors.darkTextPrimary),
        ),
        dividerColor: AppColors.darkBorder,
        iconTheme:
            const IconThemeData(color: AppColors.darkTextPrimary),
        textTheme: TextTheme(
          bodyLarge: AppText.bodyLarge,
          bodyMedium: AppText.bodyMedium,
          titleLarge: AppText.titleLarge,
          titleMedium: AppText.titleMedium,
          labelSmall: AppText.labelSmall,
        ),
      );

  static ThemeData get light => ThemeData(
        brightness: Brightness.light,
        scaffoldBackgroundColor: AppColors.background,
        colorScheme: const ColorScheme.light(
          primary: AppColors.primary,
          secondary: AppColors.accent,
          surface: AppColors.surface,
          error: AppColors.error,
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: AppColors.surface,
          elevation: 0,
          centerTitle: true,
          titleTextStyle: AppText.titleLarge.copyWith(
            color: AppColors.textPrimary,
          ),
          iconTheme:
              const IconThemeData(color: AppColors.textPrimary),
          surfaceTintColor: Colors.transparent,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.primary,
            foregroundColor: AppColors.white,
            elevation: 0,
            padding:
                const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(AppRadius.full),
            ),
            textStyle: AppText.buttonText,
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: AppColors.primary,
            side: const BorderSide(color: AppColors.primary),
            padding:
                const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(AppRadius.full),
            ),
          ),
        ),
        snackBarTheme: SnackBarThemeData(
          backgroundColor: AppColors.textPrimary,
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppRadius.md),
          ),
          contentTextStyle:
              const TextStyle(color: AppColors.white),
        ),
      );
}
