import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/user_profile.dart';

class UserPreferences {
  static const _keyOnboardingDone = 'onboarding_done';
  static const _keyProfile = 'user_profile';
  static const _keyWishlist = 'wishlist_ids';
  static const _keyRecentlyViewed = 'recently_viewed';
  static const _keyStylePoints = 'style_points';
  static const _keyBadges = 'badges';
  static const _keyThemeMode = 'theme_mode';
  static const _keyTryOnCount = 'tryon_count';

  static late SharedPreferences _prefs;

  static Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
  }

  // ── Onboarding ────────────────────────────────────────────────────────────
  static bool get isOnboardingDone => _prefs.getBool(_keyOnboardingDone) ?? false;
  static Future<void> setOnboardingDone() =>
      _prefs.setBool(_keyOnboardingDone, true);

  // ── User Profile ──────────────────────────────────────────────────────────
  static UserProfile? getProfile() {
    final json = _prefs.getString(_keyProfile);
    if (json == null) return null;
    return UserProfile.fromJson(jsonDecode(json) as Map<String, dynamic>);
  }

  static Future<void> saveProfile(UserProfile profile) =>
      _prefs.setString(_keyProfile, jsonEncode(profile.toJson()));

  static bool get hasProfile => _prefs.containsKey(_keyProfile);

  // ── Wishlist ──────────────────────────────────────────────────────────────
  static List<String> get wishlistIds =>
      _prefs.getStringList(_keyWishlist) ?? [];

  static bool isWishlisted(String id) => wishlistIds.contains(id);

  static Future<void> toggleWishlist(String id) async {
    final list = wishlistIds;
    if (list.contains(id)) {
      list.remove(id);
    } else {
      list.add(id);
    }
    await _prefs.setStringList(_keyWishlist, list);
  }

  // ── Recently Viewed ───────────────────────────────────────────────────────
  static List<String> get recentlyViewedIds =>
      _prefs.getStringList(_keyRecentlyViewed) ?? [];

  static Future<void> addRecentlyViewed(String id) async {
    final list = recentlyViewedIds;
    list.remove(id); // Remove if exists
    list.insert(0, id); // Add to front
    if (list.length > 10) list.removeLast();
    await _prefs.setStringList(_keyRecentlyViewed, list);
  }

  // ── Style Points & Gamification ───────────────────────────────────────────
  static int get stylePoints => _prefs.getInt(_keyStylePoints) ?? 0;

  static Future<void> addStylePoints(int points) =>
      _prefs.setInt(_keyStylePoints, stylePoints + points);

  static List<String> get badges =>
      _prefs.getStringList(_keyBadges) ?? [];

  static Future<void> addBadge(String badge) async {
    final list = badges;
    if (!list.contains(badge)) {
      list.add(badge);
      await _prefs.setStringList(_keyBadges, list);
    }
  }

  static int get tryOnCount => _prefs.getInt(_keyTryOnCount) ?? 0;

  static Future<void> incrementTryOnCount() async {
    final count = tryOnCount + 1;
    await _prefs.setInt(_keyTryOnCount, count);
    if (count == 1) await addBadge('first_look');
    if (count == 5) await addBadge('style_explorer');
    if (count == 10) await addBadge('trendsetter');
    if (count == 25) await addBadge('fashion_guru');
  }

  // ── Theme ─────────────────────────────────────────────────────────────────
  static bool get isDarkMode => _prefs.getBool(_keyThemeMode) ?? true;
  static Future<void> setDarkMode(bool value) =>
      _prefs.setBool(_keyThemeMode, value);

  // ── API Keys ─────────────────────────────────────────────────────────────
  static const _keyApiKey = 'fashn_api_key';
  static String get apiKey => _prefs.getString(_keyApiKey) ?? '';
  static Future<void> setApiKey(String key) =>
      _prefs.setString(_keyApiKey, key);
  static bool get hasApiKey => apiKey.isNotEmpty;

  static const _keyDecartApiKey = 'decart_api_key';
  static String get decartApiKey => _prefs.getString(_keyDecartApiKey) ?? '';
  static Future<void> setDecartApiKey(String key) =>
      _prefs.setString(_keyDecartApiKey, key);
  static bool get hasDecartApiKey => decartApiKey.isNotEmpty;
}
