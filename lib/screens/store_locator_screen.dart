import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import 'package:url_launcher/url_launcher.dart';
import '../utils/app_theme.dart';
import '../utils/haptics.dart';

class StoreLocatorScreen extends StatelessWidget {
  const StoreLocatorScreen({super.key});

  static const _stores = [
    _Store(
      name: 'LUCY Store - Connaught Place',
      address: 'Block A, Connaught Place, New Delhi 110001',
      phone: '+91 11 2334 5678',
      lat: 28.6315,
      lng: 77.2167,
      hasExclusiveOffers: true,
    ),
    _Store(
      name: 'LUCY - Linking Road',
      address: '45 Linking Road, Bandra West, Mumbai 400050',
      phone: '+91 22 2640 1234',
      lat: 19.0596,
      lng: 72.8295,
      hasExclusiveOffers: true,
    ),
    _Store(
      name: 'LUCY - MG Road',
      address: '12 MG Road, Bengaluru 560001',
      phone: '+91 80 2558 9012',
      lat: 12.9716,
      lng: 77.5946,
      hasExclusiveOffers: false,
    ),
    _Store(
      name: 'LUCY - Park Street',
      address: '78 Park Street, Kolkata 700016',
      phone: '+91 33 2229 3456',
      lat: 22.5554,
      lng: 88.3517,
      hasExclusiveOffers: false,
    ),
    _Store(
      name: 'LUCY - Phoenix Mall',
      address: 'Phoenix Marketcity, Whitefield, Bengaluru 560048',
      phone: '+91 80 4150 7890',
      lat: 12.9983,
      lng: 77.6969,
      hasExclusiveOffers: true,
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: AppColors.background,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new,
              color: AppColors.textPrimary, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text('LUCY Stores', style: AppText.titleLarge),
        centerTitle: true,
      ),
      body: ListView.builder(
        padding: const EdgeInsets.all(16),
        physics: const BouncingScrollPhysics(),
        itemCount: _stores.length,
        itemBuilder: (_, i) => FadeInUp(
          duration: const Duration(milliseconds: 400),
          delay: Duration(milliseconds: 80 * i),
          child: _StoreCard(store: _stores[i]),
        ),
      ),
    );
  }
}

class _Store {
  final String name;
  final String address;
  final String phone;
  final double lat;
  final double lng;
  final bool hasExclusiveOffers;

  const _Store({
    required this.name,
    required this.address,
    required this.phone,
    required this.lat,
    required this.lng,
    required this.hasExclusiveOffers,
  });
}

class _StoreCard extends StatelessWidget {
  final _Store store;
  const _StoreCard({required this.store});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(AppSpacing.md),
      decoration: AppDecoration.gradientCard,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: AppColors.gold.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(AppRadius.md),
                ),
                child: const Icon(Icons.store_rounded,
                    color: AppColors.gold, size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(store.name, style: AppText.titleMedium),
                    if (store.hasExclusiveOffers)
                      Container(
                        margin: const EdgeInsets.only(top: 4),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: AppColors.gold,
                          borderRadius:
                              BorderRadius.circular(AppRadius.full),
                        ),
                        child: const Text(
                          'IN-STORE OFFERS',
                          style: TextStyle(
                            fontSize: 8,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 1,
                            color: AppColors.background,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(store.address, style: AppText.bodyMedium),
          const SizedBox(height: 12),
          Row(
            children: [
              _ActionChip(
                icon: Icons.directions_rounded,
                label: 'Directions',
                onTap: () {
                  Haptics.light();
                  launchUrl(Uri.parse(
                      'https://www.google.com/maps/search/?api=1&query=${store.lat},${store.lng}'));
                },
              ),
              const SizedBox(width: 8),
              _ActionChip(
                icon: Icons.phone_outlined,
                label: 'Call',
                onTap: () {
                  Haptics.light();
                  launchUrl(Uri.parse('tel:${store.phone}'));
                },
              ),
              const SizedBox(width: 8),
              _ActionChip(
                icon: Icons.calendar_today_rounded,
                label: 'Book Stylist',
                onTap: () {
                  Haptics.light();
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      backgroundColor: AppColors.surface,
                      behavior: SnackBarBehavior.floating,
                      content: Text(
                        'Stylist booking coming soon!',
                        style: TextStyle(color: AppColors.textPrimary),
                      ),
                    ),
                  );
                },
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _ActionChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _ActionChip({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: AppColors.surfaceLight,
          borderRadius: BorderRadius.circular(AppRadius.full),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: AppColors.gold),
            const SizedBox(width: 4),
            Text(
              label,
              style: AppText.bodyMedium.copyWith(fontSize: 11),
            ),
          ],
        ),
      ),
    );
  }
}
