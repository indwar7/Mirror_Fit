import 'package:flutter/material.dart';
import 'package:confetti/confetti.dart';
import '../utils/app_theme.dart';

class CelebrationOverlay extends StatefulWidget {
  final Widget child;
  final bool trigger;

  const CelebrationOverlay({
    super.key,
    required this.child,
    required this.trigger,
  });

  @override
  State<CelebrationOverlay> createState() => _CelebrationOverlayState();
}

class _CelebrationOverlayState extends State<CelebrationOverlay> {
  late ConfettiController _confettiCtrl;

  @override
  void initState() {
    super.initState();
    _confettiCtrl = ConfettiController(duration: const Duration(seconds: 2));
  }

  @override
  void didUpdateWidget(CelebrationOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.trigger && !oldWidget.trigger) {
      _confettiCtrl.play();
    }
  }

  @override
  void dispose() {
    _confettiCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        widget.child,
        Align(
          alignment: Alignment.topCenter,
          child: ConfettiWidget(
            confettiController: _confettiCtrl,
            blastDirectionality: BlastDirectionality.explosive,
            shouldLoop: false,
            numberOfParticles: 25,
            maxBlastForce: 20,
            minBlastForce: 8,
            gravity: 0.15,
            colors: const [
              AppColors.gold,
              AppColors.goldLight,
              AppColors.goldDark,
              AppColors.white,
              Color(0xFFE8D48B),
            ],
          ),
        ),
      ],
    );
  }
}
