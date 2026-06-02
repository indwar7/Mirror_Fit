import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

/// Shows the composited try-on photo with share/save options.
class TryOnPhotoScreen extends StatelessWidget {
  final Uint8List photoBytes;
  final String garmentName;

  const TryOnPhotoScreen({
    super.key,
    required this.photoBytes,
    required this.garmentName,
  });

  Future<void> _share(BuildContext context) async {
    try {
      final dir = await getTemporaryDirectory();
      final file = File('${dir.path}/tryon_result.png');
      await file.writeAsBytes(photoBytes);
      await Share.shareXFiles(
        [XFile(file.path)],
        text: 'Check out how I look in $garmentName! 🔥',
      );
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not share: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text('You in $garmentName'),
        centerTitle: true,
      ),
      body: Column(
        children: [
          Expanded(
            child: InteractiveViewer(
              child: Center(
                child: Image.memory(photoBytes, fit: BoxFit.contain),
              ),
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 20),
            color: Colors.black,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _btn(
                  context,
                  icon: Icons.arrow_back,
                  label: 'Retake',
                  onTap: () => Navigator.pop(context),
                ),
                _btn(
                  context,
                  icon: Icons.share,
                  label: 'Share',
                  onTap: () => _share(context),
                  highlight: true,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _btn(BuildContext context, {
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    bool highlight = false,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
        decoration: BoxDecoration(
          color: highlight ? Colors.white : Colors.white12,
          borderRadius: BorderRadius.circular(30),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: highlight ? Colors.black : Colors.white, size: 20),
            const SizedBox(width: 8),
            Text(label, style: TextStyle(
              color: highlight ? Colors.black : Colors.white,
              fontWeight: FontWeight.bold,
            )),
          ],
        ),
      ),
    );
  }
}
