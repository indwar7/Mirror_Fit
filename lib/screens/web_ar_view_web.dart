// ignore: avoid_web_libraries_in_flutter, deprecated_member_use
import 'dart:html' as html;
// ignore: avoid_web_libraries_in_flutter
import 'dart:ui_web' as ui_web;
import 'package:flutter/widgets.dart';

void registerWebArView(String htmlContent, String viewId) {
  try {
    ui_web.platformViewRegistry.registerViewFactory(
      viewId,
      (int id) => html.IFrameElement()
        ..setAttribute('srcdoc', htmlContent)
        ..style.width = '100%'
        ..style.height = '100%'
        ..style.border = 'none'
        ..allow = 'camera; microphone; display-capture'
        ..setAttribute('allowfullscreen', 'true'),
    );
  } catch (_) {
    // Already registered for this viewId — safe to ignore
  }
}

Widget buildWebArView(String viewId) => HtmlElementView(viewType: viewId);
