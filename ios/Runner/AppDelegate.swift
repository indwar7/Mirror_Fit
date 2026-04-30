import Flutter
import UIKit
import ARKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  private var arChannel: FlutterMethodChannel?

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)
    setupARChannel()
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  private func setupARChannel() {
    // Try registrar first (works even before window is ready on iOS 26)
    if let registrar = self.registrar(forPlugin: "ARTryOnPlugin") {
      let channel = FlutterMethodChannel(
        name: "com.lucy.artryon",
        binaryMessenger: registrar.messenger()
      )
      self.arChannel = channel
      setHandler(on: channel)
      print("[AR] Channel set up via registrar")
      return
    }

    // Fallback: try via rootViewController
    if let controller = window?.rootViewController as? FlutterViewController {
      let channel = FlutterMethodChannel(
        name: "com.lucy.artryon",
        binaryMessenger: controller.binaryMessenger
      )
      self.arChannel = channel
      setHandler(on: channel)
      print("[AR] Channel set up via rootViewController")
      return
    }

    print("[AR] WARNING: Could not set up method channel in didFinishLaunching")
  }

  private func setHandler(on channel: FlutterMethodChannel) {
    channel.setMethodCallHandler { [weak self] call, result in
      switch call.method {
      case "launchARTryOn":
        DispatchQueue.main.async {
          self?.launchARTryOn(result: result)
        }
      case "isBodyTrackingSupported":
        result(ARBodyTrackingConfiguration.isSupported)
      default:
        result(FlutterMethodNotImplemented)
      }
    }
  }

  // Retry channel setup when app becomes active (window is guaranteed ready)
  override func applicationDidBecomeActive(_ application: UIApplication) {
    super.applicationDidBecomeActive(application)
    if arChannel == nil {
      setupARChannel()
    }
  }

  private func launchARTryOn(result: @escaping FlutterResult) {
    guard let topVC = findTopViewController() else {
      result(FlutterError(code: "NO_VC", message: "No view controller available", details: nil))
      return
    }

    let arVC = ARBodyTryOnViewController()
    arVC.modalPresentationStyle = .fullScreen
    arVC.onDismiss = { result("dismissed") }
    topVC.present(arVC, animated: true)
  }

  private func findTopViewController() -> UIViewController? {
    var rootVC: UIViewController?

    // Try self.window
    if let vc = self.window?.rootViewController {
      rootVC = vc
    }

    // Scene-based fallback for iOS 13+
    if rootVC == nil, #available(iOS 13.0, *) {
      for scene in UIApplication.shared.connectedScenes {
        guard let ws = scene as? UIWindowScene else { continue }
        if #available(iOS 15.0, *), let w = ws.keyWindow, let vc = w.rootViewController {
          rootVC = vc
          break
        }
        if let w = ws.windows.first(where: { $0.isKeyWindow }), let vc = w.rootViewController {
          rootVC = vc
          break
        }
        if let w = ws.windows.first, let vc = w.rootViewController {
          rootVC = vc
          break
        }
      }
    }

    guard var top = rootVC else { return nil }
    while let presented = top.presentedViewController {
      top = presented
    }
    return top
  }
}
