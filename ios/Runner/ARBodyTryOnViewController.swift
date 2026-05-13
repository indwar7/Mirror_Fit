import UIKit
import RealityKit
import ARKit

// MARK: - Jacket Rig (procedural fallback)

private struct ProcJacketRig {
    let root:         AnchorEntity
    let torso:        ModelEntity
    let leftSleeve:   ModelEntity
    let rightSleeve:  ModelEntity
    let collar:       ModelEntity
    let leftCuff:     ModelEntity
    let rightCuff:    ModelEntity
}

// MARK: - Joint Names (ARKit ARSkeleton3D)

private enum JointNames {
    static let spine7        = ARSkeleton.JointName(rawValue: "spine_7_joint")
    static let neck1         = ARSkeleton.JointName(rawValue: "neck_1_joint")
    static let leftShoulder  = ARSkeleton.JointName(rawValue: "left_shoulder_1_joint")
    static let rightShoulder = ARSkeleton.JointName(rawValue: "right_shoulder_1_joint")
    static let leftArm       = ARSkeleton.JointName(rawValue: "left_arm_joint")
    static let rightArm      = ARSkeleton.JointName(rawValue: "right_arm_joint")
    static let leftForeArm   = ARSkeleton.JointName(rawValue: "left_foreArm_joint")
    static let rightForeArm  = ARSkeleton.JointName(rawValue: "right_foreArm_joint")
    static let leftHip       = ARSkeleton.JointName(rawValue: "left_upLeg_joint")
    static let rightHip      = ARSkeleton.JointName(rawValue: "right_upLeg_joint")
}

// MARK: - Mode

private enum ARMode {
    case selfie   // front camera, ARFaceTrackingConfiguration, jacket hangs below face
    case body     // back camera, ARBodyTrackingConfiguration, jacket rigged to skeleton
}

// MARK: - View Controller

class ARBodyTryOnViewController: UIViewController, ARSessionDelegate {

    // MARK: - Properties

    private var arView: ARView!
    private var statusLabel: UILabel!
    private var titleLabel: UILabel!
    private var modeButton: UIButton!
    var onDismiss: (() -> Void)?

    private var mode: ARMode = .selfie   // default: front-camera so user sees themselves

    // Procedural rig used when no USDZ is provided
    private var procRig: ProcJacketRig?
    // USDZ rig used when assets/jacket.usdz is bundled
    private var usdzAnchor: AnchorEntity?
    private var usdzJacket: Entity?
    // True when a USDZ was successfully loaded
    private var hasUSDZ: Bool = false

    // Per-part smoothing state for procedural rig
    private var smoothPos = [SIMD3<Float>](repeating: .zero, count: 4)
    private var smoothRot = [simd_quatf](repeating: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), count: 4)
    private var smoothScl = [SIMD3<Float>](repeating: .one, count: 4)
    private var prevPos   = [SIMD3<Float>](repeating: .zero, count: 4)

    private let baseLerp: Float = 0.20
    private let fastLerp: Float = 0.75
    private let velocityThreshold: Float = 0.025

    private var frameCount: UInt64 = 0
    private let processEveryNthFrame: UInt64 = 2

    // Selfie-mode smoothing for the floating jacket
    private var selfieSmoothTransform: simd_float4x4?
    private let selfieLerp: Float = 0.25

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        setupARView()
        setupUI()
        loadJacketAsset()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        startSession()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        arView?.session.pause()
    }

    deinit {
        arView?.session.pause()
    }

    override var prefersStatusBarHidden: Bool { true }

    // MARK: - Asset Load

    /// Tries to load `jacket.usdz` from the app bundle.
    /// If found, uses it for both selfie and body modes (auto-scaled).
    /// If not, builds the procedural cylinder+box rig as a fallback.
    private func loadJacketAsset() {
        if let url = Bundle.main.url(forResource: "jacket", withExtension: "usdz") {
            do {
                let loaded = try Entity.load(contentsOf: url)
                hasUSDZ = true
                usdzJacket = loaded
                let root = AnchorEntity(world: simd_float4x4(1))
                root.addChild(loaded)
                root.isEnabled = false
                usdzAnchor = root
                arView.scene.addAnchor(root)
                print("[AR] Loaded jacket.usdz from bundle")
                return
            } catch {
                print("[AR] jacket.usdz exists but failed to load: \(error). Falling back to procedural rig.")
            }
        } else {
            print("[AR] No jacket.usdz in bundle — using procedural rig")
        }
        buildProceduralJacket()
    }

    // MARK: - AR Setup

    private func setupARView() {
        arView = ARView(frame: view.bounds)
        arView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        arView.renderOptions = [.disableMotionBlur, .disableDepthOfField]
        view.addSubview(arView)
        arView.session.delegate = self
    }

    private func startSession() {
        switch mode {
        case .selfie:  startSelfieTracking()
        case .body:    startBodyTracking()
        }
    }

    private func startSelfieTracking() {
        guard ARFaceTrackingConfiguration.isSupported else {
            updateStatus("Front-camera tracking unsupported — switching to back camera")
            mode = .body
            startBodyTracking()
            return
        }
        let config = ARFaceTrackingConfiguration()
        config.maximumNumberOfTrackedFaces = 1
        config.isLightEstimationEnabled = true
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        setAnchorEnabled(false)
        updateStatus("Selfie mode — face the camera, jacket follows your shoulders")
    }

    private func startBodyTracking() {
        guard ARBodyTrackingConfiguration.isSupported else {
            updateStatus("Body tracking not supported — switching to selfie")
            mode = .selfie
            startSelfieTracking()
            return
        }
        let config = ARBodyTrackingConfiguration()
        config.automaticSkeletonScaleEstimationEnabled = true
        if ARBodyTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            config.frameSemantics.insert(.personSegmentationWithDepth)
        } else if ARBodyTrackingConfiguration.supportsFrameSemantics(.personSegmentation) {
            config.frameSemantics.insert(.personSegmentation)
        }
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        setAnchorEnabled(false)
        updateStatus("Body mode — back camera, 2-3m away, full body in frame")
        DispatchQueue.main.asyncAfter(deadline: .now() + 8) { [weak self] in
            guard let self = self, self.mode == .body, !self.isAnchorEnabled() else { return }
            self.updateStatus("No body detected — back camera must see head-to-knees")
        }
    }

    @objc private func toggleMode() {
        // Reset smoothing so the jacket doesn't snap from a stale position.
        smoothPos = [SIMD3<Float>](repeating: .zero, count: 4)
        smoothScl = [SIMD3<Float>](repeating: .one,  count: 4)
        smoothRot = [simd_quatf](repeating: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), count: 4)
        prevPos   = [SIMD3<Float>](repeating: .zero, count: 4)
        selfieSmoothTransform = nil

        mode = (mode == .selfie) ? .body : .selfie
        modeButton.setTitle(mode == .selfie ? "Back Cam" : "Selfie", for: .normal)
        startSession()
    }

    // MARK: - Anchor helpers (works for both USDZ + procedural)

    private func setAnchorEnabled(_ enabled: Bool) {
        if hasUSDZ {
            usdzAnchor?.isEnabled = enabled
        } else {
            procRig?.root.isEnabled = enabled
        }
    }

    private func isAnchorEnabled() -> Bool {
        if hasUSDZ { return usdzAnchor?.isEnabled ?? false }
        return procRig?.root.isEnabled ?? false
    }

    // MARK: - UI

    private func setupUI() {
        let closeBtn = UIButton(type: .system)
        closeBtn.setImage(
            UIImage(systemName: "xmark.circle.fill")?
                .withConfiguration(UIImage.SymbolConfiguration(pointSize: 28)),
            for: .normal
        )
        closeBtn.tintColor = .white
        closeBtn.translatesAutoresizingMaskIntoConstraints = false
        closeBtn.addTarget(self, action: #selector(closeTapped), for: .touchUpInside)
        applyShadow(to: closeBtn.layer)
        view.addSubview(closeBtn)

        titleLabel = UILabel()
        titleLabel.text = "AR Try-On"
        titleLabel.textColor = .white
        titleLabel.font = .systemFont(ofSize: 17, weight: .semibold)
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        applyShadow(to: titleLabel.layer)
        view.addSubview(titleLabel)

        modeButton = UIButton(type: .system)
        modeButton.setTitle("Back Cam", for: .normal)
        modeButton.titleLabel?.font = .systemFont(ofSize: 13, weight: .semibold)
        modeButton.setTitleColor(.white, for: .normal)
        modeButton.backgroundColor = UIColor.black.withAlphaComponent(0.55)
        modeButton.layer.cornerRadius = 16
        modeButton.contentEdgeInsets = UIEdgeInsets(top: 6, left: 14, bottom: 6, right: 14)
        modeButton.translatesAutoresizingMaskIntoConstraints = false
        modeButton.addTarget(self, action: #selector(toggleMode), for: .touchUpInside)
        view.addSubview(modeButton)

        statusLabel = PaddedLabel()
        statusLabel.text = "Initializing..."
        statusLabel.textColor = .white
        statusLabel.font = .systemFont(ofSize: 13, weight: .medium)
        statusLabel.textAlignment = .center
        statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        statusLabel.layer.cornerRadius = 20
        statusLabel.layer.masksToBounds = true
        statusLabel.numberOfLines = 2
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(statusLabel)

        NSLayoutConstraint.activate([
            closeBtn.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            closeBtn.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            closeBtn.widthAnchor.constraint(equalToConstant: 44),
            closeBtn.heightAnchor.constraint(equalToConstant: 44),

            titleLabel.centerYAnchor.constraint(equalTo: closeBtn.centerYAnchor),
            titleLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),

            modeButton.centerYAnchor.constraint(equalTo: closeBtn.centerYAnchor),
            modeButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),

            statusLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),
            statusLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            statusLabel.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 24),
            statusLabel.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -24),
        ])
    }

    private func applyShadow(to layer: CALayer) {
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.5
        layer.shadowRadius = 4
        layer.shadowOffset = .zero
    }

    @objc private func closeTapped() {
        arView?.session.pause()
        dismiss(animated: true) { [weak self] in
            self?.onDismiss?()
        }
    }

    private func updateStatus(_ text: String) {
        DispatchQueue.main.async { [weak self] in
            self?.statusLabel?.text = text
        }
    }

    // MARK: - Fabric Material

    private func makeFabricMaterial() -> PhysicallyBasedMaterial {
        var mat = PhysicallyBasedMaterial()
        mat.baseColor = .init(tint: UIColor(red: 0.08, green: 0.12, blue: 0.22, alpha: 1.0))
        mat.roughness = .init(floatLiteral: 0.82)
        mat.metallic  = .init(floatLiteral: 0.0)
        return mat
    }

    // MARK: - Procedural Mesh Builders

    private func buildTaperedCylinder(
        topRadius: Float, bottomRadius: Float, segments: Int = 14
    ) throws -> MeshResource {
        var positions = [SIMD3<Float>]()
        var normals   = [SIMD3<Float>]()
        var indices   = [UInt32]()
        let segF = Float(segments)
        for i in 0...segments {
            let angle = Float(i) / segF * 2.0 * Float.pi
            let c = cos(angle), s = sin(angle)
            positions.append(SIMD3<Float>(c * topRadius,    1.0, s * topRadius))
            normals.append(normalize(SIMD3<Float>(c, 0, s)))
            positions.append(SIMD3<Float>(c * bottomRadius, 0.0, s * bottomRadius))
            normals.append(normalize(SIMD3<Float>(c, 0, s)))
        }
        for i in 0..<UInt32(segments) {
            let base = i * 2
            indices += [base, base + 1, base + 3, base, base + 3, base + 2]
        }
        let topCenterIdx = UInt32(positions.count)
        positions.append(SIMD3<Float>(0, 1, 0)); normals.append([0, 1, 0])
        let topRingStart = UInt32(positions.count)
        for i in 0..<segments {
            let angle = Float(i) / segF * 2.0 * Float.pi
            positions.append(SIMD3<Float>(cos(angle) * topRadius, 1.0, sin(angle) * topRadius))
            normals.append([0, 1, 0])
        }
        for i in 0..<UInt32(segments) {
            let next = (i + 1) % UInt32(segments)
            indices += [topCenterIdx, topRingStart + i, topRingStart + next]
        }
        let botCenterIdx = UInt32(positions.count)
        positions.append(SIMD3<Float>(0, 0, 0)); normals.append([0, -1, 0])
        let botRingStart = UInt32(positions.count)
        for i in 0..<segments {
            let angle = Float(i) / segF * 2.0 * Float.pi
            positions.append(SIMD3<Float>(cos(angle) * bottomRadius, 0.0, sin(angle) * bottomRadius))
            normals.append([0, -1, 0])
        }
        for i in 0..<UInt32(segments) {
            let next = (i + 1) % UInt32(segments)
            indices += [botCenterIdx, botRingStart + next, botRingStart + i]
        }
        var desc = MeshDescriptor(name: "cyl")
        desc.positions  = MeshBuffer(positions)
        desc.normals    = MeshBuffer(normals)
        desc.primitives = .triangles(indices)
        return try MeshResource.generate(from: [desc])
    }

    private func buildTorsoMesh(width w: Float, height h: Float, depth d: Float) throws -> MeshResource {
        let bow: Float = 0.018
        let hw = w * 0.5, hh = h * 0.5, hd = d * 0.5
        typealias V3 = SIMD3<Float>
        typealias Face = (verts: [V3], normal: V3)
        let faces: [Face] = [
            ([ [-hw, -hh, hd+bow], [hw, -hh, hd+bow], [hw, hh, hd+bow], [-hw, hh, hd+bow] ], [0, 0, 1]),
            ([ [hw, -hh, -hd], [-hw, -hh, -hd], [-hw, hh, -hd], [hw, hh, -hd] ], [0, 0, -1]),
            ([ [-hw, -hh, -hd], [-hw, -hh, hd+bow], [-hw, hh, hd+bow], [-hw, hh, -hd] ], [-1, 0, 0]),
            ([ [hw, -hh, hd+bow], [hw, -hh, -hd], [hw, hh, -hd], [hw, hh, hd+bow] ], [1, 0, 0]),
            ([ [-hw, hh, hd+bow], [hw, hh, hd+bow], [hw, hh, -hd], [-hw, hh, -hd] ], [0, 1, 0]),
            ([ [-hw, -hh, -hd], [hw, -hh, -hd], [hw, -hh, hd+bow], [-hw, -hh, hd+bow] ], [0, -1, 0]),
        ]
        var positions = [V3](); var normals = [V3](); var indices = [UInt32]()
        for face in faces {
            let base = UInt32(positions.count)
            positions.append(contentsOf: face.verts)
            normals.append(contentsOf: [V3](repeating: face.normal, count: 4))
            indices += [base, base+1, base+2, base, base+2, base+3]
        }
        var desc = MeshDescriptor(name: "torso")
        desc.positions  = MeshBuffer(positions)
        desc.normals    = MeshBuffer(normals)
        desc.primitives = .triangles(indices)
        return try MeshResource.generate(from: [desc])
    }

    private func buildProceduralJacket() {
        do {
            let mat = makeFabricMaterial()
            let torsoEnt = ModelEntity(mesh: try buildTorsoMesh(width: 0.46, height: 0.56, depth: 0.20),
                                       materials: [mat])
            let leftSleeveEnt  = ModelEntity(mesh: try buildTaperedCylinder(topRadius: 0.065, bottomRadius: 0.048),
                                             materials: [mat])
            let rightSleeveEnt = ModelEntity(mesh: try buildTaperedCylinder(topRadius: 0.065, bottomRadius: 0.048),
                                             materials: [mat])
            let collarEnt = ModelEntity(mesh: try buildTaperedCylinder(topRadius: 0.082, bottomRadius: 0.075, segments: 18),
                                        materials: [mat])
            let cuffMesh = try buildTaperedCylinder(topRadius: 0.042, bottomRadius: 0.040, segments: 12)
            let leftCuffEnt  = ModelEntity(mesh: cuffMesh, materials: [mat])
            let rightCuffEnt = ModelEntity(mesh: cuffMesh, materials: [mat])
            let root = AnchorEntity(world: simd_float4x4(1))
            for ent in [torsoEnt, leftSleeveEnt, rightSleeveEnt, collarEnt, leftCuffEnt, rightCuffEnt] {
                root.addChild(ent)
            }
            root.isEnabled = false
            arView.scene.addAnchor(root)
            procRig = ProcJacketRig(
                root: root, torso: torsoEnt,
                leftSleeve: leftSleeveEnt, rightSleeve: rightSleeveEnt,
                collar: collarEnt, leftCuff: leftCuffEnt, rightCuff: rightCuffEnt
            )
        } catch {
            print("[AR] Failed to build procedural jacket: \(error)")
            updateStatus("Mesh error — restart app")
        }
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        frameCount += 1
        guard frameCount % processEveryNthFrame == 0 else { return }
        for anchor in anchors {
            if let bodyAnchor = anchor as? ARBodyAnchor, mode == .body {
                updateForBody(bodyAnchor)
                return
            }
            if let faceAnchor = anchor as? ARFaceAnchor, mode == .selfie {
                updateForSelfie(faceAnchor)
                return
            }
        }
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for anchor in anchors {
            if anchor is ARBodyAnchor, mode == .body {
                setAnchorEnabled(true)
                updateStatus("Body detected — try it on!")
                return
            }
            if anchor is ARFaceAnchor, mode == .selfie {
                setAnchorEnabled(true)
                updateStatus("Locked on — turn your head to see the jacket")
                return
            }
        }
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for anchor in anchors {
            if anchor is ARBodyAnchor && mode == .body {
                setAnchorEnabled(false)
                updateStatus("Body lost — step back into view")
                return
            }
            if anchor is ARFaceAnchor && mode == .selfie {
                setAnchorEnabled(false)
                updateStatus("Face lost — face the camera again")
                return
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        updateStatus("AR error: \(error.localizedDescription)")
    }

    // MARK: - Selfie-mode jacket update (face-driven)
    //
    // Strategy: the face anchor gives us head position + orientation in world
    // space. We place the jacket directly below the face. Shoulder width is
    // approximated from face size (the face is ~ 1/8 of body height, shoulders
    // are ~ 2.2x face width). The jacket inherits face rotation (yaw), so
    // when you turn your head the jacket rotates with the upper body.

    private func updateForSelfie(_ face: ARFaceAnchor) {
        let faceTransform = face.transform

        // Drop straight down from face origin by ~0.18m (head→sternum)
        // and back along the face's local +Z (away from screen) by ~0.05m.
        let downOffset = SIMD3<Float>(0, -0.20, 0)
        let translatedTransform = simd_mul(
            faceTransform,
            float4x4(translation: downOffset)
        )

        // Smooth the world transform to avoid jitter.
        let t: Float = selfieLerp
        if let prev = selfieSmoothTransform {
            selfieSmoothTransform = mix4x4(prev, translatedTransform, t: t)
        } else {
            selfieSmoothTransform = translatedTransform
        }
        let finalTransform = selfieSmoothTransform!

        if hasUSDZ, let root = usdzAnchor, let jacket = usdzJacket {
            root.transform = Transform(matrix: finalTransform)
            // Scale USDZ to match expected real-world torso height (~0.55m).
            // The jacket's natural size depends on how it was authored, so we
            // measure its bounding box once and rescale to a fixed height.
            let bounds = jacket.visualBounds(relativeTo: nil)
            let heightNow = bounds.extents.y
            if heightNow > 0.001 {
                let targetH: Float = 0.55
                let s = targetH / heightNow
                // Only rescale if currently way off — avoids fighting authored scale every frame.
                if abs(jacket.scale.x - s) > 0.05 {
                    jacket.scale = SIMD3<Float>(repeating: s)
                }
            }
        } else if let rig = procRig {
            // Place the procedural rig at the offset point.
            rig.root.transform = Transform(matrix: finalTransform)
            // In selfie mode we DON'T have arm joints, so sleeves hang straight
            // down at fixed shoulder offsets. They won't follow arm motion —
            // accept this; it's the documented limit.
            let mat = makeFabricMaterial()
            _ = mat   // (silence unused warning if compiler complains; material already on entities)

            let shoulderHalfW: Float = 0.18
            let sleeveLen: Float = 0.55

            // Reset transforms each frame (root rotation handles head yaw)
            rig.torso.position = SIMD3<Float>(0, -0.10, 0)
            rig.torso.orientation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
            rig.torso.scale = SIMD3<Float>(1.0, 1.0, 1.0)

            rig.collar.position = SIMD3<Float>(0, 0.16, 0.01)
            rig.collar.orientation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
            rig.collar.scale = SIMD3<Float>(1.0, 0.06, 1.0)

            // Left sleeve hanging down from left shoulder
            rig.leftSleeve.position = SIMD3<Float>(-shoulderHalfW, 0.10, 0)
            rig.leftSleeve.orientation = simd_quatf(angle: .pi, axis: [0, 0, 1])  // point down
            rig.leftSleeve.scale = SIMD3<Float>(1.0, sleeveLen, 1.0)

            rig.rightSleeve.position = SIMD3<Float>(shoulderHalfW, 0.10, 0)
            rig.rightSleeve.orientation = simd_quatf(angle: .pi, axis: [0, 0, 1])
            rig.rightSleeve.scale = SIMD3<Float>(1.0, sleeveLen, 1.0)

            rig.leftCuff.position  = SIMD3<Float>(-shoulderHalfW, 0.10 - sleeveLen, 0)
            rig.leftCuff.orientation = rig.leftSleeve.orientation
            rig.leftCuff.scale = SIMD3<Float>(1.0, 0.03, 1.0)

            rig.rightCuff.position = SIMD3<Float>(shoulderHalfW, 0.10 - sleeveLen, 0)
            rig.rightCuff.orientation = rig.rightSleeve.orientation
            rig.rightCuff.scale = SIMD3<Float>(1.0, 0.03, 1.0)
        }
    }

    // MARK: - Body-mode jacket update (skeleton-driven)

    private func updateForBody(_ bodyAnchor: ARBodyAnchor) {
        if hasUSDZ {
            updateUSDZForBody(bodyAnchor)
        } else {
            updateProceduralForBody(bodyAnchor)
        }
    }

    /// USDZ path: root anchored to skeleton root. If the USDZ has bones whose
    /// names match ARSkeleton3D joints, we drive those bones directly. If not,
    /// we just place the whole entity at the skeleton root and rely on the
    /// jacket's authored proportions.
    private func updateUSDZForBody(_ bodyAnchor: ARBodyAnchor) {
        guard let root = usdzAnchor, let jacket = usdzJacket else { return }
        root.transform = Transform(matrix: bodyAnchor.transform)

        // Auto-fit scale based on measured shoulder width.
        let sk = bodyAnchor.skeleton
        if let lShT = sk.modelTransform(for: JointNames.leftShoulder),
           let rShT = sk.modelTransform(for: JointNames.rightShoulder) {
            let lSh = SIMD3<Float>(lShT.columns.3.x, lShT.columns.3.y, lShT.columns.3.z)
            let rSh = SIMD3<Float>(rShT.columns.3.x, rShT.columns.3.y, rShT.columns.3.z)
            let shoulderWidth = simd_distance(lSh, rSh)
            // Authored USDZ shoulder width assumption: 0.40m. Rescale.
            let bounds = jacket.visualBounds(relativeTo: nil)
            let widthNow = bounds.extents.x
            if widthNow > 0.001 {
                let s = shoulderWidth / widthNow * 1.1
                if abs(jacket.scale.x - s) > 0.05 {
                    jacket.scale = SIMD3<Float>(repeating: s)
                }
            }
            // If the USDZ exposes bones by ARKit joint name, drive them directly.
            // Sketchfab/MakeHuman rigs often don't match ARKit names, so this is
            // best-effort. When names don't match, the jacket still follows the
            // body anchor as a single rigid entity.
            driveBonesIfPresent(jacket, sk: sk)
        }
    }

    private func driveBonesIfPresent(_ entity: Entity, sk: ARSkeleton3D) {
        // Walk the entity tree, look for ModelEntity with a jointNames property.
        // RealityKit's skinned-mesh API exposes joints on ModelComponent.
        // For each joint in the skeleton, if there's a matching bone, override its transform.
        for child in entity.children {
            if let model = child as? ModelEntity, let _ = model.jointNames.first {
                applySkeletonToJoints(model, sk: sk)
            }
            driveBonesIfPresent(child, sk: sk)
        }
    }

    private func applySkeletonToJoints(_ model: ModelEntity, sk: ARSkeleton3D) {
        let names = model.jointNames
        var newTransforms = model.jointTransforms
        for (i, name) in names.enumerated() {
            // ARKit joint names use underscores: "left_arm_joint", "spine_7_joint", etc.
            // USDZ joint names from common rigs (mixamo, makehuman) use different conventions.
            // We try the name verbatim first, then a few heuristics.
            let candidates = Self.armatureCandidates(for: name)
            for cand in candidates {
                if let t = sk.modelTransform(for: ARSkeleton.JointName(rawValue: cand)) {
                    newTransforms[i] = Transform(matrix: t)
                    break
                }
            }
        }
        model.jointTransforms = newTransforms
    }

    /// Maps a USDZ bone name to plausible ARSkeleton3D joint names.
    /// Add more mappings here as you bring in new rigs.
    private static func armatureCandidates(for boneName: String) -> [String] {
        let lower = boneName.lowercased()
        var out = [boneName]
        if lower.contains("spine") || lower == "chest"  { out.append("spine_7_joint") }
        if lower.contains("neck")                       { out.append("neck_1_joint") }
        if lower.contains("leftshoulder") || lower == "l_shoulder" { out.append("left_shoulder_1_joint") }
        if lower.contains("rightshoulder") || lower == "r_shoulder" { out.append("right_shoulder_1_joint") }
        if lower.contains("leftarm") || lower == "l_upperarm" { out.append("left_arm_joint") }
        if lower.contains("rightarm") || lower == "r_upperarm" { out.append("right_arm_joint") }
        if lower.contains("leftforearm")  { out.append("left_foreArm_joint") }
        if lower.contains("rightforearm") { out.append("right_foreArm_joint") }
        return out
    }

    /// Procedural path (existing logic, unchanged).
    private func updateProceduralForBody(_ bodyAnchor: ARBodyAnchor) {
        guard let rig = procRig else { return }
        let sk = bodyAnchor.skeleton

        guard
            let t_spine7      = sk.modelTransform(for: JointNames.spine7),
            let t_neck        = sk.modelTransform(for: JointNames.neck1),
            let t_lShoulder   = sk.modelTransform(for: JointNames.leftShoulder),
            let t_rShoulder   = sk.modelTransform(for: JointNames.rightShoulder),
            let t_lForeArm    = sk.modelTransform(for: JointNames.leftForeArm),
            let t_rForeArm    = sk.modelTransform(for: JointNames.rightForeArm),
            let t_lHip        = sk.modelTransform(for: JointNames.leftHip),
            let t_rHip        = sk.modelTransform(for: JointNames.rightHip)
        else { return }

        func pos(_ m: simd_float4x4) -> SIMD3<Float> {
            SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        }

        rig.root.transform = Transform(matrix: bodyAnchor.transform)

        let lShPos = pos(t_lShoulder), rShPos = pos(t_rShoulder)
        let hipCenter = (pos(t_lHip) + pos(t_rHip)) * 0.5
        let shCenter  = (lShPos + rShPos) * 0.5
        let shoulderWidth = simd_distance(lShPos, rShPos)
        let torsoHeight   = simd_distance(shCenter, hipCenter)
        let torsoCenter   = (shCenter + hipCenter) * 0.5 + SIMD3<Float>(0, 0, 0.025)
        let xAxis = normalize(rShPos - lShPos)
        let yAxis = normalize(shCenter - hipCenter)
        let zAxis = normalize(cross(xAxis, yAxis))
        let yAxisOrtho = normalize(cross(zAxis, xAxis))
        let rotMat = simd_float3x3(columns: (xAxis, yAxisOrtho, zAxis))
        let torsoQuat = simd_quatf(rotMat)
        let torsoScale = SIMD3<Float>(
            max(shoulderWidth / 0.46, 0.5),
            max(torsoHeight   / 0.56, 0.5),
            1.0
        )
        applySmooth(rig.torso, idx: 0,
                    targetPos: torsoCenter, targetRot: torsoQuat, targetScl: torsoScale)
        placeSleeve(rig.leftSleeve,  idx: 1, from: lShPos, to: pos(t_lForeArm))
        placeSleeve(rig.rightSleeve, idx: 2, from: rShPos, to: pos(t_rForeArm))
        let neckPos  = pos(t_neck)
        let spineDir = normalize(pos(t_spine7) - hipCenter)
        let collarQ  = quaternionFromTo(from: [0, 1, 0], to: spineDir)
        applySmooth(rig.collar, idx: 3,
                    targetPos: neckPos + SIMD3<Float>(0, 0.01, 0.01),
                    targetRot: collarQ,
                    targetScl: SIMD3<Float>(1.0, 0.048, 1.0))
        let lWrist = pos(t_lForeArm), rWrist = pos(t_rForeArm)
        rig.leftCuff.position    = lWrist
        rig.leftCuff.orientation = rig.leftSleeve.orientation
        rig.leftCuff.scale       = SIMD3<Float>(1.0, 0.03, 1.0)
        rig.rightCuff.position   = rWrist
        rig.rightCuff.orientation = rig.rightSleeve.orientation
        rig.rightCuff.scale      = SIMD3<Float>(1.0, 0.03, 1.0)
    }

    private func placeSleeve(_ entity: ModelEntity, idx: Int,
                             from start: SIMD3<Float>, to end: SIMD3<Float>) {
        let delta = end - start
        let length = simd_length(delta)
        guard length > 0.01 else { return }
        let dir = normalize(delta)
        let rot = quaternionFromTo(from: [0, 1, 0], to: dir)
        let scl = SIMD3<Float>(1.0, length, 1.0)
        applySmooth(entity, idx: idx, targetPos: start, targetRot: rot, targetScl: scl)
    }

    private func applySmooth(_ entity: ModelEntity, idx: Int,
                             targetPos: SIMD3<Float>, targetRot: simd_quatf,
                             targetScl: SIMD3<Float>) {
        let velocity = simd_distance(targetPos, prevPos[idx])
        let t: Float = velocity > velocityThreshold ? fastLerp : baseLerp
        prevPos[idx] = targetPos
        let tv = SIMD3<Float>(repeating: t)
        smoothPos[idx] = mix(smoothPos[idx], targetPos, t: tv)
        smoothRot[idx] = simd_slerp(smoothRot[idx], targetRot, t)
        smoothScl[idx] = mix(smoothScl[idx], targetScl, t: tv)
        entity.position    = smoothPos[idx]
        entity.orientation = smoothRot[idx]
        entity.scale       = smoothScl[idx]
    }

    private func quaternionFromTo(from a: SIMD3<Float>, to b: SIMD3<Float>) -> simd_quatf {
        let d = dot(a, b)
        if d > 0.9999 { return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) }
        if d < -0.9999 {
            var perp = cross(a, SIMD3<Float>(1, 0, 0))
            if simd_length(perp) < 0.001 { perp = cross(a, SIMD3<Float>(0, 1, 0)) }
            return simd_quatf(angle: .pi, axis: normalize(perp))
        }
        let axis = normalize(cross(a, b))
        let angle = acos(min(max(d, -1), 1))
        return simd_quatf(angle: angle, axis: axis)
    }
}

// MARK: - simd helpers

private extension float4x4 {
    init(translation t: SIMD3<Float>) {
        self.init(
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(t.x, t.y, t.z, 1)
        )
    }
}

/// Element-wise blend of two 4x4 transforms. Not geometrically pure (doesn't
/// re-orthonormalize rotation), but plenty good for the small per-frame
/// deltas we use for selfie-mode smoothing.
private func mix4x4(_ a: simd_float4x4, _ b: simd_float4x4, t: Float) -> simd_float4x4 {
    var out = simd_float4x4()
    out.columns.0 = mix(a.columns.0, b.columns.0, t: SIMD4<Float>(repeating: t))
    out.columns.1 = mix(a.columns.1, b.columns.1, t: SIMD4<Float>(repeating: t))
    out.columns.2 = mix(a.columns.2, b.columns.2, t: SIMD4<Float>(repeating: t))
    out.columns.3 = mix(a.columns.3, b.columns.3, t: SIMD4<Float>(repeating: t))
    return out
}

// MARK: - Padded UILabel

private class PaddedLabel: UILabel {
    var insets = UIEdgeInsets(top: 6, left: 18, bottom: 6, right: 18)
    override func drawText(in rect: CGRect) {
        super.drawText(in: rect.inset(by: insets))
    }
    override var intrinsicContentSize: CGSize {
        let size = super.intrinsicContentSize
        return CGSize(width: size.width + insets.left + insets.right,
                      height: size.height + insets.top + insets.bottom)
    }
}
