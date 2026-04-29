import UIKit
import RealityKit
import ARKit

// MARK: - Jacket Rig

private struct JacketRig {
    let root:         AnchorEntity
    let torso:        ModelEntity
    let leftSleeve:   ModelEntity
    let rightSleeve:  ModelEntity
    let collar:       ModelEntity
    let leftCuff:     ModelEntity
    let rightCuff:    ModelEntity
}

// MARK: - Joint Names

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

// MARK: - View Controller

class ARBodyTryOnViewController: UIViewController, ARSessionDelegate {

    // MARK: - Properties

    private var arView: ARView!
    private var statusLabel: UILabel!
    private var titleLabel: UILabel!
    var onDismiss: (() -> Void)?

    // Jacket rig
    private var jacketRig: JacketRig?

    // Per-part smoothing state — indices: 0=torso, 1=leftSleeve, 2=rightSleeve, 3=collar
    private var smoothPos = [SIMD3<Float>](repeating: .zero, count: 4)
    private var smoothRot = [simd_quatf](repeating: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), count: 4)
    private var smoothScl = [SIMD3<Float>](repeating: .one, count: 4)
    private var prevPos   = [SIMD3<Float>](repeating: .zero, count: 4)

    private let baseLerp: Float = 0.20
    private let fastLerp: Float = 0.75
    private let velocityThreshold: Float = 0.025

    // Frame throttling
    private var frameCount: UInt64 = 0
    private let processEveryNthFrame: UInt64 = 2

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        setupARView()
        setupUI()
        buildProceduralJacket()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        startBodyTracking()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        arView?.session.pause()
    }

    deinit {
        arView?.session.pause()
        if let rig = jacketRig {
            arView?.scene.removeAnchor(rig.root)
        }
    }

    override var prefersStatusBarHidden: Bool { true }

    // MARK: - AR Setup

    private func setupARView() {
        arView = ARView(frame: view.bounds)
        arView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        arView.renderOptions = [.disableMotionBlur, .disableDepthOfField]
        view.addSubview(arView)
        arView.session.delegate = self
    }

    private func startBodyTracking() {
        guard ARBodyTrackingConfiguration.isSupported else {
            updateStatus("Body tracking not supported on this device")
            return
        }
        let config = ARBodyTrackingConfiguration()
        config.automaticSkeletonScaleEstimationEnabled = true
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        updateStatus("Stand back — show full body to camera")
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

        statusLabel = PaddedLabel()
        statusLabel.text = "Initializing..."
        statusLabel.textColor = .white
        statusLabel.font = .systemFont(ofSize: 14, weight: .medium)
        statusLabel.textAlignment = .center
        statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        statusLabel.layer.cornerRadius = 20
        statusLabel.layer.masksToBounds = true
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(statusLabel)

        NSLayoutConstraint.activate([
            closeBtn.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            closeBtn.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            closeBtn.widthAnchor.constraint(equalToConstant: 44),
            closeBtn.heightAnchor.constraint(equalToConstant: 44),

            titleLabel.centerYAnchor.constraint(equalTo: closeBtn.centerYAnchor),
            titleLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),

            statusLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),
            statusLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            statusLabel.heightAnchor.constraint(equalToConstant: 40),
            statusLabel.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 40),
            statusLabel.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -40),
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
        // Dark navy jacket color
        mat.baseColor = .init(tint: UIColor(red: 0.08, green: 0.12, blue: 0.22, alpha: 1.0))
        mat.roughness = .init(floatLiteral: 0.82)
        mat.metallic  = .init(floatLiteral: 0.0)
        return mat
    }

    // MARK: - Procedural Mesh Builders

    /// Builds a tapered cylinder along local +Y axis from y=0 (bottom) to y=1 (top).
    /// topRadius: wider end (shoulder side), bottomRadius: narrower end (wrist side).
    private func buildTaperedCylinder(
        topRadius: Float,
        bottomRadius: Float,
        segments: Int = 14
    ) throws -> MeshResource {
        var positions = [SIMD3<Float>]()
        var normals   = [SIMD3<Float>]()
        var indices   = [UInt32]()

        let segF = Float(segments)

        // Side rings
        for i in 0...segments {
            let angle = Float(i) / segF * 2.0 * Float.pi
            let c = cos(angle), s = sin(angle)

            let topP = SIMD3<Float>(c * topRadius,    1.0, s * topRadius)
            let botP = SIMD3<Float>(c * bottomRadius, 0.0, s * bottomRadius)
            let nrm  = normalize(SIMD3<Float>(c, 0, s))

            positions.append(topP); normals.append(nrm)
            positions.append(botP); normals.append(nrm)
        }

        // Side triangles
        for i in 0..<UInt32(segments) {
            let base = i * 2
            indices += [base, base + 1, base + 3,
                        base, base + 3, base + 2]
        }

        // Top cap (y=1, topRadius)
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

        // Bottom cap (y=0, bottomRadius)
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

        var descriptor = MeshDescriptor(name: "cylinder")
        descriptor.positions  = MeshBuffer(positions)
        descriptor.normals    = MeshBuffer(normals)
        descriptor.primitives = .triangles(indices)
        return try MeshResource.generate(from: [descriptor])
    }

    /// Builds a torso box: slightly bowed front face, flat back, with face normals.
    private func buildTorsoMesh(width w: Float, height h: Float, depth d: Float) throws -> MeshResource {
        let bow: Float = 0.018
        let hw = w * 0.5, hh = h * 0.5, hd = d * 0.5

        // 6 faces × 4 vertices each = 24 verts (each face has independent normals)
        typealias V3 = SIMD3<Float>
        typealias Face = (verts: [V3], normal: V3)

        let faces: [Face] = [
            // Front (bowed outward)
            ([ [-hw, -hh, hd+bow], [hw, -hh, hd+bow], [hw, hh, hd+bow], [-hw, hh, hd+bow] ], [0, 0, 1]),
            // Back
            ([ [hw, -hh, -hd], [-hw, -hh, -hd], [-hw, hh, -hd], [hw, hh, -hd] ], [0, 0, -1]),
            // Left
            ([ [-hw, -hh, -hd], [-hw, -hh, hd+bow], [-hw, hh, hd+bow], [-hw, hh, -hd] ], [-1, 0, 0]),
            // Right
            ([ [hw, -hh, hd+bow], [hw, -hh, -hd], [hw, hh, -hd], [hw, hh, hd+bow] ], [1, 0, 0]),
            // Top
            ([ [-hw, hh, hd+bow], [hw, hh, hd+bow], [hw, hh, -hd], [-hw, hh, -hd] ], [0, 1, 0]),
            // Bottom
            ([ [-hw, -hh, -hd], [hw, -hh, -hd], [hw, -hh, hd+bow], [-hw, -hh, hd+bow] ], [0, -1, 0]),
        ]

        var positions = [V3](); var normals = [V3](); var indices = [UInt32]()
        for face in faces {
            let base = UInt32(positions.count)
            positions.append(contentsOf: face.verts)
            normals.append(contentsOf: [V3](repeating: face.normal, count: 4))
            indices += [base, base+1, base+2, base, base+2, base+3]
        }

        var descriptor = MeshDescriptor(name: "torso")
        descriptor.positions  = MeshBuffer(positions)
        descriptor.normals    = MeshBuffer(normals)
        descriptor.primitives = .triangles(indices)
        return try MeshResource.generate(from: [descriptor])
    }

    // MARK: - Build Procedural Jacket

    private func buildProceduralJacket() {
        do {
            let mat = makeFabricMaterial()

            // Torso — canonical 46cm wide × 56cm tall × 20cm deep, scaled per-frame
            let torsoEnt = ModelEntity(
                mesh: try buildTorsoMesh(width: 0.46, height: 0.56, depth: 0.20),
                materials: [mat]
            )

            // Sleeves — unit-length tapered cylinders, scaled per-frame
            let leftSleeveEnt = ModelEntity(
                mesh: try buildTaperedCylinder(topRadius: 0.065, bottomRadius: 0.048),
                materials: [mat]
            )
            let rightSleeveEnt = ModelEntity(
                mesh: try buildTaperedCylinder(topRadius: 0.065, bottomRadius: 0.048),
                materials: [mat]
            )

            // Collar — short wide ring
            let collarEnt = ModelEntity(
                mesh: try buildTaperedCylinder(topRadius: 0.082, bottomRadius: 0.075, segments: 18),
                materials: [mat]
            )

            // Cuffs — thin rings at wrist
            let cuffMesh = try buildTaperedCylinder(topRadius: 0.042, bottomRadius: 0.040, segments: 12)
            let leftCuffEnt  = ModelEntity(mesh: cuffMesh, materials: [mat])
            let rightCuffEnt = ModelEntity(mesh: cuffMesh, materials: [mat])

            // Root anchor — follows bodyAnchor.transform each frame
            let root = AnchorEntity(world: simd_float4x4(1))
            for ent in [torsoEnt, leftSleeveEnt, rightSleeveEnt, collarEnt, leftCuffEnt, rightCuffEnt] {
                root.addChild(ent)
            }
            arView.scene.addAnchor(root)

            jacketRig = JacketRig(
                root:         root,
                torso:        torsoEnt,
                leftSleeve:   leftSleeveEnt,
                rightSleeve:  rightSleeveEnt,
                collar:       collarEnt,
                leftCuff:     leftCuffEnt,
                rightCuff:    rightCuffEnt
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
            guard let bodyAnchor = anchor as? ARBodyAnchor else { continue }
            updateJacket(with: bodyAnchor)
            break
        }
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for anchor in anchors where anchor is ARBodyAnchor {
            jacketRig?.root.isEnabled = true
            updateStatus("Body detected — try on your jacket!")
            break
        }
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for anchor in anchors where anchor is ARBodyAnchor {
            jacketRig?.root.isEnabled = false
            updateStatus("Body lost — step back into view")
            break
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        updateStatus("AR error: \(error.localizedDescription)")
    }

    // MARK: - Jacket Update

    private func updateJacket(with bodyAnchor: ARBodyAnchor) {
        guard let rig = jacketRig else { return }
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

        // 1. Root anchor tracks body anchor world transform
        rig.root.transform = Transform(matrix: bodyAnchor.transform)

        // 2. Torso
        let lShPos = pos(t_lShoulder)
        let rShPos = pos(t_rShoulder)
        let hipCenter = (pos(t_lHip) + pos(t_rHip)) * 0.5
        let shCenter  = (lShPos + rShPos) * 0.5

        let shoulderWidth = simd_distance(lShPos, rShPos)
        let torsoHeight   = simd_distance(shCenter, hipCenter)
        let torsoCenter   = (shCenter + hipCenter) * 0.5 + SIMD3<Float>(0, 0, 0.025)

        // Build rotation from skeleton axes
        let xAxis = normalize(rShPos - lShPos)
        let yAxis = normalize(shCenter - hipCenter)
        // Ensure orthogonality
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
                    targetPos: torsoCenter,
                    targetRot: torsoQuat,
                    targetScl: torsoScale)

        // 3. Left sleeve: shoulder → forearm (wrist end)
        placeSleeve(rig.leftSleeve,  idx: 1, from: lShPos, to: pos(t_lForeArm))

        // 4. Right sleeve: shoulder → forearm
        placeSleeve(rig.rightSleeve, idx: 2, from: rShPos, to: pos(t_rForeArm))

        // 5. Collar at neck
        let neckPos  = pos(t_neck)
        let spineDir = normalize(pos(t_spine7) - hipCenter)
        let collarQ  = quaternionFromTo(from: [0, 1, 0], to: spineDir)
        applySmooth(rig.collar, idx: 3,
                    targetPos: neckPos + SIMD3<Float>(0, 0.01, 0.01),
                    targetRot: collarQ,
                    targetScl: SIMD3<Float>(1.0, 0.048, 1.0))

        // 6. Cuffs — follow wrist positions directly from sleeve end
        let lWrist = pos(t_lForeArm)
        let rWrist = pos(t_rForeArm)
        rig.leftCuff.position    = lWrist
        rig.leftCuff.orientation = rig.leftSleeve.orientation
        rig.leftCuff.scale       = SIMD3<Float>(1.0, 0.03, 1.0)
        rig.rightCuff.position   = rWrist
        rig.rightCuff.orientation = rig.rightSleeve.orientation
        rig.rightCuff.scale      = SIMD3<Float>(1.0, 0.03, 1.0)
    }

    // MARK: - Sleeve Placement

    private func placeSleeve(
        _ entity: ModelEntity,
        idx: Int,
        from start: SIMD3<Float>,
        to end: SIMD3<Float>
    ) {
        let delta  = end - start
        let length = simd_length(delta)
        guard length > 0.01 else { return }

        // Rotate +Y → sleeve direction
        let dir = normalize(delta)
        let rot = quaternionFromTo(from: [0, 1, 0], to: dir)

        // Place entity at start point (mesh goes from y=0 to y=length after scale)
        let scl = SIMD3<Float>(1.0, length, 1.0)

        applySmooth(entity, idx: idx,
                    targetPos: start,
                    targetRot: rot,
                    targetScl: scl)
    }

    // MARK: - Smooth Helper

    private func applySmooth(
        _ entity: ModelEntity,
        idx: Int,
        targetPos: SIMD3<Float>,
        targetRot: simd_quatf,
        targetScl: SIMD3<Float>
    ) {
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

    // MARK: - Math Utility

    /// Stable quaternion from one unit vector to another, guarding against parallel/anti-parallel.
    private func quaternionFromTo(from a: SIMD3<Float>, to b: SIMD3<Float>) -> simd_quatf {
        let d = dot(a, b)
        if d > 0.9999 { return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) }
        if d < -0.9999 {
            // 180-degree flip — pick an arbitrary perpendicular axis
            var perp = cross(a, SIMD3<Float>(1, 0, 0))
            if simd_length(perp) < 0.001 { perp = cross(a, SIMD3<Float>(0, 1, 0)) }
            return simd_quatf(angle: .pi, axis: normalize(perp))
        }
        let axis = normalize(cross(a, b))
        let angle = acos(min(max(d, -1), 1))
        return simd_quatf(angle: angle, axis: axis)
    }
}

// MARK: - Padded UILabel

private class PaddedLabel: UILabel {
    var insets = UIEdgeInsets(top: 4, left: 16, bottom: 4, right: 16)
    override func drawText(in rect: CGRect) {
        super.drawText(in: rect.inset(by: insets))
    }
    override var intrinsicContentSize: CGSize {
        let size = super.intrinsicContentSize
        return CGSize(width: size.width + insets.left + insets.right,
                      height: size.height + insets.top + insets.bottom)
    }
}
