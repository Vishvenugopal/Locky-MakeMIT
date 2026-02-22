import ARKit
import Foundation
import simd

final class LiDARStreamer: NSObject, ARSessionDelegate {
    private let arSession = ARSession()
    private let wsSession = URLSession(configuration: .default)

    private var socketTask: URLSessionWebSocketTask?
    private var activeSettings: PiConnectionSettings?
    private var frameSeq: Int = 0
    private var lastSendTime: TimeInterval = 0

    var onStatus: ((String) -> Void)?

    private(set) var isStreaming: Bool = false

    override init() {
        super.init()
        arSession.delegate = self
    }

    func startStreaming(settings: PiConnectionSettings) {
        if isStreaming {
            if activeSettings == settings {
                return
            }
            stopStreaming()
        }

        activeSettings = settings
        connectWebSocket(settings: settings)

        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }

        if config.frameSemantics.isEmpty {
            onStatus?("LiDAR not supported on this device")
            socketTask?.cancel(with: .goingAway, reason: nil)
            socketTask = nil
            activeSettings = nil
            return
        }

        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
        isStreaming = true
        onStatus?("LiDAR stream started")
    }

    func stopStreaming() {
        guard isStreaming else { return }

        isStreaming = false
        arSession.pause()

        socketTask?.cancel(with: .goingAway, reason: nil)
        socketTask = nil
        activeSettings = nil

        onStatus?("LiDAR stream stopped")
    }

    private func connectWebSocket(settings: PiConnectionSettings) {
        let url = settings.websocketURLWithToken()
        let task = wsSession.webSocketTask(with: url)
        task.resume()
        socketTask = task
        receiveLoop()
        onStatus?("Connecting WS: \(url.absoluteString)")
    }

    private func receiveLoop() {
        socketTask?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                self.receiveLoop()
            case .failure(let error):
                self.onStatus?("WS receive error: \(error.localizedDescription)")
            }
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isStreaming else { return }

        let now = CFAbsoluteTimeGetCurrent()
        let minInterval = 1.0 / max(1.0, AppConfig.lidarSendFPS)
        if (now - lastSendTime) < minInterval {
            return
        }

        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else {
            return
        }

        do {
            let packet = try makeBinaryPacket(frame: frame, depthData: depthData)
            socketTask?.send(.data(packet)) { [weak self] error in
                if let error {
                    self?.onStatus?("WS send error: \(error.localizedDescription)")
                }
            }
            frameSeq += 1
            lastSendTime = now
        } catch {
            onStatus?("LiDAR encode error: \(error.localizedDescription)")
        }
    }

    private func makeBinaryPacket(frame: ARFrame, depthData: ARDepthData) throws -> Data {
        let (depthRaw, width, height) = try depthMillimeters(from: depthData.depthMap)
        let confidenceRaw = confidenceBytes(from: depthData.confidenceMap)

        let camera = frame.camera
        let intr = camera.intrinsics
        let transform = camera.transform
        let translation = transform.columns.3
        let quat = simd_quatf(transform)

        let pose = PoseJSON(
            tx: translation.x,
            ty: translation.y,
            tz: translation.z,
            qx: quat.vector.x,
            qy: quat.vector.y,
            qz: quat.vector.z,
            qw: quat.vector.w
        )

        let intrinsics = IntrinsicsJSON(
            fx: intr[0, 0],
            fy: intr[1, 1],
            cx: intr[2, 0],
            cy: intr[2, 1]
        )

        var header: [String: Any] = [
            "type": "lidar_frame",
            "seq": frameSeq,
            "timestamp_ms": Int64(frame.timestamp * 1000.0),
            "width": width,
            "height": height,
            "depth_encoding": "u16_mm_raw",
            "depth_byte_count": depthRaw.count,
            "pose": pose.asDictionary(),
            "intrinsics": intrinsics.asDictionary(),
        ]

        if let confidenceRaw {
            header["confidence_encoding"] = "u8_raw"
            header["confidence_byte_count"] = confidenceRaw.count
        }

        let headerData = try JSONSerialization.data(withJSONObject: header, options: [])

        var packet = Data()
        var headerLen = UInt32(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLen) { raw in
            packet.append(raw.bindMemory(to: UInt8.self))
        }

        packet.append(headerData)
        packet.append(depthRaw)
        if let confidenceRaw {
            packet.append(confidenceRaw)
        }

        return packet
    }

    private func depthMillimeters(from pixelBuffer: CVPixelBuffer) throws -> (Data, Int, Int) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)

        guard format == kCVPixelFormatType_DepthFloat32 else {
            throw NSError(
                domain: "LiDARStreamer",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported depth pixel format: \(format)"]
            )
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NSError(
                domain: "LiDARStreamer",
                code: -3,
                userInfo: [NSLocalizedDescriptionKey: "Depth buffer base address unavailable"]
            )
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let rowStride = bytesPerRow / MemoryLayout<Float32>.size

        var out = Data(count: width * height * MemoryLayout<UInt16>.size)
        out.withUnsafeMutableBytes { outBytes in
            let outU16 = outBytes.bindMemory(to: UInt16.self)
            let input = base.assumingMemoryBound(to: Float32.self)

            for y in 0..<height {
                let row = input.advanced(by: y * rowStride)
                for x in 0..<width {
                    let meters = max(0.0, row[x])
                    let mm = min(65535.0, meters * 1000.0)
                    outU16[y * width + x] = UInt16(mm)
                }
            }
        }

        return (out, width, height)
    }

    private func confidenceBytes(from pixelBuffer: CVPixelBuffer?) -> Data? {
        guard let pixelBuffer else { return nil }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)

        guard format == kCVPixelFormatType_OneComponent8 else {
            return nil
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let source = base.assumingMemoryBound(to: UInt8.self)

        var out = Data(count: width * height)
        out.withUnsafeMutableBytes { outBytes in
            guard let outPtr = outBytes.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return
            }
            for y in 0..<height {
                let srcRow = source.advanced(by: y * bytesPerRow)
                let dstRow = outPtr.advanced(by: y * width)
                dstRow.assign(from: srcRow, count: width)
            }
        }

        return out
    }
}
