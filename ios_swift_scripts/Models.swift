import Foundation

struct ServerUIState: Decodable {
    let showInitialScanButton: Bool
    let showScanHideButtons: Bool
    let canStartScan: Bool
    let canHide: Bool
    let allowHideDuringMapping: Bool

    enum CodingKeys: String, CodingKey {
        case showInitialScanButton = "show_initial_scan_button"
        case showScanHideButtons = "show_scan_hide_buttons"
        case canStartScan = "can_start_scan"
        case canHide = "can_hide"
        case allowHideDuringMapping = "allow_hide_during_mapping"
    }
}

struct LidarState: Decodable {
    let totalFrames: Int
    let droppedFrames: Int
    let lastSeq: Int
    let lastTimestampMs: Int
    let secondsSinceLastFrame: Double?

    enum CodingKeys: String, CodingKey {
        case totalFrames = "total_frames"
        case droppedFrames = "dropped_frames"
        case lastSeq = "last_seq"
        case lastTimestampMs = "last_timestamp_ms"
        case secondsSinceLastFrame = "seconds_since_last_frame"
    }
}

struct ServerState: Decodable {
    let phase: String
    let mappingState: String
    let statusText: String
    let returnedToOrigin: Bool
    let ui: ServerUIState
    let lidar: LidarState
    let robotLink: String

    enum CodingKeys: String, CodingKey {
        case phase
        case mappingState = "mapping_state"
        case statusText = "status_text"
        case returnedToOrigin = "returned_to_origin"
        case ui
        case lidar
        case robotLink = "robot_link"
    }
}

struct ActionResult: Decodable {
    let ok: Bool
    let message: String
    let state: ServerState?
}

struct PoseJSON {
    let tx: Float
    let ty: Float
    let tz: Float
    let qx: Float
    let qy: Float
    let qz: Float
    let qw: Float

    func asDictionary() -> [String: Any] {
        [
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
        ]
    }
}

struct IntrinsicsJSON {
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float

    func asDictionary() -> [String: Any] {
        [
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        ]
    }
}
