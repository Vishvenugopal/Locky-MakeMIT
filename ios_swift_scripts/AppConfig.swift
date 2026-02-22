import Foundation

enum AppConfig {
    // Update these values before building in Xcode.
    static let piHost = "192.168.1.50"
    static let piPort = 8765
    static let apiToken = "1234"

    static let statePollInterval: TimeInterval = 0.5
    static let lidarSendFPS: Double = 10.0

    static var baseHTTPURL: URL {
        URL(string: "http://\(piHost):\(piPort)")!
    }

    static var baseWebSocketURL: URL {
        URL(string: "ws://\(piHost):\(piPort)/stream/lidar")!
    }

    static func websocketURLWithToken() -> URL {
        guard !apiToken.isEmpty else {
            return baseWebSocketURL
        }

        var components = URLComponents(url: baseWebSocketURL, resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "token", value: apiToken),
        ]
        return components.url ?? baseWebSocketURL
    }
}
