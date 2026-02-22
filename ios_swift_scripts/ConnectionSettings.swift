import Combine
import Foundation

struct PiConnectionSettings: Equatable {
    let host: String
    let port: Int
    let apiToken: String

    var baseHTTPURL: URL {
        URL(string: "http://\(host):\(port)")!
    }

    var baseWebSocketURL: URL {
        URL(string: "ws://\(host):\(port)/stream/lidar")!
    }

    func websocketURLWithToken() -> URL {
        guard !apiToken.isEmpty else {
            return baseWebSocketURL
        }

        var components = URLComponents(url: baseWebSocketURL, resolvingAgainstBaseURL: false)
        components?.queryItems = [
            URLQueryItem(name: "token", value: apiToken),
        ]
        return components?.url ?? baseWebSocketURL
    }
}

enum ConnectionSettingsError: LocalizedError {
    case emptyHost
    case invalidHost
    case invalidPort

    var errorDescription: String? {
        switch self {
        case .emptyHost:
            return "Enter the Pi IP address or hostname."
        case .invalidHost:
            return "Host is invalid. Use only a hostname/IP (no http://, path, or spaces)."
        case .invalidPort:
            return "Port must be a number between 1 and 65535."
        }
    }
}

@MainActor
final class ConnectionSettingsStore: ObservableObject {
    @Published var hostInput: String
    @Published var portInput: String
    @Published var tokenInput: String

    private let defaults: UserDefaults

    private enum Keys {
        static let host = "locky.pi.host"
        static let port = "locky.pi.port"
        static let token = "locky.pi.token"
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.hostInput = defaults.string(forKey: Keys.host) ?? ""

        let storedPort = defaults.integer(forKey: Keys.port)
        if storedPort > 0 {
            self.portInput = String(storedPort)
        } else {
            self.portInput = String(AppConfig.defaultPiPort)
        }

        self.tokenInput = defaults.string(forKey: Keys.token) ?? ""
    }

    func saveAndBuild() throws -> PiConnectionSettings {
        let settings = try validateCurrent()

        defaults.set(settings.host, forKey: Keys.host)
        defaults.set(settings.port, forKey: Keys.port)
        defaults.set(settings.apiToken, forKey: Keys.token)

        hostInput = settings.host
        portInput = String(settings.port)
        tokenInput = settings.apiToken

        return settings
    }

    func buildIfValid() -> PiConnectionSettings? {
        try? validateCurrent()
    }

    private func validateCurrent() throws -> PiConnectionSettings {
        let host = Self.normalizeHost(hostInput)
        if host.isEmpty {
            throw ConnectionSettingsError.emptyHost
        }
        if host.contains(" ") || host.contains("/") || host.contains("://") || host.contains("?") || host.contains("#") || host.contains(":") {
            throw ConnectionSettingsError.invalidHost
        }

        let trimmedPort = portInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let port = Int(trimmedPort), (1...65535).contains(port) else {
            throw ConnectionSettingsError.invalidPort
        }

        let token = tokenInput.trimmingCharacters(in: .whitespacesAndNewlines)
        return PiConnectionSettings(host: host, port: port, apiToken: token)
    }

    private static func normalizeHost(_ raw: String) -> String {
        var value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.isEmpty {
            return ""
        }

        if value.hasPrefix("http://") || value.hasPrefix("https://") {
            if let components = URLComponents(string: value), let host = components.host {
                value = host
            }
        }

        if let slash = value.firstIndex(of: "/") {
            value = String(value[..<slash])
        }

        if let colon = value.lastIndex(of: ":"), !value.contains("]") {
            let hostPart = String(value[..<colon])
            let portPart = String(value[value.index(after: colon)...])
            if Int(portPart) != nil {
                value = hostPart
            }
        }

        value = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return value
    }
}
