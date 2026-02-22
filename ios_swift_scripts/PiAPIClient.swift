import Foundation

final class PiAPIClient {
    private let session: URLSession
    private let decoder = JSONDecoder()

    init(session: URLSession = .shared) {
        self.session = session
    }

    func fetchState(settings: PiConnectionSettings) async throws -> ServerState {
        let request = makeRequest(path: "/state", method: "GET", settings: settings)
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(ServerState.self, from: data)
    }

    func startScan(settings: PiConnectionSettings) async throws -> ActionResult {
        try await postJSON(path: "/phase/scan/start", body: [:], settings: settings)
    }

    func startHide(settings: PiConnectionSettings, allowPartialMap: Bool = true) async throws -> ActionResult {
        try await postJSON(
            path: "/phase/hide/start",
            body: ["allow_partial_map": allowPartialMap],
            settings: settings
        )
    }

    func completeScan(settings: PiConnectionSettings, returnedToOrigin: Bool) async throws -> ActionResult {
        try await postJSON(
            path: "/phase/scan/complete",
            body: ["returned_to_origin": returnedToOrigin],
            settings: settings
        )
    }

    private func postJSON(
        path: String,
        body: [String: Any],
        settings: PiConnectionSettings
    ) async throws -> ActionResult {
        var request = makeRequest(path: path, method: "POST", settings: settings)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(ActionResult.self, from: data)
    }

    private func makeRequest(path: String, method: String, settings: PiConnectionSettings) -> URLRequest {
        let cleanPath = path.hasPrefix("/") ? String(path.dropFirst()) : path
        let url = settings.baseHTTPURL.appendingPathComponent(cleanPath)
        var request = URLRequest(url: url)
        request.httpMethod = method

        if !settings.apiToken.isEmpty {
            request.setValue(settings.apiToken, forHTTPHeaderField: "X-API-Token")
        }
        return request
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "PiAPIClient", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Invalid HTTP response",
            ])
        }

        guard (200...299).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? "<no body>"
            throw NSError(domain: "PiAPIClient", code: http.statusCode, userInfo: [
                NSLocalizedDescriptionKey: "HTTP \(http.statusCode): \(body)",
            ])
        }
    }
}
