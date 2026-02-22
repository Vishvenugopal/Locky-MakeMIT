import Foundation

final class PiAPIClient {
    private let session: URLSession
    private let decoder = JSONDecoder()

    init(session: URLSession = .shared) {
        self.session = session
    }

    func fetchState() async throws -> ServerState {
        let request = makeRequest(path: "/state", method: "GET")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(ServerState.self, from: data)
    }

    func startScan() async throws -> ActionResult {
        try await postJSON(path: "/phase/scan/start", body: [:])
    }

    func startHide(allowPartialMap: Bool = true) async throws -> ActionResult {
        try await postJSON(
            path: "/phase/hide/start",
            body: ["allow_partial_map": allowPartialMap]
        )
    }

    func completeScan(returnedToOrigin: Bool) async throws -> ActionResult {
        try await postJSON(
            path: "/phase/scan/complete",
            body: ["returned_to_origin": returnedToOrigin]
        )
    }

    private func postJSON(path: String, body: [String: Any]) async throws -> ActionResult {
        var request = makeRequest(path: path, method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(ActionResult.self, from: data)
    }

    private func makeRequest(path: String, method: String) -> URLRequest {
        let cleanPath = path.hasPrefix("/") ? String(path.dropFirst()) : path
        let url = AppConfig.baseHTTPURL.appendingPathComponent(cleanPath)
        var request = URLRequest(url: url)
        request.httpMethod = method

        if !AppConfig.apiToken.isEmpty {
            request.setValue(AppConfig.apiToken, forHTTPHeaderField: "X-API-Token")
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
