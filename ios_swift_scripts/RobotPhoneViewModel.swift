import Foundation

@MainActor
final class RobotPhoneViewModel: ObservableObject {
    @Published var state: ServerState?
    @Published var statusLine: String = "Waiting for Pi service..."
    @Published var errorLine: String?
    @Published var isBusy: Bool = false

    private let api = PiAPIClient()
    private let lidarStreamer = LiDARStreamer()
    private var pollTask: Task<Void, Never>?

    init() {
        lidarStreamer.onStatus = { [weak self] text in
            Task { @MainActor in
                self?.statusLine = text
            }
        }
    }

    func onAppear() {
        startPolling()
    }

    func onDisappear() {
        pollTask?.cancel()
        pollTask = nil
        lidarStreamer.stopStreaming()
    }

    func refreshState() async {
        do {
            let latest = try await api.fetchState()
            state = latest
            errorLine = nil
        } catch {
            errorLine = error.localizedDescription
        }
    }

    func startInitialScan() {
        Task {
            await runAction {
                ensureStreaming()
                _ = try await api.startScan()
                state = try await api.fetchState()
                statusLine = "Initial room scan started"
            }
        }
    }

    func startScanAgain() {
        Task {
            await runAction {
                ensureStreaming()
                _ = try await api.startScan()
                state = try await api.fetchState()
                statusLine = "Room scan started"
            }
        }
    }

    func startHide() {
        Task {
            await runAction {
                ensureStreaming()
                _ = try await api.startHide(allowPartialMap: true)
                state = try await api.fetchState()
                statusLine = "Hide started"
            }
        }
    }

    private func startPolling() {
        pollTask?.cancel()
        pollTask = Task {
            while !Task.isCancelled {
                await refreshState()
                try? await Task.sleep(nanoseconds: UInt64(AppConfig.statePollInterval * 1_000_000_000))
            }
        }
    }

    private func runAction(_ action: () async throws -> Void) async {
        if isBusy {
            return
        }

        isBusy = true
        defer { isBusy = false }

        do {
            try await action()
            errorLine = nil
        } catch {
            errorLine = error.localizedDescription
        }
    }

    private func ensureStreaming() {
        if !lidarStreamer.isStreaming {
            lidarStreamer.startStreaming()
        }
    }
}
