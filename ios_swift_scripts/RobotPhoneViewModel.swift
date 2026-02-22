import Foundation

@MainActor
final class RobotPhoneViewModel: ObservableObject {
    @Published var connectionStore = ConnectionSettingsStore()
    @Published var state: ServerState?
    @Published var statusLine: String = "Waiting for Pi service..."
    @Published var errorLine: String?
    @Published var isBusy: Bool = false

    private let api = PiAPIClient()
    private let lidarStreamer = LiDARStreamer()
    private var pollTask: Task<Void, Never>?
    private var activeSettings: PiConnectionSettings?

    init() {
        lidarStreamer.onStatus = { [weak self] text in
            Task { @MainActor in
                self?.statusLine = text
            }
        }
    }

    func onAppear() {
        if let saved = connectionStore.buildIfValid() {
            activeSettings = saved
            statusLine = "Loaded saved Pi settings \(saved.host):\(saved.port)"
        } else {
            statusLine = "Enter Pi connection settings to begin."
        }
        startPolling()
    }

    func onDisappear() {
        pollTask?.cancel()
        pollTask = nil
        lidarStreamer.stopStreaming()
    }

    func applyConnectionSettings() {
        Task {
            activeSettings = nil
            state = nil
            startPolling()
            errorLine = nil
            statusLine = "Connecting to Pi..."

            await runAction {
                let settings = try connectionStore.saveAndBuild()
                let latestState = try await api.fetchState(settings: settings)
                activeSettings = settings
                state = latestState
                errorLine = nil
                statusLine = "Connected to Pi at \(settings.host):\(settings.port)"
                startPolling()

                if lidarStreamer.isStreaming {
                    lidarStreamer.startStreaming(settings: settings)
                }
            }
        }
    }

    var canRunActions: Bool {
        activeSettings != nil && state != nil
    }

    func refreshState() async {
        guard let settings = activeSettings else {
            return
        }

        do {
            let latest = try await api.fetchState(settings: settings)
            state = latest
            errorLine = nil
        } catch {
            state = nil
            errorLine = error.localizedDescription
        }
    }

    func startInitialScan() {
        Task {
            await runAction {
                let settings = try requireSettings()
                ensureStreaming(settings: settings)
                _ = try await api.startScan(settings: settings)
                state = try await api.fetchState(settings: settings)
                statusLine = "Initial room scan started"
            }
        }
    }

    func startScanAgain() {
        Task {
            await runAction {
                let settings = try requireSettings()
                ensureStreaming(settings: settings)
                _ = try await api.startScan(settings: settings)
                state = try await api.fetchState(settings: settings)
                statusLine = "Room scan started"
            }
        }
    }

    func startHide() {
        Task {
            await runAction {
                let settings = try requireSettings()
                ensureStreaming(settings: settings)
                _ = try await api.startHide(settings: settings, allowPartialMap: true)
                state = try await api.fetchState(settings: settings)
                statusLine = "Hide started"
            }
        }
    }

    private func startPolling() {
        guard activeSettings != nil else {
            pollTask?.cancel()
            pollTask = nil
            return
        }

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

    private func requireSettings() throws -> PiConnectionSettings {
        if let settings = activeSettings {
            return settings
        }

        throw NSError(domain: "RobotPhoneViewModel", code: -1, userInfo: [
            NSLocalizedDescriptionKey: "Save and connect to Pi before starting scan/hide actions.",
        ])
    }

    private func ensureStreaming(settings: PiConnectionSettings) {
        if !lidarStreamer.isStreaming {
            lidarStreamer.startStreaming(settings: settings)
        }
    }
}
