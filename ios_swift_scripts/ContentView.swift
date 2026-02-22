import SwiftUI

struct ContentView: View {
    @StateObject private var vm = RobotPhoneViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Locky Robot")
                .font(.largeTitle.bold())

            if let state = vm.state {
                Text("Phase: \(state.phase) / \(state.mappingState)")
                    .font(.headline)
                Text("Robot link: \(state.robotLink)")
                    .foregroundStyle(state.robotLink == "connected" ? .green : .red)
                Text(state.statusText)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Divider()

                if state.ui.showInitialScanButton {
                    Button("Start initial room scan") {
                        vm.startInitialScan()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.isBusy)
                } else if state.ui.showScanHideButtons {
                    HStack {
                        Button("Start room scan") {
                            vm.startScanAgain()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(vm.isBusy)

                        Button("Hide") {
                            vm.startHide()
                        }
                        .buttonStyle(.bordered)
                        .disabled(vm.isBusy || !state.ui.canHide)
                    }
                } else if state.phase == "mapping" {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Scanning in progress...")
                            .font(.subheadline)

                        // Keep hide-during-scan behavior from simulator semantics.
                        Button("Hide now (partial map)") {
                            vm.startHide()
                        }
                        .buttonStyle(.bordered)
                        .disabled(vm.isBusy || !state.ui.allowHideDuringMapping)
                    }
                } else {
                    Text("Waiting for next available action...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Divider()

                Text("LiDAR frames: \(state.lidar.totalFrames)")
                    .font(.footnote)
                Text("Dropped frames: \(state.lidar.droppedFrames)")
                    .font(.footnote)
            } else {
                Text("Connecting to Pi...")
                    .foregroundStyle(.secondary)
            }

            if let error = vm.errorLine {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
            } else {
                Text(vm.statusLine)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(20)
        .task {
            vm.onAppear()
        }
        .onDisappear {
            vm.onDisappear()
        }
    }
}

#Preview {
    ContentView()
}
