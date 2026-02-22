import SwiftUI

struct ContentView: View {
    @StateObject private var vm = RobotPhoneViewModel()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Locky Robot")
                    .font(.largeTitle.bold())

                VStack(alignment: .leading, spacing: 8) {
                    Text("Pi Connection")
                        .font(.headline)

                    TextField("Pi host (example: 192.168.1.50)", text: $vm.connectionStore.hostInput)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled(true)
                        .textFieldStyle(.roundedBorder)

                    TextField("Port", text: $vm.connectionStore.portInput)
                        .keyboardType(.numberPad)
                        .textFieldStyle(.roundedBorder)

                    TextField("API token (optional)", text: $vm.connectionStore.tokenInput)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled(true)
                        .textFieldStyle(.roundedBorder)

                    Button("Save and connect") {
                        vm.applyConnectionSettings()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.isBusy)
                }

                Divider()

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
                        .disabled(vm.isBusy || !vm.canRunActions)
                    } else if state.ui.showScanHideButtons || state.phase == "mapped" || state.phase == "hidden" {
                        HStack {
                            Button("Start room scan") {
                                vm.startScanAgain()
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(vm.isBusy || !vm.canRunActions)

                            Button("Hide") {
                                vm.startHide()
                            }
                            .buttonStyle(.bordered)
                            .disabled(vm.isBusy || !state.ui.canHide || !vm.canRunActions)
                        }
                    } else if state.phase == "mapping" {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Scanning in progress...")
                                .font(.subheadline)

                            Text("Warning: hiding now will use an incomplete map.")
                                .font(.footnote)
                                .foregroundStyle(.orange)

                            // Keep hide-during-scan behavior from simulator semantics.
                            Button("Hide now (partial map)") {
                                vm.startHide()
                            }
                            .buttonStyle(.bordered)
                            .disabled(vm.isBusy || !state.ui.allowHideDuringMapping || !vm.canRunActions)
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
                    Text("Set Pi connection and tap Save and connect.")
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

                Spacer(minLength: 20)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
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
