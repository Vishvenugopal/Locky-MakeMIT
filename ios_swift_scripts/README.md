# iOS Swift Scripts (for Xcode teammates)

This folder is intentionally separate so your teammates can copy these Swift files into an Xcode iOS app project.

## Stack choice

- **iOS native** with **Swift + ARKit**
- LiDAR payload: **depth + pose**
- Control path: **Pi API only** (phone never talks directly to Viam)

## Files

- `MakeMITRobotPhoneApp.swift` - app entry point
- `ContentView.swift` - phase-driven UI
- `RobotPhoneViewModel.swift` - UI state + polling + actions
- `PiAPIClient.swift` - HTTP calls to Pi service
- `LiDARStreamer.swift` - ARKit depth capture + websocket streaming
- `Models.swift` - decodable API models
- `AppConfig.swift` - Pi host/port/token config

## Required device

- LiDAR-capable iPhone/iPad (Pro model).

## Xcode setup

1. Create a new iOS SwiftUI app project.
2. Add all files from this folder.
3. Set your Pi address in `AppConfig.swift`.
4. Add Info.plist keys:
   - `NSCameraUsageDescription`
   - `NSLocalNetworkUsageDescription`
5. If using plain `http://` and `ws://` on local LAN, add ATS exception for your Pi host or local network.

## UI behavior implemented

- Initial app state: one button, **Start initial room scan**.
- After scan complete + return origin: two buttons, **Start room scan** and **Hide**.
- During mapping: includes **Hide now (partial map)** action to preserve simulator behavior.

## Notes

- The Swift files assume your Pi service runs from `pi_headless/main.py`.
- The LiDAR stream uses binary websocket messages with a JSON header + raw payload.
