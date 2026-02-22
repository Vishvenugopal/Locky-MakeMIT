# Pi Headless Service (No UI)

This service runs on the Raspberry Pi and provides:

- HTTP control endpoints for phase actions (`scan`, `hide`, status)
- WebSocket LiDAR ingest over Wi-Fi
- Optional Viam robot output publish (with clamping + fail-safe stop)

## Why this split

- **Phone app (iOS)** handles ARKit LiDAR capture and user controls.
- **Pi service** is the single control authority ("Pi API only" routing).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r pi_headless/requirements.txt
cp pi_headless_secrets.example.json pi_headless_secrets.json
python -m pi_headless.main
```

By default the API runs at `http://0.0.0.0:8765`.

## Endpoints

- `GET /health`
- `GET /state`
- `POST /phase/scan/start`
- `POST /phase/scan/complete`
- `POST /phase/hide/start`
- `POST /phase/hide/complete`
- `POST /robot/connect`
- `POST /robot/disconnect`
- `POST /robot/command`
- `WS /stream/lidar`

If `api_token` is set, include `X-API-Token` on HTTP and `?token=...` on websocket.

## LiDAR websocket protocol

### Option A (recommended): binary frames

Each websocket binary message is:

1. `uint32` little-endian header length (4 bytes)
2. UTF-8 JSON header
3. Payload bytes (`depth` first, then optional `confidence`)

Header example:

```json
{
  "type": "lidar_frame",
  "seq": 101,
  "timestamp_ms": 1739960130450,
  "width": 256,
  "height": 192,
  "depth_encoding": "u16_mm_raw",
  "depth_byte_count": 98304,
  "confidence_encoding": "u8_raw",
  "confidence_byte_count": 49152,
  "pose": {
    "tx": 0.2,
    "ty": 0.0,
    "tz": -0.1,
    "qx": 0.0,
    "qy": 0.0,
    "qz": 0.0,
    "qw": 1.0
  },
  "intrinsics": {
    "fx": 200.0,
    "fy": 200.0,
    "cx": 128.0,
    "cy": 96.0
  }
}
```

Supported encodings:

- Depth: `u16_mm_raw`, `zlib_u16_mm`
- Confidence: `u8_raw`, `zlib_u8`

### Option B: text JSON frames

Send `type=lidar_frame` with base64 depth/confidence fields (`depth_b64`, optional `confidence_b64`).

## Notes

- This is a headless control/ingest scaffold intentionally separated from `phone_app_ui.py`.
- The state machine supports hide during mapping (`allow_partial_map=true`) to match your simulator behavior.
- Map fusion and autonomous scan/hide internals are next integration steps.
