from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
import uvicorn

from pi_headless.autonomy import AutonomyConfig, AutonomyEngine
from pi_headless.config import ServiceConfig, load_service_config
from pi_headless.lidar_ingest import LidarIngestor
from pi_headless.robot_output import RobotControlConfig, ViamBaseController
from pi_headless.state_machine import PhaseStateMachine

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG, CONFIG_WARNING = load_service_config(REPO_ROOT)

STATE = PhaseStateMachine()
LIDAR = LidarIngestor(max_queue_size=CONFIG.lidar_queue_size)
ROBOT = ViamBaseController()
WATCHDOG_TASK: Optional[asyncio.Task[Any]] = None
AUTONOMY = AutonomyEngine(
    config=AutonomyConfig(
        loop_hz=CONFIG.autonomy_loop_hz,
        max_pose_stale_s=CONFIG.max_pose_stale_s,
        mapping_max_duration_s=CONFIG.mapping_max_duration_s,
        min_mapping_loops=CONFIG.min_mapping_loops,
        scan_turn_rate_rps=CONFIG.scan_turn_rate_rps,
        cruise_speed_mps=CONFIG.cruise_speed_mps,
        waypoint_tolerance_m=CONFIG.waypoint_tolerance_m,
        min_hide_distance_m=CONFIG.min_hide_distance_m,
        grid_cell_size_m=CONFIG.grid_cell_size_m,
        max_depth_m=CONFIG.max_depth_m,
    ),
    state_machine=STATE,
    lidar_ingestor=LIDAR,
    robot=ROBOT,
)
AUTONOMY_TASK: Optional[asyncio.Task[Any]] = None

app = FastAPI(title="Locky Pi Headless Service", version="0.1.0")


def _require_token(token: str) -> None:
    expected = CONFIG.api_token.strip()
    if not expected:
        return
    if token.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid API token.")


def _robot_config_from_service(defaults: ServiceConfig) -> RobotControlConfig:
    return RobotControlConfig(
        robot_address=defaults.robot_address,
        api_key_id=defaults.api_key_id,
        api_key=defaults.api_key,
        base_name=defaults.base_name,
        linear_speed_limit_mps=defaults.linear_speed_limit_mps,
        angular_speed_limit_rps=defaults.angular_speed_limit_rps,
        publish_interval_s=defaults.publish_interval_s,
    )


async def _watchdog_loop() -> None:
    while True:
        await asyncio.sleep(0.25)

        stale_for_s = STATE.seconds_since_last_lidar()
        if stale_for_s is None:
            continue

        snapshot = STATE.snapshot()
        if snapshot["phase"] not in ("mapping", "hiding"):
            continue

        if stale_for_s > CONFIG.stream_timeout_s:
            timeout_msg = (
                f"LiDAR stream timeout ({stale_for_s:.1f}s > {CONFIG.stream_timeout_s:.1f}s). "
                "Robot stopped for safety."
            )
            STATE.update_mapping_state(snapshot["mapping_state"], timeout_msg)
            if ROBOT.is_connected:
                ROBOT.stop()


@app.on_event("startup")
async def _on_startup() -> None:
    global WATCHDOG_TASK, AUTONOMY_TASK

    logging.basicConfig(level=logging.INFO)
    if CONFIG_WARNING:
        logging.warning(CONFIG_WARNING)

    if CONFIG.robot_address and CONFIG.api_key_id and CONFIG.api_key:
        ok, msg = ROBOT.connect(_robot_config_from_service(CONFIG))
        if ok:
            logging.info(msg)
        else:
            logging.warning("Robot auto-connect failed: %s", msg)

    AUTONOMY_TASK = asyncio.create_task(AUTONOMY.run())
    WATCHDOG_TASK = asyncio.create_task(_watchdog_loop())


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    global WATCHDOG_TASK, AUTONOMY_TASK

    AUTONOMY.request_stop()

    if AUTONOMY_TASK is not None:
        AUTONOMY_TASK.cancel()
        try:
            await AUTONOMY_TASK
        except asyncio.CancelledError:
            pass
        AUTONOMY_TASK = None

    if WATCHDOG_TASK is not None:
        WATCHDOG_TASK.cancel()
        try:
            await WATCHDOG_TASK
        except asyncio.CancelledError:
            pass
        WATCHDOG_TASK = None

    try:
        ROBOT.close()
    except Exception:
        pass


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "locky-pi-headless",
        "phase": STATE.snapshot()["phase"],
        "robot_connected": ROBOT.is_connected,
        "autonomy": AUTONOMY.debug_snapshot(),
    }


@app.get("/state")
def get_state(
    x_api_token: Optional[str] = Header(default="", alias="X-API-Token"),
) -> Dict[str, Any]:
    _require_token(x_api_token or "")
    snapshot = STATE.snapshot()
    snapshot["robot_link"] = "connected" if ROBOT.is_connected else "disconnected"
    snapshot["lidar_queue_size"] = LIDAR.queue_size
    snapshot["autonomy"] = AUTONOMY.debug_snapshot()
    return snapshot


@app.post("/phase/scan/start")
def phase_scan_start(
    x_api_token: Optional[str] = Header(default="", alias="X-API-Token"),
) -> Dict[str, Any]:
    _require_token(x_api_token or "")
    ok, msg = STATE.start_scan()
    return {"ok": ok, "message": msg, "state": STATE.snapshot()}


@app.post("/phase/scan/complete")
def phase_scan_complete(
    payload: Optional[Dict[str, Any]] = Body(default=None),
    x_api_token: Optional[str] = Header(default="", alias="X-API-Token"),
) -> Dict[str, Any]:
    _require_token(x_api_token or "")

    body = payload or {}
    returned_to_origin = bool(body.get("returned_to_origin", True))

    ok, msg = STATE.mark_scan_complete(returned_to_origin=returned_to_origin)
    return {"ok": ok, "message": msg, "state": STATE.snapshot()}


@app.post("/phase/hide/start")
def phase_hide_start(
    payload: Optional[Dict[str, Any]] = Body(default=None),
    x_api_token: Optional[str] = Header(default="", alias="X-API-Token"),
) -> Dict[str, Any]:
    _require_token(x_api_token or "")

    body = payload or {}
    allow_partial_map = bool(body.get("allow_partial_map", True))

    ok, msg = STATE.start_hide(allow_partial_map=allow_partial_map)
    return {"ok": ok, "message": msg, "state": STATE.snapshot()}


@app.post("/phase/hide/complete")
def phase_hide_complete(
    x_api_token: Optional[str] = Header(default="", alias="X-API-Token"),
) -> Dict[str, Any]:
    _require_token(x_api_token or "")
    ok, msg = STATE.mark_hide_complete()
    return {"ok": ok, "message": msg, "state": STATE.snapshot()}


@app.post("/robot/connect")
def robot_connect(
    payload: Optional[Dict[str, Any]] = Body(default=None),
    x_api_token: Optional[str] = Header(default="", alias="X-API-Token"),
) -> Dict[str, Any]:
    _require_token(x_api_token or "")

    body = payload or {}
    cfg = RobotControlConfig(
        robot_address=str(body.get("robot_address", CONFIG.robot_address)).strip(),
        api_key_id=str(body.get("api_key_id", CONFIG.api_key_id)).strip(),
        api_key=str(body.get("api_key", CONFIG.api_key)).strip(),
        base_name=str(body.get("base_name", CONFIG.base_name)).strip() or "viam_base",
        linear_speed_limit_mps=float(body.get("linear_speed_limit_mps", CONFIG.linear_speed_limit_mps)),
        angular_speed_limit_rps=float(body.get("angular_speed_limit_rps", CONFIG.angular_speed_limit_rps)),
        publish_interval_s=float(body.get("publish_interval_s", CONFIG.publish_interval_s)),
    )

    ok, msg = ROBOT.connect(cfg)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"ok": True, "message": msg}


@app.post("/robot/disconnect")
def robot_disconnect(
    x_api_token: Optional[str] = Header(default="", alias="X-API-Token"),
) -> Dict[str, Any]:
    _require_token(x_api_token or "")
    ROBOT.close()
    return {"ok": True, "message": "Robot disconnected."}


@app.post("/robot/command")
def robot_command(
    payload: Dict[str, Any] = Body(...),
    x_api_token: Optional[str] = Header(default="", alias="X-API-Token"),
) -> Dict[str, Any]:
    _require_token(x_api_token or "")

    linear_mps = float(payload.get("linear_mps", 0.0))
    angular_rps = float(payload.get("angular_rps", 0.0))

    ok, msg = ROBOT.send_velocity(linear_mps=linear_mps, angular_rps=angular_rps)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    return {
        "ok": True,
        "message": "",
        "linear_mps": linear_mps,
        "angular_rps": angular_rps,
    }


@app.websocket("/stream/lidar")
async def ws_lidar_stream(websocket: WebSocket) -> None:
    expected = CONFIG.api_token.strip()
    token = websocket.query_params.get("token", "")
    if expected and token.strip() != expected:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    try:
        while True:
            packet = await websocket.receive()

            if packet.get("type") == "websocket.disconnect":
                break

            text_payload = packet.get("text")
            bytes_payload = packet.get("bytes")

            try:
                if text_payload is not None:
                    await _handle_text_ws_message(websocket, text_payload)
                elif bytes_payload is not None:
                    frame = LIDAR.decode_binary_packet(bytes_payload)
                    dropped = LIDAR.enqueue(frame)
                    if dropped:
                        STATE.record_lidar_drop()
                    STATE.record_lidar_frame(frame.seq, frame.timestamp_ms)
                else:
                    await websocket.send_json({"type": "warn", "message": "Empty websocket packet."})
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})

    except WebSocketDisconnect:
        return


async def _handle_text_ws_message(websocket: WebSocket, message: str) -> None:
    trimmed = message.strip()
    if not trimmed:
        return

    if trimmed.lower() == "ping":
        await websocket.send_text("pong")
        return

    payload = json.loads(trimmed)
    if not isinstance(payload, dict):
        raise ValueError("Text websocket payload must be a JSON object.")

    msg_type = str(payload.get("type", "")).strip()
    if msg_type == "lidar_frame":
        frame = LIDAR.decode_text_payload(payload)
        dropped = LIDAR.enqueue(frame)
        if dropped:
            STATE.record_lidar_drop()
        STATE.record_lidar_frame(frame.seq, frame.timestamp_ms)
        return

    if msg_type == "scan_complete":
        returned_to_origin = bool(payload.get("returned_to_origin", True))
        STATE.mark_scan_complete(returned_to_origin=returned_to_origin)
        return

    if msg_type == "hide_complete":
        STATE.mark_hide_complete()
        return

    if msg_type == "mapping_update":
        mapping_state = str(payload.get("mapping_state", "")).strip()
        status_text = str(payload.get("status_text", "")).strip()
        STATE.update_mapping_state(mapping_state, status_text)
        return

    raise ValueError(f"Unsupported websocket message type: {msg_type}")


def main() -> None:
    uvicorn.run(
        "pi_headless.main:app",
        host=CONFIG.host,
        port=CONFIG.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
