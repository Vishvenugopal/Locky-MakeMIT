from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_SECRETS_FILENAME = "pi_headless_secrets.json"


@dataclass
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    api_token: str = ""

    lidar_queue_size: int = 6
    stream_timeout_s: float = 2.5

    robot_address: str = ""
    api_key_id: str = ""
    api_key: str = ""
    base_name: str = "viam_base"

    linear_speed_limit_mps: float = 0.30
    angular_speed_limit_rps: float = 1.20
    publish_interval_s: float = 0.10

    autonomy_loop_hz: float = 10.0
    max_pose_stale_s: float = 1.2
    mapping_max_duration_s: float = 90.0
    min_mapping_loops: int = 2
    scan_turn_rate_rps: float = 0.45
    cruise_speed_mps: float = 0.12
    waypoint_tolerance_m: float = 0.18
    min_hide_distance_m: float = 1.2
    grid_cell_size_m: float = 0.20
    max_depth_m: float = 4.0


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _load_json(path: Path) -> Tuple[Dict[str, Any], str]:
    if not path.exists():
        return {}, ""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {}, f"Failed reading {path.name}: {exc}"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"Invalid JSON in {path.name}: {exc}"

    if not isinstance(parsed, dict):
        return {}, f"{path.name} must contain a JSON object."

    return parsed, ""


def load_service_config(base_dir: Path | None = None) -> Tuple[ServiceConfig, str]:
    root = base_dir if base_dir is not None else Path.cwd()

    secrets_override = os.environ.get("PI_SERVICE_SECRETS", "").strip()
    secrets_path = Path(secrets_override).expanduser() if secrets_override else (root / DEFAULT_SECRETS_FILENAME)

    raw, warning = _load_json(secrets_path)
    notes: List[str] = []
    if warning:
        notes.append(warning)

    def pick_str(env_name: str, json_key: str, default: str = "") -> str:
        env_val = os.environ.get(env_name)
        if env_val is not None and env_val.strip():
            return env_val.strip()
        value = raw.get(json_key, default)
        if value is None:
            return default
        return str(value).strip()

    def pick_num(env_name: str, json_key: str, default: Any) -> Any:
        env_val = os.environ.get(env_name)
        if env_val is not None and env_val.strip():
            return env_val.strip()
        return raw.get(json_key, default)

    cfg = ServiceConfig(
        host=pick_str("PI_SERVICE_HOST", "host", "0.0.0.0"),
        port=_to_int(pick_num("PI_SERVICE_PORT", "port", 8765), 8765),
        api_token=pick_str("PI_SERVICE_API_TOKEN", "api_token", ""),
        lidar_queue_size=_to_int(pick_num("PI_LIDAR_QUEUE_SIZE", "lidar_queue_size", 6), 6),
        stream_timeout_s=_to_float(pick_num("PI_STREAM_TIMEOUT_S", "stream_timeout_s", 2.5), 2.5),
        robot_address=pick_str("PI_ROBOT_ADDRESS", "robot_address", ""),
        api_key_id=pick_str("PI_API_KEY_ID", "api_key_id", ""),
        api_key=pick_str("PI_API_KEY", "api_key", ""),
        base_name=pick_str("PI_BASE_NAME", "base_name", "viam_base"),
        linear_speed_limit_mps=_to_float(
            pick_num("PI_LINEAR_LIMIT_MPS", "linear_speed_limit_mps", 0.30),
            0.30,
        ),
        angular_speed_limit_rps=_to_float(
            pick_num("PI_ANGULAR_LIMIT_RPS", "angular_speed_limit_rps", 1.20),
            1.20,
        ),
        publish_interval_s=_to_float(
            pick_num("PI_PUBLISH_INTERVAL_S", "publish_interval_s", 0.10),
            0.10,
        ),
        autonomy_loop_hz=_to_float(
            pick_num("PI_AUTONOMY_LOOP_HZ", "autonomy_loop_hz", 10.0),
            10.0,
        ),
        max_pose_stale_s=_to_float(
            pick_num("PI_MAX_POSE_STALE_S", "max_pose_stale_s", 1.2),
            1.2,
        ),
        mapping_max_duration_s=_to_float(
            pick_num("PI_MAPPING_MAX_DURATION_S", "mapping_max_duration_s", 90.0),
            90.0,
        ),
        min_mapping_loops=_to_int(
            pick_num("PI_MIN_MAPPING_LOOPS", "min_mapping_loops", 2),
            2,
        ),
        scan_turn_rate_rps=_to_float(
            pick_num("PI_SCAN_TURN_RATE_RPS", "scan_turn_rate_rps", 0.45),
            0.45,
        ),
        cruise_speed_mps=_to_float(
            pick_num("PI_CRUISE_SPEED_MPS", "cruise_speed_mps", 0.12),
            0.12,
        ),
        waypoint_tolerance_m=_to_float(
            pick_num("PI_WAYPOINT_TOLERANCE_M", "waypoint_tolerance_m", 0.18),
            0.18,
        ),
        min_hide_distance_m=_to_float(
            pick_num("PI_MIN_HIDE_DISTANCE_M", "min_hide_distance_m", 1.2),
            1.2,
        ),
        grid_cell_size_m=_to_float(
            pick_num("PI_GRID_CELL_SIZE_M", "grid_cell_size_m", 0.20),
            0.20,
        ),
        max_depth_m=_to_float(
            pick_num("PI_MAX_DEPTH_M", "max_depth_m", 4.0),
            4.0,
        ),
    )

    if not cfg.host:
        cfg.host = "0.0.0.0"
    cfg.port = int(_clip(cfg.port, 1, 65535))
    cfg.lidar_queue_size = int(_clip(cfg.lidar_queue_size, 1, 60))
    cfg.stream_timeout_s = _clip(cfg.stream_timeout_s, 0.2, 30.0)

    # Keep hardware speeds conservative by default.
    cfg.linear_speed_limit_mps = _clip(cfg.linear_speed_limit_mps, 0.05, 0.30)
    cfg.angular_speed_limit_rps = _clip(cfg.angular_speed_limit_rps, 0.10, 1.20)
    cfg.publish_interval_s = _clip(cfg.publish_interval_s, 0.03, 1.0)

    cfg.autonomy_loop_hz = _clip(cfg.autonomy_loop_hz, 4.0, 30.0)
    cfg.max_pose_stale_s = _clip(cfg.max_pose_stale_s, 0.2, 5.0)
    cfg.mapping_max_duration_s = _clip(cfg.mapping_max_duration_s, 20.0, 900.0)
    cfg.min_mapping_loops = int(_clip(cfg.min_mapping_loops, 1, 12))
    cfg.scan_turn_rate_rps = _clip(cfg.scan_turn_rate_rps, 0.10, 1.20)
    cfg.cruise_speed_mps = _clip(cfg.cruise_speed_mps, 0.04, 0.30)
    cfg.waypoint_tolerance_m = _clip(cfg.waypoint_tolerance_m, 0.05, 0.60)
    cfg.min_hide_distance_m = _clip(cfg.min_hide_distance_m, 0.30, 8.0)
    cfg.grid_cell_size_m = _clip(cfg.grid_cell_size_m, 0.08, 0.60)
    cfg.max_depth_m = _clip(cfg.max_depth_m, 0.8, 8.0)

    return cfg, " ".join(notes).strip()
