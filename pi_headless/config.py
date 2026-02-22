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

    return cfg, " ".join(notes).strip()
