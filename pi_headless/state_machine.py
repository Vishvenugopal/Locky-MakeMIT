from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class LidarCounters:
    total_frames: int = 0
    dropped_frames: int = 0
    last_seq: int = -1
    last_timestamp_ms: int = 0


class PhaseStateMachine:
    """Thread-safe phase/state holder for the headless Pi service."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.phase: str = "idle"  # idle | mapping | mapped | hiding | hidden
        self.mapping_state: str = "idle"
        self.status_text: str = "Ready. Waiting for initial room scan command."

        self.scan_started_monotonic_s: Optional[float] = None
        self.scan_completed_monotonic_s: Optional[float] = None
        self.returned_to_origin: bool = False

        self.lidar = LidarCounters()
        self.last_lidar_received_monotonic_s: Optional[float] = None

    def start_scan(self) -> Tuple[bool, str]:
        with self._lock:
            self.phase = "mapping"
            self.mapping_state = "sweep"
            self.returned_to_origin = False
            self.scan_started_monotonic_s = time.monotonic()
            self.scan_completed_monotonic_s = None
            self.status_text = "Phase 1 started: scanning room."
            return True, self.status_text

    def mark_scan_complete(self, returned_to_origin: bool) -> Tuple[bool, str]:
        with self._lock:
            if self.phase != "mapping":
                return False, "Scan can only complete while phase is mapping."

            self.phase = "mapped"
            self.mapping_state = "done"
            self.returned_to_origin = bool(returned_to_origin)
            self.scan_completed_monotonic_s = time.monotonic()

            if self.returned_to_origin:
                self.status_text = "Scan complete. Robot returned to origin."
            else:
                self.status_text = "Scan complete. Robot did not return to origin."
            return True, self.status_text

    def start_hide(self, allow_partial_map: bool = True) -> Tuple[bool, str]:
        with self._lock:
            if self.phase not in ("mapping", "mapped", "hidden"):
                return False, "Hide is allowed only after scan has started."

            interrupted_mapping = self.phase == "mapping"
            if interrupted_mapping and not allow_partial_map:
                return False, "Hide while mapping requires allow_partial_map=true."

            self.phase = "hiding"
            self.mapping_state = "idle"

            if interrupted_mapping:
                self.status_text = "Hide started from active scan (partial-map mode)."
            else:
                self.status_text = "Hide started."
            return True, self.status_text

    def mark_hide_complete(self) -> Tuple[bool, str]:
        with self._lock:
            if self.phase != "hiding":
                return False, "Hide can only complete while phase is hiding."
            self.phase = "hidden"
            self.status_text = "Hide complete."
            return True, self.status_text

    def update_mapping_state(self, mapping_state: str, status_text: str = "") -> None:
        with self._lock:
            if self.phase == "mapping":
                self.mapping_state = mapping_state.strip() or self.mapping_state
            if status_text.strip():
                self.status_text = status_text.strip()

    def record_lidar_frame(self, seq: int, timestamp_ms: int) -> None:
        with self._lock:
            self.lidar.total_frames += 1
            self.lidar.last_seq = int(seq)
            self.lidar.last_timestamp_ms = int(timestamp_ms)
            self.last_lidar_received_monotonic_s = time.monotonic()

    def record_lidar_drop(self) -> None:
        with self._lock:
            self.lidar.dropped_frames += 1

    def seconds_since_last_lidar(self) -> Optional[float]:
        with self._lock:
            if self.last_lidar_received_monotonic_s is None:
                return None
            return max(0.0, time.monotonic() - self.last_lidar_received_monotonic_s)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            show_initial_scan = self.phase == "idle"
            show_scan_hide = self.phase in ("mapped", "hidden") and self.returned_to_origin
            can_hide = self.phase in ("mapping", "mapped", "hidden")

            if self.last_lidar_received_monotonic_s is None:
                since_last_frame = None
            else:
                since_last_frame = max(
                    0.0,
                    time.monotonic() - self.last_lidar_received_monotonic_s,
                )

            return {
                "phase": self.phase,
                "mapping_state": self.mapping_state,
                "status_text": self.status_text,
                "returned_to_origin": self.returned_to_origin,
                "ui": {
                    "show_initial_scan_button": show_initial_scan,
                    "show_scan_hide_buttons": show_scan_hide,
                    "can_start_scan": True,
                    "can_hide": can_hide,
                    "allow_hide_during_mapping": True,
                },
                "lidar": {
                    "total_frames": self.lidar.total_frames,
                    "dropped_frames": self.lidar.dropped_frames,
                    "last_seq": self.lidar.last_seq,
                    "last_timestamp_ms": self.lidar.last_timestamp_ms,
                    "seconds_since_last_frame": since_last_frame,
                },
            }
