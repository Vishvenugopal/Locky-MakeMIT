from __future__ import annotations

import base64
import json
import queue
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class LidarFrame:
    seq: int
    timestamp_ms: int
    width: int
    height: int
    depth_mm: np.ndarray  # uint16 [h, w]
    pose: Dict[str, float]
    intrinsics: Dict[str, float]
    confidence: Optional[np.ndarray] = None


class LidarIngestor:
    """Decode and buffer incoming LiDAR frames from websocket messages."""

    def __init__(self, max_queue_size: int = 6) -> None:
        self._queue: "queue.Queue[LidarFrame]" = queue.Queue(maxsize=max(1, int(max_queue_size)))

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def enqueue(self, frame: LidarFrame) -> bool:
        """Returns True if an old frame was dropped due to backpressure."""
        dropped = False
        if self._queue.full():
            try:
                _ = self._queue.get_nowait()
                dropped = True
            except queue.Empty:
                dropped = False

        self._queue.put_nowait(frame)
        return dropped

    def get_latest_nowait(self) -> Optional[LidarFrame]:
        latest: Optional[LidarFrame] = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        return latest

    @staticmethod
    def decode_text_payload(payload: Dict[str, Any]) -> LidarFrame:
        if payload.get("type") != "lidar_frame":
            raise ValueError("Unsupported text message type.")

        seq = int(payload["seq"])
        timestamp_ms = int(payload["timestamp_ms"])
        width = int(payload["width"])
        height = int(payload["height"])

        depth_encoding = str(payload.get("depth_encoding", "u16_mm_raw"))
        depth_blob = base64.b64decode(payload["depth_b64"])

        confidence: Optional[np.ndarray] = None
        confidence_b64 = payload.get("confidence_b64")
        confidence_encoding = str(payload.get("confidence_encoding", ""))

        depth_mm = _decode_depth_blob(depth_blob, depth_encoding, width, height)

        if isinstance(confidence_b64, str) and confidence_b64:
            confidence_blob = base64.b64decode(confidence_b64)
            confidence = _decode_confidence_blob(confidence_blob, confidence_encoding, width, height)

        pose = dict(payload.get("pose", {}))
        intrinsics = dict(payload.get("intrinsics", {}))

        return LidarFrame(
            seq=seq,
            timestamp_ms=timestamp_ms,
            width=width,
            height=height,
            depth_mm=depth_mm,
            confidence=confidence,
            pose=pose,
            intrinsics=intrinsics,
        )

    @staticmethod
    def decode_binary_packet(packet: bytes) -> LidarFrame:
        if len(packet) < 4:
            raise ValueError("Binary packet too short.")

        header_len = int.from_bytes(packet[0:4], byteorder="little", signed=False)
        if header_len <= 0:
            raise ValueError("Invalid binary header length.")

        start = 4
        end = 4 + header_len
        if end > len(packet):
            raise ValueError("Binary packet header length exceeds packet size.")

        try:
            header = json.loads(packet[start:end].decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid binary header JSON: {exc}") from exc

        if not isinstance(header, dict):
            raise ValueError("Binary header must be a JSON object.")
        if header.get("type") != "lidar_frame":
            raise ValueError("Binary packet type must be lidar_frame.")

        seq = int(header["seq"])
        timestamp_ms = int(header["timestamp_ms"])
        width = int(header["width"])
        height = int(header["height"])

        payload = memoryview(packet[end:])

        depth_byte_count = int(header.get("depth_byte_count", 0))
        if depth_byte_count <= 0 or depth_byte_count > len(payload):
            raise ValueError("Invalid depth_byte_count in binary header.")

        confidence_byte_count = int(header.get("confidence_byte_count", 0))
        expected_total = depth_byte_count + max(0, confidence_byte_count)
        if expected_total > len(payload):
            raise ValueError("Binary payload smaller than declared byte counts.")

        depth_blob = bytes(payload[:depth_byte_count])
        confidence_blob = (
            bytes(payload[depth_byte_count : depth_byte_count + confidence_byte_count])
            if confidence_byte_count > 0
            else None
        )

        depth_encoding = str(header.get("depth_encoding", "u16_mm_raw"))
        depth_mm = _decode_depth_blob(depth_blob, depth_encoding, width, height)

        confidence = None
        confidence_encoding = str(header.get("confidence_encoding", ""))
        if confidence_blob is not None:
            confidence = _decode_confidence_blob(
                confidence_blob,
                confidence_encoding,
                width,
                height,
            )

        pose_raw = header.get("pose", {})
        intrinsics_raw = header.get("intrinsics", {})
        pose = dict(pose_raw if isinstance(pose_raw, dict) else {})
        intrinsics = dict(intrinsics_raw if isinstance(intrinsics_raw, dict) else {})

        return LidarFrame(
            seq=seq,
            timestamp_ms=timestamp_ms,
            width=width,
            height=height,
            depth_mm=depth_mm,
            confidence=confidence,
            pose=pose,
            intrinsics=intrinsics,
        )


def _decode_depth_blob(blob: bytes, encoding: str, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("Depth frame dimensions must be positive.")

    if encoding == "zlib_u16_mm":
        raw = zlib.decompress(blob)
    elif encoding == "u16_mm_raw":
        raw = blob
    else:
        raise ValueError(f"Unsupported depth encoding: {encoding}")

    expected = width * height * 2
    if len(raw) != expected:
        raise ValueError(
            f"Depth payload size mismatch. expected={expected} bytes, got={len(raw)} bytes"
        )

    arr = np.frombuffer(raw, dtype="<u2")
    return arr.reshape((height, width))


def _decode_confidence_blob(
    blob: bytes,
    encoding: str,
    width: int,
    height: int,
) -> np.ndarray:
    if encoding in ("", "none"):
        raise ValueError("Confidence bytes provided but confidence_encoding is missing.")

    if encoding == "zlib_u8":
        raw = zlib.decompress(blob)
    elif encoding == "u8_raw":
        raw = blob
    else:
        raise ValueError(f"Unsupported confidence encoding: {encoding}")

    expected = width * height
    if len(raw) != expected:
        raise ValueError(
            f"Confidence payload size mismatch. expected={expected} bytes, got={len(raw)} bytes"
        )

    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr.reshape((height, width))
