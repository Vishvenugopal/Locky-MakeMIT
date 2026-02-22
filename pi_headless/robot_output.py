from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class RobotControlConfig:
    robot_address: str
    api_key_id: str
    api_key: str
    base_name: str = "viam_base"
    linear_speed_limit_mps: float = 0.30
    angular_speed_limit_rps: float = 1.20
    publish_interval_s: float = 0.10


class ViamBaseController:
    """Async-safe Viam base wrapper for the headless Pi service."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()

        self._robot: Any = None
        self._base: Any = None
        self._vector3_type: Any = None
        self._config: Optional[RobotControlConfig] = None

        self._last_publish_s = 0.0

    @property
    def is_connected(self) -> bool:
        return self._config is not None and self._base is not None

    def _ensure_loop(self) -> None:
        if self._loop is not None and self._loop_thread is not None and self._loop_thread.is_alive():
            return

        self._loop_ready.clear()

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_ready.set()
            loop.run_forever()
            loop.close()

        self._loop_thread = threading.Thread(target=_runner, name="PiViamLoop", daemon=True)
        self._loop_thread.start()

        if not self._loop_ready.wait(timeout=2.0):
            raise RuntimeError("Timed out starting async loop for robot control.")

    def _run_coro(self, coro: Any, timeout_s: float) -> Any:
        self._ensure_loop()
        if self._loop is None:
            raise RuntimeError("Async event loop is unavailable.")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout_s)

    async def _connect_async(self, config: RobotControlConfig) -> None:
        from viam.components.base import Base, Vector3
        from viam.robot.client import RobotClient

        opts = RobotClient.Options.with_api_key(
            api_key=config.api_key,
            api_key_id=config.api_key_id,
        )
        robot = await RobotClient.at_address(config.robot_address, opts)
        base = Base.from_robot(robot=robot, name=config.base_name)

        self._robot = robot
        self._base = base
        self._vector3_type = Vector3

    async def _send_power_async(self, linear_power: float, angular_power: float) -> None:
        if self._base is None or self._vector3_type is None:
            raise RuntimeError("Robot base is not connected.")

        vector3 = self._vector3_type
        await self._base.set_power(
            linear=vector3(x=0.0, y=float(linear_power), z=0.0),
            angular=vector3(x=0.0, y=0.0, z=float(angular_power)),
        )

    async def _disconnect_async(self) -> None:
        try:
            if self._base is not None:
                await self._send_power_async(0.0, 0.0)
        except Exception:
            pass

        try:
            if self._robot is not None:
                await self._robot.close()
        except Exception:
            pass

        self._robot = None
        self._base = None
        self._vector3_type = None
        self._config = None
        self._last_publish_s = 0.0

    def connect(self, config: RobotControlConfig) -> Tuple[bool, str]:
        if not config.robot_address.strip():
            return False, "robot_address is required"
        if not config.api_key_id.strip():
            return False, "api_key_id is required"
        if not config.api_key.strip():
            return False, "api_key is required"
        if not config.base_name.strip():
            return False, "base_name is required"

        try:
            self._run_coro(self._disconnect_async(), timeout_s=4.0)
            self._run_coro(self._connect_async(config), timeout_s=15.0)
            self._config = config
            self._last_publish_s = 0.0
            return True, f"Connected to base '{config.base_name}'."
        except ModuleNotFoundError:
            return False, "viam-sdk is not installed. Install with: python -m pip install viam-sdk"
        except concurrent.futures.TimeoutError:
            return False, "Timed out connecting to Viam robot."
        except Exception as exc:
            try:
                self._run_coro(self._disconnect_async(), timeout_s=2.0)
            except Exception:
                pass
            return False, f"Failed to connect: {exc}"

    def send_velocity(
        self,
        linear_mps: float,
        angular_rps: float,
        force: bool = False,
    ) -> Tuple[bool, str]:
        cfg = self._config
        if cfg is None:
            return False, "Robot controller is not connected."

        now = time.perf_counter()
        if not force and (now - self._last_publish_s) < max(0.01, float(cfg.publish_interval_s)):
            return True, ""

        linear_limit = max(1e-6, float(cfg.linear_speed_limit_mps))
        angular_limit = max(1e-6, float(cfg.angular_speed_limit_rps))

        linear_clamped = max(-linear_limit, min(linear_limit, float(linear_mps)))
        angular_clamped = max(-angular_limit, min(angular_limit, float(angular_rps)))

        linear_power = linear_clamped / linear_limit
        angular_power = angular_clamped / angular_limit

        try:
            self._run_coro(self._send_power_async(linear_power, angular_power), timeout_s=4.0)
            self._last_publish_s = now
            return True, ""
        except concurrent.futures.TimeoutError:
            return False, "Timed out sending velocity command."
        except Exception as exc:
            return False, f"Failed to send velocity command: {exc}"

    def stop(self) -> Tuple[bool, str]:
        if self._base is None:
            return True, ""
        try:
            self._run_coro(self._send_power_async(0.0, 0.0), timeout_s=4.0)
            return True, ""
        except Exception as exc:
            return False, f"Failed to stop robot: {exc}"

    def close(self) -> None:
        if self._loop is None:
            self._robot = None
            self._base = None
            self._vector3_type = None
            self._config = None
            self._last_publish_s = 0.0
            return

        try:
            self._run_coro(self._disconnect_async(), timeout_s=4.0)
        except Exception:
            pass

        loop = self._loop
        loop_thread = self._loop_thread

        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass

        if loop_thread is not None and loop_thread.is_alive():
            loop_thread.join(timeout=1.5)

        self._loop = None
        self._loop_thread = None
        self._loop_ready.clear()
