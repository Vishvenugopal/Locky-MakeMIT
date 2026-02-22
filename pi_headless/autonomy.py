from __future__ import annotations

import asyncio
import math
import random
import time
import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pi_headless.lidar_ingest import LidarFrame, LidarIngestor
from pi_headless.robot_output import ViamBaseController
from pi_headless.state_machine import PhaseStateMachine


@dataclass
class AutonomyConfig:
    loop_hz: float = 10.0
    max_pose_stale_s: float = 1.2
    mapping_max_duration_s: float = 90.0
    min_mapping_loops: int = 2
    scan_turn_rate_rps: float = 0.45
    cruise_speed_mps: float = 0.12
    waypoint_tolerance_m: float = 0.18
    min_hide_distance_m: float = 1.2
    grid_cell_size_m: float = 0.20
    max_depth_m: float = 4.0


@dataclass
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass
class OccupancyGrid2D:
    cell_size_m: float
    min_x_m: float
    min_y_m: float
    width: int
    height: int
    driveable_occ: np.ndarray  # bool [h, w]
    known_mask: np.ndarray  # bool [h, w]
    occupancy: np.ndarray  # int8 [h, w] where -1 unknown, 0 free, 1 occupied

    def world_to_grid(self, x_m: float, y_m: float) -> Optional[Tuple[int, int]]:
        gx = int((x_m - self.min_x_m) / self.cell_size_m)
        gy = int((y_m - self.min_y_m) / self.cell_size_m)
        if gx < 0 or gy < 0 or gx >= self.width or gy >= self.height:
            return None
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x_m = self.min_x_m + (gx + 0.5) * self.cell_size_m
        y_m = self.min_y_m + (gy + 0.5) * self.cell_size_m
        return x_m, y_m


class AutonomyEngine:
    """
    Runtime autonomy loop for the headless Pi service.

    This adapts the simulation phase semantics to real streamed LiDAR by:
    - integrating streamed depth+pose into a 2D occupancy representation,
    - frontier-style scan progression (sweep + navigate),
    - return-to-origin completion,
    - hide-target planning and navigation.
    """

    def __init__(
        self,
        config: AutonomyConfig,
        state_machine: PhaseStateMachine,
        lidar_ingestor: LidarIngestor,
        robot: ViamBaseController,
    ) -> None:
        self._cfg = config
        self._state = state_machine
        self._lidar = lidar_ingestor
        self._robot = robot

        self._rng = random.Random(23)

        self._stop_requested = False

        self._last_phase: str = ""
        self._latest_pose: Optional[Pose2D] = None
        self._latest_pose_received_s: Optional[float] = None
        self._origin_pose: Optional[Pose2D] = None

        self._free_evidence: Dict[Tuple[int, int], int] = {}
        self._occ_evidence: Dict[Tuple[int, int], int] = {}
        self._traversed_world: List[Tuple[float, float]] = []

        self._mapping_started_s: Optional[float] = None
        self._mapping_mode: str = "idle"  # sweep | navigate_frontier | return_home | idle
        self._mapping_loops: int = 0
        self._sweep_elapsed_s: float = 0.0
        self._return_replan_attempted: bool = False

        self._hide_target_xy: Optional[Tuple[float, float]] = None
        self._hide_initialized: bool = False

        self._nav_waypoints_xy: List[Tuple[float, float]] = []
        self._nav_index: int = 0
        self._nav_goal_label: str = ""
        self._nav_best_dist: float = float("inf")
        self._nav_last_progress_s: float = 0.0

        self._last_command_linear: float = 0.0
        self._last_command_angular: float = 0.0

    async def run(self) -> None:
        target_hz = max(1e-3, float(self._cfg.loop_hz))
        interval_s = 1.0 / target_hz
        last_loop_s = time.monotonic()

        while not self._stop_requested:
            loop_start_s = time.monotonic()
            dt_s = max(0.0, loop_start_s - last_loop_s)
            last_loop_s = loop_start_s

            try:
                self._consume_latest_lidar()
                self._step(loop_start_s, dt_s)
            except Exception as exc:
                self._state.update_mapping_state("idle", f"Autonomy loop error: {exc}")
                self._send_robot_command(0.0, 0.0, force=True)

            elapsed = time.monotonic() - loop_start_s
            sleep_s = max(0.0, interval_s - elapsed)
            if sleep_s > 0.0:
                await asyncio.sleep(sleep_s)

    def request_stop(self) -> None:
        self._stop_requested = True
        self._send_robot_command(0.0, 0.0, force=True)

    def debug_snapshot(self) -> Dict[str, Any]:
        return {
            "mapping_mode": self._mapping_mode,
            "mapping_loops": self._mapping_loops,
            "have_pose": self._latest_pose is not None,
            "have_origin": self._origin_pose is not None,
            "hide_initialized": self._hide_initialized,
            "have_hide_target": self._hide_target_xy is not None,
            "nav_goal": self._nav_goal_label,
            "nav_index": self._nav_index,
            "nav_count": len(self._nav_waypoints_xy),
            "map_free_cells": len(self._free_evidence),
            "map_occ_cells": len(self._occ_evidence),
        }

    def _step(self, now_s: float, dt_s: float) -> None:
        snapshot = self._state.snapshot()
        phase = str(snapshot.get("phase", "idle"))

        if phase != self._last_phase:
            self._on_phase_transition(phase)
            self._last_phase = phase

        if phase == "mapping":
            self._step_mapping(now_s, dt_s)
            return

        if phase == "hiding":
            self._step_hiding(now_s, dt_s)
            return

        self._send_robot_command(0.0, 0.0, force=False)

    def _on_phase_transition(self, phase: str) -> None:
        if phase == "mapping":
            self._reset_mapping_session()
            return

        if phase == "hiding":
            self._hide_initialized = False
            self._hide_target_xy = None
            self._clear_navigation()
            return

        self._clear_navigation()

    def _reset_mapping_session(self) -> None:
        self._mapping_started_s = time.monotonic()
        self._mapping_mode = "sweep"
        self._mapping_loops = 0
        self._sweep_elapsed_s = 0.0
        self._return_replan_attempted = False

        self._hide_initialized = False
        self._hide_target_xy = None

        self._free_evidence.clear()
        self._occ_evidence.clear()
        self._traversed_world.clear()

        self._clear_navigation()

        if self._latest_pose is not None:
            self._origin_pose = Pose2D(
                x_m=self._latest_pose.x_m,
                y_m=self._latest_pose.y_m,
                yaw_rad=self._latest_pose.yaw_rad,
            )
            self._record_traversed_world_point(self._latest_pose.x_m, self._latest_pose.y_m)
        else:
            self._origin_pose = None

        self._state.update_mapping_state("sweep", "Phase 1 started: scanning room.")

    def _step_mapping(self, now_s: float, dt_s: float) -> None:
        if self._mapping_started_s is None:
            self._reset_mapping_session()

        if self._latest_pose is None:
            self._send_robot_command(0.0, 0.0, force=False)
            self._state.update_mapping_state(self._mapping_mode, "Waiting for LiDAR pose before mapping.")
            return

        if self._is_pose_stale(now_s):
            self._send_robot_command(0.0, 0.0, force=False)
            self._state.update_mapping_state(
                self._mapping_mode,
                "LiDAR pose is stale. Pausing mapping motion until fresh frames arrive.",
            )
            return

        if self._origin_pose is None:
            self._origin_pose = Pose2D(
                x_m=self._latest_pose.x_m,
                y_m=self._latest_pose.y_m,
                yaw_rad=self._latest_pose.yaw_rad,
            )

        elapsed_s = now_s - max(0.0, self._mapping_started_s or now_s)
        if elapsed_s >= self._cfg.mapping_max_duration_s and self._mapping_mode != "return_home":
            self._start_return_home("Mapping complete (max scan duration reached). Returning to origin.")
            if self._state.snapshot().get("phase") != "mapping":
                return

        if self._mapping_mode == "sweep":
            self._step_mapping_sweep(dt_s)
            return

        if self._mapping_mode == "navigate_frontier":
            moved, finished, blocked = self._follow_navigation_path(now_s)
            if blocked:
                self._mapping_mode = "sweep"
                self._sweep_elapsed_s = 0.0
                self._clear_navigation()
                self._state.update_mapping_state("sweep", "Navigation blocked. Returning to sweep for replanning.")
                return
            if finished:
                self._mapping_mode = "sweep"
                self._sweep_elapsed_s = 0.0
                self._clear_navigation()
                self._state.update_mapping_state("sweep", "Reached frontier. Starting next sweep.")
                return
            if moved:
                return
            self._send_robot_command(0.0, 0.0, force=False)
            return

        if self._mapping_mode == "return_home":
            moved, finished, blocked = self._follow_navigation_path(now_s)
            if blocked:
                if not self._return_replan_attempted:
                    self._return_replan_attempted = True
                    self._start_return_home("Return-home path blocked. Replanning.")
                    return
                self._send_robot_command(0.0, 0.0, force=True)
                self._state.mark_scan_complete(returned_to_origin=False)
                self._state.update_mapping_state("done", "Mapping complete. Could not return to origin.")
                return

            if finished:
                self._send_robot_command(0.0, 0.0, force=True)
                self._state.mark_scan_complete(returned_to_origin=True)
                self._state.update_mapping_state("done", "Mapping complete. Robot returned to origin.")
                self._clear_navigation()
                return

            if moved:
                return
            self._send_robot_command(0.0, 0.0, force=False)
            return

        # Unknown mode fallback.
        self._mapping_mode = "sweep"
        self._sweep_elapsed_s = 0.0
        self._state.update_mapping_state("sweep", "Resetting mapping to sweep mode.")

    def _step_mapping_sweep(self, dt_s: float) -> None:
        turn_rate = float(self._cfg.scan_turn_rate_rps)
        self._send_robot_command(0.0, turn_rate, force=False)
        self._sweep_elapsed_s += max(0.0, dt_s)

        sweep_duration_s = (2.0 * math.pi) / max(0.10, abs(turn_rate))
        if self._sweep_elapsed_s < sweep_duration_s:
            self._state.update_mapping_state("sweep", "Scanning room (360 sweep).")
            return

        self._sweep_elapsed_s = 0.0
        self._mapping_loops += 1

        grid = self._build_occupancy_grid()
        if grid is None:
            self._state.update_mapping_state("sweep", "Collecting LiDAR evidence before planning.")
            return

        traversable = self._traversable_mask(grid)
        start_cell = grid.world_to_grid(self._latest_pose.x_m, self._latest_pose.y_m) if self._latest_pose else None
        if start_cell is None:
            self._state.update_mapping_state("sweep", "Robot pose outside map bounds. Continuing sweep.")
            return

        plan = self._choose_frontier_path(grid, traversable, start_cell)
        if plan is not None:
            target_cell, path_cells = plan
            self._set_navigation_path(path_cells, grid, "frontier")
            self._mapping_mode = "navigate_frontier"
            tx, ty = grid.grid_to_world(target_cell[0], target_cell[1])
            self._state.update_mapping_state(
                "navigate_frontier",
                f"Frontier target planned at ({tx:.2f}, {ty:.2f}). Navigating...",
            )
            return

        if self._mapping_loops >= self._cfg.min_mapping_loops:
            self._start_return_home("Mapping complete (frontiers exhausted). Returning to origin.")
        else:
            self._state.update_mapping_state(
                "sweep",
                f"No reachable frontier yet. Continuing sweep ({self._mapping_loops}/{self._cfg.min_mapping_loops}).",
            )

    def _start_return_home(self, reason: str) -> None:
        if self._latest_pose is None or self._origin_pose is None:
            self._state.mark_scan_complete(returned_to_origin=False)
            self._state.update_mapping_state("done", "Mapping complete. Missing pose for return-home.")
            self._send_robot_command(0.0, 0.0, force=True)
            return

        grid = self._build_occupancy_grid()
        if grid is None:
            self._state.mark_scan_complete(returned_to_origin=False)
            self._state.update_mapping_state("done", "Mapping complete. Occupancy map unavailable for return-home.")
            self._send_robot_command(0.0, 0.0, force=True)
            return

        traversable = self._traversable_mask(grid)
        start_cell = grid.world_to_grid(self._latest_pose.x_m, self._latest_pose.y_m)
        home_cell = grid.world_to_grid(self._origin_pose.x_m, self._origin_pose.y_m)

        if start_cell is None or home_cell is None:
            self._state.mark_scan_complete(returned_to_origin=False)
            self._state.update_mapping_state("done", "Mapping complete. Origin outside known map.")
            self._send_robot_command(0.0, 0.0, force=True)
            return

        nav_map = traversable.copy()
        nav_map[start_cell[1], start_cell[0]] = True
        nav_map[home_cell[1], home_cell[0]] = True

        path_cells = self._astar_path(nav_map, start_cell, home_cell)
        if path_cells is None or len(path_cells) < 2:
            self._state.mark_scan_complete(returned_to_origin=False)
            self._state.update_mapping_state("done", "Mapping complete. Could not path back to origin.")
            self._send_robot_command(0.0, 0.0, force=True)
            return

        self._set_navigation_path(path_cells, grid, "origin")
        self._mapping_mode = "return_home"
        self._state.update_mapping_state("return_home", reason)

    def _step_hiding(self, now_s: float, _dt_s: float) -> None:
        if self._latest_pose is None:
            self._send_robot_command(0.0, 0.0, force=False)
            self._state.update_mapping_state("idle", "Waiting for LiDAR pose before hide navigation.")
            return

        if self._is_pose_stale(now_s):
            self._send_robot_command(0.0, 0.0, force=False)
            self._state.update_mapping_state("idle", "LiDAR pose is stale. Pausing hide motion.")
            return

        if self._origin_pose is None:
            self._origin_pose = Pose2D(
                x_m=self._latest_pose.x_m,
                y_m=self._latest_pose.y_m,
                yaw_rad=self._latest_pose.yaw_rad,
            )

        if not self._hide_initialized:
            self._hide_initialized = True
            planned = self._plan_hide_path()
            if planned is None:
                self._send_robot_command(0.0, 0.0, force=True)
                self._state.mark_hide_complete()
                self._state.update_mapping_state("idle", "Hide complete (fallback at current pose).")
                return

            target_xy, path_cells, grid = planned
            self._hide_target_xy = target_xy
            self._set_navigation_path(path_cells, grid, "hide target")
            self._state.update_mapping_state(
                "idle",
                f"Phase 2 started. Hide target selected at ({target_xy[0]:.2f}, {target_xy[1]:.2f}).",
            )

        moved, finished, blocked = self._follow_navigation_path(now_s)
        if blocked:
            replanned = self._replan_hide_path()
            if replanned is None:
                self._send_robot_command(0.0, 0.0, force=True)
                self._state.mark_hide_complete()
                self._state.update_mapping_state("idle", "Hide complete (no alternate route found).")
                return
            return

        if finished:
            self._send_robot_command(0.0, 0.0, force=True)
            self._state.mark_hide_complete()
            self._state.update_mapping_state("idle", "Hide complete. Robot reached hiding spot.")
            self._clear_navigation()
            return

        if moved:
            return

        self._send_robot_command(0.0, 0.0, force=False)

    def _plan_hide_path(self) -> Optional[Tuple[Tuple[float, float], List[Tuple[int, int]], OccupancyGrid2D]]:
        if self._latest_pose is None:
            return None

        grid = self._build_occupancy_grid()
        if grid is None:
            return None

        traversable = self._traversable_mask(grid)
        start_cell = grid.world_to_grid(self._latest_pose.x_m, self._latest_pose.y_m)
        if start_cell is None:
            return None

        plan = self._choose_hiding_target_with_path(grid, traversable, start_cell)
        if plan is None:
            return None

        target_cell, path_cells = plan
        target_xy = grid.grid_to_world(target_cell[0], target_cell[1])
        return target_xy, path_cells, grid

    def _replan_hide_path(self) -> Optional[bool]:
        if self._latest_pose is None or self._hide_target_xy is None:
            return None

        grid = self._build_occupancy_grid()
        if grid is None:
            return None

        traversable = self._traversable_mask(grid)
        start_cell = grid.world_to_grid(self._latest_pose.x_m, self._latest_pose.y_m)
        goal_cell = grid.world_to_grid(self._hide_target_xy[0], self._hide_target_xy[1])
        if start_cell is None or goal_cell is None:
            return None

        nav_map = traversable.copy()
        nav_map[start_cell[1], start_cell[0]] = True
        nav_map[goal_cell[1], goal_cell[0]] = True

        path_cells = self._astar_path(nav_map, start_cell, goal_cell)
        if path_cells is None or len(path_cells) < 2:
            return None

        self._set_navigation_path(path_cells, grid, "hide target")
        self._state.update_mapping_state("idle", "Hide path re-planned around obstacle.")
        return True

    def _follow_navigation_path(self, now_s: float) -> Tuple[bool, bool, bool]:
        if self._latest_pose is None:
            return False, False, False

        if self._nav_index >= len(self._nav_waypoints_xy):
            return False, True, False

        while self._nav_index < len(self._nav_waypoints_xy):
            tx, ty = self._nav_waypoints_xy[self._nav_index]
            dx = tx - self._latest_pose.x_m
            dy = ty - self._latest_pose.y_m
            dist = math.hypot(dx, dy)

            if dist < self._cfg.waypoint_tolerance_m:
                self._nav_index += 1
                self._nav_best_dist = float("inf")
                self._nav_last_progress_s = now_s
                continue

            if dist < self._nav_best_dist - 0.02:
                self._nav_best_dist = dist
                self._nav_last_progress_s = now_s
            elif (now_s - self._nav_last_progress_s) > 4.0:
                self._send_robot_command(0.0, 0.0, force=False)
                return False, False, True

            desired_yaw = math.atan2(dy, dx)
            yaw_err = wrap_angle(desired_yaw - self._latest_pose.yaw_rad)

            if abs(yaw_err) > math.radians(12.0):
                angular_cmd = clip(yaw_err * 1.8, -self._cfg.scan_turn_rate_rps, self._cfg.scan_turn_rate_rps)
                self._send_robot_command(0.0, angular_cmd, force=False)
                return True, False, False

            linear_cmd = min(self._cfg.cruise_speed_mps, max(0.04, dist * 0.9))
            angular_cmd = clip(yaw_err * 1.1, -0.8, 0.8)
            self._send_robot_command(linear_cmd, angular_cmd, force=False)
            return True, False, False

        return False, True, False

    def _set_navigation_path(
        self,
        path_cells: List[Tuple[int, int]],
        grid: OccupancyGrid2D,
        goal_label: str,
    ) -> None:
        if len(path_cells) <= 1:
            self._nav_waypoints_xy = []
            self._nav_index = 0
            self._nav_goal_label = goal_label
            self._nav_best_dist = float("inf")
            self._nav_last_progress_s = time.monotonic()
            return

        self._nav_waypoints_xy = [
            grid.grid_to_world(gx, gy)
            for gx, gy in path_cells[1:]
        ]
        self._nav_index = 0
        self._nav_goal_label = goal_label
        self._nav_best_dist = float("inf")
        self._nav_last_progress_s = time.monotonic()

    def _clear_navigation(self) -> None:
        self._nav_waypoints_xy = []
        self._nav_index = 0
        self._nav_goal_label = ""
        self._nav_best_dist = float("inf")
        self._nav_last_progress_s = time.monotonic()

    def _consume_latest_lidar(self) -> None:
        frame = self._lidar.get_latest_nowait()
        if frame is None:
            return

        pose = self._pose_from_frame(frame)
        if pose is None:
            return

        self._latest_pose = pose
        self._latest_pose_received_s = time.monotonic()
        self._record_traversed_world_point(pose.x_m, pose.y_m)

        phase = self._state.snapshot().get("phase", "idle")
        if phase in ("mapping", "hiding"):
            self._integrate_frame_2d(frame, pose)

    def _integrate_frame_2d(self, frame: LidarFrame, pose: Pose2D) -> None:
        depth = frame.depth_mm
        if depth.size == 0:
            return

        intr = frame.intrinsics or {}
        fx = float(intr.get("fx", 0.0) or 0.0)
        cx = float(intr.get("cx", 0.0) or 0.0)
        if fx <= 1e-6:
            fx = max(10.0, 0.5 * depth.shape[1])
        if not math.isfinite(cx) or cx <= 0.0:
            cx = 0.5 * float(depth.shape[1])

        h, w = depth.shape
        stride = max(2, int(max(h, w) / 80))

        src_cell = self._world_cell(pose.x_m, pose.y_m)

        sample_budget = 1200
        sample_count = 0

        for v in range(0, h, stride):
            row = depth[v]
            for u in range(0, w, stride):
                depth_mm = int(row[u])
                if depth_mm <= 80:
                    continue

                z_m = min(self._cfg.max_depth_m, depth_mm * 0.001)
                if z_m <= 0.08:
                    continue

                # 2D horizontal projection from image bearing + robot heading.
                theta = math.atan2((float(u) - cx), fx)
                world_heading = pose.yaw_rad + theta

                hit_x = pose.x_m + math.cos(world_heading) * z_m
                hit_y = pose.y_m + math.sin(world_heading) * z_m
                hit_cell = self._world_cell(hit_x, hit_y)

                ray = bresenham_line(src_cell[0], src_cell[1], hit_cell[0], hit_cell[1])
                if len(ray) > 1:
                    for rx, ry in ray[:-1]:
                        key = (int(rx), int(ry))
                        self._free_evidence[key] = self._free_evidence.get(key, 0) + 1

                key_hit = (int(hit_cell[0]), int(hit_cell[1]))
                self._occ_evidence[key_hit] = self._occ_evidence.get(key_hit, 0) + 1

                sample_count += 1
                if sample_count >= sample_budget:
                    return

    def _pose_from_frame(self, frame: LidarFrame) -> Optional[Pose2D]:
        pose_raw = frame.pose or {}
        try:
            tx = float(pose_raw.get("tx"))
            tz = float(pose_raw.get("tz"))
            qx = float(pose_raw.get("qx"))
            qy = float(pose_raw.get("qy"))
            qz = float(pose_raw.get("qz"))
            qw = float(pose_raw.get("qw"))
        except Exception:
            return None

        if not all(math.isfinite(v) for v in (tx, tz, qx, qy, qz, qw)):
            return None

        yaw = quaternion_to_planar_yaw(qx=qx, qy=qy, qz=qz, qw=qw)
        return Pose2D(x_m=tx, y_m=tz, yaw_rad=yaw)

    def _build_occupancy_grid(self) -> Optional[OccupancyGrid2D]:
        keys = set(self._free_evidence.keys()) | set(self._occ_evidence.keys())

        if self._latest_pose is not None:
            keys.add(self._world_cell(self._latest_pose.x_m, self._latest_pose.y_m))
        if self._origin_pose is not None:
            keys.add(self._world_cell(self._origin_pose.x_m, self._origin_pose.y_m))
        if self._hide_target_xy is not None:
            keys.add(self._world_cell(self._hide_target_xy[0], self._hide_target_xy[1]))

        if len(keys) < 60:
            return None

        gx_values = [c[0] for c in keys]
        gy_values = [c[1] for c in keys]

        pad_cells = 6
        min_gx = min(gx_values) - pad_cells
        max_gx = max(gx_values) + pad_cells
        min_gy = min(gy_values) - pad_cells
        max_gy = max(gy_values) + pad_cells

        width = int(max_gx - min_gx + 1)
        height = int(max_gy - min_gy + 1)
        if width < 8 or height < 8:
            return None

        known_mask = np.zeros((height, width), dtype=bool)
        driveable_occ = np.zeros((height, width), dtype=bool)
        occupancy = np.full((height, width), -1, dtype=np.int8)

        for (cgx, cgy) in keys:
            ix = int(cgx - min_gx)
            iy = int(cgy - min_gy)
            if ix < 0 or iy < 0 or ix >= width or iy >= height:
                continue

            free_score = int(self._free_evidence.get((cgx, cgy), 0))
            occ_score = int(self._occ_evidence.get((cgx, cgy), 0))
            if free_score <= 0 and occ_score <= 0:
                continue

            known_mask[iy, ix] = True

            if occ_score >= max(1, int(round(0.7 * free_score))):
                driveable_occ[iy, ix] = True

        occupancy[known_mask] = 0
        occupancy[driveable_occ] = 1

        # Inject traversed memory as weak free evidence to keep connectivity.
        for wx, wy in self._traversed_world[-6000:]:
            ix = int(math.floor(wx / self._cfg.grid_cell_size_m)) - min_gx
            iy = int(math.floor(wy / self._cfg.grid_cell_size_m)) - min_gy
            if ix < 0 or iy < 0 or ix >= width or iy >= height:
                continue
            if driveable_occ[iy, ix]:
                continue
            known_mask[iy, ix] = True
            occupancy[iy, ix] = 0

        return OccupancyGrid2D(
            cell_size_m=float(self._cfg.grid_cell_size_m),
            min_x_m=float(min_gx * self._cfg.grid_cell_size_m),
            min_y_m=float(min_gy * self._cfg.grid_cell_size_m),
            width=width,
            height=height,
            driveable_occ=driveable_occ,
            known_mask=known_mask,
            occupancy=occupancy,
        )

    def _traversable_mask(self, grid: OccupancyGrid2D) -> np.ndarray:
        traversable = (grid.occupancy == 0) & (~grid.driveable_occ)

        # Inflate occupied cells by a conservative clearance radius.
        clearance_m = 0.24
        clearance_cells = max(0, int(math.ceil(clearance_m / grid.cell_size_m)) - 1)
        if clearance_cells <= 0:
            return traversable

        blocked = grid.driveable_occ
        inflated = blocked.copy()
        height, width = blocked.shape

        for dy in range(-clearance_cells, clearance_cells + 1):
            for dx in range(-clearance_cells, clearance_cells + 1):
                if dx == 0 and dy == 0:
                    continue
                if (dx * dx + dy * dy) > (clearance_cells * clearance_cells):
                    continue

                src_y0 = max(0, -dy)
                src_y1 = height - max(0, dy)
                src_x0 = max(0, -dx)
                src_x1 = width - max(0, dx)

                dst_y0 = max(0, dy)
                dst_y1 = height - max(0, -dy)
                dst_x0 = max(0, dx)
                dst_x1 = width - max(0, -dx)

                inflated[dst_y0:dst_y1, dst_x0:dst_x1] |= blocked[src_y0:src_y1, src_x0:src_x1]

        return traversable & (~inflated)

    def _frontier_mask(self, grid: OccupancyGrid2D) -> np.ndarray:
        unknown = grid.occupancy == -1
        free = grid.occupancy == 0

        adjacent_unknown = np.zeros_like(unknown, dtype=bool)
        adjacent_unknown[1:, :] |= unknown[:-1, :]
        adjacent_unknown[:-1, :] |= unknown[1:, :]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]

        return free & adjacent_unknown

    def _extract_frontier_clusters(self, grid: OccupancyGrid2D) -> List[List[Tuple[int, int]]]:
        frontier_mask = self._frontier_mask(grid)
        if not np.any(frontier_mask):
            return []

        visited = np.zeros_like(frontier_mask, dtype=bool)
        clusters: List[List[Tuple[int, int]]] = []
        height, width = frontier_mask.shape

        for y in range(height):
            for x in range(width):
                if not frontier_mask[y, x] or visited[y, x]:
                    continue

                cluster: List[Tuple[int, int]] = []
                stack = [(x, y)]
                visited[y, x] = True

                while stack:
                    cx, cy = stack.pop()
                    cluster.append((cx, cy))
                    for nx, ny in (
                        (cx + 1, cy),
                        (cx - 1, cy),
                        (cx, cy + 1),
                        (cx, cy - 1),
                    ):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        if visited[ny, nx] or not frontier_mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))

                clusters.append(cluster)

        clusters.sort(key=len, reverse=True)
        return clusters

    @staticmethod
    def _local_egress_count(nav_map: np.ndarray, cell: Tuple[int, int]) -> int:
        cx, cy = cell
        h, w = nav_map.shape
        count = 0
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if nav_map[ny, nx]:
                count += 1
        return count

    def _choose_frontier_path(
        self,
        grid: OccupancyGrid2D,
        traversable: np.ndarray,
        start_cell: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
        nav_map = traversable.copy()
        nav_map[start_cell[1], start_cell[0]] = True

        clusters = self._extract_frontier_clusters(grid)
        if not clusters:
            return self._choose_exploration_fallback_path(grid, nav_map, start_cell)

        for cluster in clusters:
            arr = np.asarray(cluster, dtype=np.float64)
            centroid = arr.mean(axis=0)
            cluster_sorted = sorted(
                cluster,
                key=lambda c: (c[0] - centroid[0]) ** 2 + (c[1] - centroid[1]) ** 2,
            )

            for gx, gy in cluster_sorted[: min(len(cluster_sorted), 90)]:
                if not nav_map[gy, gx]:
                    continue
                if self._local_egress_count(nav_map, (gx, gy)) < 2:
                    continue

                path_cells = self._astar_path(nav_map, start_cell, (gx, gy))
                if path_cells is None or len(path_cells) < 2:
                    continue
                return (gx, gy), path_cells

        return self._choose_exploration_fallback_path(grid, nav_map, start_cell)

    def _choose_exploration_fallback_path(
        self,
        grid: OccupancyGrid2D,
        nav_map: np.ndarray,
        start_cell: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
        unknown = grid.occupancy == -1
        if not np.any(unknown):
            return None

        neighbor_unknown = np.zeros_like(grid.occupancy, dtype=np.int16)
        unknown_i = unknown.astype(np.int16)
        neighbor_unknown[1:, :] += unknown_i[:-1, :]
        neighbor_unknown[:-1, :] += unknown_i[1:, :]
        neighbor_unknown[:, 1:] += unknown_i[:, :-1]
        neighbor_unknown[:, :-1] += unknown_i[:, 1:]

        candidates = np.argwhere(nav_map)
        scored: List[Tuple[float, Tuple[int, int]]] = []
        sx, sy = start_cell
        for gy, gx in candidates:
            gx_i = int(gx)
            gy_i = int(gy)
            if gx_i == sx and gy_i == sy:
                continue

            egress = self._local_egress_count(nav_map, (gx_i, gy_i))
            if egress < 2:
                continue

            frontier_pull = float(neighbor_unknown[gy_i, gx_i])
            if frontier_pull <= 0.0:
                continue

            dist2 = float((gx_i - sx) ** 2 + (gy_i - sy) ** 2)
            score = (frontier_pull * 90.0) + (dist2 * 0.40) + (egress * 25.0)
            scored.append((score, (gx_i, gy_i)))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        for _, candidate in scored[:220]:
            path_cells = self._astar_path(nav_map, start_cell, candidate)
            if path_cells is None or len(path_cells) < 2:
                continue
            return candidate, path_cells

        return None

    def _choose_hiding_target_with_path(
        self,
        grid: OccupancyGrid2D,
        traversable: np.ndarray,
        start_cell: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
        if self._origin_pose is None:
            return None

        origin_cell = grid.world_to_grid(self._origin_pose.x_m, self._origin_pose.y_m)
        if origin_cell is None:
            return None

        nav_map = traversable.copy()
        nav_map[start_cell[1], start_cell[0]] = True

        candidate_mask = traversable & grid.known_mask
        candidates = np.argwhere(candidate_mask)
        if candidates.shape[0] == 0:
            if nav_map[start_cell[1], start_cell[0]]:
                return start_cell, [start_cell]
            return None

        self._rng.shuffle(candidates)

        min_cells = max(1, int(self._cfg.min_hide_distance_m / grid.cell_size_m))
        scored: List[Tuple[float, Tuple[int, int]]] = []

        ox, oy = origin_cell

        for gy, gx in candidates:
            gx_i = int(gx)
            gy_i = int(gy)

            dx = gx_i - int(ox)
            dy = gy_i - int(oy)
            dist2 = float(dx * dx + dy * dy)
            if dist2 < float(min_cells * min_cells):
                continue

            cover_bonus = 0.0
            for cx, cy in bresenham_line(int(ox), int(oy), gx_i, gy_i):
                if (cx == ox and cy == oy) or (cx == gx_i and cy == gy_i):
                    continue
                if grid.driveable_occ[cy, cx]:
                    cover_bonus = 220.0
                    break

            egress = float(self._local_egress_count(nav_map, (gx_i, gy_i)))
            score = dist2 + cover_bonus + (egress * 10.0)
            scored.append((score, (gx_i, gy_i)))

            if len(scored) >= 3500:
                break

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        reachable: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = []

        for _, target in scored[:520]:
            path_cells = self._astar_path(nav_map, start_cell, target)
            if path_cells is None or len(path_cells) < 2:
                continue
            reachable.append((target, path_cells))
            if len(reachable) >= 40:
                break

        if not reachable:
            return None

        pool = reachable[: min(18, len(reachable))]
        return self._rng.choice(pool)

    def _astar_path(
        self,
        traversable: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        width = traversable.shape[1]
        height = traversable.shape[0]

        sx, sy = start
        gx, gy = goal

        if sx < 0 or sy < 0 or sx >= width or sy >= height:
            return None
        if gx < 0 or gy < 0 or gx >= width or gy >= height:
            return None
        if not traversable[gy, gx]:
            return None

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed: set[Tuple[int, int]] = set()

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed.add(current)
            cx, cy = current

            for nx, ny in (
                (cx + 1, cy),
                (cx - 1, cy),
                (cx, cy + 1),
                (cx, cy - 1),
            ):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if not traversable[ny, nx] and (nx, ny) != goal:
                    continue

                tentative_g = current_cost + 1.0
                if tentative_g >= g_score.get((nx, ny), float("inf")):
                    continue

                g_score[(nx, ny)] = tentative_g
                came_from[(nx, ny)] = current
                f_score = tentative_g + heuristic((nx, ny), goal)
                heapq.heappush(open_heap, (f_score, tentative_g, (nx, ny)))

        return None

    def _record_traversed_world_point(self, x_m: float, y_m: float) -> None:
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            return
        self._traversed_world.append((float(x_m), float(y_m)))
        if len(self._traversed_world) > 9000:
            self._traversed_world = self._traversed_world[-9000:]

    def _is_pose_stale(self, now_s: float) -> bool:
        if self._latest_pose_received_s is None:
            return True
        stale_for_s = max(0.0, now_s - self._latest_pose_received_s)
        return stale_for_s > max(0.1, float(self._cfg.max_pose_stale_s))

    def _world_cell(self, x_m: float, y_m: float) -> Tuple[int, int]:
        gx = int(math.floor(float(x_m) / self._cfg.grid_cell_size_m))
        gy = int(math.floor(float(y_m) / self._cfg.grid_cell_size_m))
        return gx, gy

    def _send_robot_command(self, linear_mps: float, angular_rps: float, force: bool = False) -> None:
        linear_mps = float(linear_mps)
        angular_rps = float(angular_rps)

        if (not force
            and abs(linear_mps - self._last_command_linear) < 1e-4
            and abs(angular_rps - self._last_command_angular) < 1e-4):
            return

        if not self._robot.is_connected:
            self._last_command_linear = 0.0
            self._last_command_angular = 0.0
            return

        ok, msg = self._robot.send_velocity(linear_mps=linear_mps, angular_rps=angular_rps, force=force)
        if not ok and msg:
            self._state.update_mapping_state("idle", f"Robot command failed: {msg}")
            self._last_command_linear = 0.0
            self._last_command_angular = 0.0
            return

        self._last_command_linear = linear_mps
        self._last_command_angular = angular_rps


def quaternion_to_planar_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    # Rotation matrix third column components.
    r02 = 2.0 * ((qx * qz) + (qy * qw))
    r22 = 1.0 - (2.0 * ((qx * qx) + (qy * qy)))

    # ARKit camera forward points along negative Z in camera coordinates.
    fwd_x = -r02
    fwd_z = -r22

    return math.atan2(fwd_z, fwd_x)


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy
    x, y = x0, y0

    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return points
