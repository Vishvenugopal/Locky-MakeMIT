"""
Simulated phone app + robot exploration/hiding sandbox.

This script lets you:
1) Select one of the local .ply scans.
2) View it in an Open3D 3D window with faded base points.
3) Drive a simulated differential-drive robot.
4) Run a 360-degree scan pass that reveals full-color points.
5) Run a hiding-spot selection phase using a 2-tier occupancy map and
   Bresenham line-of-sight checks.

Requirements:
    pip install open3d numpy

Run:
    python phone_app_ui.py
"""

from __future__ import annotations

import math
import random
import time
import heapq
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "open3d is required. Install with: pip install open3d"
    ) from exc


FEET_TO_METERS = 0.3048
INCH_TO_METERS = 0.0254

DRIVEABLE_MAX_HEIGHT_M = 1.0 * FEET_TO_METERS
SHIELD_MAX_HEIGHT_M = 6.0 * FEET_TO_METERS

ROBOT_WIDTH_M = 9.0 * INCH_TO_METERS
ROBOT_LENGTH_M = 12.0 * INCH_TO_METERS
ROBOT_BODY_HEIGHT_M = 4.0 * INCH_TO_METERS
PHONE_MOUNT_HEIGHT_M = 3.0 * INCH_TO_METERS
LIDAR_SENSOR_HEIGHT_M = ROBOT_BODY_HEIGHT_M + PHONE_MOUNT_HEIGHT_M


def wrap_angle(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Integer grid line between two points (inclusive)."""
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


@dataclass
class RobotState:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0


@dataclass
class TwoTierGrid:
    cell_size_m: float
    min_x_m: float
    min_y_m: float
    width: int
    height: int
    driveable_occ: np.ndarray  # bool [h, w]
    shield_occ: np.ndarray  # bool [h, w]
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


class SimulationModel:
    """Holds simulation state, map logic, and robot state machine."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.scene_files: Dict[str, Path] = {}
        self.refresh_scene_files()

        self.points_xyz = np.zeros((0, 3), dtype=np.float64)
        self.colors_rgb = np.zeros((0, 3), dtype=np.float64)
        self.revealed_mask = np.zeros((0,), dtype=bool)
        self.reveal_source_xy = np.zeros((0, 2), dtype=np.float64)

        self.robot = RobotState()
        self.origin_xy = np.array([0.0, 0.0], dtype=np.float64)
        self.origin_yaw_rad = 0.0

        self.phase = "idle"  # idle | mapping | mapped | hiding | hidden
        self.mapping_state = "idle"  # idle | sweep | navigate_frontier | return_home | done
        self.status_text = "Load a PLY scene to begin."

        self.manual_forward = 0  # -1, 0, +1
        self.manual_turn = 0  # -1, 0, +1

        self.linear_speed_mps = 0.55
        self.angular_speed_rps = math.radians(95.0)

        self.robot_width_m = ROBOT_WIDTH_M
        self.robot_length_m = ROBOT_LENGTH_M
        self.robot_body_height_m = ROBOT_BODY_HEIGHT_M
        self.sensor_height_m = LIDAR_SENSOR_HEIGHT_M

        self.scan_turn_rate_rps = math.radians(32.0)
        self.scan_fov_deg = 52.0
        self.scan_vert_fov_deg = 64.0
        self.scan_range_m = 2.5
        self.scan_occlusion_bin_deg = 1.0
        self.scan_occlusion_slack_m = 0.10
        self.scan_accumulated_rad = 0.0
        self.mapping_loops = 0
        self.max_mapping_loops = 36

        self.hide_target_xy: Optional[Tuple[float, float]] = None

        self.min_hide_distance_m = 1.2
        self.grid_cell_size_m = 0.12

        self.nav_waypoints_xy: List[Tuple[float, float]] = []
        self.nav_index = 0
        self.nav_grid: Optional[TwoTierGrid] = None
        self.nav_traversable: Optional[np.ndarray] = None
        self.nav_goal_label = ""

        self.latest_grid: Optional[TwoTierGrid] = None
        self.latest_traversable: Optional[np.ndarray] = None

        self.xy_bounds = (-2.0, 2.0, -2.0, 2.0)

        self.scene_dirty = False
        self.cloud_dirty = False
        self.robot_dirty = False
        self.target_dirty = False

    @property
    def loaded(self) -> bool:
        return self.points_xyz.shape[0] > 0

    @property
    def total_points(self) -> int:
        return int(self.points_xyz.shape[0])

    @property
    def revealed_points(self) -> int:
        return int(np.count_nonzero(self.revealed_mask))

    def refresh_scene_files(self) -> None:
        self.scene_files.clear()
        for path in sorted(self.base_dir.glob("*.ply")):
            self.scene_files[path.name] = path

    def load_scene(
        self,
        scene_path: Path,
        world_scale: float,
        voxel_size_m: float,
        max_points: int,
    ) -> Tuple[bool, str]:
        if not scene_path.exists():
            return False, f"Scene file not found: {scene_path}"
        if world_scale <= 0:
            return False, "Scale must be > 0"

        pcd = o3d.io.read_point_cloud(str(scene_path))
        if pcd.is_empty():
            return False, "Loaded point cloud is empty."

        if voxel_size_m > 0:
            pcd = pcd.voxel_down_sample(voxel_size_m)

        points = np.asarray(pcd.points, dtype=np.float64)
        if points.size == 0:
            return False, "Point cloud became empty after downsample."

        colors = np.asarray(pcd.colors, dtype=np.float64)
        if colors.size == 0:
            colors = np.full_like(points, 0.72)

        if max_points > 0 and points.shape[0] > max_points:
            rng = np.random.default_rng(7)
            idx = rng.choice(points.shape[0], size=max_points, replace=False)
            points = points[idx]
            colors = colors[idx]

        points = points * world_scale

        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center_xy = (min_bound[:2] + max_bound[:2]) * 0.5
        floor_z = np.percentile(points[:, 2], 1.0)

        points[:, 0] -= center_xy[0]
        points[:, 1] -= center_xy[1]
        points[:, 2] -= floor_z

        min_xy = points[:, :2].min(axis=0)
        max_xy = points[:, :2].max(axis=0)

        self.points_xyz = points
        self.colors_rgb = np.clip(colors, 0.0, 1.0)
        self.revealed_mask = np.zeros((points.shape[0],), dtype=bool)
        self.reveal_source_xy = np.full((points.shape[0], 2), np.nan, dtype=np.float64)

        self.robot = RobotState(0.0, 0.0, 0.0)
        self.origin_xy = np.array([0.0, 0.0], dtype=np.float64)
        self.origin_yaw_rad = 0.0

        self.phase = "idle"
        self.mapping_state = "idle"
        self.hide_target_xy = None
        self.scan_accumulated_rad = 0.0
        self.mapping_loops = 0

        self._clear_navigation()
        self.latest_grid = None
        self.latest_traversable = None

        margin = 0.15
        self.xy_bounds = (
            float(min_xy[0] + margin),
            float(max_xy[0] - margin),
            float(min_xy[1] + margin),
            float(max_xy[1] - margin),
        )

        scene_diag = float(np.linalg.norm(max_xy - min_xy))
        self.scan_range_m = float(np.clip(scene_diag * 0.20, 0.95, 3.0))

        self.status_text = (
            f"Loaded {scene_path.name}: {points.shape[0]} points "
            f"(scale={world_scale:.2f}, voxel={voxel_size_m:.3f}m, "
            f"scan_range={self.scan_range_m:.2f}m, lidar_h={self.sensor_height_m:.2f}m)."
        )

        self.scene_dirty = True
        self.cloud_dirty = True
        self.robot_dirty = True
        self.target_dirty = True

        return True, self.status_text

    def set_manual_control(self, forward: int, turn: int) -> None:
        self.manual_forward = int(np.clip(forward, -1, 1))
        self.manual_turn = int(np.clip(turn, -1, 1))

    def start_scan(self) -> Tuple[bool, str]:
        if not self.loaded:
            return False, "Load a scene first."

        self.robot = RobotState(0.0, 0.0, 0.0)
        self.origin_xy = np.array([0.0, 0.0], dtype=np.float64)
        self.origin_yaw_rad = 0.0

        self.revealed_mask[:] = False
        self.reveal_source_xy[:] = np.nan

        self.phase = "mapping"
        self.mapping_state = "sweep"
        self.scan_accumulated_rad = 0.0
        self.mapping_loops = 0
        self.hide_target_xy = None

        self._clear_navigation()
        self.latest_grid = None
        self.latest_traversable = None

        self.cloud_dirty = True
        self.robot_dirty = True
        self.target_dirty = True

        msg = "Phase 1 started: autonomous Scan room (Spin and Go)."
        self.status_text = msg
        return True, msg

    def start_hide(self, reset_origin: bool = True) -> Tuple[bool, str]:
        if not self.loaded:
            return False, "Load a scene first."
        if self.phase not in ("mapped", "hidden"):
            return False, "Finish Scan room first (mapping must be complete)."
        if self.revealed_points < 200:
            return False, "Not enough scanned points yet. Run scan first."

        if reset_origin:
            self.origin_xy = np.array([self.robot.x_m, self.robot.y_m], dtype=np.float64)
            self.origin_yaw_rad = self.robot.yaw_rad

        grid = self.build_two_tier_grid()
        if grid is None:
            return False, "Failed to build occupancy grid from scan."
        traversable = self._traversable_mask(grid)
        self.latest_grid = grid
        self.latest_traversable = traversable

        start_cell = grid.world_to_grid(self.robot.x_m, self.robot.y_m)
        if start_cell is None:
            return False, "Robot is outside map bounds. Reload scene and retry."

        hide_plan = self._choose_hiding_target_with_path(grid, traversable, start_cell)
        if hide_plan is None:
            return False, "No reachable hiding target found from current robot pose."

        target_cell, path_cells, plan_kind = hide_plan

        nav_map = traversable.copy()
        nav_map[start_cell[1], start_cell[0]] = True

        self._set_navigation_path(path_cells, grid, nav_map, "hide target")

        self.hide_target_xy = grid.grid_to_world(target_cell[0], target_cell[1])

        self.phase = "hiding"
        self.target_dirty = True

        msg = (
            "Phase 2 started. Origin re-zeroed and hide target selected at "
            f"({self.hide_target_xy[0]:.2f}, {self.hide_target_xy[1]:.2f}) "
            f"[{plan_kind}]."
        )
        self.status_text = msg
        return True, msg

    def step_once(self, sim_dt_s: float = 0.15) -> None:
        if sim_dt_s <= 0:
            return
        self._advance(sim_dt_s)

    def update(self, dt_real_s: float, speed_multiplier: float) -> None:
        if dt_real_s <= 0:
            return
        dt_sim = dt_real_s * max(0.0, speed_multiplier)
        if dt_sim <= 0.0:
            return
        self._advance(dt_sim)

    def _advance(self, dt_s: float) -> None:
        if not self.loaded:
            return

        moved = False

        if self.phase == "mapping":
            moved = self._advance_mapping(dt_s)

        elif self.phase == "hiding":
            moved = self._advance_hiding(dt_s)

        else:
            linear = self.manual_forward * self.linear_speed_mps
            angular = self.manual_turn * self.angular_speed_rps

            if abs(angular) > 1e-6:
                self.robot.yaw_rad = wrap_angle(self.robot.yaw_rad + angular * dt_s)
                moved = True
            if abs(linear) > 1e-6:
                next_x = self.robot.x_m + math.cos(self.robot.yaw_rad) * linear * dt_s
                next_y = self.robot.y_m + math.sin(self.robot.yaw_rad) * linear * dt_s
                if self._is_position_driveable(next_x, next_y):
                    self.robot.x_m = next_x
                    self.robot.y_m = next_y
                    moved = True
                elif self.phase in ("mapped", "hidden"):
                    self.status_text = "Movement blocked by mapped obstacle."

            if moved:
                self._clamp_robot_to_bounds()

        if moved:
            self.robot_dirty = True

    def _advance_mapping(self, dt_s: float) -> bool:
        moved = False

        if self.mapping_state == "sweep":
            dtheta = self.scan_turn_rate_rps * dt_s
            self.robot.yaw_rad = wrap_angle(self.robot.yaw_rad + dtheta)
            self.scan_accumulated_rad += abs(dtheta)
            self._reveal_visible_points(self.scan_range_m, self.scan_fov_deg)
            moved = True

            if self.scan_accumulated_rad >= (2.0 * math.pi):
                self.scan_accumulated_rad = 0.0
                self.mapping_loops += 1

                grid = self.build_two_tier_grid()
                if grid is None:
                    self.status_text = "Sweep complete. Waiting for enough scan evidence..."
                    return moved

                traversable = self._traversable_mask(grid)
                self.latest_grid = grid
                self.latest_traversable = traversable

                planned = self._choose_frontier_path(grid, traversable)

                if planned is None or self.mapping_loops >= self.max_mapping_loops:
                    if self._start_return_home(grid, traversable):
                        self.mapping_state = "return_home"
                        self.status_text = "No frontiers left. Returning to origin."
                    else:
                        self.phase = "mapped"
                        self.mapping_state = "done"
                        self.status_text = (
                            "Mapping complete. Could not path back to origin from current pose."
                        )
                else:
                    target_cell, path_cells = planned
                    self._set_navigation_path(path_cells, grid, traversable, "frontier")
                    self.mapping_state = "navigate_frontier"
                    tx, ty = grid.grid_to_world(target_cell[0], target_cell[1])
                    self.status_text = (
                        f"Frontier target planned at ({tx:.2f}, {ty:.2f}). Navigating..."
                    )

        elif self.mapping_state in ("navigate_frontier", "return_home"):
            moved, finished, blocked = self._follow_navigation_path(dt_s)

            if blocked:
                self.status_text = "Navigation blocked. Replanning from next sweep."
                self.mapping_state = "sweep"
                self.scan_accumulated_rad = 0.0
                self._clear_navigation()

            elif finished and self.mapping_state == "navigate_frontier":
                self.mapping_state = "sweep"
                self.scan_accumulated_rad = 0.0
                self.status_text = "Reached frontier. Starting next 360 sweep."

            elif finished and self.mapping_state == "return_home":
                self.phase = "mapped"
                self.mapping_state = "done"
                self.status_text = "Mapping complete. Robot returned to origin."

        return moved

    def _advance_hiding(self, dt_s: float) -> bool:
        if self.hide_target_xy is None:
            self.phase = "mapped"
            return False

        moved, finished, blocked = self._follow_navigation_path(dt_s)

        if finished:
            self.phase = "hidden"
            self.status_text = "Robot reached hiding spot."
            return moved

        if blocked:
            grid = self.build_two_tier_grid()
            if grid is None:
                self.phase = "mapped"
                self.status_text = "Hide path blocked and map unavailable."
                return moved

            traversable = self._traversable_mask(grid)
            start_cell = grid.world_to_grid(self.robot.x_m, self.robot.y_m)
            goal_cell = grid.world_to_grid(self.hide_target_xy[0], self.hide_target_xy[1])

            if start_cell is None or goal_cell is None:
                self.phase = "mapped"
                self.status_text = "Hide target out of bounds after replan attempt."
                return moved

            traversable = traversable.copy()
            traversable[start_cell[1], start_cell[0]] = True

            path_cells = self._astar_path(traversable, start_cell, goal_cell)
            if path_cells is None or len(path_cells) < 2:
                self.phase = "mapped"
                self.status_text = "Hide path blocked and no alternate route found."
                return moved

            self._set_navigation_path(path_cells, grid, traversable, "hide target")
            self.status_text = "Hide path re-planned around obstacle."

        return moved

    def _start_return_home(self, grid: TwoTierGrid, traversable: np.ndarray) -> bool:
        start_cell = grid.world_to_grid(self.robot.x_m, self.robot.y_m)
        home_cell = grid.world_to_grid(self.origin_xy[0], self.origin_xy[1])
        if start_cell is None or home_cell is None:
            return False

        nav_map = traversable.copy()
        nav_map[start_cell[1], start_cell[0]] = True
        nav_map[home_cell[1], home_cell[0]] = True

        path_cells = self._astar_path(nav_map, start_cell, home_cell)
        if path_cells is None or len(path_cells) < 2:
            return False

        self._set_navigation_path(path_cells, grid, nav_map, "origin")
        return True

    def _set_navigation_path(
        self,
        path_cells: List[Tuple[int, int]],
        grid: TwoTierGrid,
        traversable: np.ndarray,
        goal_label: str,
    ) -> None:
        self.nav_grid = grid
        self.nav_traversable = traversable
        self.nav_goal_label = goal_label
        self.nav_index = 0

        if len(path_cells) <= 1:
            self.nav_waypoints_xy = []
            return

        self.nav_waypoints_xy = [
            grid.grid_to_world(gx, gy)
            for gx, gy in path_cells[1:]
        ]

    def _clear_navigation(self) -> None:
        self.nav_waypoints_xy = []
        self.nav_index = 0
        self.nav_grid = None
        self.nav_traversable = None
        self.nav_goal_label = ""

    def _follow_navigation_path(self, dt_s: float) -> Tuple[bool, bool, bool]:
        """Returns (moved, finished, blocked)."""
        if self.nav_index >= len(self.nav_waypoints_xy):
            return False, True, False

        moved = False

        while self.nav_index < len(self.nav_waypoints_xy):
            tx, ty = self.nav_waypoints_xy[self.nav_index]
            dx = tx - self.robot.x_m
            dy = ty - self.robot.y_m
            dist = math.hypot(dx, dy)

            if dist < 0.05:
                self.nav_index += 1
                continue

            desired_yaw = math.atan2(dy, dx)
            yaw_err = wrap_angle(desired_yaw - self.robot.yaw_rad)

            if abs(yaw_err) > math.radians(4.0):
                max_turn = self.angular_speed_rps * dt_s
                turn_step = float(np.clip(yaw_err, -max_turn, max_turn))
                self.robot.yaw_rad = wrap_angle(self.robot.yaw_rad + turn_step)
                moved = True
                break

            max_step = min(self.linear_speed_mps * dt_s, self.grid_cell_size_m * 0.42)
            step_dist = min(dist, max_step)
            next_x = self.robot.x_m + math.cos(self.robot.yaw_rad) * step_dist
            next_y = self.robot.y_m + math.sin(self.robot.yaw_rad) * step_dist

            if not self._is_position_traversable(next_x, next_y):
                return moved, False, True

            self.robot.x_m = next_x
            self.robot.y_m = next_y
            self._clamp_robot_to_bounds()
            moved = True
            break

        finished = self.nav_index >= len(self.nav_waypoints_xy)
        return moved, finished, False

    def _is_position_traversable(self, x_m: float, y_m: float) -> bool:
        if self.nav_grid is None or self.nav_traversable is None:
            return True
        cell = self.nav_grid.world_to_grid(x_m, y_m)
        if cell is None:
            return False
        gx, gy = cell
        return bool(self.nav_traversable[gy, gx])

    def _is_position_driveable(self, x_m: float, y_m: float) -> bool:
        if self.latest_grid is None or self.latest_traversable is None:
            return True
        cell = self.latest_grid.world_to_grid(x_m, y_m)
        if cell is None:
            return False
        gx, gy = cell
        return bool(self.latest_traversable[gy, gx])

    def _clamp_robot_to_bounds(self) -> None:
        min_x, max_x, min_y, max_y = self.xy_bounds
        self.robot.x_m = float(np.clip(self.robot.x_m, min_x, max_x))
        self.robot.y_m = float(np.clip(self.robot.y_m, min_y, max_y))

    def _reveal_visible_points(self, range_m: float, fov_deg: float) -> int:
        if self.points_xyz.shape[0] == 0:
            return 0

        unknown_idx = np.flatnonzero(~self.revealed_mask)
        if unknown_idx.size == 0:
            return 0

        pts = self.points_xyz[unknown_idx, :]

        vec_x = pts[:, 0] - self.robot.x_m
        vec_y = pts[:, 1] - self.robot.y_m
        vec_z = pts[:, 2] - self.sensor_height_m
        horiz_dist = np.hypot(vec_x, vec_y)
        dist_3d = np.sqrt((horiz_dist * horiz_dist) + (vec_z * vec_z))
        in_range = dist_3d <= range_m

        vert_angle = np.abs(np.arctan2(vec_z, np.maximum(horiz_dist, 1e-6)))
        in_vert_fov = vert_angle <= math.radians(self.scan_vert_fov_deg * 0.5)

        heading = np.arctan2(vec_y, vec_x)
        rel_heading = np.arctan2(
            np.sin(heading - self.robot.yaw_rad),
            np.cos(heading - self.robot.yaw_rad),
        )
        if fov_deg < 359.0:
            in_fov = np.abs(rel_heading) <= math.radians(fov_deg * 0.5)
            visible = in_range & in_vert_fov & in_fov
        else:
            visible = in_range & in_vert_fov

        if not np.any(visible):
            return 0

        visible_idx = unknown_idx[visible]
        visible_rel = rel_heading[visible]
        visible_horiz = horiz_dist[visible]

        bin_count = max(120, int(round(360.0 / max(0.25, self.scan_occlusion_bin_deg))))
        bin_pos = (visible_rel + math.pi) / (2.0 * math.pi)
        az_bins = np.floor(bin_pos * bin_count).astype(np.int32)
        az_bins = np.clip(az_bins, 0, bin_count - 1)

        nearest_per_bin = np.full((bin_count,), np.inf, dtype=np.float64)
        np.minimum.at(nearest_per_bin, az_bins, visible_horiz)

        in_front_surface = visible_horiz <= (nearest_per_bin[az_bins] + self.scan_occlusion_slack_m)
        reveal_idx = visible_idx[in_front_surface]

        if reveal_idx.size == 0:
            return 0

        self.revealed_mask[reveal_idx] = True
        self.reveal_source_xy[reveal_idx, 0] = self.robot.x_m
        self.reveal_source_xy[reveal_idx, 1] = self.robot.y_m
        self.cloud_dirty = True

        return int(reveal_idx.size)

    def build_two_tier_grid(self) -> Optional[TwoTierGrid]:
        if self.points_xyz.shape[0] == 0:
            return None

        observed_idx = np.flatnonzero(self.revealed_mask)
        if observed_idx.size < 40:
            return None

        observed = self.points_xyz[observed_idx]
        observed_sources = self.reveal_source_xy[observed_idx]

        pad = 0.2
        min_xy = self.points_xyz[:, :2].min(axis=0) - pad
        max_xy = self.points_xyz[:, :2].max(axis=0) + pad

        width = int(math.ceil((max_xy[0] - min_xy[0]) / self.grid_cell_size_m)) + 1
        height = int(math.ceil((max_xy[1] - min_xy[1]) / self.grid_cell_size_m)) + 1

        if width <= 2 or height <= 2:
            return None

        driveable_occ = np.zeros((height, width), dtype=bool)
        shield_occ = np.zeros((height, width), dtype=bool)
        known_mask = np.zeros((height, width), dtype=bool)

        gx = ((observed[:, 0] - min_xy[0]) / self.grid_cell_size_m).astype(np.int32)
        gy = ((observed[:, 1] - min_xy[1]) / self.grid_cell_size_m).astype(np.int32)

        in_bounds = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
        gx = gx[in_bounds]
        gy = gy[in_bounds]
        h = observed[:, 2][in_bounds]

        known_mask[gy, gx] = True

        drive_mask = (h >= 0.0) & (h <= DRIVEABLE_MAX_HEIGHT_M)
        shield_mask = (h > DRIVEABLE_MAX_HEIGHT_M) & (h <= SHIELD_MAX_HEIGHT_M)

        if np.any(drive_mask):
            driveable_occ[gy[drive_mask], gx[drive_mask]] = True
        if np.any(shield_mask):
            shield_occ[gy[shield_mask], gx[shield_mask]] = True

        # Free-space evidence comes from raycasts between reveal-source pose and hit cell.
        if observed.shape[0] > 0:
            sample_count = min(7000, observed.shape[0])
            rng = np.random.default_rng(23)
            sample_idx = rng.choice(observed.shape[0], size=sample_count, replace=False)
            sampled = observed[sample_idx]
            sampled_src = observed_sources[sample_idx]

            hit_x = ((sampled[:, 0] - min_xy[0]) / self.grid_cell_size_m).astype(np.int32)
            hit_y = ((sampled[:, 1] - min_xy[1]) / self.grid_cell_size_m).astype(np.int32)
            src_x = ((sampled_src[:, 0] - min_xy[0]) / self.grid_cell_size_m).astype(np.int32)
            src_y = ((sampled_src[:, 1] - min_xy[1]) / self.grid_cell_size_m).astype(np.int32)

            for hx, hy, sx, sy in zip(hit_x, hit_y, src_x, src_y):
                if hx < 0 or hy < 0 or hx >= width or hy >= height:
                    continue
                if sx < 0 or sy < 0 or sx >= width or sy >= height:
                    continue

                line = bresenham_line(int(sx), int(sy), int(hx), int(hy))
                for i, (rx, ry) in enumerate(line):
                    if i == len(line) - 1:
                        break
                    known_mask[ry, rx] = True

        known_mask[driveable_occ] = True
        known_mask[shield_occ] = True

        occupancy = np.full((height, width), -1, dtype=np.int8)
        occupancy[known_mask] = 0
        occupancy[driveable_occ] = 1

        return TwoTierGrid(
            cell_size_m=self.grid_cell_size_m,
            min_x_m=float(min_xy[0]),
            min_y_m=float(min_xy[1]),
            width=width,
            height=height,
            driveable_occ=driveable_occ,
            shield_occ=shield_occ,
            known_mask=known_mask,
            occupancy=occupancy,
        )

    def _traversable_mask(self, grid: TwoTierGrid) -> np.ndarray:
        traversable = (grid.occupancy == 0) & (~grid.driveable_occ)

        # Inflate driveable obstacles by a robot clearance radius so planned paths
        # are physically navigable for the 9in x 12in footprint.
        clearance_m = 0.5 * max(self.robot_width_m, self.robot_length_m)
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

    def _frontier_mask(self, grid: TwoTierGrid) -> np.ndarray:
        unknown = grid.occupancy == -1
        free = grid.occupancy == 0

        adjacent_unknown = np.zeros_like(unknown, dtype=bool)
        adjacent_unknown[1:, :] |= unknown[:-1, :]
        adjacent_unknown[:-1, :] |= unknown[1:, :]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]

        return free & adjacent_unknown

    def _extract_frontier_clusters(self, grid: TwoTierGrid) -> List[List[Tuple[int, int]]]:
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

    def _choose_frontier_path(
        self,
        grid: TwoTierGrid,
        traversable: np.ndarray,
    ) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
        start_cell = grid.world_to_grid(self.robot.x_m, self.robot.y_m)
        if start_cell is None:
            return None

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

                path_cells = self._astar_path(nav_map, start_cell, (gx, gy))
                if path_cells is None or len(path_cells) < 2:
                    continue
                return (gx, gy), path_cells

        return self._choose_exploration_fallback_path(grid, nav_map, start_cell)

    def _choose_exploration_fallback_path(
        self,
        grid: TwoTierGrid,
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
            if int(gx) == sx and int(gy) == sy:
                continue
            frontier_pull = float(neighbor_unknown[gy, gx])
            if frontier_pull <= 0.0:
                continue
            dist2 = float((int(gx) - sx) ** 2 + (int(gy) - sy) ** 2)
            score = (frontier_pull * 90.0) + (dist2 * 0.40)
            scored.append((score, (int(gx), int(gy))))

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
        grid: TwoTierGrid,
        traversable: np.ndarray,
        start_cell: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]], str]]:
        origin_cell = grid.world_to_grid(self.origin_xy[0], self.origin_xy[1])
        if origin_cell is None:
            return None

        nav_map = traversable.copy()
        nav_map[start_cell[1], start_cell[0]] = True

        candidate_mask = traversable & grid.known_mask
        candidates = np.argwhere(candidate_mask)  # rows => [gy, gx]
        if candidates.shape[0] == 0:
            return None

        rng = np.random.default_rng()
        rng.shuffle(candidates)

        min_cells = max(1, int(self.min_hide_distance_m / grid.cell_size_m))
        ideal_scored: List[Tuple[float, Tuple[int, int]]] = []
        fallback_scored: List[Tuple[float, Tuple[int, int]]] = []

        unknown = grid.occupancy == -1
        neighbor_unknown = np.zeros_like(grid.occupancy, dtype=np.int16)
        unknown_i = unknown.astype(np.int16)
        neighbor_unknown[1:, :] += unknown_i[:-1, :]
        neighbor_unknown[:-1, :] += unknown_i[1:, :]
        neighbor_unknown[:, 1:] += unknown_i[:, :-1]
        neighbor_unknown[:, :-1] += unknown_i[:, 1:]

        ox, oy = origin_cell

        for gy, gx in candidates:
            dx = int(gx) - int(ox)
            dy = int(gy) - int(oy)
            dist2 = (dx * dx + dy * dy)
            if dist2 < (min_cells * min_cells):
                continue

            hide_under = (not grid.driveable_occ[gy, gx]) and bool(grid.shield_occ[gy, gx])

            hide_behind = False
            for cx, cy in bresenham_line(int(ox), int(oy), int(gx), int(gy)):
                if (cx == ox and cy == oy) or (cx == gx and cy == gy):
                    continue
                if grid.driveable_occ[cy, cx] or grid.shield_occ[cy, cx]:
                    hide_behind = True
                    break

            if hide_under or hide_behind:
                score = float(dist2) + (260.0 if hide_under else 180.0)
                score += float(neighbor_unknown[gy, gx]) * 35.0
                ideal_scored.append((score, (int(gx), int(gy))))
            else:
                score = float(dist2) + float(neighbor_unknown[gy, gx]) * 120.0
                if grid.shield_occ[gy, gx]:
                    score += 40.0
                fallback_scored.append((score, (int(gx), int(gy))))

            if (len(ideal_scored) + len(fallback_scored)) >= 4200:
                break

        chosen = self._pick_reachable_candidate(
            scored_cells=ideal_scored,
            nav_map=nav_map,
            start_cell=start_cell,
            max_checks=260,
            top_pool=18,
        )
        if chosen is not None:
            target_cell, path_cells = chosen
            return target_cell, path_cells, "ideal"

        chosen = self._pick_reachable_candidate(
            scored_cells=fallback_scored,
            nav_map=nav_map,
            start_cell=start_cell,
            max_checks=420,
            top_pool=24,
        )
        if chosen is not None:
            target_cell, path_cells = chosen
            return target_cell, path_cells, "fallback"

        any_cells = [
            (float(((int(gx) - int(ox)) ** 2 + (int(gy) - int(oy)) ** 2)), (int(gx), int(gy)))
            for gy, gx in candidates
            if not (int(gx) == start_cell[0] and int(gy) == start_cell[1])
        ]
        chosen = self._pick_reachable_candidate(
            scored_cells=any_cells,
            nav_map=nav_map,
            start_cell=start_cell,
            max_checks=650,
            top_pool=28,
        )
        if chosen is not None:
            target_cell, path_cells = chosen
            return target_cell, path_cells, "fallback-any"

        return None

    def _pick_reachable_candidate(
        self,
        scored_cells: List[Tuple[float, Tuple[int, int]]],
        nav_map: np.ndarray,
        start_cell: Tuple[int, int],
        max_checks: int,
        top_pool: int,
    ) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
        if not scored_cells:
            return None

        ranked = sorted(scored_cells, key=lambda item: item[0], reverse=True)
        reachable: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = []

        for _, target_cell in ranked[:max_checks]:
            path_cells = self._astar_path(nav_map, start_cell, target_cell)
            if path_cells is None or len(path_cells) < 2:
                continue
            reachable.append((target_cell, path_cells))
            if len(reachable) >= max(6, top_pool * 2):
                break

        if not reachable:
            return None

        best_pool = reachable[: min(top_pool, len(reachable))]
        return random.choice(best_pool)

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

    @staticmethod
    def _world_to_grid(
        x_m: float,
        y_m: float,
        min_x_m: float,
        min_y_m: float,
        width: int,
        height: int,
        cell_size_m: float,
    ) -> Optional[Tuple[int, int]]:
        gx = int((x_m - min_x_m) / cell_size_m)
        gy = int((y_m - min_y_m) / cell_size_m)
        if gx < 0 or gy < 0 or gx >= width or gy >= height:
            return None
        return gx, gy


class RobotVisualizer3D:
    """Open3D visualization window."""

    def __init__(self) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Robot 3D Scene", width=1280, height=820)
        self.setup_figure()

        self.base_cloud = o3d.geometry.PointCloud()
        self.scanned_cloud = o3d.geometry.PointCloud()

        self.scene_added = False

        self.robot_length_m = ROBOT_LENGTH_M
        self.robot_width_m = ROBOT_WIDTH_M
        self.robot_body_height_m = ROBOT_BODY_HEIGHT_M
        self.sensor_height_m = LIDAR_SENSOR_HEIGHT_M

        self.robot_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.robot_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.heading_line: Optional[o3d.geometry.LineSet] = None
        self.robot_added = False

        self.target_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.target_center: Optional[np.ndarray] = None

        self.closed = False

    def setup_figure(self) -> None:
        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.array([0.04, 0.045, 0.06], dtype=np.float64)
        render_opt.point_size = 3.2
        render_opt.light_on = True

    def set_scene(
        self,
        points_xyz: np.ndarray,
        colors_rgb: np.ndarray,
        revealed_mask: np.ndarray,
    ) -> None:
        self.update_layers(points_xyz, colors_rgb, revealed_mask)

        if not self.scene_added:
            self.vis.add_geometry(self.base_cloud, reset_bounding_box=True)
            self.vis.add_geometry(self.scanned_cloud, reset_bounding_box=False)
            self.scene_added = True

        if points_xyz.shape[0] > 0:
            self.vis.reset_view_point(True)

    def update_layers(
        self,
        points_xyz: np.ndarray,
        colors_rgb: np.ndarray,
        revealed_mask: np.ndarray,
    ) -> None:
        scanned_idx = np.flatnonzero(revealed_mask)
        base_idx = np.flatnonzero(~revealed_mask)

        if scanned_idx.size == 0:
            scanned_xyz = np.zeros((0, 3), dtype=np.float64)
            scanned_rgb = np.zeros((0, 3), dtype=np.float64)
        else:
            scanned_xyz = points_xyz[scanned_idx]
            scanned_rgb = colors_rgb[scanned_idx]

        if base_idx.size == 0:
            base_xyz = np.zeros((0, 3), dtype=np.float64)
            base_rgb = np.zeros((0, 3), dtype=np.float64)
        else:
            base_xyz = points_xyz[base_idx]
            base_rgb = self._fade_colors(colors_rgb[base_idx], alpha=0.07)

        self.base_cloud.points = o3d.utility.Vector3dVector(base_xyz)
        self.base_cloud.colors = o3d.utility.Vector3dVector(base_rgb)

        self.scanned_cloud.points = o3d.utility.Vector3dVector(scanned_xyz)
        self.scanned_cloud.colors = o3d.utility.Vector3dVector(scanned_rgb)

        if self.scene_added:
            self.vis.update_geometry(self.base_cloud)
            self.vis.update_geometry(self.scanned_cloud)

    def set_robot_pose(self, robot: RobotState) -> None:
        center = np.array([robot.x_m, robot.y_m, 0.0], dtype=np.float64)
        heading_start = np.array(
            [robot.x_m, robot.y_m, self.sensor_height_m],
            dtype=np.float64,
        )
        heading = np.array(
            [
                robot.x_m + math.cos(robot.yaw_rad) * 0.27,
                robot.y_m + math.sin(robot.yaw_rad) * 0.27,
                self.sensor_height_m,
            ],
            dtype=np.float64,
        )

        if not self.robot_added:
            self.robot_mesh = o3d.geometry.TriangleMesh.create_box(
                width=self.robot_length_m,
                height=self.robot_width_m,
                depth=self.robot_body_height_m,
            )
            self.robot_mesh.translate(
                np.array(
                    [
                        -0.5 * self.robot_length_m,
                        -0.5 * self.robot_width_m,
                        0.0,
                    ],
                    dtype=np.float64,
                )
            )
            self.robot_mesh.paint_uniform_color([0.95, 0.2, 0.1])
            self.robot_mesh.compute_vertex_normals()
            self.robot_mesh.translate(center)

            self.heading_line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.vstack([heading_start, heading])),
                lines=o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32)),
            )
            self.heading_line.colors = o3d.utility.Vector3dVector(
                np.array([[1.0, 0.95, 0.1]], dtype=np.float64)
            )

            self.vis.add_geometry(self.robot_mesh, reset_bounding_box=False)
            self.vis.add_geometry(self.heading_line, reset_bounding_box=False)
            self.robot_added = True
            self.robot_center = center
            return

        if self.robot_mesh is None or self.heading_line is None:
            return

        delta = center - self.robot_center
        self.robot_mesh.translate(delta)
        self.robot_center = center

        self.heading_line.points = o3d.utility.Vector3dVector(np.vstack([heading_start, heading]))

        self.vis.update_geometry(self.robot_mesh)
        self.vis.update_geometry(self.heading_line)

    def set_hide_target(self, target_xy: Optional[Tuple[float, float]]) -> None:
        if target_xy is None:
            if self.target_mesh is not None:
                self.vis.remove_geometry(self.target_mesh, reset_bounding_box=False)
                self.target_mesh = None
                self.target_center = None
            return

        center = np.array([target_xy[0], target_xy[1], 0.045], dtype=np.float64)

        if self.target_mesh is None:
            self.target_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.045)
            self.target_mesh.paint_uniform_color([0.2, 1.0, 0.35])
            self.target_mesh.compute_vertex_normals()
            self.target_mesh.translate(center)
            self.vis.add_geometry(self.target_mesh, reset_bounding_box=False)
            self.target_center = center
            return

        if self.target_center is None:
            self.target_center = center

        delta = center - self.target_center
        self.target_mesh.translate(delta)
        self.target_center = center
        self.vis.update_geometry(self.target_mesh)

    def tick(self) -> bool:
        if self.closed:
            return False

        alive = self.vis.poll_events()
        if not alive:
            self.closed = True
            return False

        self.vis.update_renderer()
        return True

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.vis.destroy_window()

    @staticmethod
    def _fade_colors(colors_rgb: np.ndarray, alpha: float) -> np.ndarray:
        bg = np.array([0.04, 0.045, 0.06], dtype=np.float64)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return np.clip(colors_rgb * alpha + bg * (1.0 - alpha), 0.0, 1.0)


class SimulatedPhoneApp:
    """Tkinter window that controls the simulation."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

        self.model = SimulationModel(base_dir)
        self.visualizer = RobotVisualizer3D()

        self.root = tk.Tk()
        self.root.title("Simulated Phone App")

        self.scene_var = tk.StringVar()
        self.scale_var = tk.DoubleVar(value=1.0)
        self.voxel_var = tk.DoubleVar(value=0.04)
        self.max_points_var = tk.IntVar(value=180000)
        self.sim_speed_var = tk.DoubleVar(value=1.0)

        self.phase_var = tk.StringVar(value="Phase: idle")
        self.pose_var = tk.StringVar(value="Pose: x=0.00  y=0.00  yaw=0 deg")
        self.reveal_var = tk.StringVar(value="Revealed points: 0 / 0")
        self.status_var = tk.StringVar(value="Load one of the .ply scans.")

        self.manual_flags = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
        }

        self._build_ui()
        self._bind_keyboard_controls()

        scenes = list(self.model.scene_files.keys())
        if scenes:
            self.scene_var.set(scenes[0])

        self.capture_dir = self.base_dir / "captures"
        moved_on_startup = self._organize_capture_files(silent=True)

        self.last_tick_t = time.perf_counter()
        self.last_capture_cleanup_t = self.last_tick_t
        self.running = True

        if moved_on_startup > 0:
            self.status_var.set(
                f"Moved {moved_on_startup} capture file(s) to {self.capture_dir.name}/."
            )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Scene load controls
        scene_box = ttk.LabelFrame(main, text="Scene")
        scene_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        scene_box.columnconfigure(1, weight=1)

        ttk.Label(scene_box, text="PLY file:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        scene_combo = ttk.Combobox(
            scene_box,
            textvariable=self.scene_var,
            values=list(self.model.scene_files.keys()),
            state="readonly",
            width=30,
        )
        scene_combo.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(scene_box, text="Scale:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(scene_box, textvariable=self.scale_var, width=10).grid(
            row=1, column=1, sticky="w", padx=4, pady=4
        )

        ttk.Label(scene_box, text="Voxel(m):").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(scene_box, textvariable=self.voxel_var, width=10).grid(
            row=2, column=1, sticky="w", padx=4, pady=4
        )

        ttk.Label(scene_box, text="Max points:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(scene_box, textvariable=self.max_points_var, width=10).grid(
            row=3, column=1, sticky="w", padx=4, pady=4
        )

        ttk.Button(scene_box, text="Load Scene", command=self._load_scene).grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=4, pady=(6, 4)
        )

        # Manual movement
        move_box = ttk.LabelFrame(main, text="Robot Drive (hold buttons or use WASD)")
        move_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        btn_forward = ttk.Button(move_box, text="Forward")
        btn_back = ttk.Button(move_box, text="Backward")
        btn_left = ttk.Button(move_box, text="Turn Left")
        btn_right = ttk.Button(move_box, text="Turn Right")

        btn_forward.grid(row=0, column=1, padx=4, pady=4)
        btn_left.grid(row=1, column=0, padx=4, pady=4)
        btn_right.grid(row=1, column=2, padx=4, pady=4)
        btn_back.grid(row=2, column=1, padx=4, pady=4)

        self._bind_hold_button(btn_forward, "forward")
        self._bind_hold_button(btn_back, "backward")
        self._bind_hold_button(btn_left, "left")
        self._bind_hold_button(btn_right, "right")

        # Phase/state actions
        phase_box = ttk.LabelFrame(main, text="Phases")
        phase_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        ttk.Button(phase_box, text="Scan room", command=self._start_scan).grid(
            row=0, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(phase_box, text="Hide", command=self._start_hide).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )

        for idx in range(2):
            phase_box.columnconfigure(idx, weight=1)

        # Simulation timing controls
        sim_box = ttk.LabelFrame(main, text="Simulation Time")
        sim_box.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        sim_box.columnconfigure(1, weight=1)

        ttk.Label(sim_box, text="Speed:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Scale(
            sim_box,
            from_=0.0,
            to=4.0,
            variable=self.sim_speed_var,
            orient="horizontal",
        ).grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        ttk.Button(sim_box, text="Step +0.15s", command=self._step_once).grid(
            row=0, column=2, sticky="ew", padx=4, pady=4
        )

        # Status
        status_box = ttk.LabelFrame(main, text="Status")
        status_box.grid(row=4, column=0, sticky="ew")

        ttk.Label(status_box, textvariable=self.phase_var).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(status_box, textvariable=self.pose_var).grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(status_box, textvariable=self.reveal_var).grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(status_box, textvariable=self.status_var, wraplength=520).grid(
            row=3, column=0, sticky="w", padx=4, pady=(2, 6)
        )

    def _bind_keyboard_controls(self) -> None:
        def press(name: str) -> None:
            self.manual_flags[name] = True

        def release(name: str) -> None:
            self.manual_flags[name] = False

        self.root.bind("<KeyPress-w>", lambda _e: press("forward"))
        self.root.bind("<KeyRelease-w>", lambda _e: release("forward"))
        self.root.bind("<KeyPress-s>", lambda _e: press("backward"))
        self.root.bind("<KeyRelease-s>", lambda _e: release("backward"))
        self.root.bind("<KeyPress-a>", lambda _e: press("left"))
        self.root.bind("<KeyRelease-a>", lambda _e: release("left"))
        self.root.bind("<KeyPress-d>", lambda _e: press("right"))
        self.root.bind("<KeyRelease-d>", lambda _e: release("right"))

        self.root.focus_set()

    def _bind_hold_button(self, btn: ttk.Button, flag_name: str) -> None:
        def press(_event: tk.Event) -> None:
            self.manual_flags[flag_name] = True

        def release(_event: tk.Event) -> None:
            self.manual_flags[flag_name] = False

        btn.bind("<ButtonPress-1>", press)
        btn.bind("<ButtonRelease-1>", release)
        btn.bind("<Leave>", release)

    def _apply_manual_controls(self) -> None:
        forward = int(self.manual_flags["forward"]) - int(self.manual_flags["backward"])
        turn = int(self.manual_flags["left"]) - int(self.manual_flags["right"])
        self.model.set_manual_control(forward, turn)

    def _organize_capture_files(self, silent: bool = True) -> int:
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        moved_count = 0
        patterns = ("DepthCapture_*.png", "DepthCapture_*.jpg", "DepthCapture_*.jpeg")

        for pattern in patterns:
            for src in self.base_dir.glob(pattern):
                if not src.is_file():
                    continue

                dst = self.capture_dir / src.name
                if dst.exists():
                    stem = dst.stem
                    suffix = dst.suffix
                    i = 1
                    while True:
                        candidate = self.capture_dir / f"{stem}_{i}{suffix}"
                        if not candidate.exists():
                            dst = candidate
                            break
                        i += 1

                try:
                    shutil.move(str(src), str(dst))
                    moved_count += 1
                except OSError:
                    continue

        if moved_count > 0 and not silent:
            self.status_var.set(
                f"Moved {moved_count} capture file(s) to {self.capture_dir.name}/."
            )

        return moved_count

    def _load_scene(self) -> None:
        scene_name = self.scene_var.get().strip()
        if not scene_name:
            messagebox.showwarning("Scene", "Choose a .ply scene first.")
            return

        scene_path = self.model.scene_files.get(scene_name)
        if scene_path is None:
            messagebox.showerror("Scene", f"Scene not found: {scene_name}")
            return

        ok, msg = self.model.load_scene(
            scene_path=scene_path,
            world_scale=float(self.scale_var.get()),
            voxel_size_m=float(self.voxel_var.get()),
            max_points=int(self.max_points_var.get()),
        )

        if not ok:
            messagebox.showerror("Load failed", msg)
            return

        moved = self._organize_capture_files(silent=True)
        if moved > 0:
            msg = f"{msg} Moved {moved} capture file(s) to {self.capture_dir.name}/."

        self.model.status_text = msg

        self.status_var.set(msg)
        self._sync_visualizer(force_scene=True)
        self._refresh_labels()

    def _start_scan(self) -> None:
        ok, msg = self.model.start_scan()
        if not ok:
            messagebox.showwarning("Scan", msg)
        self.status_var.set(msg)

    def _start_hide(self) -> None:
        confirm = messagebox.askyesno(
            "Hide",
            "Place the robot in your reference orientation now.\n"
            "Press Yes to re-zero origin and start Hide phase.",
        )
        if not confirm:
            self.status_var.set("Hide cancelled. Orientation reset not applied.")
            return

        ok, msg = self.model.start_hide(reset_origin=True)
        if not ok:
            messagebox.showwarning("Hide", msg)
        self.status_var.set(msg)

    def _step_once(self) -> None:
        self._apply_manual_controls()
        self.model.step_once(0.15)
        self._sync_visualizer()
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        phase_text = self.model.phase
        if self.model.phase == "mapping":
            phase_text = f"mapping / {self.model.mapping_state}"
        self.phase_var.set(f"Phase: {phase_text}")
        self.pose_var.set(
            "Pose: "
            f"x={self.model.robot.x_m:.2f}  "
            f"y={self.model.robot.y_m:.2f}  "
            f"yaw={math.degrees(self.model.robot.yaw_rad):.1f} deg"
        )
        self.reveal_var.set(
            f"Revealed points: {self.model.revealed_points} / {self.model.total_points}"
        )

        # Keep latest model status unless a newer direct status was set from button actions.
        if self.model.status_text:
            self.status_var.set(self.model.status_text)

    def _sync_visualizer(self, force_scene: bool = False) -> None:
        if not self.model.loaded:
            return

        if force_scene or self.model.scene_dirty:
            self.visualizer.set_scene(
                self.model.points_xyz,
                self.model.colors_rgb,
                self.model.revealed_mask,
            )
            self.model.scene_dirty = False
            self.model.cloud_dirty = False

        elif self.model.cloud_dirty:
            self.visualizer.update_layers(
                self.model.points_xyz,
                self.model.colors_rgb,
                self.model.revealed_mask,
            )
            self.model.cloud_dirty = False

        if self.model.robot_dirty or force_scene:
            self.visualizer.set_robot_pose(self.model.robot)
            self.model.robot_dirty = False

        if self.model.target_dirty or force_scene:
            self.visualizer.set_hide_target(self.model.hide_target_xy)
            self.model.target_dirty = False

    def _tick(self) -> None:
        if not self.running:
            return

        now = time.perf_counter()
        dt = now - self.last_tick_t
        self.last_tick_t = now

        if (now - self.last_capture_cleanup_t) >= 2.0:
            self.last_capture_cleanup_t = now
            self._organize_capture_files(silent=True)

        self._apply_manual_controls()
        self.model.update(dt_real_s=dt, speed_multiplier=float(self.sim_speed_var.get()))

        self._sync_visualizer()
        self._refresh_labels()

        if not self.visualizer.tick():
            self._on_close()
            return

        self.root.after(16, self._tick)

    def _on_close(self) -> None:
        if not self.running:
            return
        self.running = False
        self.visualizer.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.after(16, self._tick)
        self.root.mainloop()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    app = SimulatedPhoneApp(base_dir)
    app.run()


if __name__ == "__main__":
    main()
