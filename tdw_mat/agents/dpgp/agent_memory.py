"""TDW-specific spatial memory used only by DPGP.

The formal benchmark uses ground-truth segmentation masks. This module owns
geometric projection and the shared regional path/cost/controller contract.
"""

import copy
import heapq
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import pyastar2d as pyastar
except ModuleNotFoundError:  # The TDW environment provides it; unit tests need not.
    pyastar = None

from .config import MethodConfig, FORWARD_DISTANCE


class AgentMemory:
    """Observed object locations, occupancy grids, and deterministic A*."""

    def __init__(
        self,
        agent_id: int,
        agent_color: Sequence[int],
        env_api: Optional[dict] = None,
        config: Optional[MethodConfig] = None,
        gt_mask: bool = True,
    ):
        if agent_id not in (0, 1):
            raise ValueError("agent_id must be 0 or 1")
        if not gt_mask:
            raise ValueError("TDW DPGP supports the formal GT_MASK=True setting only")
        self.agent_id = int(agent_id)
        self.agent_color = np.asarray(agent_color)
        if self.agent_color.shape != (3,):
            raise ValueError("agent_color must be one RGB triplet")
        self.env_api = env_api or {}
        self.config = config or MethodConfig()
        shape = self.config.map_size
        self.occupancy_map = np.zeros(shape, dtype=np.int32)
        self.local_occupancy_map = np.zeros(shape, dtype=np.int32)
        self.known_map = np.zeros(shape, dtype=np.int32)
        self.object_map = np.zeros(shape, dtype=np.int32)
        self.id_map = np.zeros(shape, dtype=np.int64)
        self.wall_map = np.zeros(shape, dtype=np.int32)
        self.true_wall_map = np.zeros(shape, dtype=np.int32)
        self.current_obstacle_map = np.zeros(shape, dtype=np.float32)
        self.object_info: Dict[int, dict] = {}
        self.color_to_id: Dict[Tuple[int, int, int], int] = {}
        self.obs: Optional[dict] = None
        self.position: Optional[np.ndarray] = None
        self.forward: Optional[np.ndarray] = None
        self.opponent_position: Optional[np.ndarray] = None
        self.opponent_visible = False
        self.target_grid: Optional[Tuple[int, int]] = None
        self.local_step = 0
        self._projection_cache = {}
        self._navigation_endpoint = None
        self._navigation_waypoint = None
        self._navigation_waypoint_origin = None
        self._navigation_avoid = None
        self._navigation_recovery_origin = None
        self.navigation_diagnostics = {}

    def update(self, obs: dict) -> None:
        """Consume one formal public observation."""
        for key in ("current_frames", "agent", "visible_objects", "depth",
                    "seg_mask", "camera_matrix", "FOV"):
            if key not in obs:
                raise KeyError("observation is missing {}".format(key))
        if len(obs["agent"]) < 6:
            raise ValueError("observation agent pose must contain position and forward")
        self.obs = obs
        self.position = np.asarray(obs["agent"][:3], dtype=np.float64)
        self.forward = np.asarray(obs["agent"][3:6], dtype=np.float64)
        self.local_step += 1
        self._update_objects()
        self.known_map = np.maximum(self.known_map, self._depth_to_maps())
        if self.local_step % self.config.map_update_frequency == 0:
            self.local_occupancy_map = copy.deepcopy(self.occupancy_map)

    def world_to_grid(self, position: Sequence[float]) -> Tuple[int, int]:
        x, z = self._xz(position)
        i = int(round((x - self.config.scene_x_min) / self.config.cell_size))
        j = int(round((z - self.config.scene_z_min) / self.config.cell_size))
        return i, j

    def grid_to_world(self, index: Sequence[int]) -> Tuple[float, float]:
        if len(index) != 2:
            raise ValueError("grid index must contain two coordinates")
        return (
            float(index[0]) * self.config.cell_size + self.config.scene_x_min,
            float(index[1]) * self.config.cell_size + self.config.scene_z_min,
        )

    def clamp_grid(self, index: Sequence[int]) -> Tuple[int, int]:
        rows, columns = self.local_occupancy_map.shape
        return (
            max(0, min(int(index[0]), rows - 1)),
            max(0, min(int(index[1]), columns - 1)),
        )

    def get_object_position(self, object_id: int):
        item = self.object_info.get(int(object_id))
        return None if item is None else item.get("position")

    def get_navigation_path_info(self, start, goal, *, radius=0.0, room_name=None) -> dict:
        """One map/query contract for planning and execution; no state mutation."""
        path, cost = self.find_shortest_path(start, goal, radius=radius, room_name=room_name)
        if path is None or cost < 0:
            return {"path": None, "path_len": -1.0,
                    "known_grid_count": 0, "total_grid_count": 0, "endpoint": None,
                    "reason": "no_reachable_endpoint"}
        indices = [(int(item[0]), int(item[1])) for item in path]
        known = sum(self.known_map[i, j] != 0 for i, j in indices)
        return {"path": indices, "path_len": float(cost),
                "known_grid_count": int(known),
                "total_grid_count": len(indices), "endpoint": (
                    tuple(start) if len(indices) == 1 else
                    (self.grid_to_world(indices[-1])[0], float(start[1]),
                     self.grid_to_world(indices[-1])[1])), "reason": None}

    def select_scan_viewpoint(self, room_name: str, center,
                              previous_viewpoints) -> tuple:
        """Choose one reachable supplementary view without adding map state."""
        center = tuple(float(value) for value in center[:3])
        previous = [tuple(float(value) for value in point[:3])
                    for point in previous_viewpoints]
        if not previous:
            return center
        belongs = self.env_api.get("belongs_to_which_room")
        start = self.position if self.position is not None else center
        candidates = []
        for radius in (1.5, 2.5):
            for direction in range(8):
                angle = direction * math.pi / 4.0
                point = (
                    center[0] + radius * math.cos(angle),
                    center[1],
                    center[2] + radius * math.sin(angle),
                )
                if callable(belongs) and belongs(point) != room_name:
                    continue
                if not self.is_navigable_position(point):
                    continue
                info = self.get_navigation_path_info(start, point)
                path_len = float(info.get("path_len", -1.0))
                if path_len < 0:
                    continue
                separation = min(math.hypot(
                    point[0] - old[0], point[2] - old[2]
                ) for old in previous)
                candidates.append((separation, -path_len, point))
        return max(candidates)[2] if candidates else center

    def find_shortest_path(self, start, goal, avoid=None, *, radius=0.0,
                           room_name=None, preferred=None):
        """Reach a free cell in the requested region, never teleport the start."""
        start_grid = self.world_to_grid(start)
        if self.clamp_grid(start_grid) != start_grid:
            return None, -1.0
        if radius > 0 and self._in_region(start, goal, radius, room_name):
            return np.asarray([start_grid], dtype=np.int32), 0.0
        cost_map = self._navigation_cost_map()
        # A noisy observation at the occupied ego cell must not relocate ego.
        cost_map[start_grid] = 1.0
        if avoid is not None and avoid != start_grid:
            cost_map[avoid] += 200000
        goal_grid = self.world_to_grid(goal)
        goals = None
        if radius > 0:
            goals = np.zeros(self.config.map_size, dtype=bool)
            extent = int(math.ceil(radius / self.config.cell_size)) + 1
            for i in range(max(0, goal_grid[0] - extent),
                           min(goals.shape[0], goal_grid[0] + extent + 1)):
                for j in range(max(0, goal_grid[1] - extent),
                               min(goals.shape[1], goal_grid[1] + extent + 1)):
                    x, z = self.grid_to_world((i, j))
                    point = (x, float(start[1]), z)
                    if self._in_region(point, goal, radius, room_name) and not self._occupied(i, j):
                        goals[i, j] = True
            # Actual arrival was checked above. Rounding an outside pose to an
            # inside cell must not produce a zero-cost, uncompleted Move.
            goals[start_grid] = False
            if not goals.any():
                return None, -1.0
            # Keep an already selected valid endpoint, but retry the whole region
            # if a newly observed obstacle makes its path unavailable.
            if preferred is not None:
                cell = self.world_to_grid(preferred)
                if self.clamp_grid(cell) == cell and goals[cell]:
                    path = self._astar_path(cost_map, start_grid, cell)
                    if path is not None:
                        return path, sum(float(cost_map[tuple(p)]) for p in path[1:])
        else:
            if self.clamp_grid(goal_grid) != goal_grid:
                return None, -1.0
            goal_grid = self._nearest_unoccupied_grid(*goal_grid)
            if goal_grid is None:
                return None, -1.0
        path = self._astar_path(cost_map, start_grid, goal_grid, goals)
        if path is None or len(path) == 0:
            return None, -1.0
        return path, sum(float(cost_map[tuple(p)]) for p in path[1:])

    def _in_region(self, point, goal, radius, room_name):
        belongs = self.env_api.get("belongs_to_which_room")
        return (math.hypot(point[0] - goal[0], point[2] - goal[2]) < radius
                and (room_name is None or (callable(belongs) and belongs(point) == room_name)))

    def reset_navigation(self):
        """New semantic destination, not merely a new communication epoch."""
        self._navigation_endpoint = None
        self._navigation_waypoint = None
        self._navigation_avoid = self._navigation_recovery_origin = None

    def _segment_clear(self, start, end):
        """Conservative swept grid rectangle; includes ordinary obstacle cores."""
        a, b = self.world_to_grid(start), self.world_to_grid(end)
        if self.clamp_grid(a) != a or self.clamp_grid(b) != b:
            return False
        for i in range(min(a[0], b[0]), max(a[0], b[0]) + 1):
            for j in range(min(a[1], b[1]), max(a[1], b[1]) + 1):
                if (i, j) != a and (self.local_occupancy_map[i, j] > 0
                                    or self.wall_map[i, j] > 0
                                    or self.true_wall_map[i, j] > 0):
                    return False
        return True

    def move_to_pos(self, target, *, radius=0.0, room_name=None):
        """Follow the same region query, checking the full public forward stride."""
        if self.obs is None or self.position is None or self.forward is None:
            raise RuntimeError("AgentMemory must be updated before navigation")
        target = tuple(float(x) for x in target)
        if (self._navigation_recovery_origin is not None
                and np.linalg.norm(self.position[[0, 2]] - self._navigation_recovery_origin)
                >= 2 * self.config.cell_size):
            self._navigation_avoid = self._navigation_recovery_origin = None
        path, _ = self.find_shortest_path(
            self.position, target, avoid=self._navigation_avoid,
            radius=radius, room_name=room_name, preferred=self._navigation_endpoint)
        if path is None:
            self._navigation_endpoint = self._navigation_waypoint = None
            self.navigation_diagnostics = self._navigation_snapshot(target, "no_reachable_endpoint")
            return None, -1
        self.target_grid = (int(path[-1][0]), int(path[-1][1]))
        x, z = self.grid_to_world(self.target_grid)
        endpoint = (x, float(self.position[1]), z)
        reason = "selected" if self._navigation_waypoint is None else "held"
        if self._navigation_endpoint != endpoint:
            self._navigation_waypoint = None
            reason = "target_changed"
        self._navigation_endpoint = endpoint
        waypoint = self._navigation_waypoint
        if waypoint is not None:
            x, z = self.grid_to_world(waypoint)
            if (np.linalg.norm(self.position[[0, 2]] - self._navigation_waypoint_origin)
                    >= self.config.cell_size):
                reason = "moved"
            elif math.hypot(x - self.position[0], z - self.position[2]) <= self.config.cell_size:
                reason = "arrived"
            elif not self._segment_clear(self.position, (x, 0.0, z)):
                reason = "blocked_segment"
            if reason != "held":
                waypoint = None
        if waypoint is None:
            # Do not straighten an A* bend through furniture or walls.
            for cell in reversed(path[1:min(6, len(path))]):
                x, z = self.grid_to_world(cell)
                if self._segment_clear(self.position, (x, 0.0, z)):
                    waypoint = tuple(int(v) for v in cell)
                    break
            self._navigation_waypoint = waypoint
            self._navigation_waypoint_origin = self.position[[0, 2]].copy()
        if waypoint is None:
            self.navigation_diagnostics = self._navigation_snapshot(target, "no_safe_segment")
            return None, -1
        x, z = self.grid_to_world(waypoint)
        angle = self._signed_angle(self.forward, self.position, (x, 0.0, z))
        facing = self.forward[[0, 2]]
        norm = np.linalg.norm(facing)
        end = self.position.copy()
        if norm > 0:
            end[[0, 2]] += FORWARD_DISTANCE * facing / norm
        if abs(angle) < self.config.turn_angle_degrees * 1.2 and norm > 0 and self._segment_clear(self.position, end):
            action = {"type": 0}
        elif abs(angle) > 1e-6:
            action = {"type": 1 if angle > 0 else 2}
        else:
            reason, action = "blocked_stride", None
        self.navigation_diagnostics = self._navigation_snapshot(target, reason)
        return action, len(path)

    def _navigation_snapshot(self, target, reason):
        return dict(frame=self.obs.get("current_frames"), position=self.position.tolist(),
                    target=list(target), endpoint=self._navigation_endpoint,
                    waypoint=self._navigation_waypoint,
                    reason=reason, avoid=self._navigation_avoid)

    def recover_navigation(self):
        """Replan around one stalled waypoint, without changing factual maps."""
        self._navigation_avoid = self._navigation_waypoint or self._navigation_avoid
        self._navigation_recovery_origin = self.position[[0, 2]].copy()
        self._navigation_waypoint = self._navigation_endpoint = None

    def is_navigable_position(self, position) -> bool:
        cell = self.world_to_grid(position)
        return self.clamp_grid(cell) == cell and not self._occupied(*cell)

    def _update_objects(self) -> None:
        assert self.obs is not None
        self.opponent_visible = False
        for raw in self.obs.get("visible_objects") or []:
            if not isinstance(raw, dict) or raw.get("id") is None:
                continue
            object_id = int(raw["id"])
            color = tuple(int(value) for value in raw["seg_color"])
            self.color_to_id[color] = object_id
            object_position = self._object_position(color)
            if object_position is None:
                continue
            item = dict(raw)
            item["position"] = object_position
            self.object_info[object_id] = item
            i, j = self.world_to_grid(object_position)
            object_type = int(raw.get("type", -1))
            if 0 <= i < self.object_map.shape[0] and 0 <= j < self.object_map.shape[1]:
                if self.object_map[i, j] == 0 and object_type != 3:
                    self.object_map[i, j] = object_type + 1
                    self.id_map[i, j] = object_id
            if object_type == 3 and object_id != self.agent_id:
                self.opponent_position = object_position
                self.opponent_visible = True

        self.current_obstacle_map.fill(0)
        if self.agent_id == 0:
            cells = np.where(self.object_map == 5)
            self.current_obstacle_map[cells] = self.id_map[cells]
            self.current_obstacle_map = self._convolve(
                self.current_obstacle_map, kernel=12
            ).astype(np.float32)

    def _object_position(self, color: Tuple[int, int, int]):
        points = self._point_cloud(color)
        if points.shape[1] < 5:
            return None
        return points.mean(axis=1)

    def _point_cloud(self, color: Tuple[int, int, int]) -> np.ndarray:
        assert self.obs is not None
        depth = np.asarray(self.obs["depth"], dtype=np.float64).copy()
        mask = np.any(np.asarray(self.obs["seg_mask"]) != np.asarray(color), axis=-1)
        depth[mask] = 1e9
        xx_factor, yy_factor = self._projection_factors(
            depth.shape, self.obs["FOV"]
        )
        xx = xx_factor * depth
        yy = yy_factor * depth
        selected = np.where((depth > 0) & (depth < 10))
        camera = np.stack((xx[selected], yy[selected], depth[selected],
                           np.ones_like(depth[selected])))
        inverse = np.linalg.inv(np.asarray(self.obs["camera_matrix"]).reshape(4, 4))
        inverse = inverse.dot(np.diag((1.0, -1.0, -1.0, 1.0)))
        return inverse.dot(camera)[:3]

    def _color_ids(self) -> np.ndarray:
        assert self.obs is not None
        mask = np.asarray(self.obs["seg_mask"])
        flat = mask.reshape(-1, mask.shape[-1])
        known_colors = (self.agent_color, *self.color_to_id.keys())
        byte_rgb = (
            flat.shape[1] == 3
            and np.all(flat >= 0)
            and np.all(flat <= 255)
            and np.all(flat == np.floor(flat))
            and all(
                np.asarray(color).shape == (3,)
                and np.all(np.asarray(color) >= 0)
                and np.all(np.asarray(color) <= 255)
                and np.all(np.asarray(color) == np.floor(np.asarray(color)))
                for color in known_colors
            )
        )
        if byte_rgb:
            values = flat.astype(np.uint32, copy=False)
            pixel_keys = (
                (values[:, 0] << 16) | (values[:, 1] << 8) | values[:, 2]
            )

            def pack(color):
                return ((int(color[0]) << 16)
                        | (int(color[1]) << 8)
                        | int(color[2]))

            lookup = {pack(self.agent_color): self.agent_id}
            lookup.update({
                pack(color): object_id
                for color, object_id in self.color_to_id.items()
            })
            known_keys = np.asarray(sorted(lookup), dtype=np.uint32)
            known_ids = np.asarray(
                [lookup[int(key)] for key in known_keys], dtype=np.int64
            )
            indices = np.searchsorted(known_keys, pixel_keys)
            matched = indices < len(known_keys)
            matched[matched] &= (
                known_keys[indices[matched]] == pixel_keys[matched]
            )
            result = np.full(flat.shape[0], -100, dtype=np.int64)
            result[matched] = known_ids[indices[matched]]
        else:
            result = np.full(flat.shape[0], -100, dtype=np.int64)
            result[np.all(flat == self.agent_color, axis=1)] = self.agent_id
            for color, object_id in self.color_to_id.items():
                result[np.all(flat == color, axis=1)] = object_id
        return result.reshape(mask.shape[:2])

    def _projection_factors(self, depth_shape, fov):
        """Cache only the immutable pixel geometry for one camera shape/FOV."""
        key = (tuple(depth_shape), float(fov))
        factors = self._projection_cache.get(key)
        if factors is None:
            height, width = depth_shape
            cx, cy = width / 2.0, height / 2.0
            focal = cx / math.tan(math.radians(float(fov) / 2.0))
            xx, yy = np.meshgrid(
                np.linspace(0, width - 1, width),
                np.linspace(0, height - 1, height),
            )
            factors = ((xx - cx) / focal, (yy - cy) / focal)
            for factor in factors:
                factor.flags.writeable = False
            self._projection_cache.clear()
            self._projection_cache[key] = factors
        return factors

    def _depth_to_maps(self) -> np.ndarray:
        assert self.obs is not None
        depth = np.asarray(self.obs["depth"], dtype=np.float64).copy()
        color_ids = self._color_ids()
        dynamic_ids = {0, 1}
        for name in ("held_objects", "oppo_held_objects"):
            dynamic_ids.update(int(item["id"]) for item in self.obs.get(name, [])
                               if isinstance(item, dict) and item.get("id") is not None)
        coop = self.obs.get("coop_info") or {}
        if coop.get("active") and coop.get("object_id") is not None:
            dynamic_ids.add(int(coop["object_id"]))
        static = ~np.isin(color_ids.reshape(-1), tuple(dynamic_ids))
        boundary = self._convolve(color_ids, 3) - color_ids * 9 != 0
        depth = np.maximum(depth, self._convolve(np.where(boundary, 1e9, 0.0), 5))
        xx_factor, yy_factor = self._projection_factors(
            depth.shape, self.obs["FOV"]
        )
        xx = xx_factor * depth
        yy = yy_factor * depth
        points = np.stack((xx, yy, depth, np.ones_like(depth))).reshape(4, -1)
        inverse = np.linalg.inv(np.asarray(self.obs["camera_matrix"]).reshape(4, 4))
        inverse = inverse.dot(np.diag((1.0, -1.0, -1.0, 1.0)))
        world = inverse.dot(points)
        flat_depth = depth.reshape(-1)
        grid_x = np.rint((world[0] - self.config.scene_x_min) / self.config.cell_size)
        grid_z = np.rint((world[2] - self.config.scene_z_min) / self.config.cell_size)
        grid_x = np.clip(grid_x, 0, self.config.map_size[0] - 1).astype(np.int32)
        grid_z = np.clip(grid_z, 0, self.config.map_size[1] - 1).astype(np.int32)
        local_known = np.zeros_like(self.known_map)

        selected = (static & (flat_depth > 0) & (flat_depth < self.config.detection_distance)
                    & (world[1] < 2 * self.config.wall_height))
        local_known[grid_x[selected], grid_z[selected]] = 1
        low = (static & (flat_depth > 0) & (flat_depth < self.config.navigation_distance)
               & (world[1] < 0.05))
        self.occupancy_map[grid_x[low], grid_z[low]] = 0
        occupied = (static & (flat_depth > 0) & (flat_depth < self.config.navigation_distance)
                    & (world[1] > 0.1) & (world[1] <= self.config.wall_height))
        self.occupancy_map[grid_x[occupied], grid_z[occupied]] = 1
        self.local_occupancy_map[grid_x[occupied], grid_z[occupied]] = 1
        removable = low & (self.local_occupancy_map[grid_x, grid_z] == 0)
        self.wall_map[grid_x[removable], grid_z[removable]] = 0
        wall = (static & (flat_depth > 0) & (flat_depth < 5)
                & (world[1] > self.config.wall_height)
                & (world[1] < 2 * self.config.wall_height))
        self.wall_map[grid_x[wall], grid_z[wall]] = 1
        true_wall = (static & (flat_depth > 0) & (flat_depth < 5)
                     & (world[1] > self.config.true_wall_height)
                     & (world[1] < 2 * self.config.wall_height))
        self.true_wall_map[grid_x[true_wall], grid_z[true_wall]] = 1
        # A peer is a current obstacle, never a permanent wall. The existing
        # dynamic layer is rebuilt on every observation and stays a finite cost.
        peer = ((color_ids.reshape(-1) == 1 - self.agent_id)
                & (flat_depth > 0) & (flat_depth < self.config.navigation_distance))
        self.current_obstacle_map[grid_x[peer], grid_z[peer]] = 1
        return local_known

    def _navigation_cost_map(self) -> np.ndarray:
        cost = np.ones_like(self.local_occupancy_map, dtype=np.float32)
        cost[self._convolve(self.local_occupancy_map, 5) > 0] = 5
        cost[self._convolve(self.local_occupancy_map, 3) > 0] = 10
        cost[self.known_map == 0] += 2.5
        cost[self.current_obstacle_map > 0] += 200000
        # Observed solid cores are forbidden; unknown cells and clearance
        # buffers retain their finite penalties and exploration capability.
        cost[(self.local_occupancy_map > 0) | (self.wall_map > 0)
             | (self.true_wall_map > 0)] = np.inf
        return cost

    @staticmethod
    def _astar_path(cost_map: np.ndarray, start, goal, goals=None):
        """Point A* or multi-goal A* with an admissible region lower bound."""
        if pyastar is not None and goals is None:
            try:
                return pyastar.astar_path(cost_map, start, goal, allow_diagonal=False)
            except RuntimeError:
                return None
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        if (start == goal if goals is None else goals[start]):
            return np.asarray([start], dtype=np.int32)
        rows, columns = cost_map.shape
        minimum_cost = float(np.min(cost_map))
        distances = None
        if goals is not None:
            from scipy.ndimage import distance_transform_cdt

            # Exact distance to the goal set with obstacles relaxed. Reuse the
            # existing SciPy dependency; avoid one point A* per candidate cell.
            distances = distance_transform_cdt(~goals, metric="taxicab")
        frontier = [(0.0, 0.0, start)]
        previous = {}
        best = {start: 0.0}
        while frontier:
            _, negative_cost, current = heapq.heappop(frontier)
            path_cost = -negative_cost
            if path_cost != best.get(current):
                continue
            if (current == goal if goals is None else goals[current]):
                path = [current]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                path.reverse()
                return np.asarray(path, dtype=np.int32)
            i, j = current
            for neighbor in ((i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j)):
                ni, nj = neighbor
                if not (0 <= ni < rows and 0 <= nj < columns):
                    continue
                candidate = path_cost + float(cost_map[ni, nj])
                if candidate >= best.get(neighbor, math.inf):
                    continue
                best[neighbor] = candidate
                previous[neighbor] = current
                heuristic = (abs(ni - goal[0]) + abs(nj - goal[1])
                             if distances is None else int(distances[ni, nj])) * minimum_cost
                heapq.heappush(frontier, (candidate + heuristic, -candidate, neighbor))
        return None

    def _nearest_unoccupied_grid(self, i: int, j: int):
        if not self._occupied(i, j):
            return i, j
        rows, columns = self.local_occupancy_map.shape
        max_radius = max(i + j, i + columns - 1 - j,
                         rows - 1 - i + j, rows - 1 - i + columns - 1 - j)
        for radius in range(1, max_radius + 1):
            candidates = []
            for delta_i in range(-radius, radius + 1):
                delta_j = radius - abs(delta_i)
                row = i + delta_i
                if not 0 <= row < rows:
                    continue
                columns_at_radius = (j,) if delta_j == 0 else (j - delta_j, j + delta_j)
                for column in columns_at_radius:
                    if 0 <= column < columns and not self._occupied(row, column):
                        candidates.append((row, column))
            if candidates:
                return min(candidates)
        return None

    def _occupied(self, i: int, j: int) -> bool:
        return bool(
            self.local_occupancy_map[i, j] > 0
            or self.wall_map[i, j] > 0
            or self.true_wall_map[i, j] > 0
            or self.current_obstacle_map[i, j] > 0
        )

    @staticmethod
    def _signed_angle(forward, origin, target) -> float:
        direction = np.asarray((target[0] - origin[0], target[2] - origin[2]),
                               dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            return 0.0
        direction /= norm
        facing = np.asarray((forward[0], forward[2]), dtype=np.float64)
        return math.degrees(math.atan2(
            facing[0] * direction[1] - facing[1] * direction[0],
            facing[0] * direction[0] + facing[1] * direction[1],
        ))

    @staticmethod
    def _convolve(values: np.ndarray, kernel: int) -> np.ndarray:
        from scipy.signal import convolve2d

        return convolve2d(values, np.ones((kernel, kernel)),
                          mode="same", boundary="fill")

    @staticmethod
    def _xz(value) -> Tuple[float, float]:
        if len(value) == 2:
            return float(value[0]), float(value[1])
        if len(value) == 3:
            return float(value[0]), float(value[2])
        raise ValueError("position must contain two or three coordinates")
