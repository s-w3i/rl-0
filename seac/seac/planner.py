"""Planner-guided per-agent recurrent local actor-critic with relevance-gated SEAC transfer for decentralized local conflict resolution."""

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch


_DIR_OFFSETS = (
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
)

_PLANNER_FEATURE_MODES = {"basic", "local"}
_POS_DIMS = 2
_NEXT_DIR_DIMS = 4


def _info_array(info, key, size):
    if key not in info:
        return None
    arr = np.asarray(info[key], dtype=np.float32).reshape(-1)
    if arr.size < size:
        padded = np.zeros(size, dtype=np.float32)
        padded[: arr.size] = arr
        return padded
    return arr[:size]


def planner_hint_dim(prefix_len):
    prefix_len = int(prefix_len)
    return 12 + 3 * prefix_len


def planner_pair_feature_dim():
    return 22


def _planner_slices(prefix_len):
    prefix_len = int(prefix_len)
    offset = 0
    slices = {}
    slices["has_plan"] = slice(offset, offset + 1)
    offset += 1
    slices["next_dir"] = slice(offset, offset + _NEXT_DIR_DIMS)
    offset += _NEXT_DIR_DIMS
    slices["waypoint_distance"] = slice(offset, offset + 1)
    offset += 1
    slices["planned_next_blocked"] = slice(offset, offset + 1)
    offset += 1
    slices["on_path"] = slice(offset, offset + 1)
    offset += 1
    slices["position"] = slice(offset, offset + _POS_DIMS)
    offset += _POS_DIMS
    slices["next_cell"] = slice(offset, offset + _POS_DIMS)
    offset += _POS_DIMS
    slices["prefix_mask"] = slice(offset, offset + prefix_len)
    offset += prefix_len
    slices["prefix_cells"] = slice(offset, offset + (2 * prefix_len))
    offset += 2 * prefix_len
    assert offset == planner_hint_dim(prefix_len)
    return slices


def _check_planner_feature_mode(planner_feature_mode):
    if planner_feature_mode not in _PLANNER_FEATURE_MODES:
        raise ValueError(
            f"Unsupported planner_feature_mode='{planner_feature_mode}'. "
            f"Supported modes: {sorted(_PLANNER_FEATURE_MODES)}."
        )


def _planner_feature_bounds(prefix_len):
    prefix_len = int(prefix_len)
    low = np.concatenate(
        [
            np.zeros(1, dtype=np.float32),
            np.zeros(_NEXT_DIR_DIMS, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(_POS_DIMS, dtype=np.float32),
            np.zeros(_POS_DIMS, dtype=np.float32),
            np.zeros(prefix_len, dtype=np.float32),
            np.zeros(2 * prefix_len, dtype=np.float32),
        ]
    )
    high = np.concatenate(
        [
            np.ones(1, dtype=np.float32),
            np.ones(_NEXT_DIR_DIMS, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.ones(_POS_DIMS, dtype=np.float32),
            np.ones(_POS_DIMS, dtype=np.float32),
            np.ones(prefix_len, dtype=np.float32),
            np.ones(2 * prefix_len, dtype=np.float32),
        ]
    )
    return low, high


def _space_uses_normalized_coordinates(space):
    high = np.asarray(space.high, dtype=np.float32).reshape(-1)
    low = np.asarray(space.low, dtype=np.float32).reshape(-1)
    return bool(high.size >= 2 and low.size >= 2 and high[0] <= 1.0 and high[1] <= 1.0)


def _local_planner_feature_bounds(prefix_len, base_space=None):
    prefix_len = int(prefix_len)
    low, high = _planner_feature_bounds(prefix_len)
    slices = _planner_slices(prefix_len)
    low[slices["position"]] = 0.0
    high[slices["position"]] = 0.0
    uses_normalized_coordinates = (
        True
        if base_space is None
        else _space_uses_normalized_coordinates(base_space)
    )
    if uses_normalized_coordinates:
        low[slices["next_cell"]] = -1.0
        high[slices["next_cell"]] = 1.0
        low[slices["prefix_cells"]] = -1.0
        high[slices["prefix_cells"]] = 1.0
    else:
        low[slices["next_cell"]] = -1.0
        high[slices["next_cell"]] = 1.0
        low[slices["prefix_cells"]] = -float(prefix_len)
        high[slices["prefix_cells"]] = float(prefix_len)
    return low, high


def _concat_box_space(space, extra_low, extra_high):
    if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
        raise ValueError("Planner hints require per-agent flattened Box observations.")
    low = np.concatenate(
        [np.asarray(space.low, dtype=np.float32).reshape(-1), extra_low], axis=0
    )
    high = np.concatenate(
        [np.asarray(space.high, dtype=np.float32).reshape(-1), extra_high], axis=0
    )
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def append_planner_observation_space(obs_space, prefix_len, planner_feature_mode="local"):
    if not isinstance(obs_space, gym.spaces.Tuple):
        raise ValueError("Planner hints expect a tuple observation space.")
    _check_planner_feature_mode(planner_feature_mode)
    return gym.spaces.Tuple(
        tuple(
            _concat_box_space(
                space,
                *(
                    _local_planner_feature_bounds(prefix_len, space)
                    if planner_feature_mode == "local"
                    else _planner_feature_bounds(prefix_len)
                ),
            )
            for space in obs_space.spaces
        )
    )


def extract_planner_hints(obs, prefix_len):
    slices = _planner_slices(prefix_len)
    hint_size = planner_hint_dim(prefix_len)
    planner_obs = obs[..., -hint_size:]
    return {key: planner_obs[..., value] for key, value in slices.items()}


def build_planner_context_pair(
    target_obs, source_obs, prefix_len, planner_feature_mode="local"
):
    _check_planner_feature_mode(planner_feature_mode)
    target = extract_planner_hints(target_obs, prefix_len)
    source = extract_planner_hints(source_obs, prefix_len)

    target_pos = _pair_position(target_obs, target, planner_feature_mode)
    source_pos = _pair_position(source_obs, source, planner_feature_mode)
    rel = source_pos - target_pos
    rel_dx = rel[..., 0:1]
    rel_dy = rel[..., 1:2]
    manhattan = rel_dx.abs() + rel_dy.abs()

    target_next = _pair_cell(target_pos, target["next_cell"], planner_feature_mode)
    source_next = _pair_cell(source_pos, source["next_cell"], planner_feature_mode)
    target_has_plan = target["has_plan"]
    source_has_plan = source["has_plan"]

    overlap_next = (
        (target_has_plan > 0.5)
        & (source_has_plan > 0.5)
        & torch.isclose(target_next[..., 0:1], source_next[..., 0:1], atol=1e-6)
        & torch.isclose(target_next[..., 1:2], source_next[..., 1:2], atol=1e-6)
    ).float()
    cross_next = (
        (target_has_plan > 0.5)
        & (source_has_plan > 0.5)
        & torch.isclose(target_next[..., 0:1], source_pos[..., 0:1], atol=1e-6)
        & torch.isclose(target_next[..., 1:2], source_pos[..., 1:2], atol=1e-6)
        & torch.isclose(source_next[..., 0:1], target_pos[..., 0:1], atol=1e-6)
        & torch.isclose(source_next[..., 1:2], target_pos[..., 1:2], atol=1e-6)
    ).float()

    shared_prefix_zone = _shared_prefix_zone(
        target,
        source,
        prefix_len,
        target_pos=target_pos,
        source_pos=source_pos,
        planner_feature_mode=planner_feature_mode,
    )

    return torch.cat(
        [
            rel_dx,
            rel_dy,
            manhattan,
            target["next_dir"],
            source["next_dir"],
            target["waypoint_distance"],
            source["waypoint_distance"],
            target["planned_next_blocked"],
            source["planned_next_blocked"],
            overlap_next,
            cross_next,
            shared_prefix_zone,
            target["on_path"],
            source["on_path"],
            target_has_plan,
            source_has_plan,
        ],
        dim=-1,
    )


def planner_context_risk_score(pair_features):
    """Estimate when cross-agent transfer should matter from planner pair features."""
    target_blocked = pair_features[..., 13:14]
    source_blocked = pair_features[..., 14:15]
    overlap_next = pair_features[..., 15:16]
    cross_next = pair_features[..., 16:17]
    shared_prefix_zone = pair_features[..., 17:18]
    target_has_plan = pair_features[..., 20:21]
    source_has_plan = pair_features[..., 21:22]
    both_planned = (target_has_plan > 0.5).float() * (source_has_plan > 0.5).float()
    blocked = torch.maximum(target_blocked, source_blocked).clamp(0.0, 1.0)
    return both_planned * torch.maximum(
        torch.maximum(overlap_next, cross_next),
        torch.maximum(shared_prefix_zone, 0.75 * blocked),
    ).clamp(0.0, 1.0)


def planner_context_gate_target(pair_features, min_weight=0.25, high_weight=0.95):
    risk = planner_context_risk_score(pair_features.detach())
    min_weight = float(min_weight)
    low_weight = min(1.0, min_weight + 0.10 * (1.0 - min_weight))
    high_weight = max(low_weight, min(1.0, float(high_weight)))
    return low_weight + (high_weight - low_weight) * risk


def _pair_position(obs, hints, planner_feature_mode):
    if planner_feature_mode == "local" and obs.shape[-1] >= 2:
        return obs[..., 0:2]
    return hints["position"]


def _pair_cell(position, cell, planner_feature_mode):
    if planner_feature_mode == "local":
        return position + cell
    return cell


def _shared_prefix_zone(
    target,
    source,
    prefix_len,
    target_pos=None,
    source_pos=None,
    planner_feature_mode="local",
):
    if prefix_len <= 0:
        return torch.zeros_like(target["has_plan"])

    target_mask = target["prefix_mask"]
    source_mask = source["prefix_mask"]
    target_cells = target["prefix_cells"].view(*target["prefix_mask"].shape[:-1], prefix_len, 2)
    source_cells = source["prefix_cells"].view(*source["prefix_mask"].shape[:-1], prefix_len, 2)
    if planner_feature_mode == "local":
        target_cells = target_cells + target_pos.unsqueeze(-2)
        source_cells = source_cells + source_pos.unsqueeze(-2)

    matches = []
    for t_idx in range(prefix_len):
        for s_idx in range(prefix_len):
            both_valid = (
                (target_mask[..., t_idx : t_idx + 1] > 0.5)
                & (source_mask[..., s_idx : s_idx + 1] > 0.5)
            )
            same_cell = (
                torch.isclose(
                    target_cells[..., t_idx, 0:1], source_cells[..., s_idx, 0:1], atol=1e-6
                )
                & torch.isclose(
                    target_cells[..., t_idx, 1:2], source_cells[..., s_idx, 1:2], atol=1e-6
                )
            )
            matches.append((both_valid & same_cell).float())

    if not matches:
        return torch.zeros_like(target["has_plan"])
    return torch.stack(matches, dim=0).amax(dim=0)


class AStarPlanner:
    def __init__(self, planner_type="astar"):
        if planner_type != "astar":
            raise ValueError(f"Unsupported planner_type='{planner_type}'.")
        self.planner_type = planner_type

    def plan(self, start, goal, obstacles, allowed_dirs):
        start = tuple(int(v) for v in start)
        goal = tuple(int(v) for v in goal)
        if start == goal:
            return [start]

        blocked = np.asarray(obstacles, dtype=np.bool_).copy()
        if (
            0 <= goal[0] < blocked.shape[1]
            and 0 <= goal[1] < blocked.shape[0]
            and blocked[goal[1], goal[0]]
        ):
            blocked[goal[1], goal[0]] = False

        frontier = []
        heappush(frontier, (self._heuristic(start, goal), 0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current_cost, current = heappop(frontier)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current, blocked, allowed_dirs):
                new_cost = current_cost + 1
                if neighbor in cost_so_far and new_cost >= cost_so_far[neighbor]:
                    continue
                cost_so_far[neighbor] = new_cost
                priority = new_cost + self._heuristic(neighbor, goal)
                heappush(frontier, (priority, new_cost, neighbor))
                came_from[neighbor] = current

        return [start]

    def _heuristic(self, node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def _neighbors(self, current, obstacles, allowed_dirs):
        x, y = current
        max_y, max_x = obstacles.shape
        for dir_idx, (dx, dy) in enumerate(_DIR_OFFSETS):
            if not bool(allowed_dirs[y, x, dir_idx]):
                continue
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= max_x or ny >= max_y:
                continue
            if obstacles[ny, nx]:
                continue
            yield (nx, ny)

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


@dataclass
class PlannerState:
    goal: Optional[Tuple[int, int]] = None
    path: List[Tuple[int, int]] = field(default_factory=list)
    steps_since_replan: int = 0
    last_on_path: bool = True


class PlannerObservationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        planner_type="astar",
        planner_recompute_interval=1,
        planner_prefix_len=4,
        planner_feature_mode="local",
        blocked_penalty=0.0,
        agent_blocked_penalty=0.0,
        swap_penalty=0.0,
        blocked_wait_bonus=0.0,
        unblocked_deviation_penalty=0.0,
        planner_follow_bonus=0.0,
        conflict_clear_bonus=0.0,
        persistent_agent_blocked_penalty=0.0,
        persistent_agent_block_threshold=5,
        planner_action_follow_bonus=0.0,
        planner_turn_to_plan_bonus=0.0,
        task_progress_bonus=0.0,
        task_regress_penalty=0.0,
        pickup_bonus=0.0,
        delivery_bonus=0.0,
        no_progress_penalty=0.0,
        no_progress_threshold=25,
        rotation_penalty=0.0,
        idle_penalty=0.0,
        toggle_penalty=0.0,
    ):
        _check_planner_feature_mode(planner_feature_mode)
        super().__init__(env)
        self._backend = _make_backend(
            env.unwrapped,
            planner_type=planner_type,
            planner_recompute_interval=planner_recompute_interval,
            planner_prefix_len=planner_prefix_len,
            planner_feature_mode=planner_feature_mode,
        )
        self.blocked_penalty = float(blocked_penalty)
        self.agent_blocked_penalty = float(agent_blocked_penalty)
        self.swap_penalty = float(swap_penalty)
        self.blocked_wait_bonus = float(blocked_wait_bonus)
        self.unblocked_deviation_penalty = float(unblocked_deviation_penalty)
        self.planner_follow_bonus = float(planner_follow_bonus)
        self.conflict_clear_bonus = float(conflict_clear_bonus)
        self.persistent_agent_blocked_penalty = float(
            persistent_agent_blocked_penalty
        )
        self.persistent_agent_block_threshold = max(
            1, int(persistent_agent_block_threshold)
        )
        self.planner_action_follow_bonus = float(planner_action_follow_bonus)
        self.planner_turn_to_plan_bonus = float(planner_turn_to_plan_bonus)
        self.task_progress_bonus = float(task_progress_bonus)
        self.task_regress_penalty = float(task_regress_penalty)
        self.pickup_bonus = float(pickup_bonus)
        self.delivery_bonus = float(delivery_bonus)
        self.no_progress_penalty = float(no_progress_penalty)
        self.no_progress_threshold = max(1, int(no_progress_threshold))
        self.rotation_penalty = float(rotation_penalty)
        self.idle_penalty = float(idle_penalty)
        self.toggle_penalty = float(toggle_penalty)
        self.observation_space = append_planner_observation_space(
            env.observation_space, planner_prefix_len, planner_feature_mode
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._backend.reset(obs)
        return obs, self._backend.attach_info(info)

    def step(self, action):
        step_cache = self._backend.before_step(action)
        task_cache = self._task_cache()
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = np.asarray(reward, dtype=np.float32)
        shaped = reward.copy()

        pre_next_cells = step_cache["next_cells"]
        pre_has_plan = step_cache["has_plan"]
        pre_blocked = step_cache["blocked"]
        action_values = step_cache["action_values"]
        shaping_by_agent = np.zeros_like(shaped, dtype=np.float32)
        action_shaping_by_agent = np.zeros_like(shaped, dtype=np.float32)
        task_shaping_by_agent = np.zeros_like(shaped, dtype=np.float32)
        loop_shaping_by_agent = np.zeros_like(shaped, dtype=np.float32)
        blocked_static_by_agent = _info_array(
            info, "step_blocked_static_by_agent", len(shaped)
        )
        blocked_agent_by_agent = _info_array(
            info, "step_blocked_agent_by_agent", len(shaped)
        )
        vertex_by_agent = _info_array(info, "step_vertex_by_agent", len(shaped))
        swap_by_agent = _info_array(info, "step_swap_by_agent", len(shaped))
        conflict_clear_by_agent = _info_array(
            info, "step_conflict_clear_by_agent", len(shaped)
        )
        consecutive_agent_blocked = _info_array(
            info, "agent_consecutive_agent_blocked", len(shaped)
        )
        consecutive_no_progress = _info_array(
            info, "agent_consecutive_no_progress", len(shaped)
        )
        moved_by_agent = _info_array(info, "step_moved_by_agent", len(shaped))
        has_block_breakdown = (
            blocked_static_by_agent is not None or blocked_agent_by_agent is not None
        )
        for idx, agent in enumerate(self._backend.env.agents):
            if pre_has_plan[idx] <= 0.5:
                continue
            agent_conflict_blocked = (
                (
                    blocked_agent_by_agent is not None
                    and blocked_agent_by_agent[idx] > 0
                )
                or (vertex_by_agent is not None and vertex_by_agent[idx] > 0)
            )
            if (
                not has_block_breakdown
                and pre_blocked[idx] > 0.5
                and self.blocked_penalty != 0.0
            ):
                shaping_by_agent[idx] -= self.blocked_penalty
            if blocked_static_by_agent is not None and blocked_static_by_agent[idx] > 0:
                shaping_by_agent[idx] -= self.blocked_penalty
            if agent_conflict_blocked:
                shaping_by_agent[idx] -= self.agent_blocked_penalty
                if (
                    consecutive_agent_blocked is not None
                    and self.persistent_agent_blocked_penalty != 0.0
                    and consecutive_agent_blocked[idx]
                    >= self.persistent_agent_block_threshold
                ):
                    over_threshold = (
                        consecutive_agent_blocked[idx]
                        - self.persistent_agent_block_threshold
                        + 1
                    )
                    shaping_by_agent[idx] -= (
                        self.persistent_agent_blocked_penalty * over_threshold
                    )
            elif pre_blocked[idx] > 0.5 and action_values[idx] != 0:
                shaping_by_agent[idx] -= self.agent_blocked_penalty
            if swap_by_agent is not None and swap_by_agent[idx] > 0:
                shaping_by_agent[idx] -= self.swap_penalty
            if (
                conflict_clear_by_agent is not None
                and conflict_clear_by_agent[idx] > 0
            ):
                shaping_by_agent[idx] += self.conflict_clear_bonus

            follows_plan_action = self._backend.action_follows_plan(
                idx, action_values[idx], step_cache
            )
            turns_to_plan = self._backend.action_turns_to_plan(
                idx, action_values[idx], step_cache
            )
            stepped_to_plan = np.allclose(
                np.asarray(pre_next_cells[idx], dtype=np.float32),
                np.asarray(
                    self._backend._normalized_position(agent.x, agent.y),
                    dtype=np.float32,
                ),
                atol=1e-6,
            )
            if stepped_to_plan and self.planner_follow_bonus != 0.0:
                action_shaping_by_agent[idx] += self.planner_follow_bonus
            if follows_plan_action and self.planner_action_follow_bonus != 0.0:
                action_shaping_by_agent[idx] += self.planner_action_follow_bonus
            elif turns_to_plan and self.planner_turn_to_plan_bonus != 0.0:
                action_shaping_by_agent[idx] += self.planner_turn_to_plan_bonus
            elif (
                pre_blocked[idx] > 0.5
                and action_values[idx] == 0
                and self.blocked_wait_bonus != 0.0
            ):
                action_shaping_by_agent[idx] += self.blocked_wait_bonus
            elif (
                pre_blocked[idx] <= 0.5
                and not follows_plan_action
                and not turns_to_plan
                and self.unblocked_deviation_penalty != 0.0
            ):
                action_shaping_by_agent[idx] -= self.unblocked_deviation_penalty

            task_delta = self._task_distance_delta(task_cache[idx], idx)
            if task_delta > 0 and self.task_progress_bonus != 0.0:
                task_shaping_by_agent[idx] += self.task_progress_bonus * task_delta
            elif task_delta < 0 and self.task_regress_penalty != 0.0:
                task_shaping_by_agent[idx] -= self.task_regress_penalty * abs(task_delta)

            if self._picked_up_requested_shelf(task_cache[idx], idx):
                task_shaping_by_agent[idx] += self.pickup_bonus
            deliveries = self._delivery_delta(task_cache[idx], idx)
            if deliveries > 0:
                task_shaping_by_agent[idx] += self.delivery_bonus * deliveries

            no_progress_count = (
                0
                if consecutive_no_progress is None
                else int(consecutive_no_progress[idx])
            )
            if (
                self.no_progress_penalty != 0.0
                and no_progress_count >= self.no_progress_threshold
            ):
                over_threshold = min(
                    10, no_progress_count - self.no_progress_threshold + 1
                )
                loop_shaping_by_agent[idx] -= self.no_progress_penalty * over_threshold
            if (
                self.rotation_penalty != 0.0
                and action_values[idx] in {2, 3}
                and (moved_by_agent is None or moved_by_agent[idx] <= 0)
            ):
                loop_shaping_by_agent[idx] -= self.rotation_penalty
            if (
                self.idle_penalty != 0.0
                and action_values[idx] == 0
                and pre_blocked[idx] <= 0.5
                and (moved_by_agent is None or moved_by_agent[idx] <= 0)
            ):
                loop_shaping_by_agent[idx] -= self.idle_penalty
            if (
                self.toggle_penalty != 0.0
                and action_values[idx] == 4
                and not self._toggle_changed_load(task_cache[idx], idx)
                and deliveries <= 0
            ):
                loop_shaping_by_agent[idx] -= self.toggle_penalty

        shaping_by_agent += (
            action_shaping_by_agent + task_shaping_by_agent + loop_shaping_by_agent
        )
        shaped += shaping_by_agent
        obs = self._backend.step(obs, step_cache=step_cache)
        info = self._backend.attach_info(info)
        info["conflict_shaping_sum"] = float(shaping_by_agent.sum())
        info["conflict_shaping_mean"] = float(shaping_by_agent.mean())
        info["conflict_shaping_by_agent"] = [
            float(v) for v in np.asarray(shaping_by_agent).reshape(-1)
        ]
        info["planner_action_shaping_sum"] = float(action_shaping_by_agent.sum())
        info["planner_task_shaping_sum"] = float(task_shaping_by_agent.sum())
        info["planner_loop_shaping_sum"] = float(loop_shaping_by_agent.sum())
        info["planner_action_shaping_by_agent"] = [
            float(v) for v in np.asarray(action_shaping_by_agent).reshape(-1)
        ]
        info["planner_task_shaping_by_agent"] = [
            float(v) for v in np.asarray(task_shaping_by_agent).reshape(-1)
        ]
        info["planner_loop_shaping_by_agent"] = [
            float(v) for v in np.asarray(loop_shaping_by_agent).reshape(-1)
        ]
        return obs, shaped, terminated, truncated, info

    def _task_cache(self):
        cache = []
        for idx, agent in enumerate(self._backend.env.agents):
            target = self._task_target(idx)
            distance = self._task_distance(agent, target)
            cache.append(
                {
                    "target": target,
                    "distance": distance,
                    "carrying": agent.carrying_shelf,
                    "delivery_count": self._delivery_count(idx),
                }
            )
        return cache

    def _task_target(self, agent_idx):
        env = self._backend.env
        agent = env.agents[agent_idx]
        if agent.carrying_shelf is not None:
            shelf = agent.carrying_shelf
            if getattr(shelf, "delivered", False):
                return (int(shelf.home_x), int(shelf.home_y))
            if env.goals:
                return min(
                    (tuple(goal) for goal in env.goals),
                    key=lambda goal: abs(goal[0] - agent.x) + abs(goal[1] - agent.y),
                )
            return None
        if getattr(env, "dedicated_requests", False) and getattr(
            env, "assigned_shelves", None
        ):
            shelf = env.assigned_shelves[agent_idx]
            return (int(shelf.x), int(shelf.y))
        candidates = list(getattr(env, "request_queue", []) or [])
        if not candidates:
            return None
        shelf = min(
            candidates,
            key=lambda item: abs(int(item.x) - agent.x) + abs(int(item.y) - agent.y),
        )
        return (int(shelf.x), int(shelf.y))

    def _task_distance(self, agent, target):
        if target is None:
            return None
        return abs(int(target[0]) - int(agent.x)) + abs(int(target[1]) - int(agent.y))

    def _task_distance_delta(self, cached, agent_idx):
        if cached["distance"] is None:
            return 0
        target = cached["target"]
        if target is None:
            return 0
        distance = self._task_distance(self._backend.env.agents[agent_idx], target)
        if distance is None:
            return 0
        return int(cached["distance"]) - int(distance)

    def _delivery_count(self, agent_idx):
        values = getattr(self._backend.env, "_delivery_count_by_agent", None)
        if values is None or agent_idx >= len(values):
            return 0
        return int(values[agent_idx])

    def _delivery_delta(self, cached, agent_idx):
        return max(0, self._delivery_count(agent_idx) - int(cached["delivery_count"]))

    def _picked_up_requested_shelf(self, cached, agent_idx):
        env = self._backend.env
        agent = env.agents[agent_idx]
        if cached["carrying"] is not None or agent.carrying_shelf is None:
            return False
        try:
            return bool(env._is_requested_shelf(agent, agent.carrying_shelf))
        except AttributeError:
            return False

    def _toggle_changed_load(self, cached, agent_idx):
        agent = self._backend.env.agents[agent_idx]
        return cached["carrying"] is not agent.carrying_shelf


def maybe_add_planner_hints(
    env,
    use_global_planner=False,
    planner_type="astar",
    planner_recompute_interval=1,
    planner_prefix_len=4,
    planner_feature_mode="local",
    blocked_penalty=0.0,
    agent_blocked_penalty=0.0,
    swap_penalty=0.0,
    blocked_wait_bonus=0.0,
    unblocked_deviation_penalty=0.0,
    planner_follow_bonus=0.0,
    conflict_clear_bonus=0.0,
    persistent_agent_blocked_penalty=0.0,
    persistent_agent_block_threshold=5,
    planner_action_follow_bonus=0.0,
    planner_turn_to_plan_bonus=0.0,
    task_progress_bonus=0.0,
    task_regress_penalty=0.0,
    pickup_bonus=0.0,
    delivery_bonus=0.0,
    no_progress_penalty=0.0,
    no_progress_threshold=25,
    rotation_penalty=0.0,
    idle_penalty=0.0,
    toggle_penalty=0.0,
):
    if not use_global_planner:
        return env
    return PlannerObservationWrapper(
        env,
        planner_type=planner_type,
        planner_recompute_interval=planner_recompute_interval,
        planner_prefix_len=planner_prefix_len,
        planner_feature_mode=planner_feature_mode,
        blocked_penalty=blocked_penalty,
        agent_blocked_penalty=agent_blocked_penalty,
        swap_penalty=swap_penalty,
        blocked_wait_bonus=blocked_wait_bonus,
        unblocked_deviation_penalty=unblocked_deviation_penalty,
        planner_follow_bonus=planner_follow_bonus,
        conflict_clear_bonus=conflict_clear_bonus,
        persistent_agent_blocked_penalty=persistent_agent_blocked_penalty,
        persistent_agent_block_threshold=persistent_agent_block_threshold,
        planner_action_follow_bonus=planner_action_follow_bonus,
        planner_turn_to_plan_bonus=planner_turn_to_plan_bonus,
        task_progress_bonus=task_progress_bonus,
        task_regress_penalty=task_regress_penalty,
        pickup_bonus=pickup_bonus,
        delivery_bonus=delivery_bonus,
        no_progress_penalty=no_progress_penalty,
        no_progress_threshold=no_progress_threshold,
        rotation_penalty=rotation_penalty,
        idle_penalty=idle_penalty,
        toggle_penalty=toggle_penalty,
    )


def _make_backend(
    env,
    planner_type,
    planner_recompute_interval,
    planner_prefix_len,
    planner_feature_mode="local",
):
    required = (
        "agents",
        "shelfs",
        "goals",
        "grid_size",
        "obstacles",
        "_cell_allowed_dirs",
        "request_queue",
    )
    missing = [name for name in required if not hasattr(env, name)]
    if missing:
        raise ValueError(
            "Global planner requires a warehouse-style grid env with attributes "
            + ", ".join(missing)
        )
    return RwarePlannerBackend(
        env,
        planner_type=planner_type,
        planner_recompute_interval=planner_recompute_interval,
        planner_prefix_len=planner_prefix_len,
        planner_feature_mode=planner_feature_mode,
    )


class RwarePlannerBackend:
    def __init__(
        self,
        env,
        planner_type="astar",
        planner_recompute_interval=1,
        planner_prefix_len=4,
        planner_feature_mode="local",
    ):
        self.env = env
        self.planner = AStarPlanner(planner_type)
        self.planner_recompute_interval = max(1, int(planner_recompute_interval))
        self.planner_prefix_len = max(1, int(planner_prefix_len))
        _check_planner_feature_mode(planner_feature_mode)
        self.planner_feature_mode = planner_feature_mode
        self.states = [PlannerState() for _ in range(self.env.n_agents)]
        self._episode_steps = 0
        self._path_adherence_total = 0
        self._planned_agent_step_total = 0
        self._planner_follow_total = 0
        self._blocked_planned_step_total = 0
        self._blocked_wait_total = 0
        self._blocked_rotate_total = 0
        self._blocked_other_total = 0
        self._unblocked_deviation_total = 0
        self._last_raw_obs = None
        self._last_augmented_obs = None
        self._last_step_metrics = None

    def reset(self, obs):
        self.states = [PlannerState() for _ in range(self.env.n_agents)]
        self._episode_steps = 0
        self._path_adherence_total = 0
        self._planned_agent_step_total = 0
        self._planner_follow_total = 0
        self._blocked_planned_step_total = 0
        self._blocked_wait_total = 0
        self._blocked_rotate_total = 0
        self._blocked_other_total = 0
        self._unblocked_deviation_total = 0
        self._last_step_metrics = {
            "planner_step_followed": 0,
            "planner_step_blocked_wait": 0,
            "planner_step_blocked_rotate": 0,
            "planner_step_blocked_other": 0,
            "planner_step_unblocked_deviation": 0,
        }
        self._refresh_plans()
        self._last_augmented_obs = self._augment_obs(obs)
        return self._last_augmented_obs

    def before_step(self, action=None):
        raw_obs = self._last_raw_obs
        return {
            "next_cells": [self._normalized_next_cell(state) for state in self.states],
            "current_grid_cells": [
                (int(agent.x), int(agent.y)) for agent in self.env.agents
            ],
            "next_grid_cells": [
                self._grid_next_cell(state, idx)
                for idx, state in enumerate(self.states)
            ],
            "directions": [agent.dir for agent in self.env.agents],
            "has_plan": [float(len(state.path) > 1) for state in self.states],
            "blocked": [
                self._planned_next_blocked(
                    idx, state, None if raw_obs is None else raw_obs[idx]
                )
                for idx, state in enumerate(self.states)
            ],
            "action_values": self._base_action_values(action),
        }

    def step(self, obs, step_cache=None):
        if step_cache is None:
            step_cache = self.before_step()
        pre_next_cells = step_cache["next_cells"]
        pre_has_plan = step_cache["has_plan"]
        pre_blocked = step_cache["blocked"]
        action_values = step_cache["action_values"]

        self._refresh_plans()
        self._episode_steps += 1
        self._path_adherence_total += sum(int(state.last_on_path) for state in self.states)
        self._blocked_planned_step_total += sum(pre_blocked)

        planned_agent_steps = 0
        followed = 0
        blocked_wait = 0
        blocked_rotate = 0
        blocked_other = 0
        unblocked_deviation = 0
        for idx, agent in enumerate(self.env.agents):
            if pre_has_plan[idx] <= 0.5:
                continue
            planned_agent_steps += 1
            stepped_to_plan = np.allclose(
                np.asarray(pre_next_cells[idx], dtype=np.float32),
                np.asarray(self._normalized_position(agent.x, agent.y), dtype=np.float32),
                atol=1e-6,
            )
            if stepped_to_plan:
                followed += 1
            elif pre_blocked[idx] > 0.5 and action_values[idx] == 0:
                blocked_wait += 1
            elif pre_blocked[idx] > 0.5 and action_values[idx] in {2, 3}:
                blocked_rotate += 1
            elif pre_blocked[idx] > 0.5:
                blocked_other += 1
            else:
                unblocked_deviation += 1
        self._planned_agent_step_total += planned_agent_steps
        self._planner_follow_total += followed
        self._blocked_wait_total += blocked_wait
        self._blocked_rotate_total += blocked_rotate
        self._blocked_other_total += blocked_other
        self._unblocked_deviation_total += unblocked_deviation
        self._last_step_metrics = {
            "planner_step_followed": int(followed),
            "planner_step_blocked_wait": int(blocked_wait),
            "planner_step_blocked_rotate": int(blocked_rotate),
            "planner_step_blocked_other": int(blocked_other),
            "planner_step_unblocked_deviation": int(unblocked_deviation),
        }

        self._last_augmented_obs = self._augment_obs(obs)
        return self._last_augmented_obs

    def attach_info(self, info):
        info = dict(info)
        total_agent_steps = max(1, self._episode_steps * self.env.n_agents)
        current_hints = [
            self._build_hint(
                idx, None if self._last_raw_obs is None else self._last_raw_obs[idx]
            )
            for idx in range(self.env.n_agents)
        ]
        on_path_by_agent = [int(h[7] > 0.5) for h in current_hints]
        blocked_by_agent = [int(h[6] > 0.5) for h in current_hints]
        info["path_adherence_rate"] = float(self._path_adherence_total / total_agent_steps)
        info["blocked_planned_step_frequency"] = float(
            self._blocked_planned_step_total / total_agent_steps
        )
        planned_agent_steps = max(1, self._planned_agent_step_total)
        info["planner_follow_rate"] = float(self._planner_follow_total / planned_agent_steps)
        # This counts "blocked and did not follow the nominal next cell", which may be
        # waiting/yielding via NOOP. Rotation is reported separately because it does
        # not physically clear a blocked cell.
        info["blocked_wait_rate"] = float(self._blocked_wait_total / planned_agent_steps)
        info["blocked_rotate_rate"] = float(
            self._blocked_rotate_total / planned_agent_steps
        )
        info["blocked_other_rate"] = float(
            self._blocked_other_total / planned_agent_steps
        )
        info["unblocked_deviation_rate"] = float(
            self._unblocked_deviation_total / planned_agent_steps
        )
        info["local_deviation_count"] = int(self._unblocked_deviation_total)
        info["planner_step_on_path"] = int(sum(on_path_by_agent))
        info["planner_step_blocked_planned"] = int(sum(blocked_by_agent))
        if self._last_step_metrics is not None:
            info.update(self._last_step_metrics)
        info["planner_on_path_by_agent"] = on_path_by_agent
        info["planner_blocked_planned_by_agent"] = blocked_by_agent
        return info

    def _augment_obs(self, obs):
        raw_obs = tuple(np.asarray(agent_obs, dtype=np.float32) for agent_obs in obs)
        self._last_raw_obs = raw_obs
        hints = [self._build_hint(idx, raw_obs[idx]) for idx in range(self.env.n_agents)]
        return tuple(
            np.concatenate(
                [raw_obs[idx], hints[idx]],
                axis=-1,
            ).astype(np.float32)
            for idx in range(self.env.n_agents)
        )

    def _base_action_values(self, action):
        if action is None:
            return [-1 for _ in range(self.env.n_agents)]
        values = []
        for agent_action in action:
            arr = np.asarray(agent_action)
            if arr.shape == ():
                values.append(int(arr))
            else:
                values.append(int(arr.reshape(-1)[0]))
        return values

    def action_follows_plan(self, agent_idx, action_value, step_cache):
        return self._plan_action_kind(agent_idx, action_value, step_cache) == "forward"

    def action_turns_to_plan(self, agent_idx, action_value, step_cache):
        return self._plan_action_kind(agent_idx, action_value, step_cache) == "turn"

    def _plan_action_kind(self, agent_idx, action_value, step_cache):
        if action_value not in {1, 2, 3}:
            return None
        if step_cache["has_plan"][agent_idx] <= 0.5:
            return None
        current = step_cache["current_grid_cells"][agent_idx]
        next_cell = step_cache["next_grid_cells"][agent_idx]
        desired_dir = self._direction_between(current, next_cell)
        if desired_dir is None:
            return None
        current_dir = step_cache["directions"][agent_idx]
        if current_dir == desired_dir and action_value == 1:
            return "forward"
        if action_value in {2, 3}:
            turn_dir = self._direction_after_turn(current_dir, action_value)
            if turn_dir == desired_dir:
                return "turn"
        return None

    def _grid_next_cell(self, state, agent_idx):
        if len(state.path) > 1:
            return tuple(int(v) for v in state.path[1])
        agent = self.env.agents[agent_idx]
        return (int(agent.x), int(agent.y))

    def _direction_between(self, current, next_cell):
        dx = int(next_cell[0]) - int(current[0])
        dy = int(next_cell[1]) - int(current[1])
        direction_cls = type(self.env.agents[0].dir)
        if dx == 0 and dy == -1:
            return direction_cls.UP
        if dx == 0 and dy == 1:
            return direction_cls.DOWN
        if dx == -1 and dy == 0:
            return direction_cls.LEFT
        if dx == 1 and dy == 0:
            return direction_cls.RIGHT
        return None

    def _direction_after_turn(self, direction, action_value):
        direction_cls = type(direction)
        wraplist = [
            direction_cls.UP,
            direction_cls.RIGHT,
            direction_cls.DOWN,
            direction_cls.LEFT,
        ]
        offset = 1 if action_value == 3 else -1
        return wraplist[(wraplist.index(direction) + offset) % len(wraplist)]

    def _refresh_plans(self):
        for idx, agent in enumerate(self.env.agents):
            state = self.states[idx]
            current = (agent.x, agent.y)
            was_on_path = bool(state.path) and current in state.path
            state.last_on_path = was_on_path if state.path else True

            goal, path = self._best_goal_and_path(idx)
            should_replan = (
                not state.path
                or state.goal != goal
                or not was_on_path
                or not self._path_is_valid(state.path, idx)
                or state.steps_since_replan >= self.planner_recompute_interval
            )
            if should_replan:
                state.goal = goal
                state.path = path
                state.steps_since_replan = 0
            else:
                path_index = state.path.index(current)
                state.path = state.path[path_index:]
                state.steps_since_replan += 1

    def _best_goal_and_path(self, agent_idx):
        agent = self.env.agents[agent_idx]
        current = (agent.x, agent.y)
        candidates = self._candidate_goals(agent_idx)
        if not candidates:
            return current, [current]

        obstacles = self._static_obstacles(agent_idx)
        allowed_dirs = np.asarray(self.env._cell_allowed_dirs, dtype=np.uint8)

        best_goal = current
        best_path = [current]
        best_cost = None
        for goal in candidates:
            path = self.planner.plan(current, goal, obstacles, allowed_dirs)
            if not path:
                continue
            if path[-1] != goal:
                continue
            cost = len(path)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_goal = goal
                best_path = path

        return best_goal, best_path

    def _candidate_goals(self, agent_idx):
        agent = self.env.agents[agent_idx]
        if agent.carrying_shelf:
            shelf = agent.carrying_shelf
            if getattr(shelf, "delivered", False):
                return [(int(shelf.home_x), int(shelf.home_y))]
            return [tuple(goal) for goal in self.env.goals]

        if getattr(self.env, "dedicated_requests", False) and getattr(
            self.env, "assigned_shelves", None
        ):
            assigned = self.env.assigned_shelves[agent_idx]
            return [(int(assigned.x), int(assigned.y))]

        return [(int(shelf.x), int(shelf.y)) for shelf in self.env.request_queue]

    def _static_obstacles(self, agent_idx):
        obstacles = np.asarray(self.env.obstacles, dtype=np.bool_).copy()
        agent = self.env.agents[agent_idx]
        if not agent.carrying_shelf:
            return obstacles
        for shelf in self.env.shelfs:
            if shelf is agent.carrying_shelf:
                continue
            obstacles[shelf.y, shelf.x] = True
        obstacles[agent.y, agent.x] = False
        return obstacles

    def _path_is_valid(self, path, agent_idx):
        if not path:
            return False
        agent = self.env.agents[agent_idx]
        current = (agent.x, agent.y)
        if current not in path:
            return False

        obstacles = self._static_obstacles(agent_idx)
        allowed_dirs = np.asarray(self.env._cell_allowed_dirs, dtype=np.uint8)
        path_index = path.index(current)
        trimmed = path[path_index:]
        for start, end in zip(trimmed, trimmed[1:]):
            dir_idx = self._dir_index(start, end)
            if dir_idx is None:
                return False
            if not bool(allowed_dirs[start[1], start[0], dir_idx]):
                return False
            if obstacles[end[1], end[0]]:
                return False
        return True

    def _build_hint(self, agent_idx, raw_obs=None):
        state = self.states[agent_idx]
        agent = self.env.agents[agent_idx]
        current = (agent.x, agent.y)
        next_cell = state.path[1] if len(state.path) > 1 else current
        next_dir = np.zeros(_NEXT_DIR_DIMS, dtype=np.float32)
        if len(state.path) > 1:
            dir_idx = self._dir_index(current, next_cell)
            if dir_idx is not None:
                next_dir[dir_idx] = 1.0
        waypoint, waypoint_distance = self._next_waypoint(state.path)
        has_plan = float(len(state.path) > 1)
        blocked = float(self._planned_next_blocked(agent_idx, state, raw_obs))
        prefix_mask = np.zeros(self.planner_prefix_len, dtype=np.float32)
        prefix_cells = np.zeros((self.planner_prefix_len, 2), dtype=np.float32)
        for idx, cell in enumerate(state.path[1 : 1 + self.planner_prefix_len]):
            prefix_mask[idx] = 1.0
            prefix_cells[idx] = self._hint_cell(current, cell)

        return np.concatenate(
            [
                np.array([has_plan], dtype=np.float32),
                next_dir,
                np.array([waypoint_distance], dtype=np.float32),
                np.array([blocked], dtype=np.float32),
                np.array([float(state.last_on_path)], dtype=np.float32),
                np.asarray(self._hint_current_position(current), dtype=np.float32),
                np.asarray(self._hint_cell(current, next_cell), dtype=np.float32),
                prefix_mask,
                prefix_cells.reshape(-1),
            ]
        )

    def _hint_current_position(self, current):
        if self.planner_feature_mode == "local":
            return (0.0, 0.0)
        return self._normalized_position(*current)

    def _hint_cell(self, current, cell):
        if self.planner_feature_mode == "local":
            if not getattr(self.env, "normalised_coordinates", False):
                return (
                    float(cell[0] - current[0]),
                    float(cell[1] - current[1]),
                )
            max_x = max(1, self.env.grid_size[1] - 1)
            max_y = max(1, self.env.grid_size[0] - 1)
            return (
                float((cell[0] - current[0]) / max_x),
                float((cell[1] - current[1]) / max_y),
            )
        return self._normalized_position(*cell)

    def _next_waypoint(self, path):
        if len(path) <= 1:
            if path:
                return path[0], 0.0
            return (0, 0), 0.0

        current = path[0]
        waypoint = path[1]
        prev_dir = self._dir_index(path[0], path[1])
        steps = 1
        for cell_idx in range(2, len(path)):
            current_dir = self._dir_index(path[cell_idx - 1], path[cell_idx])
            if current_dir != prev_dir:
                break
            waypoint = path[cell_idx]
            steps += 1

        max_dist = max(1, self.env.grid_size[0] + self.env.grid_size[1])
        distance = abs(waypoint[0] - current[0]) + abs(waypoint[1] - current[1])
        return waypoint, float(distance / max_dist)

    def _planned_next_blocked(self, agent_idx, state, raw_obs=None):
        if len(state.path) <= 1:
            return 0

        agent = self.env.agents[agent_idx]
        next_cell = state.path[1]
        x, y = int(next_cell[0]), int(next_cell[1])
        if x < 0 or y < 0 or x >= self.env.grid_size[1] or y >= self.env.grid_size[0]:
            return 1
        if self.env._is_obstacle(x, y):
            return 1

        # Use simulator state instead of parsing flattened observations. The old
        # parser was brittle across observation variants and could over-label
        # planner next cells as blocked, which made NOOP overly attractive.
        agent_id = int(self.env.grid[0, y, x])
        if agent_id > 0 and agent_id != agent.id:
            return 1

        shelf_id = int(self.env.grid[1, y, x])
        carrying_shelf = getattr(agent, "carrying_shelf", None)
        if carrying_shelf is not None and shelf_id > 0 and shelf_id != carrying_shelf.id:
            return 1
        return 0

    def _normalized_position(self, x, y):
        max_x = max(1, self.env.grid_size[1] - 1)
        max_y = max(1, self.env.grid_size[0] - 1)
        return (float(x) / max_x, float(y) / max_y)

    def _normalized_next_cell(self, state):
        if len(state.path) <= 1:
            if state.path:
                return self._normalized_position(*state.path[0])
            return (0.0, 0.0)
        return self._normalized_position(*state.path[1])

    def _dir_index(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        for idx, (off_x, off_y) in enumerate(_DIR_OFFSETS):
            if dx == off_x and dy == off_y:
                return idx
        return None
