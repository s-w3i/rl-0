from collections import OrderedDict
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set

import gymnasium as gym
from gymnasium.utils import seeding
import networkx as nx
import numpy as np


_COLLISION_LAYERS = 2

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1


class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx : self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits


class Action(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


_LAYOUT_DIRECTION_CHARS = {
    "|": (Direction.UP, Direction.DOWN),
    "-": (Direction.LEFT, Direction.RIGHT),
    "^": (Direction.UP,),
    "v": (Direction.DOWN,),
    "<": (Direction.LEFT,),
    ">": (Direction.RIGHT,),
}

_DIRECTION_TOKEN_MAP = {
    "U": Direction.UP,
    "UP": Direction.UP,
    "^": Direction.UP,
    "D": Direction.DOWN,
    "DOWN": Direction.DOWN,
    "V": Direction.DOWN,
    "v": Direction.DOWN,
    "L": Direction.LEFT,
    "LEFT": Direction.LEFT,
    "<": Direction.LEFT,
    "R": Direction.RIGHT,
    "RIGHT": Direction.RIGHT,
    ">": Direction.RIGHT,
}

_DIRECTION_AVAILABILITY_BITS = {
    Direction.UP: 1,
    Direction.DOWN: 2,
    Direction.LEFT: 4,
    Direction.RIGHT: 8,
}


class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class ObservationType(Enum):
    DICT = 0
    FLATTENED = 1
    IMAGE = 2
    IMAGE_DICT = 3


class ImageLayer(Enum):
    """
    Input layers of image-style observations
    """

    SHELVES = 0  # binary layer indicating shelves (also indicates carried shelves)
    REQUESTS = 1  # binary layer indicating requested shelves
    AGENTS = 2  # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENT_DIRECTION = 3  # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_LOAD = 4  # binary layer indicating agents with load
    GOALS = 5  # binary layer indicating goal/ delivery locations
    ACCESSIBLE = 6  # binary layer indicating accessible cells (all but occupied cells/ out of map)
    AVAILABLE_DIRECTIONS = 7  # bitmask layer for allowed movement dirs (U=1,D=2,L=4,R=8)


class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y


class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, msg_bits: int):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.message = np.zeros(msg_bits)
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False

    @property
    def collision_layers(self):
        if self.loaded:
            return (_LAYER_AGENTS, _LAYER_SHELFS)
        else:
            return (_LAYER_AGENTS,)

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir


class Shelf(Entity):
    counter = 0

    def __init__(self, x, y):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)
        self.home_x = x
        self.home_y = y
        self.delivered = False

    @property
    def collision_layers(self):
        return (_LAYER_SHELFS,)


class Warehouse(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        n_agents: int,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        dedicated_requests: bool = False,
        layout: Optional[str] = None,
        observation_type: ObservationType = ObservationType.FLATTENED,
        image_observation_layers: List[ImageLayer] = [
            ImageLayer.SHELVES,
            ImageLayer.REQUESTS,
            ImageLayer.AGENTS,
            ImageLayer.GOALS,
            ImageLayer.ACCESSIBLE,
        ],
        image_observation_directional: bool = True,
        normalised_coordinates: bool = False,
        render_mode: Optional[str] = None,
        reward_delivery_weight: float = 1.0,
        lane_observation: bool = False,
        cell_direction_constraints: Optional[List] = None,
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param n_agents: Number of spawned and controlled agents
        :type n_agents: int
        :param msg_bits: Number of communication bits for each agent
        :type msg_bits: int
        :param sensor_range: Range of each agents observation
        :type sensor_range: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param reward_delivery_weight: Weight applied to delivery rewards
        :type reward_delivery_weight: float
        :param lane_observation: Include local lane-direction availability maps in observations.
        :type lane_observation: bool
        :param cell_direction_constraints: Optional per-cell allowed movement directions.
        :type cell_direction_constraints: Optional[List]
        :param layout: A string for a custom warehouse layout. X are shelve locations, dots are corridors, g are goal locations, O are obstacles, and |,-,^,v,<,> define directional corridor constraints. Ignores shelf_columns, shelf_height and shelf_rows when used.
        :type layout: str
        :param observation_type: Specifies type of observations
        :param image_observation_layers: Specifies types of layers observed if image-observations
            are used
        :type image_observation_layers: List[ImageLayer]
        :param image_observation_directional: Specifies whether image observations should be
            rotated to be directional (agent perspective) if image-observations are used
        :type image_observation_directional: bool
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        if not layout:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        else:
            self._make_layout_from_str(layout)

        self.n_agents = n_agents
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_delivery_weight = float(reward_delivery_weight)
        self.lane_observation = bool(lane_observation)
        self.cell_direction_constraints = self._normalize_cell_direction_constraints(
            cell_direction_constraints
        )
        if self.cell_direction_constraints:
            self._layout_direction_overrides.update(self.cell_direction_constraints)
            self._init_cell_allowed_dirs()
        self._conflict_detected_count = 0
        self._conflict_resolved_count = 0
        self._active_conflict_episodes = 0
        self._active_conflict_pairs: Set[Tuple[int, int]] = set()
        self._delivery_count_total = 0
        self._delivery_count_by_agent = [0 for _ in range(self.n_agents)]
        self._task_completed_total = 0
        self._task_completed_by_agent = [0 for _ in range(self.n_agents)]
        self._steps_since_task_progress = 0
        self._last_termination_reason = "running"
        self.reward_range = (0, 1)

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps

        self.normalised_coordinates = normalised_coordinates

        sa_action_space = [len(Action), *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = gym.spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = gym.spaces.MultiDiscrete(sa_action_space)
        self.action_space = gym.spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.request_queue_size = request_queue_size
        self.request_queue = []
        self.dedicated_requests = dedicated_requests
        self.assigned_shelves: List[Shelf] = []

        self.agents: List[Agent] = []

        # default values:
        self.fast_obs = None
        self.image_obs = None
        self.image_dict_obs = None
        if observation_type == ObservationType.IMAGE:
            self.observation_space = self._use_image_obs(
                image_observation_layers, image_observation_directional
            )
        elif observation_type == ObservationType.IMAGE_DICT:
            self.observation_space = self._use_image_dict_obs(
                image_observation_layers, image_observation_directional
            )

        else:
            # used for DICT observation type and needed as preceeding stype to generate
            # FLATTENED observations as well
            self.observation_space = self._use_slow_obs()

            # for performance reasons we
            # can flatten the obs vector
            if observation_type == ObservationType.FLATTENED:
                self.observation_space = self._use_fast_obs()

        self.global_image = None
        self.renderer = None
        self.render_mode = render_mode

    def _normalize_dir_token(self, value):
        if value is None:
            return None
        if isinstance(value, Direction):
            return value
        if isinstance(value, str):
            token = value.strip()
            if not token:
                return None
            mapped = _DIRECTION_TOKEN_MAP.get(token)
            if mapped is not None:
                return mapped
            mapped = _DIRECTION_TOKEN_MAP.get(token.upper())
            if mapped is not None:
                return mapped
            try:
                return Direction(int(token))
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid direction '{value}'. Use 0-3 or UP/DOWN/LEFT/RIGHT."
                )
        try:
            return Direction(int(value))
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid direction '{value}'. Use 0-3 or UP/DOWN/LEFT/RIGHT."
            )

    def _normalize_direction_sequence(self, value):
        if value is None:
            return tuple()
        if isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        elif isinstance(value, str):
            token = value.strip()
            if not token:
                raw_values = []
            elif token == "|":
                raw_values = ["UP", "DOWN"]
            elif token == "-":
                raw_values = ["LEFT", "RIGHT"]
            elif any(sep in token for sep in (",", ";", "|", "/", " ")):
                cleaned = token
                for sep in (";", "|", "/", " "):
                    cleaned = cleaned.replace(sep, ",")
                raw_values = [part for part in cleaned.split(",") if part]
            elif len(token) > 1 and all(ch in "^v<>" for ch in token):
                raw_values = list(token)
            else:
                raw_values = [token]
        else:
            raw_values = [value]

        dirs = []
        seen = set()
        for item in raw_values:
            direction = self._normalize_dir_token(item)
            if direction is None or direction in seen:
                continue
            dirs.append(direction)
            seen.add(direction)
        return tuple(dirs)

    def _normalize_cell_direction_constraints(self, constraints):
        if not constraints:
            return {}
        normalized: Dict[Tuple[int, int], Tuple[Direction, ...]] = {}
        for entry in constraints:
            if entry is None:
                continue
            x = y = None
            dirs_raw = None
            if isinstance(entry, dict):
                x = entry.get("x")
                y = entry.get("y")
                dirs_raw = entry.get("dirs")
                if dirs_raw is None:
                    dirs_raw = entry.get("directions")
            elif isinstance(entry, (list, tuple)):
                if len(entry) >= 3:
                    x, y, dirs_raw = entry[0], entry[1], entry[2]
                elif len(entry) == 2 and isinstance(entry[1], (list, tuple, set, str)):
                    point = entry[0]
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        x, y = point[0], point[1]
                        dirs_raw = entry[1]
            if x is None or y is None or dirs_raw is None:
                raise ValueError(
                    "Invalid cell_direction_constraints entry. Expected x/y and dirs."
                )
            dirs = self._normalize_direction_sequence(dirs_raw)
            if not dirs:
                continue
            point = (int(x), int(y))
            normalized[point] = dirs
        return normalized

    def _init_cell_allowed_dirs(self):
        self._cell_allowed_dirs = np.ones((*self.grid_size, len(Direction)), dtype=np.uint8)
        obstacle_mask = self.obstacles.astype(bool)
        self._cell_allowed_dirs[obstacle_mask, :] = 0
        for (x, y), dirs in self._layout_direction_overrides.items():
            self._set_cell_allowed_dirs(x, y, dirs)

    def _set_cell_allowed_dirs(self, x: int, y: int, dirs: Tuple[Direction, ...]):
        if x < 0 or y < 0 or x >= self.grid_size[1] or y >= self.grid_size[0]:
            return
        if not dirs:
            return
        if self._is_obstacle(x, y):
            return
        self._cell_allowed_dirs[y, x, :] = 0
        for direction in dirs:
            self._cell_allowed_dirs[y, x, direction.value] = 1

    def _is_direction_allowed(self, x: int, y: int, direction: Direction) -> bool:
        if x < 0 or y < 0 or x >= self.grid_size[1] or y >= self.grid_size[0]:
            return False
        if self._is_obstacle(x, y):
            return False
        return bool(self._cell_allowed_dirs[y, x, direction.value])

    def _is_obstacle(self, x: int, y: int) -> bool:
        return bool(self.obstacles[y, x])

    def _make_layout_from_params(self, shelf_columns, shelf_rows, column_height):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        self.grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        self.column_height = column_height
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.goals = [
            (self.grid_size[1] // 2 - 1, self.grid_size[0] - 1),
            (self.grid_size[1] // 2, self.grid_size[0] - 1),
        ]

        self.highways = np.zeros(self.grid_size, dtype=np.uint8)
        self.obstacles = np.zeros(self.grid_size, dtype=np.uint8)
        self._layout_direction_overrides: Dict[Tuple[int, int], Tuple[Direction, ...]] = {}

        def highway_func(x, y):
            is_on_vertical_highway = x % 3 == 0
            is_on_horizontal_highway = y % (column_height + 1) == 0
            is_on_delivery_row = y == self.grid_size[0] - 1
            is_on_queue = (y > self.grid_size[0] - (column_height + 3)) and (
                x == self.grid_size[1] // 2 - 1 or x == self.grid_size[1] // 2
            )
            return (
                is_on_vertical_highway
                or is_on_horizontal_highway
                or is_on_delivery_row
                or is_on_queue
            )

        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = int(highway_func(x, y))
        self._init_cell_allowed_dirs()

    def _make_layout_from_str(self, layout):
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.uint8)
        self.obstacles = np.zeros(self.grid_size, dtype=np.uint8)
        self._layout_direction_overrides = {}

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                char = char.lower()
                assert char in "gxo.|-^v<>"
                if char == "g":
                    self.goals.append((x, y))
                    self.highways[y, x] = 1
                elif char == ".":
                    self.highways[y, x] = 1
                elif char == "o":
                    self.obstacles[y, x] = 1
                elif char in _LAYOUT_DIRECTION_CHARS:
                    self.highways[y, x] = 1
                    self._layout_direction_overrides[(x, y)] = _LAYOUT_DIRECTION_CHARS[char]

        assert len(self.goals) >= 1, "At least one goal is required"
        self._init_cell_allowed_dirs()

    def _use_image_obs(self, image_observation_layers, directional=True):
        """
        Set image observation space
        :param image_observation_layers (List[ImageLayer]): list of layers to use as image channels
        :param directional (bool): flag whether observations should be directional (pointing in
            direction of agent or north-wise)
        """
        self.image_obs = True
        self.fast_obs = False
        self.image_dict_obs = True
        self.image_observation_directional = directional
        self.image_observation_layers = image_observation_layers

        observation_shape = (1 + 2 * self.sensor_range, 1 + 2 * self.sensor_range)

        layers_min = []
        layers_max = []
        for layer in image_observation_layers:
            if layer == ImageLayer.AGENT_DIRECTION:
                # directions as int
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32) * max(
                    [d.value + 1 for d in Direction]
                )
            elif layer == ImageLayer.AVAILABLE_DIRECTIONS:
                # direction bitmask (U=1, D=2, L=4, R=8)
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32) * 15.0
            else:
                # binary layer
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32)
            layers_min.append(layer_min)
            layers_max.append(layer_max)

        # total observation
        min_obs = np.stack(layers_min)
        max_obs = np.stack(layers_max)
        return gym.spaces.Tuple(
            tuple([gym.spaces.Box(min_obs, max_obs, dtype=np.float32)] * self.n_agents)
        )

    def _use_image_dict_obs(self, image_observation_layers, directional=True):
        """
        Get image dictionary observation with image and flattened feature vector
        :param image_observation_layers (List[ImageLayer]): list of layers to use as image channels
        :param directional (bool): flag whether observations should be directional (pointing in
            direction of agent or north-wise)
        """
        image_obs_space = self._use_image_obs(image_observation_layers, directional)[0]
        self.image_obs = False
        self.image_dict_obs = True
        feature_dict = OrderedDict(
            {
                "direction": gym.spaces.Discrete(4),
                "on_highway": gym.spaces.MultiBinary(1),
                "carrying_shelf": gym.spaces.MultiBinary(1),
            }
        )
        local_cells = (1 + 2 * self.sensor_range) ** 2
        if self.lane_observation:
            feature_dict.update(
                {
                    "lane_up_map": gym.spaces.MultiBinary(local_cells),
                    "lane_down_map": gym.spaces.MultiBinary(local_cells),
                    "lane_left_map": gym.spaces.MultiBinary(local_cells),
                    "lane_right_map": gym.spaces.MultiBinary(local_cells),
                }
            )
        feature_space = gym.spaces.Dict(feature_dict)

        feature_flat_dim = gym.spaces.flatdim(feature_space)
        feature_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(feature_flat_dim,),
            dtype=np.float32,
        )

        return gym.spaces.Tuple(
            tuple(
                [
                    gym.spaces.Dict(
                        {"image": image_obs_space, "features": feature_space}
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

    def _use_slow_obs(self):
        self.fast_obs = False

        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2
        extra_lane_bits = 0
        if self.lane_observation:
            extra_lane_bits = 4 * self._obs_sensor_locations
        self._obs_bits_for_self = 4 + len(Direction) + extra_lane_bits
        self._obs_bits_per_agent = 1 + len(Direction) + self.msg_bits
        self._obs_bits_per_shelf = 2
        self._obs_bits_for_requests = 2

        self._obs_length = (
            self._obs_bits_for_self
            + self._obs_sensor_locations * self._obs_bits_per_agent
            + self._obs_sensor_locations * self._obs_bits_per_shelf
        )

        max_grid_val = max(self.grid_size)
        low = np.zeros(2)
        if self.normalised_coordinates:
            high = np.ones(2)
            dtype = np.float32
        else:
            high = np.ones(2) * max_grid_val
            dtype = np.int32
        location_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=dtype,
        )

        self_observation_dict = OrderedDict(
            {
                "location": location_space,
                "carrying_shelf": gym.spaces.MultiBinary(1),
                "direction": gym.spaces.Discrete(4),
                "on_highway": gym.spaces.MultiBinary(1),
            }
        )
        if self.lane_observation:
            self_observation_dict.update(
                {
                    "lane_up_map": gym.spaces.MultiBinary(self._obs_sensor_locations),
                    "lane_down_map": gym.spaces.MultiBinary(self._obs_sensor_locations),
                    "lane_left_map": gym.spaces.MultiBinary(self._obs_sensor_locations),
                    "lane_right_map": gym.spaces.MultiBinary(self._obs_sensor_locations),
                }
            )
        self_observation_dict_space = gym.spaces.Dict(self_observation_dict)
        sensor_per_location_dict = OrderedDict(
            {
                "has_agent": gym.spaces.MultiBinary(1),
                "direction": gym.spaces.Discrete(4),
            }
        )
        if self.msg_bits > 0:
            sensor_per_location_dict["local_message"] = gym.spaces.MultiBinary(
                self.msg_bits
            )
        sensor_per_location_dict.update(
            {
                "has_shelf": gym.spaces.MultiBinary(1),
                "shelf_requested": gym.spaces.MultiBinary(1),
            }
        )
        return gym.spaces.Tuple(
            tuple(
                [
                    gym.spaces.Dict(
                        OrderedDict(
                            {
                                "self": self_observation_dict_space,
                                "sensors": gym.spaces.Tuple(
                                    self._obs_sensor_locations
                                    * (gym.spaces.Dict(sensor_per_location_dict),)
                                ),
                            }
                        )
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

    def _use_fast_obs(self):
        if self.fast_obs:
            return self.observation_space

        self.fast_obs = True
        ma_spaces = []
        for sa_obs in self.observation_space:
            flatdim = gym.spaces.flatdim(sa_obs)
            ma_spaces += [
                gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        return gym.spaces.Tuple(tuple(ma_spaces))

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def _is_requested_shelf(self, agent, shelf) -> bool:
        if not self.dedicated_requests:
            return shelf in self.request_queue
        if not self.assigned_shelves:
            return False
        return self.assigned_shelves[agent.id - 1] == shelf

    def _assign_new_shelf(self, agent_index: int):
        if not self.assigned_shelves:
            return
        if agent_index < 0 or agent_index >= len(self.assigned_shelves):
            return
        candidates = [s for s in self.shelfs if s not in self.assigned_shelves]
        if not candidates:
            return
        new_request = self.np_random.choice(candidates)
        self.assigned_shelves[agent_index] = new_request
        self.request_queue = list(self.assigned_shelves)

    def _next_cell_for_action(self, agent: Agent, action: Action) -> Tuple[int, int]:
        if action != Action.FORWARD:
            return agent.x, agent.y
        if not self._is_direction_allowed(agent.x, agent.y, agent.dir):
            return agent.x, agent.y
        if agent.dir == Direction.UP:
            target = (agent.x, max(0, agent.y - 1))
        elif agent.dir == Direction.DOWN:
            target = (agent.x, min(self.grid_size[0] - 1, agent.y + 1))
        elif agent.dir == Direction.LEFT:
            target = (max(0, agent.x - 1), agent.y)
        elif agent.dir == Direction.RIGHT:
            target = (min(self.grid_size[1] - 1, agent.x + 1), agent.y)
        else:
            target = (agent.x, agent.y)
        if self._is_obstacle(target[0], target[1]):
            return agent.x, agent.y
        return target

    def _conflict_pairs(self, actions: List[Action]) -> Set[Tuple[int, int]]:
        positions = [(agent.x, agent.y) for agent in self.agents]
        targets = [
            self._next_cell_for_action(agent, action)
            for agent, action in zip(self.agents, actions)
        ]
        pairs: Set[Tuple[int, int]] = set()
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                pair_conflict = False
                if targets[i] == targets[j] and targets[i] != positions[i]:
                    pair_conflict = True
                elif (
                    targets[i] == positions[j]
                    and targets[j] == positions[i]
                    and targets[i] != positions[i]
                    and targets[j] != positions[j]
                ):
                    pair_conflict = True
                elif (
                    targets[i] == positions[j]
                    and targets[j] == positions[j]
                    and targets[i] != positions[i]
                ):
                    pair_conflict = True
                if pair_conflict:
                    pair = tuple(sorted((self.agents[i].id, self.agents[j].id)))
                    pairs.add(pair)
        return pairs

    def _make_img_obs(self, agent):
        # write image observations
        if agent.id == 1:
            self.global_layers = {}
            # first agent's observation --> update global observation layers
            for layer_type in self.image_observation_layers:
                if layer_type == ImageLayer.REQUESTS and self.dedicated_requests:
                    continue
                if layer_type == ImageLayer.SHELVES:
                    layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                    # set all occupied shelf cells to 1.0 (instead of shelf ID)
                    layer[layer > 0.0] = 1.0
                    # print("SHELVES LAYER")
                elif layer_type == ImageLayer.REQUESTS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for requested_shelf in self.request_queue:
                        layer[requested_shelf.y, requested_shelf.x] = 1.0
                    # print("REQUESTS LAYER")
                elif layer_type == ImageLayer.AGENTS:
                    layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                    # set all occupied agent cells to 1.0 (instead of agent ID)
                    layer[layer > 0.0] = 1.0
                    # print("AGENTS LAYER")
                elif layer_type == ImageLayer.AGENT_DIRECTION:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        agent_direction = ag.dir.value + 1
                        layer[ag.y, ag.x] = float(agent_direction)
                    # print("AGENT DIRECTIONS LAYER")
                elif layer_type == ImageLayer.AGENT_LOAD:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        if ag.carrying_shelf is not None:
                            layer[ag.y, ag.x] = 1.0
                    # print("AGENT LOAD LAYER")
                elif layer_type == ImageLayer.GOALS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for goal_y, goal_x in self.goals:
                        layer[goal_x, goal_y] = 1.0
                    # print("GOALS LAYER")
                elif layer_type == ImageLayer.ACCESSIBLE:
                    layer = np.ones(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        layer[ag.y, ag.x] = 0.0
                    layer[self.obstacles.astype(bool)] = 0.0
                elif layer_type == ImageLayer.AVAILABLE_DIRECTIONS:
                    layer = self._direction_availability_layer()
                else:
                    raise ValueError(f"Unknown image layer type: {layer_type}")

                # pad with 0s for out-of-map cells
                layer = np.pad(layer, self.sensor_range, mode="constant")
                self.global_layers[layer_type] = layer

        layers = []
        for layer_type in self.image_observation_layers:
            if layer_type == ImageLayer.REQUESTS and self.dedicated_requests:
                layer = np.zeros(self.grid_size, dtype=np.float32)
                if self.assigned_shelves:
                    requested_shelf = self.assigned_shelves[agent.id - 1]
                    layer[requested_shelf.y, requested_shelf.x] = 1.0
                layer = np.pad(layer, self.sensor_range, mode="constant")
            else:
                layer = self.global_layers[layer_type]
            layers.append(layer)
        obs = np.stack(layers)

        # global information was generated --> get information for agent
        start_x = agent.y
        end_x = agent.y + 2 * self.sensor_range + 1
        start_y = agent.x
        end_y = agent.x + 2 * self.sensor_range + 1
        obs = obs[:, start_x:end_x, start_y:end_y]

        if self.image_observation_directional:
            # rotate image to be in direction of agent
            if agent.dir == Direction.DOWN:
                # rotate by 180 degrees (clockwise)
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif agent.dir == Direction.LEFT:
                # rotate by 90 degrees (clockwise)
                obs = np.rot90(obs, k=3, axes=(1, 2))
            elif agent.dir == Direction.RIGHT:
                # rotate by 270 degrees (clockwise)
                obs = np.rot90(obs, k=1, axes=(1, 2))
            # no rotation needed for UP direction
        return obs

    def _local_lane_direction_maps(self, agent):
        window = 1 + 2 * self.sensor_range
        min_x = agent.x - self.sensor_range
        max_x = agent.x + self.sensor_range + 1
        min_y = agent.y - self.sensor_range
        max_y = agent.y + self.sensor_range + 1

        lane_up = np.zeros(window * window, dtype=np.float32)
        lane_down = np.zeros(window * window, dtype=np.float32)
        lane_left = np.zeros(window * window, dtype=np.float32)
        lane_right = np.zeros(window * window, dtype=np.float32)

        idx = 0
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if 0 <= x < self.grid_size[1] and 0 <= y < self.grid_size[0]:
                    lane_up[idx] = float(self._is_direction_allowed(x, y, Direction.UP))
                    lane_down[idx] = float(self._is_direction_allowed(x, y, Direction.DOWN))
                    lane_left[idx] = float(self._is_direction_allowed(x, y, Direction.LEFT))
                    lane_right[idx] = float(self._is_direction_allowed(x, y, Direction.RIGHT))
                idx += 1

        return lane_up, lane_down, lane_left, lane_right

    def _cell_direction_availability_mask(self, x: int, y: int) -> int:
        if self._is_obstacle(x, y):
            return 0
        mask = 0
        for direction, bit in _DIRECTION_AVAILABILITY_BITS.items():
            if not self._is_direction_allowed(x, y, direction):
                continue
            if direction == Direction.UP:
                nx, ny = x, y - 1
            elif direction == Direction.DOWN:
                nx, ny = x, y + 1
            elif direction == Direction.LEFT:
                nx, ny = x - 1, y
            else:
                nx, ny = x + 1, y
            if nx < 0 or ny < 0 or nx >= self.grid_size[1] or ny >= self.grid_size[0]:
                continue
            if self._is_obstacle(nx, ny):
                continue
            mask |= bit
        return mask

    def _direction_availability_layer(self):
        layer = np.zeros(self.grid_size, dtype=np.float32)
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                layer[y, x] = float(self._cell_direction_availability_mask(x, y))
        return layer

    def _get_default_obs(self, agent):
        min_x = agent.x - self.sensor_range
        max_x = agent.x + self.sensor_range + 1

        min_y = agent.y - self.sensor_range
        max_y = agent.y + self.sensor_range + 1

        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)

        if self.fast_obs:
            # write flattened observations
            flatdim = gym.spaces.flatdim(self.observation_space[agent.id - 1])
            obs = _VectorWriter(flatdim)

            if self.normalised_coordinates:
                agent_x = agent.x / (self.grid_size[1] - 1)
                agent_y = agent.y / (self.grid_size[0] - 1)
            else:
                agent_x = agent.x
                agent_y = agent.y

            obs.write([agent_x, agent_y, int(agent.carrying_shelf is not None)])
            direction = np.zeros(4)
            direction[agent.dir.value] = 1.0
            obs.write(direction)
            obs.write([int(self._is_highway(agent.x, agent.y))])
            if self.lane_observation:
                lane_up, lane_down, lane_left, lane_right = self._local_lane_direction_maps(
                    agent
                )
                obs.write(lane_up)
                obs.write(lane_down)
                obs.write(lane_left)
                obs.write(lane_right)

            # 'has_agent': MultiBinary(1),
            # 'direction': Discrete(4),
            # 'local_message': MultiBinary(2)
            # 'has_shelf': MultiBinary(1),
            # 'shelf_requested': MultiBinary(1),

            for i, (id_agent, id_shelf) in enumerate(zip(agents, shelfs)):
                if id_agent == 0:
                    # no agent, direction, or message
                    obs.write([0.0])  # no agent present
                    obs.write([1.0, 0.0, 0.0, 0.0])  # agent direction
                    obs.skip(self.msg_bits)  # agent message
                else:
                    obs.write([1.0])  # agent present
                    direction = np.zeros(4)
                    direction[self.agents[id_agent - 1].dir.value] = 1.0
                    obs.write(direction)  # agent direction as onehot
                    if self.msg_bits > 0:
                        obs.write(self.agents[id_agent - 1].message)  # agent message
                if id_shelf == 0:
                    obs.write([0.0, 0.0])  # no shelf or requested shelf
                else:
                    shelf = self.shelfs[id_shelf - 1]
                    obs.write(
                        [1.0, int(self._is_requested_shelf(agent, shelf))]
                    )  # shelf presence and request status
            return obs.vector

        # write dictionary observations
        obs = {}
        if self.normalised_coordinates:
            agent_x = agent.x / (self.grid_size[1] - 1)
            agent_y = agent.y / (self.grid_size[0] - 1)
        else:
            agent_x = agent.x
            agent_y = agent.y
        # --- self data
        obs["self"] = {
            "location": np.array([agent_x, agent_y], dtype=np.int32),
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(self._is_highway(agent.x, agent.y))],
        }
        if self.lane_observation:
            lane_up, lane_down, lane_left, lane_right = self._local_lane_direction_maps(
                agent
            )
            obs["self"]["lane_up_map"] = lane_up.astype(np.int8).tolist()
            obs["self"]["lane_down_map"] = lane_down.astype(np.int8).tolist()
            obs["self"]["lane_left_map"] = lane_left.astype(np.int8).tolist()
            obs["self"]["lane_right_map"] = lane_right.astype(np.int8).tolist()
        # --- sensor data
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))

        # find neighboring agents
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs["sensors"][i]["has_agent"] = [0]
                obs["sensors"][i]["direction"] = 0
                obs["sensors"][i]["local_message"] = (
                    self.msg_bits * [0] if self.msg_bits > 0 else None
                )
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message"] = (
                    self.agents[id_ - 1].message if self.msg_bits > 0 else None
                )

        # find neighboring shelfs:
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                shelf = self.shelfs[id_ - 1]
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(self._is_requested_shelf(agent, shelf))
                ]

        return obs

    def _make_obs(self, agent):
        if self.image_obs:
            return self._make_img_obs(agent)
        elif self.image_dict_obs:
            image_obs = self._make_img_obs(agent)
            feature_obs = _VectorWriter(
                self.observation_space[agent.id - 1]["features"].shape[0]
            )
            direction = np.zeros(4)
            direction[agent.dir.value] = 1.0
            feature_obs.write(direction)
            feature_obs.write(
                [
                    int(self._is_highway(agent.x, agent.y)),
                    int(agent.carrying_shelf is not None),
                ]
            )
            if self.lane_observation:
                lane_up, lane_down, lane_left, lane_right = self._local_lane_direction_maps(
                    agent
                )
                feature_obs.write(lane_up)
                feature_obs.write(lane_down)
                feature_obs.write(lane_left)
                feature_obs.write(lane_right)
            return {
                "image": image_obs,
                "features": feature_obs.vector,
            }
        else:
            return self._get_default_obs(agent)

    def _is_carrying_shelf_static_blocked(
        self, agent: Agent, start: Tuple[int, int], target: Tuple[int, int]
    ) -> bool:
        return bool(
            agent.carrying_shelf
            and start != target
            and self.grid[_LAYER_SHELFS, target[1], target[0]]
            and not (
                self.grid[_LAYER_AGENTS, target[1], target[0]]
                and self.agents[
                    self.grid[_LAYER_AGENTS, target[1], target[0]] - 1
                ].carrying_shelf
            )
        )

    def _forward_move_static_context(self, agent: Agent) -> Dict[str, object]:
        start = (agent.x, agent.y)
        target = agent.req_location(self.grid_size)
        forward_intent = agent.req_action == Action.FORWARD
        non_clamp = target != start
        direction_allowed = self._is_direction_allowed(start[0], start[1], agent.dir)
        target_is_obstacle = self._is_obstacle(target[0], target[1])
        carrying_shelf_blocked = self._is_carrying_shelf_static_blocked(
            agent, start, target
        )
        valid_mover = bool(
            forward_intent
            and non_clamp
            and direction_allowed
            and not target_is_obstacle
            and not carrying_shelf_blocked
        )
        return {
            "start": start,
            "target": target,
            "forward_intent": forward_intent,
            "non_clamp": non_clamp,
            "direction_allowed": direction_allowed,
            "target_is_obstacle": target_is_obstacle,
            "carrying_shelf_blocked": carrying_shelf_blocked,
            "valid_mover": valid_mover,
        }

    def _get_info(self):
        return {}

    def _recalc_grid(self):
        self.grid[:] = 0
        for s in self.shelfs:
            self.grid[_LAYER_SHELFS, s.y, s.x] = s.id

        for a in self.agents:
            self.grid[_LAYER_AGENTS, a.y, a.x] = a.id

    def reset(self, seed=None, options=None):
        if seed is not None:
            # setting seed
            super().reset(seed=seed, options=options)

        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self._conflict_detected_count = 0
        self._conflict_resolved_count = 0
        self._active_conflict_episodes = 0
        self._active_conflict_pairs = set()
        self._delivery_count_total = 0
        self._delivery_count_by_agent = [0 for _ in range(self.n_agents)]
        self._task_completed_total = 0
        self._task_completed_by_agent = [0 for _ in range(self.n_agents)]
        self._steps_since_task_progress = 0
        self._last_termination_reason = "running"

        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9

        # make the shelfs
        self.shelfs = [
            Shelf(x, y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y) and not self._is_obstacle(x, y)
        ]

        # spawn agents at random locations
        available = np.flatnonzero(self.obstacles.ravel() == 0)
        if len(available) < self.n_agents:
            raise ValueError("Not enough free cells for agent spawns.")
        agent_locs = self.np_random.choice(
            available,
            size=self.n_agents,
            replace=False,
        )
        agent_locs = np.unravel_index(agent_locs, self.grid_size)
        # and direction
        agent_dirs = self.np_random.choice([d for d in Direction], size=self.n_agents)
        self.agents = [
            Agent(x, y, dir_, self.msg_bits)
            for y, x, dir_ in zip(*agent_locs, agent_dirs)
        ]

        self._recalc_grid()

        if self.dedicated_requests:
            if len(self.shelfs) < self.n_agents:
                raise ValueError("Not enough shelves to assign one per agent.")
            self.assigned_shelves = list(
                self.np_random.choice(self.shelfs, size=self.n_agents, replace=False)
            )
            self.request_queue = list(self.assigned_shelves)
        else:
            self.request_queue = list(
                self.np_random.choice(
                    self.shelfs, size=self.request_queue_size, replace=False
                )
            )
            self.assigned_shelves = []

        return tuple([self._make_obs(agent) for agent in self.agents]), self._get_info()

    def step(
        self, actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        assert len(actions) == len(self.agents)

        for agent, action in zip(self.agents, actions):
            if isinstance(action, Action):
                action = action.value
            if self.msg_bits > 0:
                if isinstance(action, (list, tuple, np.ndarray)):
                    arr = np.asarray(action)
                    if arr.shape == ():
                        agent.req_action = Action(int(arr))
                        agent.message[:] = 0
                    else:
                        agent.req_action = Action(int(arr[0]))
                        msg = arr[1:]
                        if msg.size:
                            agent.message[:] = msg[: self.msg_bits]
                        else:
                            agent.message[:] = 0
                else:
                    agent.req_action = Action(int(action))
                    agent.message[:] = 0
            else:
                agent.req_action = Action(int(action))

        commited_agents = set()
        G = nx.DiGraph()
        pre_positions = [(agent.x, agent.y) for agent in self.agents]

        move_contexts = [self._forward_move_static_context(agent) for agent in self.agents]
        valid_mover = np.array(
            [bool(ctx["valid_mover"]) for ctx in move_contexts], dtype=bool
        )
        step_targets = [ctx["target"] for ctx in move_contexts]

        target_counts: Dict[Tuple[int, int], int] = {}
        for idx, is_valid in enumerate(valid_mover):
            if not is_valid:
                continue
            target = step_targets[idx]
            target_counts[target] = target_counts.get(target, 0) + 1

        step_vertex_by_agent = np.zeros(self.n_agents, dtype=np.int32)
        for idx, agent in enumerate(self.agents):
            if not valid_mover[idx]:
                continue
            target = step_targets[idx]
            occupied_id = int(self.grid[_LAYER_AGENTS, target[1], target[0]])
            contested = target_counts.get(target, 0) >= 2
            occupied_by_other = occupied_id > 0 and occupied_id != agent.id
            if contested or occupied_by_other:
                step_vertex_by_agent[idx] = 1

        step_swap_by_agent = np.zeros(self.n_agents, dtype=np.int32)
        step_swap_attempts = 0
        for i in range(self.n_agents):
            if not valid_mover[i]:
                continue
            for j in range(i + 1, self.n_agents):
                if not valid_mover[j]:
                    continue
                if (
                    step_targets[i] == pre_positions[j]
                    and step_targets[j] == pre_positions[i]
                ):
                    step_swap_attempts += 1
                    step_swap_by_agent[i] = 1
                    step_swap_by_agent[j] = 1

        for idx, agent in enumerate(self.agents):
            start = move_contexts[idx]["start"]
            target = move_contexts[idx]["target"]
            if move_contexts[idx]["forward_intent"] and not move_contexts[idx]["valid_mover"]:
                agent.req_action = Action.NOOP
                G.add_edge(start, start)
            else:
                G.add_edge(start, target)

        # Track pairwise action conflicts from current move intents.
        prev_conflict_pairs = set(self._active_conflict_pairs)
        step_conflict_pairs = self._conflict_pairs(
            [agent.req_action for agent in self.agents]
        )
        started_pairs = step_conflict_pairs - prev_conflict_pairs
        resolved_pairs = prev_conflict_pairs - step_conflict_pairs

        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

        for comp in wcomps:
            try:
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible. so skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[_LAYER_AGENTS, start_node[1], start_node[0]]
                    if agent_id > 0:
                        commited_agents.add(agent_id)
            except nx.NetworkXNoCycle:
                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(self.agents) - commited_agents

        for agent in failed_agents:
            assert agent.req_action == Action.FORWARD
            agent.req_action = Action.NOOP

        rewards = np.zeros(self.n_agents)
        task_completed_before_step = int(self._task_completed_total)

        def apply_reward(agent_id, value):
            if value == 0:
                return
            if self.reward_type == RewardType.GLOBAL:
                rewards[:] += value
            else:
                rewards[agent_id - 1] += value

        manually_dropped_agents = set()

        for agent in self.agents:
            agent.prev_x, agent.prev_y = agent.x, agent.y

            if agent.req_action == Action.FORWARD:
                agent.x, agent.y = agent.req_location(self.grid_size)
                if agent.carrying_shelf:
                    agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                if shelf_id:
                    shelf = self.shelfs[shelf_id - 1]
                    agent.carrying_shelf = shelf
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf:
                if not self._is_highway(agent.x, agent.y):
                    shelf = agent.carrying_shelf
                    correct_return = bool(
                        shelf.delivered
                        and shelf.home_x == agent.x
                        and shelf.home_y == agent.y
                    )
                    if correct_return:
                        shelf.delivered = False
                        self._task_completed_total += 1
                        self._task_completed_by_agent[agent.id - 1] += 1
                        if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                            apply_reward(agent.id, 0.5 * self.reward_delivery_weight)
                    agent.carrying_shelf = None
                    agent.has_delivered = False
                    manually_dropped_agents.add(agent)

        step_moved_by_agent = np.array(
            [
                int((agent.x, agent.y) != pre_positions[idx])
                for idx, agent in enumerate(self.agents)
            ],
            dtype=np.int32,
        )
        step_blocked_static_by_agent = np.array(
            [
                int(move_contexts[idx]["forward_intent"] and not valid_mover[idx])
                for idx in range(self.n_agents)
            ],
            dtype=np.int32,
        )
        step_blocked_agent_by_agent = np.array(
            [
                int(valid_mover[idx] and not step_moved_by_agent[idx])
                for idx in range(self.n_agents)
            ],
            dtype=np.int32,
        )
        step_blocked_static = int(step_blocked_static_by_agent.sum())
        step_blocked_agent = int(step_blocked_agent_by_agent.sum())
        step_blocked_total = int(step_blocked_static + step_blocked_agent)
        step_moved_total = int(step_moved_by_agent.sum())
        step_vertex_conflicts = int(step_vertex_by_agent.sum())

        dropped_agents = set(manually_dropped_agents)
        for agent in self.agents:
            if not agent.carrying_shelf:
                continue
            if not self._is_highway(agent.x, agent.y):
                shelf = agent.carrying_shelf
                if (
                    shelf.delivered
                    and shelf.home_x == agent.x
                    and shelf.home_y == agent.y
                ):
                    shelf.delivered = False
                    agent.carrying_shelf = None
                    dropped_agents.add(agent)
                    self._task_completed_total += 1
                    self._task_completed_by_agent[agent.id - 1] += 1
                    if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                        apply_reward(agent.id, 0.5 * self.reward_delivery_weight)
                    agent.has_delivered = False

        self._recalc_grid()

        carried_shelves = {
            agent.carrying_shelf for agent in self.agents if agent.carrying_shelf
        }
        for agent in self.agents:
            if agent in dropped_agents or agent.carrying_shelf:
                continue
            shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]
            if shelf in carried_shelves:
                continue
            if self._is_requested_shelf(agent, shelf):
                agent.carrying_shelf = shelf
                carried_shelves.add(shelf)

        shelf_delivered = False
        for y, x in self.goals:
            shelf_id = self.grid[_LAYER_SHELFS, x, y]
            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]

            if shelf not in self.request_queue:
                continue
            agent_id = self.grid[_LAYER_AGENTS, x, y]
            if self.dedicated_requests:
                if agent_id <= 0:
                    continue
                if self.assigned_shelves and self.assigned_shelves[agent_id - 1] != shelf:
                    continue
            # a shelf was successfully delived.
            shelf_delivered = True
            shelf.delivered = True
            if agent_id > 0:
                self._delivery_count_total += 1
                self._delivery_count_by_agent[agent_id - 1] += 1
            # remove from queue and replace it
            if self.dedicated_requests:
                self._assign_new_shelf(agent_id - 1)
            else:
                candidates = [s for s in self.shelfs if s not in self.request_queue]
                new_request = self.np_random.choice(candidates)
                self.request_queue[self.request_queue.index(shelf)] = new_request
            # also reward the agents
            if self.reward_type == RewardType.GLOBAL:
                rewards += 1 * self.reward_delivery_weight
            elif self.reward_type == RewardType.INDIVIDUAL:
                rewards[agent_id - 1] += 1 * self.reward_delivery_weight
            elif self.reward_type == RewardType.TWO_STAGE:
                self.agents[agent_id - 1].has_delivered = True
                rewards[agent_id - 1] += 0.5 * self.reward_delivery_weight

        if shelf_delivered:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1
        if self._task_completed_total > task_completed_before_step:
            self._steps_since_task_progress = 0
        else:
            self._steps_since_task_progress += 1

        termination_reason = "running"
        if self.max_inactivity_steps and self._cur_inactive_steps >= self.max_inactivity_steps:
            done = True
            termination_reason = "max_inactivity"
        elif self.max_steps and self._cur_steps >= self.max_steps:
            done = True
            termination_reason = "max_steps"
        else:
            done = False
        truncated = False

        step_conflict_detected = len(started_pairs)
        step_conflict_resolved = len(resolved_pairs)
        self._conflict_detected_count += int(step_conflict_detected)
        self._conflict_resolved_count += int(step_conflict_resolved)
        self._active_conflict_pairs = set(step_conflict_pairs)
        self._active_conflict_episodes = int(len(self._active_conflict_pairs))
        unresolved_conflicts = max(
            0, int(self._conflict_detected_count - self._conflict_resolved_count)
        )
        self._last_termination_reason = termination_reason

        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
        info = self._get_info()
        info["step_swap_attempts"] = int(step_swap_attempts)
        info["step_vertex_conflicts"] = int(step_vertex_conflicts)
        info["step_blocked_agent"] = int(step_blocked_agent)
        info["step_blocked_static"] = int(step_blocked_static)
        info["step_blocked_total"] = int(step_blocked_total)
        info["step_moved_total"] = int(step_moved_total)
        info["step_swap_by_agent"] = [int(v) for v in step_swap_by_agent]
        info["step_vertex_by_agent"] = [int(v) for v in step_vertex_by_agent]
        info["step_blocked_agent_by_agent"] = [
            int(v) for v in step_blocked_agent_by_agent
        ]
        info["step_blocked_static_by_agent"] = [
            int(v) for v in step_blocked_static_by_agent
        ]
        info["step_moved_by_agent"] = [int(v) for v in step_moved_by_agent]
        info["step_conflict_detected"] = int(step_conflict_detected)
        info["step_conflict_resolved"] = int(step_conflict_resolved)
        info["delivery_count"] = int(self._delivery_count_total)
        info["task_completed"] = int(self._task_completed_total)
        info["steps_since_task_progress"] = int(self._steps_since_task_progress)
        info["agent_delivery_count"] = [int(v) for v in self._delivery_count_by_agent]
        info["agent_task_completed"] = [int(v) for v in self._task_completed_by_agent]
        info["termination_reason"] = termination_reason
        info["conflict_detected"] = int(self._conflict_detected_count)
        info["conflict_resolved"] = int(self._conflict_resolved_count)
        info["conflict_unresolved"] = int(unresolved_conflicts)
        info["active_conflict_episodes"] = int(self._active_conflict_episodes)
        return new_obs, list(rewards), done, truncated, info

    def render(self):
        if not self.renderer:
            from rware.rendering import Viewer
            self.renderer = Viewer(self.grid_size)

        return self.renderer.render(self, return_rgb_array=self.render_mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def get_global_image(
        self,
        image_layers=[
            ImageLayer.SHELVES,
            ImageLayer.GOALS,
        ],
        recompute=False,
        pad_to_shape=None,
    ):
        """
        Get global image observation
        :param image_layers: image layers to include in global image
        :param recompute: bool whether image should be recomputed or taken from last computation
            (for default params, image will be constant for environment so no recomputation needed
             but if agent or request information is included, then should be recomputed)
         :param pad_to_shape: if given than pad environment global image shape into this
             shape (if doesn't fit throw exception)
        """
        if recompute or self.global_image is None:
            layers = []
            for layer_type in image_layers:
                if layer_type == ImageLayer.SHELVES:
                    layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                    # set all occupied shelf cells to 1.0 (instead of shelf ID)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.REQUESTS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for requested_shelf in self.request_queue:
                        layer[requested_shelf.y, requested_shelf.x] = 1.0
                elif layer_type == ImageLayer.AGENTS:
                    layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                    # set all occupied agent cells to 1.0 (instead of agent ID)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.AGENT_DIRECTION:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        agent_direction = ag.dir.value + 1
                        layer[ag.y, ag.x] = float(agent_direction)
                elif layer_type == ImageLayer.AGENT_LOAD:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        if ag.carrying_shelf is not None:
                            layer[ag.y, ag.x] = 1.0
                elif layer_type == ImageLayer.GOALS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for goal_y, goal_x in self.goals:
                        layer[goal_x, goal_y] = 1.0
                elif layer_type == ImageLayer.ACCESSIBLE:
                    layer = np.ones(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        layer[ag.y, ag.x] = 0.0
                    layer[self.obstacles.astype(bool)] = 0.0
                elif layer_type == ImageLayer.AVAILABLE_DIRECTIONS:
                    layer = self._direction_availability_layer()
                else:
                    raise ValueError(f"Unknown image layer type: {layer_type}")
                layers.append(layer)
            self.global_image = np.stack(layers)
            if pad_to_shape is not None:
                padding_dims = [
                    pad_dim - global_dim
                    for pad_dim, global_dim in zip(
                        pad_to_shape, self.global_image.shape
                    )
                ]
                assert all([dim >= 0 for dim in padding_dims])
                pad_before = [pad_dim // 2 for pad_dim in padding_dims]
                pad_after = [
                    pad_dim // 2 if pad_dim % 2 == 0 else pad_dim // 2 + 1
                    for pad_dim in padding_dims
                ]
                self.global_image = np.pad(
                    self.global_image,
                    pad_width=tuple(zip(pad_before, pad_after)),
                    mode="constant",
                    constant_values=0,
                )
        return self.global_image


if __name__ == "__main__":
    env = Warehouse(9, 8, 3, 10, 3, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    from tqdm import tqdm

    # env.render()

    for _ in tqdm(range(1000000)):
        # time.sleep(0.05)
        # env.render()
        actions = env.action_space.sample()
        env.step(actions)
