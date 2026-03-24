import json
from pathlib import Path

import gym
from gym.envs.registration import register, registry
from gym import spaces as gym_spaces
from gymnasium import make as gymnasium_make
from gymnasium.envs.registration import registry as gymnasium_registry
from gymnasium import spaces as gymnasium_spaces

import rware  # noqa: F401
from rware.warehouse import ImageLayer, ObservationType, RewardType


def _convert_space(space):
    if isinstance(space, gymnasium_spaces.Box):
        return gym_spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype,
        )
    if isinstance(space, gymnasium_spaces.Discrete):
        return gym_spaces.Discrete(space.n)
    if isinstance(space, gymnasium_spaces.MultiBinary):
        return gym_spaces.MultiBinary(space.n)
    if isinstance(space, gymnasium_spaces.MultiDiscrete):
        return gym_spaces.MultiDiscrete(space.nvec)
    if isinstance(space, gymnasium_spaces.Tuple):
        return gym_spaces.Tuple(tuple(_convert_space(s) for s in space.spaces))
    if isinstance(space, gymnasium_spaces.Dict):
        return gym_spaces.Dict(
            {key: _convert_space(value) for key, value in space.spaces.items()}
        )
    return space


def _parse_reward_type(value):
    if value is None or isinstance(value, RewardType):
        return value
    name = str(value).strip()
    if "." in name:
        name = name.split(".")[-1]
    return RewardType[name.upper()]


def _parse_observation_type(value):
    if value is None or isinstance(value, ObservationType):
        return value
    name = str(value).strip()
    if "." in name:
        name = name.split(".")[-1]
    return ObservationType[name.upper()]


def _parse_image_layer(value):
    if isinstance(value, ImageLayer):
        return value
    if isinstance(value, int):
        return ImageLayer(value)
    name = str(value).strip()
    if "." in name:
        name = name.split(".")[-1]
    return ImageLayer[name.upper()]


def load_env_config(env_config_path):
    path = Path(env_config_path).expanduser().resolve()
    data = json.loads(path.read_text())
    env_id = str(data.get("env_id") or "").strip()
    kwargs = dict(data.get("kwargs") or {})
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if "reward_type" in kwargs:
        kwargs["reward_type"] = _parse_reward_type(kwargs["reward_type"])
    if "observation_type" in kwargs:
        kwargs["observation_type"] = _parse_observation_type(kwargs["observation_type"])
    if "image_observation_layers" in kwargs and kwargs["image_observation_layers"] is not None:
        kwargs["image_observation_layers"] = [
            _parse_image_layer(layer) for layer in kwargs["image_observation_layers"]
        ]
    return env_id, kwargs


class RwareLegacyGymWrapper(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, gymnasium_id, **kwargs):
        self._env = gymnasium_make(gymnasium_id, disable_env_checker=True, **kwargs)
        self.n_agents = self._env.unwrapped.n_agents
        self.observation_space = _convert_space(self._env.observation_space)
        self.action_space = _convert_space(self._env.action_space)
        self.reward_range = getattr(self._env, "reward_range", None)
        self.spec = None
        self._pending_seed = None

    def seed(self, seed=None):
        self._pending_seed = seed
        return [seed]

    def reset(self, *, seed=None, options=None, **kwargs):
        if seed is None and self._pending_seed is not None:
            seed = self._pending_seed
            self._pending_seed = None
        obs, _ = self._env.reset(seed=seed, options=options, **kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = [bool(terminated or truncated)] * self.n_agents
        if truncated:
            info = dict(info)
            info["TimeLimit.truncated"] = not terminated
        return obs, reward, done, info

    def render(self, mode="human"):
        return self._env.render()

    def close(self):
        return self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def __getattr__(self, name):
        return getattr(self._env, name)


def _register_legacy_envs():
    for env_id in gymnasium_registry.keys():
        if not env_id.startswith("rware-") or not env_id.endswith("-v2"):
            continue
        legacy_id = env_id[:-2] + "v1"
        if legacy_id in registry:
            continue
        register(
            id=legacy_id,
            entry_point="robotic_warehouse:RwareLegacyGymWrapper",
            kwargs={"gymnasium_id": env_id},
            disable_env_checker=True,
        )


_register_legacy_envs()
