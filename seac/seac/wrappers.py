import math
from collections import deque
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, spaces


class RecordEpisodeStatistics(gym.Wrapper):
    """ Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.n_agents = getattr(self.unwrapped, "n_agents")
        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0
        self.t0 = perf_counter()

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.episode_reward += np.array(reward, dtype=np.float64)
        self.episode_length += 1
        if terminated or truncated:
            info["episode_reward"] = self.episode_reward
            for i, agent_reward in enumerate(self.episode_reward):
                info[f"agent{i}/episode_reward"] = agent_reward
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0

            self.reward_queue.append(self.episode_reward)
            self.length_queue.append(self.episode_length)
        return observation, reward, terminated, truncated, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple([
            spaces.flatten(obs_space, obs)
            for obs_space, obs in zip(self.env.observation_space, observation)
        ])

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self.observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info


class SquashDones(gym.Wrapper):
    r"""Compatibility no-op for older config entries."""

    def step(self, action):
        return self.env.step(action)


class GlobalizeReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = getattr(self.unwrapped, "n_agents")

    def reward(self, reward):
        return self.n_agents * [sum(reward)]


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = self.env.spec.max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
            info = dict(info)
            info["TimeLimit.truncated"] = not terminated
        return observation, reward, terminated, truncated, info

class ClearInfo(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, {}


class Monitor(gym.Wrapper):
    def __init__(self, env, directory, video_callable=None, force=False, uid=None, mode=None):
        super().__init__(env)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
