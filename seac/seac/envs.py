import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces as gym_spaces

from robotic_warehouse import load_env_config
from wrappers import TimeLimit, Monitor, FlattenObservation


def _gymnasium_env_name(env_id):
    env_id = (env_id or "").strip()
    if env_id.startswith("rware-") and env_id.endswith("-v1"):
        return env_id[:-3] + "-v2"
    return env_id


def _needs_flatten(env):
    obs_space = env.observation_space
    if not isinstance(obs_space, gym_spaces.Tuple):
        return False
    return any(isinstance(space, (gym_spaces.Dict, gym_spaces.Tuple)) for space in obs_space.spaces)


def make_env(env_id, seed, rank, time_limit, wrappers, monitor_dir, env_config=None):
    if env_config:
        config_env_id, config_kwargs = load_env_config(env_config)
        env_name = _gymnasium_env_name(config_env_id)
        env = gym.make(env_name, disable_env_checker=True, **config_kwargs)
    else:
        env = gym.make(_gymnasium_env_name(env_id), disable_env_checker=True)

    obs, info = env.reset(seed=seed + rank)
    if _needs_flatten(env):
        env = FlattenObservation(env)
        env.reset(seed=seed + rank)

    if time_limit:
        env = TimeLimit(env, time_limit)
    for wrapper in wrappers:
        env = wrapper(env)
    if monitor_dir:
        env = Monitor(env, monitor_dir, lambda ep: int(ep == 0), force=True, uid=str(rank))
    return env


class MAGymnasiumVecEnv:
    def __init__(self, envs, device):
        self.envs = envs
        self.num_envs = len(envs)
        self.device = device
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.n_agents = len(self.observation_space)

    def _stack_obs(self, obs_per_env):
        return [
            torch.from_numpy(
                np.stack([np.asarray(obs[agent_idx]) for obs in obs_per_env], axis=0)
            ).float().to(self.device)
            for agent_idx in range(self.n_agents)
        ]

    def reset(self):
        obs_per_env = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_per_env.append(obs)
        return self._stack_obs(obs_per_env)

    def step(self, actions):
        obs_per_env = []
        rewards = []
        dones = []
        infos = []

        for env_idx, env in enumerate(self.envs):
            env_actions = []
            for agent_action in actions:
                act = np.asarray(agent_action[env_idx].detach().cpu().numpy())
                if act.size == 1:
                    env_actions.append(int(act.reshape(-1)[0]))
                else:
                    env_actions.append(act.astype(np.int64))

            obs, reward, terminated, truncated, info = env.step(env_actions)
            done = bool(terminated or truncated)
            if truncated:
                info = dict(info)
                info["TimeLimit.truncated"] = not terminated
            if done:
                terminal_info = dict(info)
                obs, _ = env.reset()
                info = terminal_info

            obs_per_env.append(obs)
            rewards.append(np.asarray(reward, dtype=np.float32))
            dones.append(done)
            infos.append(info)

        return (
            self._stack_obs(obs_per_env),
            torch.from_numpy(np.stack(rewards, axis=0)).float().to(self.device),
            torch.from_numpy(np.asarray(dones, dtype=np.bool_)).to(self.device),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()


def make_vec_envs(
    env_name,
    seed,
    dummy_vecenv,
    parallel,
    time_limit,
    wrappers,
    device,
    monitor_dir=None,
    env_config=None,
):
    envs = [
        make_env(env_name, seed, i, time_limit, wrappers, monitor_dir, env_config)
        for i in range(parallel)
    ]
    return MAGymnasiumVecEnv(envs, device)
