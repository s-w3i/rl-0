from argparse import ArgumentParser

import gymnasium as gym
import torch
from gymnasium import spaces as gym_spaces

import robotic_warehouse  # noqa: F401
import lbforaging  # noqa: F401

from robotic_warehouse import load_env_config
from a2c import A2C
from wrappers import FlattenObservation, Monitor, RecordEpisodeStatistics, TimeLimit


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--env", default="rware-small-4ag-v1")
    parser.add_argument("--path", default="pretrained/rware-small-4ag")
    parser.add_argument("--time_limit", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--render", action="store_true", help="Render each evaluation step.")
    parser.add_argument("--env-config", default=None)
    return parser.parse_args()


def _gymnasium_env_name(env_id):
    env_id = (env_id or "").strip()
    if env_id.startswith("rware-") and env_id.endswith("-v1"):
        return env_id[:-3] + "-v2"
    return env_id


def make_env(env_name, env_config):
    if env_config:
        config_env_id, config_kwargs = load_env_config(env_config)
        env = gym.make(_gymnasium_env_name(config_env_id), disable_env_checker=True, **config_kwargs)
    else:
        env = gym.make(_gymnasium_env_name(env_name), disable_env_checker=True)
    if isinstance(env.observation_space, gym_spaces.Tuple) and any(
        isinstance(space, (gym_spaces.Dict, gym_spaces.Tuple))
        for space in env.observation_space.spaces
    ):
        env = FlattenObservation(env)
    return env


def main():
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    env = make_env(args.env, args.env_config)
    agents = [
        A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, device)
        for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
    ]
    for agent in agents:
        agent.restore(args.path + f"/agent{agent.agent_id}")

    for ep in range(args.episodes):
        env = make_env(args.env, args.env_config)
        env = Monitor(env, f"seac_eval/{ep + 1}", mode="evaluation")
        env = TimeLimit(env, args.time_limit)
        env = RecordEpisodeStatistics(env)

        obs, _ = env.reset()
        done = False

        while not done:
            obs = [torch.from_numpy(o).float().to(device) for o in obs]
            _, actions, _, _ = zip(
                *[agent.model.act(obs[agent.agent_id], None, None) for agent in agents]
            )
            actions = [
                a.squeeze(0).cpu().numpy().astype("int64") if a.numel() > 1 else int(a.item())
                for a in actions
            ]
            if args.render:
                env.render()
            obs, _, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)

        print("--- Episode Finished ---")
        print(f"Episode rewards: {sum(info['episode_reward'])}")
        print(info)
        print(" --- ")
        env.close()


if __name__ == "__main__":
    main()
